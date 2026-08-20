import argparse
import json
import os
import pickle

import networkx as nx
import numpy as np
from tqdm import tqdm

from dfg_comp import build_parser, get_dfg_adj, load_run_info
from experiment_protocol import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_PRIMARY_THRESHOLD,
    bootstrap_mean_ci,
    confidence_intervals_to_dict,
    protocol_metadata,
    write_protocol,
)
from graph_utils import get_ast_tokens_and_prog_graphs, traverse_node
from utils import make_nodes_unique


DISTANCE_MODES = ('fixed', 'legacy')
SIMILARITY_DIRECTORIES = {
    'fixed': 'similarity',
    'legacy': 'similarity_legacy',
}
DISTANCE_METHODS = {
    'fixed': 'exact_fixed_node_edge_edit_distance',
    'legacy': 'legacy_networkx_optimize_first_candidate',
}
PAPER_NETWORKX_VERSION = '3.0'


def is_identifier_type(node_type):
    """Use tree-sitter's grammatical token type, including language variants."""
    return 'identifier' in node_type


def ast_without_identifiers(artifact, parser, lang):
    """Return the stored motif AST with every identifier-incident edge removed."""
    code = artifact['code']
    code_tokens = artifact['code_tokens']
    byte_code = code.encode('utf-8')
    tree = parser.parse(byte_code)
    if tree.root_node.has_error:
        raise ValueError('Tree-sitter reported a parse error')

    collected_tokens = []
    traverse_node(
        tree.root_node,
        collected_tokens,
        byte_code,
        include_comments=False,
    )
    ast_info, _, is_error = get_ast_tokens_and_prog_graphs(
        collected_tokens,
        code_tokens,
        code_tokens,
        byte_code,
        (0, 0),
    )
    aligned_tokens = [info['token'] for info in ast_info]
    if is_error or aligned_tokens != code_tokens:
        raise ValueError('AST tokens do not exactly match artifact tokens')

    nonidentifier = np.asarray(
        [not is_identifier_type(info['type']) for info in ast_info],
        dtype=bool,
    )
    keep_edges = nonidentifier[:, None] & nonidentifier[None, :]
    return (np.asarray(artifact['ast_graph']) != 0) & keep_edges, [
        info['type'] for info in ast_info
    ]


def normalized_fixed_node_edge_distance(prediction, truth):
    """Exact edge-edit distance under the stored token-node alignment."""
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError(
            f'Graph shape mismatch: {prediction.shape} versus {truth.shape}'
        )
    if prediction.shape[0] == 0:
        return 0.0
    return float(np.count_nonzero(prediction != truth) / prediction.shape[0])


def distances_for_layers(model_graphs, truth_graphs, threshold):
    """Vectorized exact fixed-node edge edit distance for layers and heads."""
    predictions = np.asarray(model_graphs) > threshold
    num_nodes = predictions.shape[-1]
    output = {}
    for name, truth in truth_graphs.items():
        truth = np.asarray(truth, dtype=bool)
        if truth.shape != predictions.shape[-2:]:
            raise ValueError(
                f'{name} shape {truth.shape} does not match attention '
                f'shape {predictions.shape[-2:]}'
            )
        output[name] = np.count_nonzero(
            predictions != truth[None, None, :, :],
            axis=(-2, -1),
        ) / num_nodes
    return output


def legacy_node_match(left, right):
    """Historical node comparison from the paper's similarity script."""
    return left['name'] == right['name']


def legacy_edge_substitution_cost(left, right):
    """The paper ignored edge attributes during GED calculation."""
    return 0


def legacy_edge_deletion_cost(edge):
    return 1


def legacy_edge_insertion_cost(edge):
    return 1


def legacy_unique_token_names(tokens):
    """Reproduce the paper's in-place token-name disambiguation safely."""
    return make_nodes_unique(list(tokens))


def _legacy_networkx_distance(model_graph, truth_graph, num_nodes):
    candidates = nx.optimize_graph_edit_distance(
        model_graph,
        truth_graph,
        node_match=legacy_node_match,
        edge_del_cost=legacy_edge_deletion_cost,
        edge_ins_cost=legacy_edge_insertion_cost,
        edge_subst_cost=legacy_edge_substitution_cost,
    )
    try:
        first_candidate = next(candidates)
    except StopIteration as exc:
        raise ValueError('NetworkX did not produce a GED candidate') from exc
    return float(first_candidate / num_nodes)


def normalized_legacy_networkx_distance(prediction, truth, tokens):
    """Return the first normalized GED candidate exactly as in the paper.

    This intentionally does not claim to be the globally optimal GED. The
    historical implementation called ``optimize_graph_edit_distance`` and
    immediately consumed its first yielded candidate.
    """
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError(
            f'Graph shape mismatch: {prediction.shape} versus {truth.shape}'
        )
    if prediction.shape[0] != len(tokens):
        raise ValueError(
            f'Graph has {prediction.shape[0]} nodes but received '
            f'{len(tokens)} tokens'
        )
    if prediction.shape[0] == 0:
        return 0.0

    node_attributes = {
        index: {'name': name}
        for index, name in enumerate(legacy_unique_token_names(tokens))
    }
    model_graph = nx.from_numpy_array(
        prediction, create_using=nx.DiGraph
    )
    truth_graph = nx.from_numpy_array(truth, create_using=nx.DiGraph)
    nx.set_node_attributes(model_graph, node_attributes)
    nx.set_node_attributes(truth_graph, node_attributes)
    return _legacy_networkx_distance(
        model_graph, truth_graph, prediction.shape[0]
    )


def legacy_distances_for_layers(
    model_graphs, truth_graphs, threshold, tokens
):
    """Paper-compatible NetworkX GED estimates for layers and heads."""
    predictions = np.asarray(model_graphs) > threshold
    if predictions.ndim != 4:
        raise ValueError(
            'Legacy GED expects attention shaped as layers, heads, nodes, nodes'
        )
    num_nodes = predictions.shape[-1]
    if predictions.shape[-2] != num_nodes or len(tokens) != num_nodes:
        raise ValueError('Attention and token dimensions do not match')
    if num_nodes == 0:
        return {
            name: np.zeros(predictions.shape[:2], dtype=np.float64)
            for name in truth_graphs
        }

    node_attributes = {
        index: {'name': name}
        for index, name in enumerate(legacy_unique_token_names(tokens))
    }
    networkx_truths = {}
    for name, truth in truth_graphs.items():
        truth = np.asarray(truth)
        if truth.shape != predictions.shape[-2:]:
            raise ValueError(
                f'{name} shape {truth.shape} does not match attention '
                f'shape {predictions.shape[-2:]}'
            )
        graph = nx.from_numpy_array(truth, create_using=nx.DiGraph)
        nx.set_node_attributes(graph, node_attributes)
        networkx_truths[name] = graph

    output = {
        name: np.zeros(predictions.shape[:2], dtype=np.float64)
        for name in truth_graphs
    }
    for layer in range(predictions.shape[0]):
        for head in range(predictions.shape[1]):
            model_graph = nx.from_numpy_array(
                predictions[layer, head], create_using=nx.DiGraph
            )
            nx.set_node_attributes(model_graph, node_attributes)
            for name, truth_graph in networkx_truths.items():
                output[name][layer, head] = _legacy_networkx_distance(
                    model_graph, truth_graph, num_nodes
                )
    return output


def similarity_directory(distance_mode):
    try:
        return SIMILARITY_DIRECTORIES[distance_mode]
    except KeyError as exc:
        raise ValueError(
            f'Unknown distance mode {distance_mode!r}; expected one of '
            f'{DISTANCE_MODES}'
        ) from exc


def evaluate_similarity(
    graphs_dir,
    code_file,
    save_dir,
    exp_name,
    parser,
    lang='python',
    threshold=0.05,
    layers=None,
    max_artifacts=None,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    distance_mode='fixed',
):
    similarity_subdirectory = similarity_directory(distance_mode)
    if distance_mode == 'legacy' and nx.__version__ != PAPER_NETWORKX_VERSION:
        print(
            '[LEGACY GED WARNING] The paper environment declared NetworkX '
            f'{PAPER_NETWORKX_VERSION}, but this run uses {nx.__version__}. '
            'The historical API call is reproduced and the version is '
            'recorded; use the Python reference comparison as an empirical '
            'compatibility check.'
        )
    run_info = load_run_info(graphs_dir, lang, code_file=code_file)
    num_layers = run_info['num_layers']
    num_heads = run_info['num_heads']
    layers = list(range(num_layers)) if layers is None else list(layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError('At least one unique layer is required')
    if any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f'Layers must be between 0 and {num_layers - 1}')
    if max_artifacts is not None and max_artifacts <= 0:
        raise ValueError('--max_artifacts must be positive')

    artifacts = run_info['artifacts']
    if max_artifacts is not None:
        artifacts = artifacts[:max_artifacts]

    graph_names = ('ast', 'dfg', 'common', 'ast_wo_identifiers')
    totals = {
        name: np.zeros((len(layers), num_heads), dtype=np.float64)
        for name in graph_names
    }
    failures = []
    evaluated = 0
    selected_records = []
    program_results = []

    for artifact_name in tqdm(artifacts):
        try:
            with open(os.path.join(graphs_dir, artifact_name), 'rb') as handle:
                artifact = pickle.load(handle)
            tokens = artifact['code_tokens']
            model_graphs = np.asarray(artifact['model_graphs'])
            expected_shape = (num_layers, num_heads, len(tokens), len(tokens))
            if model_graphs.shape != expected_shape:
                raise ValueError(
                    f'Attention shape {model_graphs.shape}, expected {expected_shape}'
                )

            ast = np.asarray(artifact['ast_graph'], dtype=bool)
            ast_wo_identifiers, token_types = ast_without_identifiers(
                artifact, parser, lang
            )
            dfg, dfg_tokens = get_dfg_adj(
                artifact['code'],
                parser,
                lang=lang,
                expected_tokens=tokens,
            )
            if dfg_tokens != tokens:
                raise ValueError('Aligned DFG tokens differ from artifact tokens')
            truth_graphs = {
                'ast': ast,
                'dfg': np.asarray(dfg, dtype=bool),
                'common': ast | np.asarray(dfg, dtype=bool),
                'ast_wo_identifiers': ast_wo_identifiers,
            }
            selected_model_graphs = model_graphs[np.asarray(layers)]
            if distance_mode == 'legacy':
                values = legacy_distances_for_layers(
                    selected_model_graphs,
                    truth_graphs,
                    threshold,
                    tokens,
                )
            else:
                values = distances_for_layers(
                    selected_model_graphs, truth_graphs, threshold
                )
            for name in graph_names:
                totals[name] += values[name]
            program_results.append(values)
            evaluated += 1
            selected_records.append({
                'artifact': artifact_name,
                'sample_index': artifact.get('sample_index'),
                'source_index': artifact.get('source_index'),
                'file_name': artifact.get('file_name'),
                'num_tokens': len(tokens),
                'num_identifiers': sum(is_identifier_type(t) for t in token_types),
            })
        except Exception as exc:
            failures.append({
                'artifact': artifact_name,
                'reason': f'{type(exc).__name__}: {exc}',
            })

    if evaluated == 0:
        raise ValueError('No graph artifacts could be evaluated')

    output_dir = os.path.join(save_dir, similarity_subdirectory)
    if exp_name:
        output_dir = os.path.join(output_dir, exp_name)
    output_dir = os.path.join(output_dir, run_info['model_name'])
    os.makedirs(output_dir, exist_ok=True)

    program_arrays = {
        name: np.stack([result[name] for result in program_results])
        for name in graph_names
    }
    program_arrays['ast_minus_ast_wo_identifiers'] = (
        program_arrays['ast'] - program_arrays['ast_wo_identifiers']
    )
    confidence_limits = {
        name: bootstrap_mean_ci(
            values,
            num_resamples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        for name, values in program_arrays.items()
    }
    program_metrics_path = os.path.join(
        output_dir, f'program_metrics_threshold_{threshold}.npz'
    )
    np.savez_compressed(
        program_metrics_path,
        **program_arrays,
        artifact=np.asarray([
            record['artifact'] for record in selected_records
        ]),
        sample_index=np.asarray([
            -1 if record['sample_index'] is None else record['sample_index']
            for record in selected_records
        ], dtype=np.int64),
        source_index=np.asarray([
            -1 if record['source_index'] is None else record['source_index']
            for record in selected_records
        ], dtype=np.int64),
        file_name=np.asarray([
            '' if record['file_name'] is None else record['file_name']
            for record in selected_records
        ]),
        layers=np.asarray(layers, dtype=np.int64),
        threshold=np.asarray(threshold),
    )
    analysis_name = (
        'ast_dfg_legacy_networkx_ged'
        if distance_mode == 'legacy'
        else 'ast_dfg_fixed_node_distance'
    )
    protocol = protocol_metadata(
        analysis_name,
        model=run_info['model_name'],
        language=lang,
        primary_threshold=threshold,
        thresholds=[threshold],
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    if distance_mode == 'legacy':
        protocol.update({
            'distance_mode': distance_mode,
            'distance': (
                'first candidate yielded by '
                'networkx.optimize_graph_edit_distance'
            ),
            'distance_normalization': 'first candidate divided by nodes',
            'paper_compatible': True,
            'networkx_version': nx.__version__,
            'paper_reference_networkx_version': PAPER_NETWORKX_VERSION,
            'networkx_version_matches_declared_paper_environment': (
                nx.__version__ == PAPER_NETWORKX_VERSION
            ),
            'node_labels': (
                'dataset tokens disambiguated with the historical '
                'make_nodes_unique function'
            ),
            'node_match': "node attribute 'name' equality",
            'edge_costs': {
                'deletion': 1,
                'insertion': 1,
                'substitution': 0,
            },
            'known_methodological_limitation': (
                'The first yielded value is a legacy GED estimate and is not '
                'guaranteed to be the globally minimal graph edit distance.'
            ),
        })
    else:
        protocol.update({
            'distance_mode': distance_mode,
            'distance': 'exact fixed-node edge edit distance',
            'distance_normalization': (
                'edge insertions plus deletions divided by number of nodes'
            ),
            'paper_compatible': False,
        })
    protocol_path = write_protocol(output_dir, protocol)

    cohort = {
        'status': 'complete',
        'distance_mode': distance_mode,
        'method': DISTANCE_METHODS[distance_mode],
        'paper_compatible': distance_mode == 'legacy',
        'normalization': protocol['distance_normalization'],
        'networkx_version': (
            nx.__version__ if distance_mode == 'legacy' else None
        ),
        'paper_reference_networkx_version': (
            PAPER_NETWORKX_VERSION if distance_mode == 'legacy' else None
        ),
        'networkx_version_matches_declared_paper_environment': (
            nx.__version__ == PAPER_NETWORKX_VERSION
            if distance_mode == 'legacy' else None
        ),
        'model': run_info['model_name'],
        'language': lang,
        'threshold': threshold,
        'layers': layers,
        'num_heads': num_heads,
        'num_requested_artifacts': len(artifacts),
        'num_evaluated': evaluated,
        'num_failures': len(failures),
        'num_selected_in_graph_run': run_info['selected_count'],
        'attempted_fraction_of_graph_run': (
            len(artifacts) / run_info['selected_count']
            if run_info['selected_count'] else None
        ),
        'evaluation_success_rate': evaluated / len(artifacts),
        'end_to_end_coverage': (
            evaluated / run_info['selected_count']
            if run_info['selected_count'] else None
        ),
        'graph_manifest': os.path.abspath(run_info['manifest_path']),
        'program_metrics_file': os.path.abspath(program_metrics_path),
        'evaluation_protocol': protocol_path,
        'bootstrap': {
            'unit': 'program',
            'method': 'percentile',
            'samples': bootstrap_samples,
            'confidence_level': confidence_level,
            'seed': bootstrap_seed,
        },
        'artifacts': selected_records,
        'failures': failures,
    }
    cohort_path = os.path.join(output_dir, 'similarity_manifest.json')
    with open(cohort_path, 'w') as handle:
        json.dump(cohort, handle, indent=2)

    outputs = []
    for layer_index, layer in enumerate(layers):
        cost = {
            name: {
                str(head): float(totals[name][layer_index, head] / evaluated)
                for head in range(num_heads)
            }
            for name in graph_names
        }
        output = {
            **cost,
            'ast_minus_ast_wo_identifiers': {
                str(head): float(
                    cost['ast'][str(head)]
                    - cost['ast_wo_identifiers'][str(head)]
                )
                for head in range(num_heads)
            },
            'model': run_info['model_name'],
            'language': lang,
            'layer': layer,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'threshold': threshold,
            'distance_mode': distance_mode,
            'networkx_version': cohort['networkx_version'],
            'paper_reference_networkx_version': (
                cohort['paper_reference_networkx_version']
            ),
            'networkx_version_matches_declared_paper_environment': (
                cohort[
                    'networkx_version_matches_declared_paper_environment'
                ]
            ),
            'confidence_intervals': {
                name: confidence_intervals_to_dict(
                    confidence_limits[name][0][layer_index]
                    if confidence_limits[name][0] is not None else None,
                    confidence_limits[name][1][layer_index]
                    if confidence_limits[name][1] is not None else None,
                )
                for name in program_arrays
            },
            'bootstrap': cohort['bootstrap'],
            'num_evaluated': evaluated,
            'num_attempted': len(artifacts),
            'method': cohort['method'],
            'similarity_manifest': os.path.abspath(cohort_path),
            'program_metrics_file': os.path.abspath(program_metrics_path),
            'evaluation_protocol': protocol_path,
        }
        output_path = os.path.join(
            output_dir, f'layer_{layer}_threshold_{threshold}.json'
        )
        with open(output_path, 'w') as handle:
            json.dump(output, handle, indent=2)
        outputs.append(output)

    print(
        f'[SIMILARITY COMPLETE] {lang} | mode {distance_mode} | '
        f'{evaluated}/{len(artifacts)} artifacts | {len(layers)} layers | '
        f'threshold {threshold}'
    )
    return outputs


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--graphs_dir', required=True)
    cli.add_argument('--code_file', default='attention/exp_data/exp_0.jsonl')
    cli.add_argument('--save_dir', required=True)
    cli.add_argument('--exp_name')
    cli.add_argument('--layer', default=-1, type=int)
    cli.add_argument('--layers', nargs='+', type=int)
    cli.add_argument('--all_layers', action='store_true')
    cli.add_argument('--num_layers', type=int)
    cli.add_argument(
        '--threshold', default=DEFAULT_PRIMARY_THRESHOLD, type=float
    )
    cli.add_argument('--max_artifacts', type=int)
    cli.add_argument(
        '--bootstrap_samples', type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    cli.add_argument(
        '--confidence_level', type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    cli.add_argument(
        '--bootstrap_seed', type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    cli.add_argument(
        '--lang',
        default='python',
        choices=['python', 'java', 'go', 'javascript'],
    )
    cli.add_argument(
        '--distance_mode',
        choices=DISTANCE_MODES,
        default='fixed',
        help=(
            "'legacy' reproduces the paper's first NetworkX GED candidate; "
            "'fixed' keeps the newer exact fixed-node edge distance."
        ),
    )
    args = cli.parse_args()

    run_info = load_run_info(args.graphs_dir, args.lang, args.code_file)
    if args.num_layers is not None and args.num_layers != run_info['num_layers']:
        raise ValueError(
            f'--num_layers={args.num_layers} does not match stored '
            f"layer count {run_info['num_layers']}"
        )
    if args.all_layers:
        layers = list(range(run_info['num_layers']))
    elif args.layers is not None:
        layers = args.layers
    else:
        layers = [run_info['num_layers'] - 1 if args.layer == -1 else args.layer]

    evaluate_similarity(
        args.graphs_dir,
        args.code_file,
        args.save_dir,
        args.exp_name,
        build_parser(args.lang),
        lang=args.lang,
        threshold=args.threshold,
        layers=layers,
        max_artifacts=args.max_artifacts,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        distance_mode=args.distance_mode,
    )


if __name__ == '__main__':
    main()
