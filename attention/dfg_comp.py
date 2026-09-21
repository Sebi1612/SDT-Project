import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm
from tree_sitter import Language, Parser

from comment_preprocessing import strip_comments
from dfg.DFG import DFG_go, DFG_java, DFG_javascript, DFG_python
from dfg.utils import (
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
)
from utils import load_codesearchnet

from experiment_protocol import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_PRIMARY_THRESHOLD,
    DEFAULT_THRESHOLDS,
    bootstrap_mean_ci,
    confidence_intervals_to_dict,
    protocol_metadata,
    threshold_index,
    write_protocol,
)


class SafeDict(dict):
    """Return an ignored sentinel for parser nodes absent from token indexes."""

    def __missing__(self, key):
        return (-1, '__IGNORED_TOKEN__')


def align_dfg_to_tokens(dfg_adj, source_tokens, target_tokens):
    """Project a DFG across equivalent tree-sitter/dataset token groupings."""
    if source_tokens == target_tokens:
        return dfg_adj, source_tokens

    source_to_target = {}
    source_index = 0
    target_index = 0

    while source_index < len(source_tokens) and target_index < len(target_tokens):
        source_group = [source_index]
        target_group = [target_index]
        source_text = source_tokens[source_index]
        target_text = target_tokens[target_index]

        while source_text != target_text:
            # CodeSearchNet Java represents a character literal such as '.' as
            # two quote tokens and omits the character contents.
            if (
                len(source_group) == 1
                and len(source_text) >= 2
                and source_text[0] == source_text[-1] == "'"
                and target_text == "'"
                and target_index + 1 < len(target_tokens)
                and target_tokens[target_index + 1] == "'"
            ):
                target_index += 1
                target_group.append(target_index)
                target_text = source_text
                continue

            if len(source_text) < len(target_text):
                source_index += 1
                if source_index >= len(source_tokens):
                    raise ValueError(
                        'Could not align tree-sitter tokens to dataset tokens'
                    )
                source_group.append(source_index)
                source_text += source_tokens[source_index]
            elif len(target_text) < len(source_text):
                target_index += 1
                if target_index >= len(target_tokens):
                    raise ValueError(
                        'Could not align tree-sitter tokens to dataset tokens'
                    )
                target_group.append(target_index)
                target_text += target_tokens[target_index]
            else:
                raise ValueError(
                    f'Token text differs: tree-sitter {source_text!r}, '
                    f'dataset {target_text!r}'
                )

        for index in source_group:
            source_to_target[index] = target_group

        source_index += 1
        target_index += 1

    if source_index != len(source_tokens) or target_index != len(target_tokens):
        raise ValueError('Token streams have different trailing content')

    aligned_adj = np.zeros((len(target_tokens), len(target_tokens)))
    source_rows, source_cols = np.nonzero(dfg_adj)
    for source_row, source_col in zip(source_rows, source_cols):
        for target_row in source_to_target[source_row]:
            for target_col in source_to_target[source_col]:
                aligned_adj[target_row, target_col] = dfg_adj[
                    source_row,
                    source_col,
                ]

    return aligned_adj, target_tokens


def align_dfg_to_python_spans(
    dfg_adj,
    token_indexes,
    code_string,
    root_node,
    target_tokens,
):
    """Project Python DFG nodes using the AST/token alignment used at extraction.

    CodeSearchNet's Python token stream omits function docstrings and optional
    semicolons, but can retain equivalent strings in other statement contexts.
    Removing text before parsing changes source positions and cannot reproduce
    those distinctions reliably.  The extraction pipeline already proves an
    exact mapping from parser spans to dataset tokens, so reuse that mapping
    here and discard only DFG nodes outside the retained spans.
    """
    from graph_utils import get_ast_tokens_and_prog_graphs, traverse_node

    byte_code = code_string.encode('utf-8')
    collected = []
    traverse_node(
        root_node,
        collected,
        byte_code,
        include_comments=False,
    )
    target_info, _, is_error = get_ast_tokens_and_prog_graphs(
        collected,
        target_tokens,
        target_tokens,
        byte_code,
        (0, 0),
    )
    if is_error or [item['token'] for item in target_info] != target_tokens:
        raise ValueError(
            'Could not reproduce exact Python AST/dataset token alignment'
        )

    line_starts = []
    cursor = 0
    for line in code_string.splitlines(keepends=True):
        line_starts.append(cursor)
        cursor += len(line.encode('utf-8'))
    if not line_starts:
        line_starts.append(0)

    def absolute_byte(point):
        row, column = point
        return line_starts[row] + column

    source_to_target = {}
    target_index = 0
    for source_index, (start_point, end_point) in enumerate(token_indexes):
        start_byte = absolute_byte(start_point)
        end_byte = absolute_byte(end_point)
        while (
            target_index < len(target_info)
            and target_info[target_index]['end_byte'] <= start_byte
        ):
            target_index += 1
        if target_index >= len(target_info):
            break
        target = target_info[target_index]
        if (
            target['start_byte'] <= start_byte
            and end_byte <= target['end_byte']
        ):
            source_to_target[source_index] = target_index

    aligned_adj = np.zeros((len(target_tokens), len(target_tokens)))
    source_rows, source_cols = np.nonzero(dfg_adj)
    for source_row, source_col in zip(source_rows, source_cols):
        target_row = source_to_target.get(int(source_row))
        target_col = source_to_target.get(int(source_col))
        if target_row is not None and target_col is not None:
            aligned_adj[target_row, target_col] = dfg_adj[
                source_row,
                source_col,
            ]
    return aligned_adj, target_tokens


def get_dfg_adj(
    code_string,
    parser,
    lang='python',
    expected_tokens=None,
    typed=False,
):
    """Build a GraphCodeBERT-style DFG aligned to dataset tokens.

    By default the result is binary for exact graph comparison. With
    ``typed=True``, ComesFrom edges are 1 and ComputedFrom edges are -1,
    matching the labels emitted by each repository extractor for DirectProbe.
    The extractors do not assign these labels identically to equivalent source
    constructs, so signed labels must not be compared across languages.
    """
    include_comments = False
    if expected_tokens is None:
        if lang == 'python':
            code_string = remove_comments_and_docstrings(code_string, lang)
        else:
            code_string, _ = strip_comments(code_string, parser)

    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    if root_node.has_error and expected_tokens is None:
        raise ValueError('Tree-sitter reported a parse error')

    token_indexes = tree_to_token_index(
        root_node,
        include_comments=False,
        atomic_string_literals=(lang == 'java'),
    )
    source_lines = code_string.split('\n')
    code_tokens = [
        index_to_code_token(index, source_lines) for index in token_indexes
    ]
    index_to_code = SafeDict()
    for index, (token_index, token) in enumerate(
        zip(token_indexes, code_tokens)
    ):
        index_to_code[token_index] = (index, token)

    extractor = {
        'python': DFG_python,
        'java': DFG_java,
        'go': DFG_go,
        'javascript': DFG_javascript,
    }.get(lang)
    if extractor is None:
        raise ValueError(f'Unsupported language: {lang}')
    dfg, _ = extractor(root_node, index_to_code, {})

    dfg_adj = np.zeros((len(code_tokens), len(code_tokens)))
    for edges in sorted(dfg, key=lambda entry: entry[1]):
        row = edges[1]
        relation_value = -1 if typed and edges[2] == 'computedFrom' else 1
        for column in edges[-1]:
            if row >= 0 and column >= 0:
                dfg_adj[row, column] = relation_value

    # Remove zero-width parser artifacts. Explicit Go semicolons have a
    # non-empty span and remain because CodeSearchNet includes them.
    indexes_to_remove = [
        index for index, token in enumerate(code_tokens) if token.strip() == ''
    ]
    if indexes_to_remove:
        dfg_adj = np.delete(dfg_adj, indexes_to_remove, axis=0)
        dfg_adj = np.delete(dfg_adj, indexes_to_remove, axis=1)
        code_tokens = [
            token
            for index, token in enumerate(code_tokens)
            if index not in indexes_to_remove
        ]

    if expected_tokens is not None:
        if lang == 'python':
            retained_indexes = [
                token_index
                for index, token_index in enumerate(token_indexes)
                if index not in indexes_to_remove
            ]
            dfg_adj, code_tokens = align_dfg_to_python_spans(
                dfg_adj,
                retained_indexes,
                code_string,
                root_node,
                expected_tokens,
            )
        else:
            dfg_adj, code_tokens = align_dfg_to_tokens(
                dfg_adj,
                code_tokens,
                expected_tokens,
            )

    return dfg_adj, code_tokens


def load_run_info(graph_loc, lang, code_file=None):
    manifest_path = os.path.join(graph_loc, 'graph_manifest.json')
    legacy_codes = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get('status') not in {None, 'complete'}:
            raise ValueError(
                f"Graph manifest is not complete: {manifest.get('status')!r}"
            )
        if manifest.get('lang') != lang:
            raise ValueError(
                f"Graph manifest language is {manifest.get('lang')!r}, "
                f'but --lang is {lang!r}'
            )
        artifacts = manifest.get('artifacts', [])
        selected_count = manifest.get('selected_num_codes')
        model_name = manifest.get('model')
    else:
        if code_file is None:
            raise ValueError('--code_file is required for legacy graph artifacts')
        manifest = None
        artifacts = sorted(
            name for name in os.listdir(graph_loc) if name.endswith('.pkl')
        )
        selected_count = len(artifacts)
        model_name = None
        legacy_codes = load_codesearchnet(code_file)
        print(
            f'Warning: {manifest_path} does not exist; evaluating all '
            f'{len(artifacts)} pickle files in the directory.'
        )

    if not artifacts:
        raise ValueError(f'No graph artifacts found in {graph_loc}')

    with open(os.path.join(graph_loc, artifacts[0]), 'rb') as graph_file:
        first_artifact = pickle.load(graph_file)
    model_graphs = first_artifact['model_graphs']
    if model_graphs.ndim != 4:
        raise ValueError(
            f'Expected a four-dimensional attention tensor, got {model_graphs.shape}'
        )

    if not model_name:
        model_name = os.path.basename(os.path.normpath(graph_loc))

    return {
        'manifest': manifest,
        'manifest_path': manifest_path,
        'artifacts': artifacts,
        'selected_count': selected_count,
        'model_name': model_name,
        'language': lang,
        'num_layers': model_graphs.shape[0],
        'num_heads': model_graphs.shape[1],
        'legacy_codes': legacy_codes,
    }


def resolve_legacy_source(artifact, legacy_codes):
    matches = [
        code
        for code in legacy_codes
        if code['code_file'] == artifact['file_name']
        and code['code_tokens'] == artifact['code_tokens']
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one legacy dataset match, found {len(matches)}"
        )
    return matches[0]['code']


def metrics_for_layers(model_graphs, dfg_graph, layers, thresholds):
    """Compute original macro edge metrics for selected layers and all heads."""
    layer_graphs = model_graphs[np.asarray(layers)]
    truth = dfg_graph == 1
    truth_count = truth.sum()
    shape = (len(layers), model_graphs.shape[1], len(thresholds))
    f_scores = np.zeros(shape, dtype=np.float64)
    recalls = np.zeros(shape, dtype=np.float64)
    precisions = np.zeros(shape, dtype=np.float64)

    for threshold_index, threshold in enumerate(thresholds):
        predictions = layer_graphs > threshold
        true_positives = np.logical_and(
            predictions,
            truth[None, None, :, :],
        ).sum(axis=(-2, -1))
        predicted_count = predictions.sum(axis=(-2, -1))

        precisions[:, :, threshold_index] = np.divide(
            true_positives,
            predicted_count,
            out=np.zeros(true_positives.shape, dtype=np.float64),
            where=predicted_count != 0,
        )
        if truth_count:
            recalls[:, :, threshold_index] = true_positives / truth_count
        precision_recall_sum = (
            precisions[:, :, threshold_index]
            + recalls[:, :, threshold_index]
        )
        f_scores[:, :, threshold_index] = np.divide(
            2
            * precisions[:, :, threshold_index]
            * recalls[:, :, threshold_index],
            precision_recall_sum,
            out=np.zeros(true_positives.shape, dtype=np.float64),
            where=precision_recall_sum != 0,
        )

    return f_scores, recalls, precisions


def metric_array_to_dict(values, thresholds):
    return {
        head: {
            threshold: float(values[head, threshold_index])
            for threshold_index, threshold in enumerate(thresholds)
        }
        for head in range(values.shape[0])
    }


def evaluate_dfg_stats(
    code_file,
    graph_loc,
    save_dir,
    layers,
    exp_name,
    parser,
    lang='python',
    thresholds=None,
    primary_threshold=DEFAULT_PRIMARY_THRESHOLD,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
):
    """Align each DFG once, then evaluate every requested attention layer."""
    thresholds = list(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if len(thresholds) != len(set(thresholds)):
        raise ValueError('Threshold values must be unique')
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError('Threshold values must be non-negative')
    primary_index = threshold_index(thresholds, primary_threshold)

    run_info = load_run_info(graph_loc, lang, code_file=code_file)
    num_layers = run_info['num_layers']
    num_heads = run_info['num_heads']
    normalized_layers = [num_layers - 1 if layer == -1 else layer for layer in layers]
    if len(normalized_layers) != len(set(normalized_layers)):
        raise ValueError('Layer values must be unique')
    for layer in normalized_layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f'Wrong layer index {layer}; expected 0 to {num_layers - 1}'
            )

    totals = [
        np.zeros(
            (len(normalized_layers), num_heads, len(thresholds)),
            dtype=np.float64,
        )
        for _ in range(3)
    ]
    attempted = 0
    aligned = 0
    empty_dfgs = 0
    total_dfg_edges = 0
    total_dfg_density = 0.0
    failures = []
    program_results = []
    program_records = []

    for artifact_name in tqdm(run_info['artifacts']):
        artifact_path = os.path.join(graph_loc, artifact_name)
        with open(artifact_path, 'rb') as graph_file:
            artifact = pickle.load(graph_file)
        attempted += 1

        code_tokens = artifact['code_tokens']
        code_string = artifact.get('code')
        if code_string is None:
            try:
                code_string = resolve_legacy_source(
                    artifact,
                    run_info['legacy_codes'],
                )
            except Exception as exc:
                failures.append({
                    'artifact': artifact_name,
                    'file_name': artifact.get('file_name'),
                    'reason': f'{type(exc).__name__}: {exc}',
                })
                continue

        try:
            dfg_graph, dfg_tokens = get_dfg_adj(
                code_string,
                parser,
                lang=lang,
                expected_tokens=code_tokens,
            )
            model_graphs = artifact['model_graphs']
            expected_shape = (
                num_layers,
                num_heads,
                len(code_tokens),
                len(code_tokens),
            )
            if model_graphs.shape != expected_shape:
                raise ValueError(
                    f'Attention shape {model_graphs.shape}, expected {expected_shape}'
                )
            if dfg_graph.shape != (len(code_tokens), len(code_tokens)):
                raise ValueError(
                    f'DFG shape {dfg_graph.shape}, expected '
                    f'{(len(code_tokens), len(code_tokens))}'
                )
            if dfg_tokens != code_tokens:
                raise ValueError('Aligned DFG tokens differ from dataset tokens')
        except Exception as exc:
            failures.append({
                'artifact': artifact_name,
                'sample_index': artifact.get('sample_index'),
                'source_index': artifact.get('source_index'),
                'file_name': artifact.get('file_name'),
                'reason': f'{type(exc).__name__}: {exc}',
            })
            continue

        program_metrics = metrics_for_layers(
            model_graphs,
            dfg_graph,
            normalized_layers,
            thresholds,
        )
        for total, program_metric in zip(totals, program_metrics):
            total += program_metric
        program_results.append(tuple(
            metric[:, :, primary_index] for metric in program_metrics
        ))
        program_records.append({
            'artifact': artifact_name,
            'sample_index': artifact.get('sample_index'),
            'source_index': artifact.get('source_index'),
            'file_name': artifact.get('file_name'),
        })

        edge_count = int(dfg_graph.sum())
        total_dfg_edges += edge_count
        total_dfg_density += edge_count / dfg_graph.size if dfg_graph.size else 0
        empty_dfgs += edge_count == 0
        aligned += 1

    if aligned == 0:
        raise ValueError(
            'No valid aligned DFGs were found. Check the graph directory, '
            'language, parser, and token alignment.'
        )

    averaged = [total / aligned for total in totals]
    alignment_rate = aligned / attempted if attempted else 0.0
    selected_count = run_info['selected_count']
    end_to_end_rate = (
        aligned / selected_count if selected_count else alignment_rate
    )

    output_dir = os.path.join(save_dir, 'dfg')
    if exp_name is not None:
        output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    model_name = run_info['model_name']
    failure_path = os.path.join(output_dir, f'{model_name}_dfg_failures.json')
    with open(failure_path, 'w') as failure_file:
        json.dump(failures, failure_file, indent=2)

    metric_names = ('fscore', 'recall', 'precision')
    program_arrays = {
        name: np.stack([
            result[metric_index]
            for result in program_results
        ])
        for metric_index, name in enumerate(metric_names)
    }
    confidence_limits = {}
    for metric_name, values in program_arrays.items():
        confidence_limits[metric_name] = bootstrap_mean_ci(
            values,
            num_resamples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )

    layer_label = '-'.join(str(layer) for layer in normalized_layers)
    program_metrics_path = os.path.join(
        output_dir, f'{model_name}_layers_{layer_label}_program_metrics.npz'
    )
    np.savez_compressed(
        program_metrics_path,
        **program_arrays,
        artifact=np.asarray([record['artifact'] for record in program_records]),
        sample_index=np.asarray(
            [
                -1 if record['sample_index'] is None else record['sample_index']
                for record in program_records
            ],
            dtype=np.int64,
        ),
        source_index=np.asarray(
            [
                -1 if record['source_index'] is None else record['source_index']
                for record in program_records
            ],
            dtype=np.int64,
        ),
        file_name=np.asarray(
            [
                '' if record['file_name'] is None else record['file_name']
                for record in program_records
            ]
        ),
        layers=np.asarray(normalized_layers, dtype=np.int64),
        primary_threshold=np.asarray(primary_threshold),
    )
    protocol = protocol_metadata(
        'dfg_attention_overlap',
        model=model_name,
        language=lang,
        primary_threshold=primary_threshold,
        thresholds=thresholds,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    protocol_path = write_protocol(output_dir, protocol)

    outputs = []
    for layer_index, layer in enumerate(normalized_layers):
        f_scores, recalls, precisions = [
            metric_array_to_dict(values[layer_index], thresholds)
            for values in averaged
        ]
        output = {
            'fscore': f_scores,
            'recall': recalls,
            'precision': precisions,
            'model': model_name,
            'language': lang,
            'layer': layer,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'thresholds': thresholds,
            'primary_threshold': primary_threshold,
            'primary_threshold_confidence_intervals': {
                name: confidence_intervals_to_dict(
                    confidence_limits[name][0][layer_index]
                    if confidence_limits[name][0] is not None else None,
                    confidence_limits[name][1][layer_index]
                    if confidence_limits[name][1] is not None else None,
                )
                for name in metric_names
            },
            'bootstrap': {
                'unit': 'program',
                'method': 'percentile',
                'samples': bootstrap_samples,
                'confidence_level': confidence_level,
                'seed': bootstrap_seed,
            },
            'num_aligned': aligned,
            'num_attempted': attempted,
            'alignment_rate': alignment_rate,
            'num_selected': selected_count,
            'end_to_end_alignment_rate': end_to_end_rate,
            'num_graph_generation_failures': selected_count - attempted,
            'num_dfg_failures': len(failures),
            'num_empty_dfgs': empty_dfgs,
            'mean_dfg_edges': total_dfg_edges / aligned,
            'mean_dfg_density': total_dfg_density / aligned,
            'graph_manifest': os.path.abspath(run_info['manifest_path']),
            'dfg_failure_file': os.path.abspath(failure_path),
            'program_metrics_file': os.path.abspath(program_metrics_path),
            'evaluation_protocol': protocol_path,
        }
        output_path = os.path.join(
            output_dir,
            f'{model_name}_layer_{layer}.json',
        )
        with open(output_path, 'w') as output_file:
            json.dump(output, output_file, indent=2)
        outputs.append(output)

        print(
            f'[DFG EVALUATION COMPLETE] Layer {layer} | Aligned '
            f'{aligned}/{attempted} artifacts ({alignment_rate:.2%}) | '
            f'End-to-end {aligned}/{selected_count} ({end_to_end_rate:.2%})'
        )

    return outputs


def save_dfg_stats(
    code_file,
    graph_loc,
    save_dir,
    layer,
    exp_name,
    parser,
    lang='python',
    thresholds=None,
    primary_threshold=DEFAULT_PRIMARY_THRESHOLD,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
):
    """Backward-compatible single-layer entry point."""
    return evaluate_dfg_stats(
        code_file,
        graph_loc,
        save_dir,
        [layer],
        exp_name,
        parser,
        lang=lang,
        thresholds=thresholds,
        primary_threshold=primary_threshold,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )[0]


def build_parser(lang):
    attention_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(attention_dir)
    grammar_root = attention_dir if lang == 'python' else repo_root
    grammar_repo = os.path.join(grammar_root, f'tree-sitter-{lang}')
    build_dir = os.path.join(repo_root, 'build')
    language_library = os.path.join(build_dir, f'my-languages-{lang}.so')
    os.makedirs(build_dir, exist_ok=True)

    language_libraries = [language_library]
    if lang == 'python':
        language_libraries.insert(
            0,
            os.path.join(attention_dir, 'build', 'my-languages.so'),
        )

    load_errors = []
    for candidate in language_libraries:
        if not os.path.exists(candidate):
            continue
        try:
            language = Language(candidate, lang)
            parser = Parser()
            parser.set_language(language)
            return parser
        except ValueError as exc:
            load_errors.append(f'{candidate}: {exc}')

    if not os.path.exists(grammar_repo):
        raise FileNotFoundError(
            f'Tree-sitter grammar repository not found: {grammar_repo}'
        )
    if not os.path.exists(language_library):
        Language.build_library(language_library, [grammar_repo])

    try:
        language = Language(language_library, lang)
        parser = Parser()
        parser.set_language(language)
        return parser
    except ValueError as exc:
        load_errors.append(f'{language_library}: {exc}')
        raise RuntimeError(
            f'No compatible tree-sitter library found for {lang}: '
            + '; '.join(load_errors)
        ) from exc


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--code_file', default='exp_data/exp_0.jsonl')
    cli_parser.add_argument('--graph_loc', required=True)
    cli_parser.add_argument('--save_dir', required=True)
    cli_parser.add_argument('--exp_name')
    cli_parser.add_argument('--layer', default=-1, type=int)
    cli_parser.add_argument('--all_layers', action='store_true')
    cli_parser.add_argument('--num_layers', type=int)
    cli_parser.add_argument(
        '--lang',
        default='python',
        choices=['python', 'java', 'go', 'javascript'],
    )
    cli_parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=DEFAULT_THRESHOLDS,
        help='Attention thresholds; defaults to the original Python grid.',
    )
    cli_parser.add_argument(
        '--primary_threshold', type=float, default=DEFAULT_PRIMARY_THRESHOLD
    )
    cli_parser.add_argument(
        '--bootstrap_samples', type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    cli_parser.add_argument(
        '--confidence_level', type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    cli_parser.add_argument(
        '--bootstrap_seed', type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    args = cli_parser.parse_args()

    tree_sitter_parser = build_parser(args.lang)
    run_info = load_run_info(args.graph_loc, args.lang, code_file=args.code_file)
    if args.all_layers:
        if (
            args.num_layers is not None
            and args.num_layers != run_info['num_layers']
        ):
            raise ValueError(
                f'--num_layers={args.num_layers} does not match the '
                f"{run_info['num_layers']} layers stored in the artifacts"
            )
        selected_layers = list(range(run_info['num_layers']))
    else:
        selected_layers = [args.layer]

    evaluate_dfg_stats(
        args.code_file,
        args.graph_loc,
        args.save_dir,
        selected_layers,
        args.exp_name,
        tree_sitter_parser,
        lang=args.lang,
        thresholds=args.thresholds,
        primary_threshold=args.primary_threshold,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
