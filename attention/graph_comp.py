import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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


def load_run_info(graph_loc):
    """Return the exact artifact cohort represented by a graph directory."""
    manifest_path = os.path.join(graph_loc, 'graph_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get('status') not in {None, 'complete'}:
            raise ValueError(
                f"Graph manifest is not complete: {manifest.get('status')!r}"
            )
        artifacts = manifest.get('artifacts', [])
        selected_count = manifest.get('selected_num_codes')
        model_name = manifest.get('model')
        language = manifest.get('lang')
    else:
        artifacts = sorted(
            name for name in os.listdir(graph_loc) if name.endswith('.pkl')
        )
        selected_count = len(artifacts)
        model_name = None
        language = None
        manifest = None
        print(
            f'Warning: {manifest_path} does not exist; evaluating all '
            f'{len(artifacts)} pickle files in the directory.'
        )

    if not artifacts:
        raise ValueError(f'No graph artifacts found in {graph_loc}')

    first_path = os.path.join(graph_loc, artifacts[0])
    with open(first_path, 'rb') as graph_file:
        first_graph = pickle.load(graph_file)
    model_graphs = first_graph['model_graphs']
    if model_graphs.ndim != 4:
        raise ValueError(
            f'Expected a four-dimensional attention tensor, got {model_graphs.shape}'
        )

    if not model_name:
        model_name = os.path.basename(os.path.normpath(graph_loc))

    return {
        'manifest': manifest,
        'artifacts': artifacts,
        'selected_count': selected_count,
        'model_name': model_name,
        'language': language,
        'num_layers': model_graphs.shape[0],
        'num_heads': model_graphs.shape[1],
    }


def metrics_for_layer(layer_graph, ast_graph, thresholds):
    """Compute the original edge metrics for every head and threshold."""
    truth = ast_graph == 1
    truth_count = truth.sum()
    num_heads = layer_graph.shape[0]
    num_thresholds = len(thresholds)

    f_scores = np.zeros((num_heads, num_thresholds), dtype=np.float64)
    recalls = np.zeros_like(f_scores)
    precisions = np.zeros_like(f_scores)
    ious = np.zeros_like(f_scores)

    for threshold_index, threshold in enumerate(thresholds):
        predictions = layer_graph > threshold
        true_positives = np.logical_and(predictions, truth[None, :, :]).sum(
            axis=(1, 2)
        )
        predicted_count = predictions.sum(axis=(1, 2))
        union_count = predicted_count + truth_count - true_positives

        precisions[:, threshold_index] = np.divide(
            true_positives,
            predicted_count,
            out=np.zeros(num_heads, dtype=np.float64),
            where=predicted_count != 0,
        )
        if truth_count:
            recalls[:, threshold_index] = true_positives / truth_count
        ious[:, threshold_index] = np.divide(
            true_positives,
            union_count,
            out=np.zeros(num_heads, dtype=np.float64),
            where=union_count != 0,
        )
        precision_recall_sum = (
            precisions[:, threshold_index] + recalls[:, threshold_index]
        )
        f_scores[:, threshold_index] = np.divide(
            2
            * precisions[:, threshold_index]
            * recalls[:, threshold_index],
            precision_recall_sum,
            out=np.zeros(num_heads, dtype=np.float64),
            where=precision_recall_sum != 0,
        )

    return f_scores, recalls, precisions, ious


def metric_array_to_dict(values, thresholds):
    return {
        head: {
            threshold: float(values[head, threshold_index])
            for threshold_index, threshold in enumerate(thresholds)
        }
        for head in range(values.shape[0])
    }


def save_plot(output_dir, model_name, layer, metrics, thresholds, num_heads):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    plot_specs = [
        ('fscore', 'F-score', axes[0, 0]),
        ('iou', 'IoU', axes[0, 1]),
        ('recall', 'Recall', axes[1, 0]),
        ('precision', 'Precision', axes[1, 1]),
    ]
    colour_map = plt.get_cmap('jet_r')

    for metric_name, label, axis in plot_specs:
        for head in range(num_heads):
            values = [metrics[metric_name][head][threshold] for threshold in thresholds]
            axis.plot(
                thresholds,
                values,
                marker='x',
                color=colour_map(float(head / num_heads)),
                markersize=4,
                label=f'head {head}',
            )
        axis.set_title(f'{label} for {model_name}')
        axis.set_xlabel('Threshold')
        axis.set_ylabel(label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{model_name}_layer_{layer}.png'))
    plt.close(fig)


def save_ast_stats(
    graph_loc,
    save_dir,
    layer,
    exp_name=None,
    thresholds=None,
    run_info=None,
    primary_threshold=DEFAULT_PRIMARY_THRESHOLD,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
):
    thresholds = list(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if len(thresholds) != len(set(thresholds)):
        raise ValueError('Threshold values must be unique')
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError('Threshold values must be non-negative')
    primary_index = threshold_index(thresholds, primary_threshold)

    run_info = load_run_info(graph_loc) if run_info is None else run_info
    num_layers = run_info['num_layers']
    num_heads = run_info['num_heads']
    if layer == -1:
        layer = num_layers - 1
    if layer < 0 or layer >= num_layers:
        raise ValueError(
            f'Wrong layer index {layer}; expected a value from 0 to {num_layers - 1}'
        )

    totals = [
        np.zeros((num_heads, len(thresholds)), dtype=np.float64)
        for _ in range(4)
    ]
    evaluated = 0
    program_results = []
    program_records = []

    for artifact_name in tqdm(run_info['artifacts']):
        artifact_path = os.path.join(graph_loc, artifact_name)
        with open(artifact_path, 'rb') as graph_file:
            graph_info = pickle.load(graph_file)

        model_graphs = graph_info['model_graphs']
        ast_graph = graph_info['ast_graph']
        token_count = len(graph_info['code_tokens'])
        expected_attention_shape = (num_layers, num_heads, token_count, token_count)
        if model_graphs.shape != expected_attention_shape:
            raise ValueError(
                f'{artifact_name} has attention shape {model_graphs.shape}; '
                f'expected {expected_attention_shape}'
            )
        if ast_graph.shape != (token_count, token_count):
            raise ValueError(
                f'{artifact_name} has AST shape {ast_graph.shape}; '
                f'expected {(token_count, token_count)}'
            )

        program_metrics = metrics_for_layer(
            model_graphs[layer],
            ast_graph,
            thresholds,
        )
        for total, program_metric in zip(totals, program_metrics):
            total += program_metric
        program_results.append(tuple(
            metric[:, primary_index] for metric in program_metrics
        ))
        program_records.append({
            'artifact': artifact_name,
            'sample_index': graph_info.get('sample_index'),
            'source_index': graph_info.get('source_index'),
            'file_name': graph_info.get('file_name'),
        })
        evaluated += 1

    if evaluated == 0:
        raise ValueError(f'No valid AST artifacts were evaluated from {graph_loc}')

    averaged = [total / evaluated for total in totals]
    f_scores, recalls, precisions, ious = [
        metric_array_to_dict(values, thresholds) for values in averaged
    ]
    metrics = {
        'fscore': f_scores,
        'recall': recalls,
        'precision': precisions,
        'iou': ious,
    }
    metric_names = ('fscore', 'recall', 'precision', 'iou')
    program_arrays = {
        name: np.stack([
            result[metric_index]
            for result in program_results
        ])
        for metric_index, name in enumerate(metric_names)
    }
    confidence_intervals = {}
    for metric_name, values in program_arrays.items():
        lower, upper = bootstrap_mean_ci(
            values,
            num_resamples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        confidence_intervals[metric_name] = confidence_intervals_to_dict(
            lower, upper
        )

    selected_count = run_info['selected_count']
    coverage = evaluated / selected_count if selected_count else None
    output_dir = os.path.join(save_dir, 'ast')
    if exp_name is not None:
        output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    model_name = run_info['model_name']
    program_metrics_path = os.path.join(
        output_dir, f'{model_name}_layer_{layer}_program_metrics.npz'
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
        primary_threshold=np.asarray(primary_threshold),
    )
    protocol = protocol_metadata(
        'ast_attention_overlap',
        model=model_name,
        language=run_info['language'],
        primary_threshold=primary_threshold,
        thresholds=thresholds,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    protocol_path = write_protocol(output_dir, protocol)

    output = {
        **metrics,
        'model': model_name,
        'language': run_info['language'],
        'layer': layer,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'thresholds': thresholds,
        'primary_threshold': primary_threshold,
        'primary_threshold_confidence_intervals': confidence_intervals,
        'bootstrap': {
            'unit': 'program',
            'method': 'percentile',
            'samples': bootstrap_samples,
            'confidence_level': confidence_level,
            'seed': bootstrap_seed,
        },
        'num_evaluated': evaluated,
        'num_selected': selected_count,
        'coverage': coverage,
        'graph_manifest': os.path.abspath(
            os.path.join(graph_loc, 'graph_manifest.json')
        ),
        'program_metrics_file': os.path.abspath(program_metrics_path),
        'evaluation_protocol': protocol_path,
    }
    data_path = os.path.join(output_dir, f'{model_name}_layer_{layer}.json')
    with open(data_path, 'w') as output_file:
        json.dump(output, output_file, indent=2)
    save_plot(output_dir, model_name, layer, metrics, thresholds, num_heads)

    print(
        f'[AST EVALUATION COMPLETE] Layer {layer} | '
        f'Evaluated {evaluated}/{selected_count} programs '
        f'({coverage:.2%})' if coverage is not None else
        f'[AST EVALUATION COMPLETE] Layer {layer} | Evaluated {evaluated} programs'
    )
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_loc', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--exp_name')
    parser.add_argument('--layer', default=-1, type=int)
    parser.add_argument('--all_layers', action='store_true')
    parser.add_argument('--num_layers', type=int)
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=DEFAULT_THRESHOLDS,
        help='Attention thresholds; defaults to the original Python grid.',
    )
    parser.add_argument(
        '--primary_threshold', type=float, default=DEFAULT_PRIMARY_THRESHOLD
    )
    parser.add_argument(
        '--bootstrap_samples', type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        '--confidence_level', type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    parser.add_argument(
        '--bootstrap_seed', type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    args = parser.parse_args()

    run_info = load_run_info(args.graph_loc)
    if args.all_layers:
        detected_layers = run_info['num_layers']
        if args.num_layers is not None and args.num_layers != detected_layers:
            raise ValueError(
                f'--num_layers={args.num_layers} does not match the '
                f'{detected_layers} layers stored in the artifacts'
            )
        layers = range(detected_layers)
    else:
        layers = [args.layer]

    for layer in layers:
        print(f'Evaluating layer {layer}...')
        save_ast_stats(
            args.graph_loc,
            args.save_dir,
            layer,
            exp_name=args.exp_name,
            thresholds=args.thresholds,
            run_info=run_info,
            primary_threshold=args.primary_threshold,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
