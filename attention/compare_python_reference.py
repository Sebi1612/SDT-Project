"""Compare a recomputed Python run with the paper's stored results."""

import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from summarize_attention_analysis import best_fscore_head, lowest_ged_head


OVERLAP_GRAPHS = ('ast', 'dfg')
OVERLAP_METRICS = ('fscore', 'precision', 'recall')
GED_GRAPHS = ('ast', 'ast_wo_identifiers', 'dfg')


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_reference(
    reference_root, model, num_layers, threshold, include_ged=True
):
    layers = []
    for layer in range(num_layers):
        ast = load_json(os.path.join(
            reference_root, 'ast', 'exp_0', f'{model}_layer_{layer}.json'
        ))
        dfg = load_json(os.path.join(
            reference_root, 'dfg', 'exp_0', f'{model}_layer_{layer}.json'
        ))
        layer_data = {
            'layer_index': layer,
            'layer_number_one_based': layer + 1,
            'overlap': {
                'ast': best_fscore_head(ast, threshold),
                'dfg': best_fscore_head(dfg, threshold),
            },
        }
        if include_ged:
            similarity = load_json(os.path.join(
                reference_root,
                'similarity',
                'exp_0',
                model,
                f'layer_{layer}_threshold_{threshold}.json',
            ))
            layer_data['minimum_ged_per_node'] = {
                graph_name: lowest_ged_head(similarity, graph_name)
                for graph_name in GED_GRAPHS
            }
        layers.append(layer_data)
    return layers


def curve_statistics(pilot_values, reference_values):
    pilot = np.asarray(pilot_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    differences = pilot - reference
    if np.std(pilot) == 0 or np.std(reference) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(pilot, reference)[0, 1])
    return {
        'mean_absolute_difference': float(np.mean(np.abs(differences))),
        'root_mean_squared_difference': float(
            math.sqrt(np.mean(differences ** 2))
        ),
        'mean_signed_difference': float(np.mean(differences)),
        'pearson_layer_correlation': correlation,
    }


def save_comparison_plots(
    pilot_layers,
    reference_layers,
    output_path,
    pilot_label='Recomputed run',
    include_ged=True,
):
    base = os.path.splitext(output_path)[0]
    layer_numbers = [layer['layer_number_one_based'] for layer in pilot_layers]

    overlap_path = base + '_overlap.png'
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, graph_name in enumerate(OVERLAP_GRAPHS):
        for column, metric in enumerate(OVERLAP_METRICS):
            axis = axes[row, column]
            axis.plot(
                layer_numbers,
                [layer['overlap'][graph_name][metric] for layer in pilot_layers],
                marker='o',
                label=pilot_label,
            )
            axis.plot(
                layer_numbers,
                [
                    layer['overlap'][graph_name][metric]
                    for layer in reference_layers
                ],
                marker='x',
                linestyle='--',
                label='Stored 3,000-program reference',
            )
            axis.set_title(f'{graph_name.upper()} {metric}')
            axis.set_xlabel('Layer')
            axis.set_ylabel(metric.capitalize())
            axis.set_xticks(layer_numbers)
            axis.set_ylim(bottom=0)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(overlap_path, dpi=200)
    plt.close(figure)

    if not include_ged:
        return overlap_path, None

    ged_path = base + '_ged.png'
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for axis, graph_name in zip(axes, GED_GRAPHS):
        axis.plot(
            layer_numbers,
            [
                layer['minimum_ged_per_node'][graph_name]['ged_per_node']
                for layer in pilot_layers
            ],
            marker='o',
            label=pilot_label,
        )
        axis.plot(
            layer_numbers,
            [
                layer['minimum_ged_per_node'][graph_name]['ged_per_node']
                for layer in reference_layers
            ],
            marker='x',
            linestyle='--',
            label='Stored 3,000-program reference',
        )
        axis.set_title(graph_name)
        axis.set_xlabel('Layer')
        axis.set_ylabel('GED per node')
        axis.set_xticks(layer_numbers)
        axis.set_ylim(bottom=0)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(ged_path, dpi=200)
    plt.close(figure)
    return overlap_path, ged_path


def compare(pilot_summary_path, reference_root, output_path, skip_ged=False):
    pilot = load_json(pilot_summary_path)
    if pilot.get('language') != 'python':
        raise ValueError('The reference sanity check requires a Python pilot')
    model = pilot.get('model')
    if not model:
        raise ValueError('Pilot summary does not record a model')
    if not skip_ged and pilot.get('ged_mode') != 'legacy':
        raise ValueError(
            'The stored paper GED reference requires a summary generated '
            "with --ged_mode legacy; fixed-node distances are not comparable"
        )
    threshold = float(pilot['primary_threshold'])
    pilot_layers = pilot['layers']
    reference_layers = load_reference(
        reference_root,
        pilot['model'],
        int(pilot['num_layers']),
        threshold,
        include_ged=not skip_ged,
    )
    if len(pilot_layers) != len(reference_layers):
        raise ValueError('Pilot and reference layer counts differ')

    overlap_rows = []
    ged_rows = []
    aggregate = {'overlap': {}, 'ged': {}}
    for graph_name in OVERLAP_GRAPHS:
        aggregate['overlap'][graph_name] = {}
        for metric in OVERLAP_METRICS:
            pilot_values = [
                layer['overlap'][graph_name][metric] for layer in pilot_layers
            ]
            reference_values = [
                layer['overlap'][graph_name][metric]
                for layer in reference_layers
            ]
            aggregate['overlap'][graph_name][metric] = curve_statistics(
                pilot_values, reference_values
            )
            for layer, pilot_value, reference_value in zip(
                pilot_layers, pilot_values, reference_values
            ):
                overlap_rows.append({
                    'layer_index': layer['layer_index'],
                    'layer_number_one_based': layer['layer_number_one_based'],
                    'graph': graph_name,
                    'metric': metric,
                    'pilot': pilot_value,
                    'reference': reference_value,
                    'difference': pilot_value - reference_value,
                    'absolute_difference': abs(pilot_value - reference_value),
                })

    if not skip_ged:
        for graph_name in GED_GRAPHS:
            pilot_values = [
                layer['minimum_ged_per_node'][graph_name]['ged_per_node']
                for layer in pilot_layers
            ]
            reference_values = [
                layer['minimum_ged_per_node'][graph_name]['ged_per_node']
                for layer in reference_layers
            ]
            aggregate['ged'][graph_name] = curve_statistics(
                pilot_values, reference_values
            )
            for layer, pilot_value, reference_value in zip(
                pilot_layers, pilot_values, reference_values
            ):
                ged_rows.append({
                    'layer_index': layer['layer_index'],
                    'layer_number_one_based': layer['layer_number_one_based'],
                    'graph': graph_name,
                    'pilot': pilot_value,
                    'reference': reference_value,
                    'difference': pilot_value - reference_value,
                    'absolute_difference': abs(pilot_value - reference_value),
                })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base = os.path.splitext(output_path)[0]
    overlap_csv = base + '_overlap.csv'
    ged_csv = base + '_ged.csv'
    expected_programs = pilot.get('expected_programs')
    run_label = (
        f'Recomputed {expected_programs:,}-program run'
        if isinstance(expected_programs, int)
        else 'Recomputed run'
    )
    overlap_plot, ged_plot = save_comparison_plots(
        pilot_layers,
        reference_layers,
        output_path,
        pilot_label=run_label,
        include_ged=not skip_ged,
    )
    with open(overlap_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=overlap_rows[0].keys())
        writer.writeheader()
        writer.writerows(overlap_rows)
    if not skip_ged:
        with open(ged_csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=ged_rows[0].keys())
            writer.writeheader()
            writer.writerows(ged_rows)

    output = {
        'status': 'complete',
        'purpose': 'Python recomputation check against stored paper results',
        'model': model,
        'primary_threshold': threshold,
        'pilot_ged_mode': pilot.get('ged_mode'),
        'pilot_ged_method': pilot.get('ged_method'),
        'pilot_ged_provenance': pilot.get('ged_provenance'),
        'ged_comparison_is_paper_compatible': not skip_ged,
        'ged_comparison_status': 'skipped' if skip_ged else 'included',
        'pilot_summary': os.path.abspath(pilot_summary_path),
        'pilot_expected_programs': pilot.get('expected_programs'),
        'candidate_run_label': run_label,
        'reference_root': os.path.abspath(reference_root),
        'reference_dataset': 'CodeSearchNet Python test subset exp_0.jsonl',
        'reference_dataset_rows': 3000,
        'reference_evaluated_rows': (
            'not recorded in the legacy result files; individual alignment '
            'failures may have been excluded by the original scripts'
        ),
        'interpretation': (
            'This is a descriptive reproducibility check, not a statistical '
            'equivalence test. Compare curve direction, magnitude, '
            'correlation, and absolute differences. GED is compared only '
            'when the candidate uses the paper-compatible legacy NetworkX '
            'mode.'
        ),
        'aggregate_curve_comparison': aggregate,
        'outputs': {
            'overlap_csv': os.path.abspath(overlap_csv),
            'overlap_plot': os.path.abspath(overlap_plot),
        },
    }
    if not skip_ged:
        output['outputs'].update({
            'ged_csv': os.path.abspath(ged_csv),
            'ged_plot': os.path.abspath(ged_plot),
        })
    with open(output_path, 'w') as handle:
        json.dump(output, handle, indent=2)
    print(f'[PYTHON REFERENCE COMPARISON COMPLETE] {output_path}')
    return output


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--pilot_summary', required=True)
    cli.add_argument(
        '--reference_root', default='attention/graph_comparision'
    )
    cli.add_argument('--output', required=True)
    cli.add_argument('--skip_ged', action='store_true')
    args = cli.parse_args()
    compare(
        args.pilot_summary,
        args.reference_root,
        args.output,
        skip_ged=args.skip_ged,
    )


if __name__ == '__main__':
    main()
