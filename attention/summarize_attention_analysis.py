"""Create paper-style Section 3.2 summaries from attention results."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt

from similarity import DISTANCE_METHODS, similarity_directory


PRIMARY_GRAPHS = ('ast', 'dfg')
GED_GRAPHS = ('ast', 'ast_wo_identifiers', 'dfg')


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def threshold_value(head_values, threshold):
    for key, value in head_values.items():
        if abs(float(key) - threshold) <= 1e-12:
            return float(value)
    raise ValueError(f'Threshold {threshold} is missing from metric output')


def best_fscore_head(layer_data, threshold):
    heads = sorted(int(head) for head in layer_data['fscore'])
    if not heads:
        raise ValueError('No attention heads found in overlap results')
    best_head = max(
        heads,
        key=lambda head: (
            threshold_value(layer_data['fscore'][str(head)], threshold),
            -head,
        ),
    )
    output = {
        'head_index': best_head,
        'head_number_one_based': best_head + 1,
        'fscore': threshold_value(
            layer_data['fscore'][str(best_head)], threshold
        ),
        'precision': threshold_value(
            layer_data['precision'][str(best_head)], threshold
        ),
        'recall': threshold_value(
            layer_data['recall'][str(best_head)], threshold
        ),
    }
    if 'iou' in layer_data:
        output['iou'] = threshold_value(
            layer_data['iou'][str(best_head)], threshold
        )
    return output


def lowest_ged_head(layer_data, graph_name):
    values = layer_data[graph_name]
    heads = sorted(int(head) for head in values)
    if not heads:
        raise ValueError(f'No attention heads found for {graph_name} GED')
    best_head = min(heads, key=lambda head: (float(values[str(head)]), head))
    return {
        'head_index': best_head,
        'head_number_one_based': best_head + 1,
        'ged_per_node': float(values[str(best_head)]),
    }


def validate_layer_metadata(data, model, language, layer, threshold):
    if data.get('model') != model:
        raise ValueError(
            f"Layer {layer} model is {data.get('model')!r}, expected {model!r}"
        )
    if data.get('language') != language:
        raise ValueError(
            f"Layer {layer} language is {data.get('language')!r}, "
            f'expected {language!r}'
        )
    if data.get('layer') != layer:
        raise ValueError(
            f"Stored layer is {data.get('layer')!r}, expected {layer}"
        )
    if abs(float(data.get('primary_threshold', threshold)) - threshold) > 1e-12:
        raise ValueError(f'Layer {layer} uses a different primary threshold')


def require_count(value, expected, label):
    if expected is not None and value != expected:
        raise ValueError(f'{label} is {value}, expected {expected}')


def save_plots(layers, output_path, ged_mode, skip_ged=False):
    base = os.path.splitext(output_path)[0]
    layer_numbers = [layer['layer_number_one_based'] for layer in layers]

    overlap_plot = base + '_overlap.png'
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for axis, metric in zip(axes, ('fscore', 'precision', 'recall')):
        for graph_name, label in (('ast', 'Syntax graph'), ('dfg', 'DFG')):
            axis.plot(
                layer_numbers,
                [layer['overlap'][graph_name][metric] for layer in layers],
                marker='o',
                label=label,
            )
        axis.set_xlabel('Layer')
        axis.set_ylabel(metric.capitalize())
        axis.set_xticks(layer_numbers)
        axis.set_ylim(bottom=0)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(overlap_plot, dpi=200)
    plt.close(figure)

    if skip_ged:
        return overlap_plot, None

    ged_plot = base + '_ged.png'
    figure, axis = plt.subplots(figsize=(7, 4))
    labels = {
        'ast': 'Syntax graph',
        'ast_wo_identifiers': 'Non-identifier syntax graph',
        'dfg': 'DFG',
    }
    for graph_name in GED_GRAPHS:
        axis.plot(
            layer_numbers,
            [
                layer['minimum_ged_per_node'][graph_name]['ged_per_node']
                for layer in layers
            ],
            marker='o',
            label=labels[graph_name],
        )
    axis.set_xlabel('Layer')
    axis.set_ylabel(
        'Legacy GED estimate per node'
        if ged_mode == 'legacy'
        else 'Fixed-node edge distance per node'
    )
    axis.set_xticks(layer_numbers)
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(ged_plot, dpi=200)
    plt.close(figure)
    return overlap_plot, ged_plot


def validate_ged_mode(data, ged_mode, layer):
    stored_mode = data.get('distance_mode')
    if stored_mode is None:
        method = data.get('method')
        stored_mode = next(
            (
                mode for mode, expected_method in DISTANCE_METHODS.items()
                if method == expected_method
            ),
            None,
        )
    if stored_mode != ged_mode:
        raise ValueError(
            f'GED layer {layer} uses distance mode {stored_mode!r}; '
            f'expected {ged_mode!r}'
        )
    expected_method = DISTANCE_METHODS[ged_mode]
    if data.get('method') != expected_method:
        raise ValueError(
            f"GED layer {layer} uses method {data.get('method')!r}; "
            f'expected {expected_method!r}'
        )


def summarize(
    results_dir,
    output_path,
    model,
    language,
    threshold=0.05,
    expected_programs=None,
    ged_mode='fixed',
    skip_ged=False,
):
    ast_dir = os.path.join(results_dir, 'ast')
    dfg_dir = os.path.join(results_dir, 'dfg')
    similarity_dir = None
    if not skip_ged:
        similarity_dir = os.path.join(
            results_dir, similarity_directory(ged_mode), model
        )

    first_ast = load_json(os.path.join(ast_dir, f'{model}_layer_0.json'))
    num_layers = int(first_ast['num_layers'])
    num_heads = int(first_ast['num_heads'])
    layers = []
    ged_provenance = None

    for layer in range(num_layers):
        ast = load_json(os.path.join(ast_dir, f'{model}_layer_{layer}.json'))
        dfg = load_json(os.path.join(dfg_dir, f'{model}_layer_{layer}.json'))
        similarity = None
        if not skip_ged:
            similarity = load_json(os.path.join(
                similarity_dir,
                f'layer_{layer}_threshold_{threshold}.json',
            ))
        for data in (ast, dfg):
            validate_layer_metadata(data, model, language, layer, threshold)
            if int(data['num_layers']) != num_layers:
                raise ValueError(f'Layer-count mismatch in layer {layer}')
            if int(data['num_heads']) != num_heads:
                raise ValueError(f'Head-count mismatch in layer {layer}')
        if not skip_ged:
            validate_layer_metadata(
                similarity, model, language, layer, threshold
            )
            validate_ged_mode(similarity, ged_mode, layer)
            current_provenance = {
                'networkx_version': similarity.get('networkx_version'),
                'paper_reference_networkx_version': similarity.get(
                    'paper_reference_networkx_version'
                ),
                'networkx_version_matches_declared_paper_environment': (
                    similarity.get(
                        'networkx_version_matches_declared_paper_environment'
                    )
                ),
            }
            if ged_provenance is None:
                ged_provenance = current_provenance
            elif current_provenance != ged_provenance:
                raise ValueError('GED provenance differs between layers')

        require_count(
            ast.get('num_evaluated'), expected_programs, 'AST evaluated programs'
        )
        require_count(
            dfg.get('num_aligned'), expected_programs, 'DFG aligned programs'
        )
        if not skip_ged:
            require_count(
                similarity.get('num_evaluated'),
                expected_programs,
                'GED evaluated programs',
            )

        layer_summary = {
            'layer_index': layer,
            'layer_number_one_based': layer + 1,
            'overlap': {
                'ast': best_fscore_head(ast, threshold),
                'dfg': best_fscore_head(dfg, threshold),
            },
        }
        if not skip_ged:
            layer_summary['minimum_ged_per_node'] = {
                graph_name: lowest_ged_head(similarity, graph_name)
                for graph_name in GED_GRAPHS
            }
        layers.append(layer_summary)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    overlap_csv = os.path.splitext(output_path)[0] + '_overlap.csv'
    ged_csv = os.path.splitext(output_path)[0] + '_ged.csv'
    overlap_plot, ged_plot = save_plots(
        layers, output_path, ged_mode, skip_ged=skip_ged
    )
    summary = {
        'status': 'complete',
        'paper_section': '3.2',
        'model': model,
        'language': language,
        'primary_threshold': threshold,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'expected_programs': expected_programs,
        'ged_status': 'skipped' if skip_ged else 'included',
        'ged_mode': None if skip_ged else ged_mode,
        'ged_method': None if skip_ged else DISTANCE_METHODS[ged_mode],
        'ged_paper_comparable': False if skip_ged else ged_mode == 'legacy',
        'ged_results_directory': (
            None if skip_ged else os.path.abspath(similarity_dir)
        ),
        'ged_provenance': ged_provenance,
        'head_selection': (
            'highest macro-mean F-score independently for AST and DFG; '
            'smallest zero-based head index breaks ties'
        ),
        'ged_head_selection': (
            'lowest macro-mean GED per node independently for each code graph; '
            'smallest zero-based head index breaks ties'
        ),
        'outputs': {
            'overlap_csv': os.path.abspath(overlap_csv),
            'overlap_plot': os.path.abspath(overlap_plot),
        },
        'layers': layers,
    }
    if not skip_ged:
        summary['outputs'].update({
            'ged_csv': os.path.abspath(ged_csv),
            'ged_plot': os.path.abspath(ged_plot),
        })
    with open(output_path, 'w') as handle:
        json.dump(summary, handle, indent=2)

    with open(overlap_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'language', 'model', 'layer_index', 'layer_number_one_based',
            'graph', 'head_index', 'head_number_one_based',
            'fscore', 'precision', 'recall',
        ))
        writer.writeheader()
        for layer in layers:
            for graph_name in PRIMARY_GRAPHS:
                metrics = layer['overlap'][graph_name]
                writer.writerow({
                    'language': language,
                    'model': model,
                    'layer_index': layer['layer_index'],
                    'layer_number_one_based': layer['layer_number_one_based'],
                    'graph': graph_name,
                    **{
                        key: metrics[key]
                        for key in (
                            'head_index', 'head_number_one_based',
                            'fscore', 'precision', 'recall',
                        )
                    },
                })

    if not skip_ged:
        with open(ged_csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                'language', 'model', 'layer_index', 'layer_number_one_based',
                'distance_mode', 'method', 'graph', 'head_index',
                'head_number_one_based', 'ged_per_node',
            ))
            writer.writeheader()
            for layer in layers:
                for graph_name in GED_GRAPHS:
                    metrics = layer['minimum_ged_per_node'][graph_name]
                    writer.writerow({
                        'language': language,
                        'model': model,
                        'layer_index': layer['layer_index'],
                        'layer_number_one_based': layer['layer_number_one_based'],
                        'distance_mode': ged_mode,
                        'method': DISTANCE_METHODS[ged_mode],
                        'graph': graph_name,
                        **metrics,
                    })

    print(
        f'[SECTION 3.2 SUMMARY COMPLETE] {language} {model} | '
        f'{num_layers} layers | {output_path}'
    )
    return summary


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--results_dir', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--model', default='codebert')
    cli.add_argument(
        '--lang', required=True, choices=['python', 'java', 'go', 'javascript']
    )
    cli.add_argument('--threshold', default=0.05, type=float)
    cli.add_argument('--expected_programs', type=int)
    cli.add_argument(
        '--ged_mode', choices=('fixed', 'legacy'), default='fixed'
    )
    cli.add_argument('--skip_ged', action='store_true')
    args = cli.parse_args()
    summarize(
        args.results_dir,
        args.output,
        args.model,
        args.lang,
        threshold=args.threshold,
        expected_programs=args.expected_programs,
        ged_mode=args.ged_mode,
        skip_ged=args.skip_ged,
    )


if __name__ == '__main__':
    main()
