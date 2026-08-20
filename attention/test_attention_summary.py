import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO


ATTENTION_DIR = os.path.dirname(os.path.abspath(__file__))
if ATTENTION_DIR not in sys.path:
    sys.path.insert(0, ATTENTION_DIR)

from run_section_3_2 import validate_codebert_dataset
from compare_python_reference import compare
from summarize_attention_analysis import summarize
from validate_graph_run import validate


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(value, handle)


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()


class AttentionSummaryTests(unittest.TestCase):
    def make_results(self, root, ged_mode='fixed'):
        for layer in range(2):
            base = {
                'model': 'codebert',
                'language': 'java',
                'layer': layer,
                'num_layers': 2,
                'num_heads': 2,
                'primary_threshold': 0.05,
                'fscore': {
                    '0': {'0.05': 0.4 + layer * 0.01},
                    '1': {'0.05': 0.7 + layer * 0.01},
                },
                'precision': {
                    '0': {'0.05': 0.5},
                    '1': {'0.05': 0.8},
                },
                'recall': {
                    '0': {'0.05': 0.3},
                    '1': {'0.05': 0.6},
                },
            }
            ast = {**base, 'num_evaluated': 100}
            dfg = {**base, 'num_aligned': 100}
            write_json(
                os.path.join(root, 'ast', f'codebert_layer_{layer}.json'), ast
            )
            write_json(
                os.path.join(root, 'dfg', f'codebert_layer_{layer}.json'), dfg
            )
            similarity = {
                'model': 'codebert',
                'language': 'java',
                'layer': layer,
                'num_evaluated': 100,
                'distance_mode': ged_mode,
                'method': (
                    'legacy_networkx_optimize_first_candidate'
                    if ged_mode == 'legacy'
                    else 'exact_fixed_node_edge_edit_distance'
                ),
                'ast': {'0': 1.2, '1': 0.8},
                'ast_wo_identifiers': {'0': 0.6, '1': 0.7},
                'dfg': {'0': 0.9, '1': 0.4},
            }
            write_json(
                os.path.join(
                    root,
                    (
                        'similarity_legacy'
                        if ged_mode == 'legacy'
                        else 'similarity'
                    ),
                    'codebert',
                    f'layer_{layer}_threshold_0.05.json',
                ),
                similarity,
            )

    def test_summary_uses_paper_head_selection(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_results(root)
            output = os.path.join(root, 'summary.json')
            result = summarize(
                root,
                output,
                'codebert',
                'java',
                expected_programs=100,
            )
            first = result['layers'][0]
            self.assertEqual(first['overlap']['ast']['head_index'], 1)
            self.assertEqual(first['overlap']['dfg']['head_index'], 1)
            self.assertEqual(
                first['minimum_ged_per_node']['ast']['head_index'], 1
            )
            self.assertEqual(
                first['minimum_ged_per_node']['ast_wo_identifiers'][
                    'head_index'
                ],
                0,
            )
            self.assertEqual(result['ged_mode'], 'fixed')
            self.assertFalse(result['ged_paper_comparable'])
            self.assertTrue(os.path.exists(output))
            self.assertTrue(os.path.exists(output[:-5] + '_overlap.csv'))
            self.assertTrue(os.path.exists(output[:-5] + '_ged.csv'))
            self.assertTrue(os.path.exists(output[:-5] + '_overlap.png'))
            self.assertTrue(os.path.exists(output[:-5] + '_ged.png'))

    def test_summary_rejects_incomplete_strict_pilot(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_results(root)
            ast_path = os.path.join(root, 'ast', 'codebert_layer_0.json')
            with open(ast_path) as handle:
                ast = json.load(handle)
            ast['num_evaluated'] = 99
            write_json(ast_path, ast)
            with self.assertRaisesRegex(ValueError, 'AST evaluated programs'):
                summarize(
                    root,
                    os.path.join(root, 'summary.json'),
                    'codebert',
                    'java',
                    expected_programs=100,
                )

    def test_summary_can_skip_ged(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_results(root)
            similarity_root = os.path.join(root, 'similarity')
            for directory, _, files in os.walk(similarity_root, topdown=False):
                for name in files:
                    os.remove(os.path.join(directory, name))
                if directory != similarity_root:
                    os.rmdir(directory)
            os.rmdir(similarity_root)

            output = os.path.join(root, 'summary.json')
            result = summarize(
                root,
                output,
                'codebert',
                'java',
                expected_programs=100,
                skip_ged=True,
            )
            self.assertEqual(result['ged_status'], 'skipped')
            self.assertIsNone(result['ged_mode'])
            self.assertNotIn('ged_csv', result['outputs'])
            self.assertNotIn('minimum_ged_per_node', result['layers'][0])
            self.assertTrue(os.path.exists(output[:-5] + '_overlap.csv'))
            self.assertTrue(os.path.exists(output[:-5] + '_overlap.png'))
            self.assertFalse(os.path.exists(output[:-5] + '_ged.csv'))
            self.assertFalse(os.path.exists(output[:-5] + '_ged.png'))

    def test_dataset_preflight_enforces_count_and_length(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'data.jsonl')
            with open(path, 'w') as handle:
                handle.write(json.dumps({
                    'code_tokens': ['a', 'b'], 'partition': 'train'
                }) + '\n')
            info = validate_codebert_dataset(path, 1, FakeTokenizer())
            self.assertEqual(info['maximum_codebert_tokens'], 2)
            self.assertEqual(info['partitions'], {'train': 1})
            with self.assertRaisesRegex(ValueError, 'expected 2'):
                validate_codebert_dataset(path, 2, FakeTokenizer())

    def test_graph_validation_can_require_full_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            write_json(os.path.join(root, 'graph_manifest.json'), {
                'status': 'complete',
                'selected_num_codes': 1,
                'artifacts': [],
                'failures': [{'reason': 'synthetic failure'}],
            })
            with redirect_stdout(StringIO()):
                self.assertFalse(validate(root, minimum_coverage=1.0))
                self.assertTrue(validate(root, minimum_coverage=0.0))

    def test_python_reference_comparison_uses_stored_legacy_layout(self):
        with tempfile.TemporaryDirectory() as root:
            pilot_results = os.path.join(root, 'pilot')
            self.make_results(pilot_results, ged_mode='legacy')
            pilot_summary = os.path.join(root, 'pilot_summary.json')
            summary = summarize(
                pilot_results,
                pilot_summary,
                'codebert',
                'java',
                expected_programs=100,
                ged_mode='legacy',
            )
            summary['language'] = 'python'
            with open(pilot_summary, 'w') as handle:
                json.dump(summary, handle)

            reference = os.path.join(root, 'reference')
            for layer in range(2):
                for graph_name in ('ast', 'dfg'):
                    source = os.path.join(
                        pilot_results,
                        graph_name,
                        f'codebert_layer_{layer}.json',
                    )
                    with open(source) as handle:
                        data = json.load(handle)
                    write_json(os.path.join(
                        reference,
                        graph_name,
                        'exp_0',
                        f'codebert_layer_{layer}.json',
                    ), data)
                similarity_path = os.path.join(
                    pilot_results,
                    'similarity_legacy',
                    'codebert',
                    f'layer_{layer}_threshold_0.05.json',
                )
                with open(similarity_path) as handle:
                    similarity = json.load(handle)
                write_json(os.path.join(
                    reference,
                    'similarity',
                    'exp_0',
                    'codebert',
                    f'layer_{layer}_threshold_0.05.json',
                ), similarity)

            output_path = os.path.join(root, 'comparison.json')
            result = compare(pilot_summary, reference, output_path)
            self.assertEqual(
                result['aggregate_curve_comparison']['overlap']['ast'][
                    'fscore'
                ]['mean_absolute_difference'],
                0.0,
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.exists(output_path[:-5] + '_overlap.png'))

    def test_python_reference_comparison_rejects_fixed_ged(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_results(root)
            summary_path = os.path.join(root, 'summary.json')
            summary = summarize(
                root,
                summary_path,
                'codebert',
                'java',
                expected_programs=100,
            )
            summary['language'] = 'python'
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, '--ged_mode legacy'):
                compare(summary_path, os.path.join(root, 'reference'), root)

    def test_python_reference_overlap_comparison_supports_other_models(self):
        with tempfile.TemporaryDirectory() as root:
            pilot_results = os.path.join(root, 'pilot')
            self.make_results(pilot_results)
            for graph_name in ('ast', 'dfg'):
                for layer in range(2):
                    path = os.path.join(
                        pilot_results, graph_name, f'codebert_layer_{layer}.json'
                    )
                    with open(path) as handle:
                        data = json.load(handle)
                    data['model'] = 'graphcodebert'
                    new_path = os.path.join(
                        pilot_results,
                        graph_name,
                        f'graphcodebert_layer_{layer}.json',
                    )
                    write_json(new_path, data)
                    write_json(os.path.join(
                        root,
                        'reference',
                        graph_name,
                        'exp_0',
                        f'graphcodebert_layer_{layer}.json',
                    ), data)
            summary_path = os.path.join(root, 'summary.json')
            summary = summarize(
                pilot_results,
                summary_path,
                'graphcodebert',
                'java',
                expected_programs=100,
                skip_ged=True,
            )
            summary['language'] = 'python'
            write_json(summary_path, summary)
            output = os.path.join(root, 'comparison.json')
            result = compare(
                summary_path,
                os.path.join(root, 'reference'),
                output,
                skip_ged=True,
            )
            self.assertEqual(result['model'], 'graphcodebert')
            self.assertEqual(result['ged_comparison_status'], 'skipped')
            self.assertNotIn('ged_csv', result['outputs'])


if __name__ == '__main__':
    unittest.main()
