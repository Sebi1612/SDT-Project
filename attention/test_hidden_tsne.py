import unittest
import json
import tempfile
from pathlib import Path

from hidden_tsne import (
    balanced_indices,
    load_artifact_token_lengths,
    resolve_layers,
    seeded_subset,
    select_program_subset,
    selected_token_type,
    token_role,
)


class HiddenTsneTests(unittest.TestCase):
    def test_language_specific_roles(self):
        self.assertEqual(token_role('def', 'def', 'python'), 'declaration')
        self.assertEqual(token_role('func', 'func', 'go'), 'declaration')
        self.assertEqual(token_role('x', 'field_identifier', 'go'), 'identifier')
        self.assertEqual(token_role(':=', ':=', 'go'), 'assignment_operator')
        self.assertEqual(token_role('?.', 'optional_chain', 'javascript'), None)

    def test_balancing_is_deterministic_and_bounded(self):
        labels = ['a'] * 20 + ['b'] * 5 + ['c'] * 10
        first = balanced_indices(labels, 15, 7)
        second = balanced_indices(labels, 15, 7)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertLessEqual(len(first), 15)
        selected = [labels[index] for index in first]
        self.assertEqual(selected.count('b'), 5)

    def test_paper_style_token_types(self):
        self.assertEqual(selected_token_type('def', 'def', 'python'), 'def')
        self.assertEqual(
            selected_token_type('field', 'field_identifier', 'go'), 'identifier'
        )
        self.assertEqual(selected_token_type('some string', '"', 'java'), '"')
        self.assertIsNone(selected_token_type('public', 'public', 'java'))

    def test_seeded_program_subset_is_deterministic(self):
        values = [f'{index}.pkl' for index in range(20)]
        self.assertEqual(seeded_subset(values, 5, 11), seeded_subset(values, 5, 11))
        self.assertEqual(len(seeded_subset(values, 5, 11)), 5)

    def test_shortest_program_selection_matches_python_notebook_intent(self):
        values = ['a.pkl', 'b.pkl', 'c.pkl']
        lengths = {'a.pkl': 120, 'b.pkl': 101, 'c.pkl': 110}
        self.assertEqual(
            select_program_subset(values, lengths, 2, 'shortest', 0),
            ['b.pkl', 'c.pkl'],
        )

    def test_artifact_lengths_come_from_aligned_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code_file = root / 'codes.jsonl'
            code_file.write_text(
                json.dumps({'code_tokens': ['a', 'b']}) + '\n'
                + json.dumps({'code_tokens': ['c']}) + '\n'
            )
            graph_manifest = root / 'graph_manifest.json'
            graph_manifest.write_text(json.dumps({'code_file': str(code_file)}))
            manifest = {
                'graph_manifest': str(graph_manifest),
                'artifacts': ['000000_a.pkl', '000001_b.pkl'],
            }
            lengths, source = load_artifact_token_lengths(root, manifest)
            self.assertEqual(lengths, {'000000_a.pkl': 2, '000001_b.pkl': 1})
            self.assertEqual(source, str(code_file.resolve()))

    def test_layers_are_resolved_from_model_artifact_shape(self):
        import pickle
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'sample.pkl'
            with artifact.open('wb') as handle:
                pickle.dump({'hidden_repr': np.zeros((7, 2, 8))}, handle)
            manifest = {'model': 'plbart', 'artifacts': [artifact.name]}
            layers, states = resolve_layers(root, manifest, None)
            self.assertEqual(states, 7)
            self.assertEqual(layers, [0, 3, 6])
            with self.assertRaisesRegex(ValueError, '0..6'):
                resolve_layers(root, manifest, [7])


if __name__ == '__main__':
    unittest.main()
