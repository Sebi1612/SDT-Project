import unittest

import numpy as np

from create_multilingual_dp import (
    TASK_LABELS,
    balanced_sample,
    distance_candidates,
    is_identifier,
    seeded_program_subset,
)


class MultilingualDirectProbeTests(unittest.TestCase):
    def test_identifier_variants_use_parser_types(self):
        self.assertTrue(is_identifier({'type': 'field_identifier'}))
        self.assertFalse(is_identifier({'type': 'string_literal'}))

    def test_distance_candidates_apply_keyword_and_identifier_constraints(self):
        data = {
            'language': 'go',
            'code_token_info': [
                {'token': 'if', 'type': 'if'},
                {'token': 'x', 'type': 'identifier'},
                {'token': '>', 'type': '>'},
            ],
            'tree_dist': np.asarray([[0, 2, 3], [2, 0, 1], [3, 1, 0]]),
        }
        all_pairs = distance_candidates('a.pkl', data, False)
        id_pairs = distance_candidates('a.pkl', data, True)
        self.assertEqual([pair['label'] for pair in all_pairs], ['2', '3'])
        self.assertEqual([pair['label'] for pair in id_pairs], ['2'])

    def test_balanced_sampling_is_deterministic(self):
        records = []
        for label in TASK_LABELS['siblings']:
            for column in range(10):
                records.append({
                    'artifact': 'a', 'row': 0, 'column': column,
                    'label': label,
                })
        first = balanced_sample(
            records, TASK_LABELS['siblings'], 4, np.random.default_rng(3)
        )[0]
        second = balanced_sample(
            records, TASK_LABELS['siblings'], 4, np.random.default_rng(3)
        )[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_program_sampling_is_deterministic(self):
        artifacts = [f'{index}.pkl' for index in range(30)]
        first = seeded_program_subset(artifacts, 7, np.random.default_rng(5))
        second = seeded_program_subset(artifacts, 7, np.random.default_rng(5))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)


if __name__ == '__main__':
    unittest.main()
