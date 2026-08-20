import unittest

import numpy as np
import torch

from dfg_comp import build_parser
from save_word_embedding import (
    aligned_ast_structure,
    merge_codebert_hidden_states,
)


class WordEmbeddingHelpersTests(unittest.TestCase):
    def test_tree_distances_are_symmetric_on_real_fixture(self):
        parser = build_parser('python')
        tokens = ['def', 'f', '(', 'x', ')', ':', 'return', 'x']
        matrix, token_info = aligned_ast_structure(
            'def f(x):\n    return x', tokens, parser, 'python'
        )
        self.assertEqual(matrix.shape, (len(tokens), len(tokens)))
        self.assertEqual([item['token'] for item in token_info], tokens)
        self.assertTrue(np.array_equal(matrix, matrix.T))
        self.assertTrue(np.array_equal(np.diag(matrix), np.zeros(len(tokens))))
        self.assertGreater(matrix.max(), 0)

    def test_subtoken_hidden_states_are_averaged(self):
        # Two lexical tokens: "hello" -> "he", "llo" and "x" -> "Ġx".
        states = tuple(
            torch.tensor([[[[0.0]], [[1.0]], [[3.0]], [[7.0]], [[0.0]]]]).reshape(1, 5, 1)
            + layer
            for layer in range(2)
        )
        merged = merge_codebert_hidden_states(
            states, ['he', 'llo', 'Ġx'], ['hello', 'x']
        )
        self.assertEqual(merged.shape, (2, 2, 1))
        self.assertAlmostEqual(float(merged[0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(merged[0, 1, 0]), 7.0)


if __name__ == '__main__':
    unittest.main()
