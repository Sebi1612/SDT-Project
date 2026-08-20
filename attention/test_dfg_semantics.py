import os
import sys
import unittest

import numpy as np


ATTENTION_DIR = os.path.dirname(os.path.abspath(__file__))
if ATTENTION_DIR not in sys.path:
    sys.path.insert(0, ATTENTION_DIR)

from dfg_comp import build_parser, get_dfg_adj, metrics_for_layers
from graph_comp_utils import f_score, precision, recall
from utils import get_max_edges


def occurrence(tokens, token, number):
    indexes = [index for index, value in enumerate(tokens) if value == token]
    return indexes[number]


class DFGSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsers = {
            language: build_parser(language)
            for language in ('python', 'java', 'go', 'javascript')
        }

    def graph(self, language, code):
        adjacency, tokens = get_dfg_adj(
            code, self.parsers[language], lang=language
        )
        self.assertEqual(adjacency.shape, (len(tokens), len(tokens)))
        self.assertTrue(np.isin(adjacency, [0, 1]).all())
        return adjacency, tokens

    def typed_graph(self, language, code):
        adjacency, tokens = get_dfg_adj(
            code, self.parsers[language], lang=language, typed=True
        )
        self.assertTrue(np.isin(adjacency, [-1, 0, 1]).all())
        return adjacency, tokens

    def assert_edge(
        self,
        adjacency,
        tokens,
        source_token,
        source_occurrence,
        dependency_token,
        dependency_occurrence,
    ):
        source = occurrence(tokens, source_token, source_occurrence)
        dependency = occurrence(
            tokens, dependency_token, dependency_occurrence
        )
        self.assertEqual(
            adjacency[source, dependency],
            1,
            f'Missing {source_token}[{source_occurrence}] -> '
            f'{dependency_token}[{dependency_occurrence}] in {tokens}',
        )

    def test_java_declaration_assignment_and_use(self):
        adjacency, tokens = self.graph(
            'java',
            'void f(){ int a = 1; int b = 2; b = a; }',
        )
        self.assert_edge(adjacency, tokens, 'a', 0, '1', 0)
        self.assert_edge(adjacency, tokens, 'b', 0, '2', 0)
        self.assert_edge(adjacency, tokens, 'b', 1, 'a', 1)
        self.assert_edge(adjacency, tokens, 'a', 1, 'a', 0)

    def test_python_declaration_assignment_and_use(self):
        adjacency, tokens = self.graph(
            'python',
            'def f():\n    a = 1\n    b = 2\n    b = a',
        )
        self.assert_edge(adjacency, tokens, 'a', 0, '1', 0)
        self.assert_edge(adjacency, tokens, 'b', 0, '2', 0)
        self.assert_edge(adjacency, tokens, 'b', 1, 'a', 1)
        self.assert_edge(adjacency, tokens, 'a', 1, 'a', 0)

    def test_python_branch_merges_reaching_definitions(self):
        adjacency, tokens = self.graph(
            'python',
            'def f(c):\n    a = 0\n    if c:\n        a = 1\n    b = a',
        )
        self.assert_edge(adjacency, tokens, 'a', 2, 'a', 0)
        self.assert_edge(adjacency, tokens, 'a', 2, 'a', 1)
        self.assert_edge(adjacency, tokens, 'b', 0, 'a', 2)

    def test_java_branch_merges_reaching_definitions(self):
        adjacency, tokens = self.graph(
            'java',
            'void f(boolean c){ int a=0; if(c){ a=1; } int b=a; }',
        )
        self.assert_edge(adjacency, tokens, 'a', 2, 'a', 0)
        self.assert_edge(adjacency, tokens, 'a', 2, 'a', 1)
        self.assert_edge(adjacency, tokens, 'b', 0, 'a', 2)

    def test_java_enhanced_for_binding(self):
        adjacency, tokens = self.graph(
            'java',
            'void f(int[] xs){ for(int x: xs){ int y=x; } }',
        )
        self.assert_edge(adjacency, tokens, 'x', 0, 'xs', 1)
        self.assert_edge(adjacency, tokens, 'y', 0, 'x', 1)

    def test_go_short_declaration(self):
        adjacency, tokens = self.graph(
            'go',
            'func f() { a := 1; b := a }',
        )
        self.assert_edge(adjacency, tokens, 'a', 0, '1', 0)
        self.assert_edge(adjacency, tokens, 'b', 0, 'a', 1)
        self.assert_edge(adjacency, tokens, 'a', 1, 'a', 0)

    def test_go_range_binding(self):
        adjacency, tokens = self.graph(
            'go',
            'func f(xs []int) { for _, x := range xs { y := x; _ = y } }',
        )
        self.assert_edge(adjacency, tokens, 'x', 0, 'xs', 1)
        self.assert_edge(adjacency, tokens, 'y', 0, 'x', 1)

    def test_javascript_assignment(self):
        adjacency, tokens = self.graph(
            'javascript',
            'function f(){ let a=1; let b=2; b=a; }',
        )
        self.assert_edge(adjacency, tokens, 'a', 0, '1', 0)
        self.assert_edge(adjacency, tokens, 'b', 0, '2', 0)
        self.assert_edge(adjacency, tokens, 'b', 1, 'a', 1)
        self.assert_edge(adjacency, tokens, 'a', 1, 'a', 0)

    def test_computed_value_depends_on_every_identifier_operand(self):
        examples = {
            'python': 'def f(x):\n    y = x\n    z = y + x',
            'java': 'void f(int x){ int y=x; int z=y+x; }',
            'go': 'func f(x int){ y := x; z := y+x; _ = z }',
            'javascript': 'function f(x){ let y=x; let z=y+x; }',
        }
        for language, code in examples.items():
            with self.subTest(language=language):
                adjacency, tokens = self.graph(language, code)
                self.assert_edge(adjacency, tokens, 'z', 0, 'y', 1)
                self.assert_edge(adjacency, tokens, 'z', 0, 'x', 2)

    def test_binary_graph_is_absolute_native_typed_graph(self):
        examples = {
            'python': 'def f(x):\n    y = x',
            'java': 'void f(int x){ int y=x; }',
            'go': 'func f(x int){ y := x; _ = y }',
            'javascript': 'function f(x){ let y=x; }',
        }
        native_declaration_labels = {
            'python': -1,
            'java': 1,
            'go': -1,
            'javascript': 1,
        }
        for language, code in examples.items():
            with self.subTest(language=language):
                binary, tokens = self.graph(language, code)
                typed, typed_tokens = self.typed_graph(language, code)
                self.assertEqual(tokens, typed_tokens)
                np.testing.assert_array_equal(binary, np.abs(typed))
                y = occurrence(tokens, 'y', 0)
                x = occurrence(tokens, 'x', 1)
                self.assertEqual(
                    typed[y, x], native_declaration_labels[language]
                )

    def test_update_expression_has_self_dependency(self):
        for language, code in (
            ('java', 'void f(){ int a=1; a++; }'),
            ('go', 'func f(){ var a=1; a++ }'),
            ('javascript', 'function f(){ let a=1; a++; }'),
        ):
            with self.subTest(language=language):
                adjacency, tokens = self.graph(language, code)
                self.assert_edge(adjacency, tokens, 'a', 1, 'a', 1)

    def test_vectorized_metrics_match_original_implementation(self):
        random = np.random.default_rng(0)
        model_graphs = random.random((2, 3, 7, 7))
        dfg_graph = random.integers(0, 2, size=(7, 7))
        thresholds = [0, 0.05, 0.3, 0.9]
        actual = metrics_for_layers(
            model_graphs,
            dfg_graph,
            [0, 1],
            thresholds,
        )

        for metric_index, metric_function in enumerate(
            (f_score, recall, precision)
        ):
            for layer in range(model_graphs.shape[0]):
                for head in range(model_graphs.shape[1]):
                    for threshold_index, threshold in enumerate(thresholds):
                        thresholded = get_max_edges(
                            model_graphs[layer, head],
                            mode='threshold',
                            threshold=threshold,
                        )
                        expected = metric_function(thresholded, dfg_graph)
                        self.assertAlmostEqual(
                            actual[metric_index][layer, head, threshold_index],
                            expected,
                        )

    @unittest.expectedFailure
    def test_known_limitation_block_shadowing_restores_outer_definition(self):
        """GraphCodeBERT tracks names, not lexical scope, after a nested block."""
        adjacency, tokens = self.graph(
            'java',
            'void f(){ int x=1; { int x=2; int y=x; } int z=x; }',
        )
        self.assert_edge(adjacency, tokens, 'x', 3, 'x', 0)


if __name__ == '__main__':
    unittest.main()
