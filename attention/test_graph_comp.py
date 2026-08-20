import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(__file__))

from graph_comp import metrics_for_layer
from graph_comp_utils import IoU, f_score, precision, recall
from experiment_protocol import bootstrap_mean_ci, threshold_index
from utils import get_max_edges


class GraphComparisonMetricsTest(unittest.TestCase):
    def test_program_bootstrap_is_deterministic_and_contains_observed_mean(self):
        values = np.arange(60, dtype=float).reshape(20, 3)
        lower_1, upper_1 = bootstrap_mean_ci(values, 500, 0.95, seed=7)
        lower_2, upper_2 = bootstrap_mean_ci(values, 500, 0.95, seed=7)
        np.testing.assert_array_equal(lower_1, lower_2)
        np.testing.assert_array_equal(upper_1, upper_2)
        observed = values.mean(axis=0)
        self.assertTrue(np.all(lower_1 <= observed))
        self.assertTrue(np.all(observed <= upper_1))

    def test_primary_threshold_must_be_in_sweep_once(self):
        self.assertEqual(threshold_index([0, 0.05, 0.1], 0.05), 1)
        with self.assertRaises(ValueError):
            threshold_index([0, 0.1], 0.05)

    def test_vectorized_metrics_match_original_implementation(self):
        random = np.random.default_rng(0)
        layer_graph = random.random((3, 7, 7))
        ast_graph = random.integers(0, 2, size=(7, 7))
        thresholds = [0, 0.05, 0.3, 0.9]

        actual = metrics_for_layer(layer_graph, ast_graph, thresholds)
        metric_functions = [f_score, recall, precision, IoU]

        for metric_index, metric_function in enumerate(metric_functions):
            for head in range(layer_graph.shape[0]):
                for threshold_index, threshold in enumerate(thresholds):
                    thresholded = get_max_edges(
                        layer_graph[head],
                        mode='threshold',
                        threshold=threshold,
                    )
                    expected = metric_function(thresholded, ast_graph)
                    self.assertAlmostEqual(
                        actual[metric_index][head, threshold_index],
                        expected,
                    )


if __name__ == '__main__':
    unittest.main()
