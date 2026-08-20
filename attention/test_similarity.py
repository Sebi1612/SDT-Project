import unittest

import networkx as nx
import numpy as np

from similarity import (
    distances_for_layers,
    legacy_distances_for_layers,
    legacy_unique_token_names,
    normalized_fixed_node_edge_distance,
    normalized_legacy_networkx_distance,
    similarity_directory,
)


class SimilarityTests(unittest.TestCase):
    def test_fixed_node_distance_counts_insertions_and_deletions(self):
        prediction = np.asarray([[0, 1], [1, 0]])
        truth = np.asarray([[0, 1], [0, 1]])
        self.assertEqual(
            normalized_fixed_node_edge_distance(prediction, truth), 1.0
        )

    def test_vectorized_distance_matches_scalar_definition(self):
        rng = np.random.default_rng(9)
        attention = rng.random((3, 2, 5, 5))
        truth = rng.integers(0, 2, size=(5, 5), dtype=np.int8)
        result = distances_for_layers(attention, {'ast': truth}, 0.4)['ast']
        for layer in range(3):
            for head in range(2):
                expected = normalized_fixed_node_edge_distance(
                    attention[layer, head] > 0.4, truth
                )
                self.assertAlmostEqual(result[layer, head], expected)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_fixed_node_edge_distance(
                np.zeros((2, 2)), np.zeros((3, 3))
            )

    def test_legacy_distance_matches_historical_networkx_call(self):
        prediction = np.asarray([
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        truth = np.asarray([
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])
        tokens = ['x', '+', 'x', 'y']

        model_graph = nx.from_numpy_array(
            prediction, create_using=nx.DiGraph
        )
        truth_graph = nx.from_numpy_array(truth, create_using=nx.DiGraph)
        names = legacy_unique_token_names(tokens)
        attributes = {
            index: {'name': name} for index, name in enumerate(names)
        }
        nx.set_node_attributes(model_graph, attributes)
        nx.set_node_attributes(truth_graph, attributes)
        historical = next(nx.optimize_graph_edit_distance(
            model_graph,
            truth_graph,
            node_match=lambda left, right: left['name'] == right['name'],
            edge_del_cost=lambda edge: 1,
            edge_ins_cost=lambda edge: 1,
            edge_subst_cost=lambda left, right: 0,
        )) / len(tokens)

        self.assertEqual(tokens, ['x', '+', 'x', 'y'])
        self.assertAlmostEqual(
            normalized_legacy_networkx_distance(
                prediction, truth, tokens
            ),
            historical,
        )

    def test_layered_legacy_distance_matches_scalar_definition(self):
        attention = np.asarray([[[
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.7],
            [0.8, 0.0, 0.0],
        ]]])
        truth = np.asarray([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = legacy_distances_for_layers(
            attention, {'ast': truth}, 0.5, ['a', 'b', 'c']
        )['ast']
        expected = normalized_legacy_networkx_distance(
            attention[0, 0] > 0.5, truth, ['a', 'b', 'c']
        )
        self.assertAlmostEqual(result[0, 0], expected)

    def test_distance_modes_use_separate_directories(self):
        self.assertEqual(similarity_directory('fixed'), 'similarity')
        self.assertEqual(
            similarity_directory('legacy'), 'similarity_legacy'
        )


if __name__ == '__main__':
    unittest.main()
