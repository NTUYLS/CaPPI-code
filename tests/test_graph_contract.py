"""Synthetic walk contracts; no embedding training or real datasets are used."""
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cappi_graph import GraphConfig, make_walks  # noqa: E402


def undirected(u, v):
    return (min(u, v), max(u, v))


class GraphWalkContractTests(unittest.TestCase):
    # Two triangles joined at node 1 and a tail: graph structure gives all three
    # policies a choice between returning, revisiting, and exploring new nodes.
    edges = np.asarray([
        [0, 1], [1, 2], [0, 2], [1, 3], [3, 4], [1, 4], [4, 5], [5, 6],
    ], dtype=np.int64)
    n_nodes = 8  # Node 7 is absent from the training graph.
    methods = ('drw', 'deepwalk', 'node2vec')

    def config(self, method, **overrides):
        values = dict(method=method, walks_per_node=8, walk_length=20,
                      p=0.25, q=4.0)
        values.update(overrides)
        return GraphConfig(**values)

    def test_walk_steps_use_only_train_adjacency_and_exclude_held_out_edges(self):
        train = {undirected(*edge) for edge in self.edges}
        held_out = {(0, 3), (2, 4), (6, 7)}
        self.assertTrue(train.isdisjoint(held_out))
        original = self.edges.copy()
        for method in self.methods:
            with self.subTest(method=method):
                config = self.config(method)
                walks, _ = make_walks(self.edges, self.n_nodes, config, seed=19)
                self.assertEqual(len(walks), 7 * config.walks_per_node)
                starts = Counter(walk[0] for walk in walks)
                self.assertEqual(starts, Counter({node: config.walks_per_node
                                                 for node in range(7)}))
                for walk in walks:
                    self.assertEqual(len(walk), config.walk_length)
                    self.assertTrue(all(0 <= node < self.n_nodes for node in walk))
                    for u, v in zip(walk, walk[1:]):
                        pair = undirected(u, v)
                        self.assertIn(pair, train)
                        self.assertNotIn(pair, held_out)
        np.testing.assert_array_equal(self.edges, original)

    def test_isolated_nodes_never_enter_walks_and_mask_matches_train_degree(self):
        for method in self.methods:
            with self.subTest(method=method):
                walks, present = make_walks(self.edges, self.n_nodes,
                                            self.config(method), seed=19)
                self.assertEqual(present.dtype, np.dtype(bool))
                self.assertEqual(present.tolist(), [True] * 7 + [False])
                visited = {node for walk in walks for node in walk}
                self.assertEqual(visited, set(range(7)))
                self.assertNotIn(7, visited)

    def test_empty_training_graph_produces_no_walks_and_all_false_mask(self):
        for method in self.methods:
            with self.subTest(method=method):
                walks, present = make_walks(np.empty((0, 2), dtype=np.int64),
                                            self.n_nodes, self.config(method), seed=4)
                self.assertEqual(walks, [])
                np.testing.assert_array_equal(present, np.zeros(self.n_nodes, dtype=bool))

    def test_seed_determinism_and_independence_from_global_random_state(self):
        for method in self.methods:
            with self.subTest(method=method):
                config = self.config(method)
                original_state = random.getstate()
                first, mask = make_walks(self.edges, self.n_nodes, config, seed=91)
                self.assertEqual(random.getstate(), original_state)
                random.seed(127)
                try:
                    second, second_mask = make_walks(self.edges, self.n_nodes,
                                                     config, seed=91)
                finally:
                    random.setstate(original_state)
                third, _ = make_walks(self.edges, self.n_nodes, config, seed=92)
                self.assertEqual(first, second)
                np.testing.assert_array_equal(mask, second_mask)
                self.assertNotEqual(first, third)

    def test_edge_order_orientation_and_duplicates_do_not_change_walks(self):
        equivalent_edges = np.concatenate([self.edges[::-1, ::-1], self.edges[:2]])
        for method in self.methods:
            with self.subTest(method=method):
                first, first_mask = make_walks(self.edges, self.n_nodes,
                                               self.config(method), seed=91)
                second, second_mask = make_walks(equivalent_edges, self.n_nodes,
                                                 self.config(method), seed=91)
                self.assertEqual(first, second)
                np.testing.assert_array_equal(first_mask, second_mask)

    def test_drw_visits_unseen_neighbor_whenever_one_exists(self):
        adjacency = {node: set() for node in range(self.n_nodes)}
        for u, v in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        walks, _ = make_walks(self.edges, self.n_nodes, self.config('drw'), seed=19)
        choices_checked = 0
        for walk in walks:
            seen = {walk[0]}
            for current, following in zip(walk, walk[1:]):
                if adjacency[current] - seen:
                    self.assertNotIn(following, seen)
                    choices_checked += 1
                seen.add(following)
        self.assertGreater(choices_checked, 100)

    def test_three_methods_produce_distinct_walks_and_node2vec_bias_has_effect(self):
        outputs = {}
        for method in self.methods:
            walks, _ = make_walks(self.edges, self.n_nodes,
                                  self.config(method, walks_per_node=40), seed=31)
            outputs[method] = walks
        for left, right in [('drw', 'deepwalk'), ('drw', 'node2vec'),
                            ('deepwalk', 'node2vec')]:
            self.assertNotEqual(outputs[left], outputs[right], (left, right))

        def backtracks(walks):
            return sum(walk[index - 1] == walk[index + 1]
                       for walk in walks for index in range(1, len(walk) - 1))

        # p=.25, q=4 strongly favors the preceding node. The fixed corpus is
        # large enough to distinguish the configured bias without flaky draws.
        self.assertGreater(backtracks(outputs['node2vec']),
                           backtracks(outputs['deepwalk']) * 1.4)
        neutral, _ = make_walks(self.edges, self.n_nodes,
                                self.config('node2vec', walks_per_node=40, p=1, q=1),
                                seed=31)
        self.assertGreater(backtracks(outputs['node2vec']), backtracks(neutral) * 1.4)

    def test_unsupported_method_nonpositive_bias_and_self_loop_are_rejected(self):
        with self.assertRaises(ValueError):
            make_walks(self.edges, self.n_nodes, self.config('unsupported'), seed=1)
        for p, q in [(0, 1), (1, 0), (-1, 1), (1, -1)]:
            with self.subTest(p=p, q=q), self.assertRaises(ValueError):
                make_walks(self.edges, self.n_nodes,
                           self.config('node2vec', p=p, q=q), seed=1)
        for method in self.methods:
            with self.subTest(method=method), self.assertRaises(ValueError):
                make_walks(np.asarray([[0, 0]]), self.n_nodes,
                           self.config(method), seed=1)


if __name__ == '__main__':
    unittest.main()
