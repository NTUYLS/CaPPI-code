"""Leakage and provenance tests using synthetic IDs unrelated to row offsets."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cappi_data import load_fold, prepare_dataset


def pair_set(pairs):
    return set(map(tuple, np.asarray(pairs).tolist()))


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / "raw"
        cls.raw.mkdir()
        cls.ids = np.arange(80) * 17 + 101
        rng = np.random.default_rng(92)
        cls.positives = {(u, v) for u in range(80) for v in range(u + 1, 80) if rng.random() < .25}
        with (cls.raw / "肝癌 sequence.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "sequence"])
            writer.writerows((int(node_id), "ACDEFGHIKLMNPQRSTVWY" + "A" * index)
                             for index, node_id in enumerate(cls.ids))
        with (cls.raw / "肝癌 interact.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["node1", "node2"])
            writer.writerows((cls.ids[u], cls.ids[v]) for u, v in sorted(cls.positives))
            u, v = sorted(cls.positives)[0]
            writer.writerow([cls.ids[v], cls.ids[u]])
            writer.writerow([cls.ids[u], cls.ids[v]])
            writer.writerow([cls.ids[0], cls.ids[0]])
        cls.prepared = cls.root / "prepared"
        cls.manifest = prepare_dataset(cls.raw, cls.prepared, seed=23)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_mapping_and_cleaning(self):
        fold = load_fold(self.prepared, "random", "pair", 0)
        np.testing.assert_array_equal(fold["node_ids"], self.ids)
        self.assertEqual(self.manifest["cleaning"]["duplicate_undirected_rows_removed"], 2)
        self.assertEqual(self.manifest["cleaning"]["self_loops_removed"], 1)
        self.assertEqual(self.manifest["n_positives"], len(self.positives))

    def test_pair_outer_folds_cover_each_pair_once_and_graph_has_no_holdouts(self):
        positive_memberships = {}
        for mode in ("random", "degree_matched"):
            all_tests = set()
            all_test_positives = set()
            for index in range(5):
                fold = load_fold(self.prepared, mode, "pair", index)
                splits = [pair_set(fold[f"{split}_pairs"]) for split in ("train", "val", "test")]
                self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
                self.assertFalse(all_tests & splits[2])
                all_tests |= splits[2]
                test_positive = pair_set(fold["test_pairs"][fold["test_labels"] == 1])
                all_test_positives |= test_positive
                if mode == "random":
                    positive_memberships[index] = test_positive
                else:
                    self.assertEqual(positive_memberships[index], test_positive)
                self.assertEqual(pair_set(fold["graph_edges"]), pair_set(fold["train_pairs"][fold["train_labels"] == 1]))
                self.assertFalse(pair_set(fold["graph_edges"]) & (splits[1] | splits[2]))
                for split in ("train", "val", "test"):
                    pairs, labels = fold[f"{split}_pairs"], fold[f"{split}_labels"]
                    self.assertEqual(len(pairs), len(pair_set(pairs)))
                    self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
                    self.assertFalse(pair_set(pairs[labels == 0]) & self.positives)
            with np.load(self.prepared / f"negatives_{mode}.npz") as data:
                negative_pool = pair_set(data["pairs"])
            self.assertEqual(all_tests, self.positives | negative_pool)
            self.assertEqual(all_test_positives, self.positives)
            self.assertEqual(len(negative_pool), len(self.positives))

    def test_protein_disjoint_and_same_assignments_across_negative_modes(self):
        tested_nodes = set()
        for index in range(5):
            random_fold = load_fold(self.prepared, "random", "protein", index)
            matched_fold = load_fold(self.prepared, "degree_matched", "protein", index)
            partitions = [set(random_fold[f"{split}_nodes"]) for split in ("train", "val", "test")]
            self.assertEqual(set.union(*partitions), set(range(80)))
            self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
            self.assertFalse(tested_nodes & partitions[2])
            tested_nodes |= partitions[2]
            for split in ("train", "val", "test"):
                np.testing.assert_array_equal(random_fold[f"{split}_nodes"], matched_fold[f"{split}_nodes"])
                self.assertTrue(set(random_fold[f"{split}_pairs"].reshape(-1)).issubset(set(random_fold[f"{split}_nodes"])))
                self.assertTrue(set(matched_fold[f"{split}_pairs"].reshape(-1)).issubset(set(matched_fold[f"{split}_nodes"])))
                self.assertEqual(pair_set(random_fold[f"{split}_pairs"][random_fold[f"{split}_labels"] == 1]),
                                 pair_set(matched_fold[f"{split}_pairs"][matched_fold[f"{split}_labels"] == 1]))
            self.assertTrue(set(random_fold["graph_edges"].reshape(-1)).issubset(partitions[0]))
        self.assertEqual(tested_nodes, set(range(80)))

    def test_existing_outputs_reject_changed_config_and_tampered_artifacts(self):
        same = prepare_dataset(self.raw, self.prepared, seed=23)
        self.assertEqual(same["dataset_hash"], self.manifest["dataset_hash"])
        with self.assertRaisesRegex(ValueError, "different inputs/configuration"):
            prepare_dataset(self.raw, self.prepared, seed=24)
        artifact = self.prepared / "interaction_cleaning.csv"
        original = artifact.read_bytes()
        try:
            artifact.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "modified"):
                load_fold(self.prepared, "random", "pair", 0)
        finally:
            artifact.write_bytes(original)

    def test_manifest_fingerprint_rejects_altered_protocol_metadata(self):
        artifact = self.prepared / "manifest.json"
        original = artifact.read_bytes()
        try:
            altered = json.loads(original)
            altered["seed"] = 999
            artifact.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration fingerprint"):
                load_fold(self.prepared, "random", "pair", 0)
        finally:
            artifact.write_bytes(original)

    def test_negative_pool_index_assignments_identical_across_modes(self):
        indexes = {}
        for mode in ("random", "degree_matched"):
            with np.load(self.prepared / f"negatives_{mode}.npz") as data:
                lookup = {tuple(pair): i for i, pair in enumerate(data["pairs"])}
            for index in range(5):
                fold = load_fold(self.prepared, mode, "pair", index)
                for split in ("train", "val", "test"):
                    pairs = fold[f"{split}_pairs"][fold[f"{split}_labels"] == 0]
                    membership = {lookup[tuple(pair)] for pair in pairs}
                    key = (index, split)
                    if mode == "random":
                        indexes[key] = membership
                    else:
                        self.assertEqual(indexes[key], membership)

    def test_reproducibility_across_fresh_output_directories(self):
        second = prepare_dataset(self.raw, self.root / "prepared_again", seed=23)
        self.assertEqual(second["dataset_hash"], self.manifest["dataset_hash"])
        self.assertEqual(second["files"], self.manifest["files"])
        self.assertIn("complete positive network", second["negative_sampling"]["benchmark_design_disclosure"])

    def test_degree_matching_discloses_unavailable_exact_matches(self):
        # A star has no nonedges incident on its central high-degree node.
        from cappi_data import _negative_pools
        star = np.asarray([(0, other) for other in range(1, 15)], dtype=np.int64)
        pools, audit = _negative_pools(star, 15, 45)
        self.assertFalse(audit["matching_is_exact"])
        self.assertEqual(audit["fallback_count"], len(star))
        self.assertEqual(audit["exact_bin_matches"], 0)
        self.assertEqual(len(pools["degree_matched"]), len(star))
        self.assertFalse(pair_set(star) & pair_set(pools["degree_matched"]))


if __name__ == "__main__":
    unittest.main()
