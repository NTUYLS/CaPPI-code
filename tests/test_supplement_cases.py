"""Scientific contracts for new target ranks and validation-only trajectories."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_supplement_cases as cases
from cappi_model import CaPPI, ModelConfig


def fixture():
    return {"n_nodes": 6, "node_ids": np.array([11, 91, 35, 62, 20, 7]),
            "sequences": ["ACDE", "CDEF", "DEFG", "EFGH", "FGHI", "GHIK"],
            "train_pairs": np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4]]),
            "train_labels": np.array([1, 0, 1, 0, 1, 0]),
            "val_pairs": np.array([[0, 3], [1, 4], [2, 4]]), "val_labels": np.array([1, 1, 0]),
            "test_pairs": np.array([[0, 4], [0, 5]]), "test_labels": np.array([1, 0]),
            "train_nodes": np.arange(5), "val_nodes": np.arange(5), "test_nodes": np.array([0, 4, 5]),
            "graph_edges": np.array([[0, 1], [1, 2], [2, 3]]), "split_hash": "base-fixture"}


class SupplementCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mapping_uses_sequence_not_numeric_name(self):
        data = fixture()
        ids = ["s.X271_00015", "s.X271_00001", "s.C", "s.D", "s.E", "s.F"]
        fasta = dict(zip(reversed(ids), reversed(data["sequences"])))
        positive = np.array([[0, 1], [2, 3]])
        rows, targets = cases.map_identifiers(data, fasta, positive, [(ids[1], ids[0]), (ids[2], ids[3])])
        self.assertEqual(targets["X271_00001"]["node_id"], 91)
        self.assertEqual(targets["X271_00015"]["index"], 0)
        self.assertEqual(rows[0]["source_id"], "s.X271_00015")

    def test_mapping_rejects_ambiguous_sequence(self):
        data = fixture()
        fasta = {"a": data["sequences"][0], "b": data["sequences"][0]}
        with self.assertRaisesRegex(ValueError, "2 exact FASTA"):
            cases.map_identifiers(data, fasta, np.array([]), [])

    def test_mapping_rejects_source_edge_mismatch(self):
        data = fixture(); fasta = {str(i): seq for i, seq in enumerate(data["sequences"])}
        with self.assertRaisesRegex(ValueError, "positive edge set differ"):
            cases.map_identifiers(data, fasta, np.array([[0, 1]]), [])

    def test_target_removal_preserves_other_membership_and_all_candidates(self):
        base = fixture(); original = base["train_pairs"].copy()
        positives = np.array([[0, 1], [0, 3], [0, 4], [1, 2], [1, 4], [2, 3]])
        data, audit = cases.target_split(base, 0, positives)
        self.assertFalse(np.any(data["train_pairs"] == 0))
        self.assertFalse(np.any(data["val_pairs"] == 0))
        self.assertFalse(np.any(data["graph_edges"] == 0))
        np.testing.assert_array_equal(data["train_pairs"], original[2:])
        np.testing.assert_array_equal(base["train_pairs"], original)
        self.assertEqual(len(data["test_pairs"]), 5)
        self.assertEqual(int(data["test_labels"].sum()), 3)
        self.assertEqual(audit["excluded_pairs"], {"train": 2, "val": 1})
        self.assertNotEqual(data["split_hash"], base["split_hash"])

    def test_target_split_identity_covers_labels(self):
        base = fixture(); positives = np.array([[0, 1], [1, 2]])
        first, _ = cases.target_split(base, 0, positives)
        second, _ = cases.target_split(base, 0, np.array([[0, 2], [1, 2]]))
        self.assertNotEqual(first["split_hash"], second["split_hash"])

    def test_symmetric_hidden_is_order_invariant_and_recovers_logits(self):
        model = CaPPI(ModelConfig(seq_dim=16, topo_dim=4, filters=4, max_length=8)).eval()
        vectors = torch.randn(5, 16); pairs = torch.tensor([[0, 1], [3, 2], [1, 4]])
        with torch.no_grad():
            hidden = cases.symmetric_hidden(model, vectors, pairs)
            reverse = cases.symmetric_hidden(model, vectors, pairs.flip(1))
            logits = model.pair_head.classifier[-1](hidden).squeeze(-1)
            reference = model.score_pairs(vectors, pairs)
        self.assertEqual(tuple(hidden.shape), (3, 32))
        torch.testing.assert_close(hidden, reverse, rtol=0, atol=0)
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-6)
        model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            cases.symmetric_hidden(model, vectors, pairs)

    def test_joint_umap_rejects_changed_validation_membership_before_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            for epoch in cases.SNAPSHOTS:
                np.savez_compressed(Path(folder) / f"validation_hidden_epoch_{epoch}.npz",
                    features=np.ones((2, 32)), pairs=np.array([[0, 1], [1, 2 if epoch != 600 else 3]]),
                    labels=np.array([0, 1]))
            with self.assertRaisesRegex(ValueError, "different validation samples"):
                cases.plot_trajectory(folder, 42)

    def test_tiny_trajectory_saves_real_snapshots_and_guards_completed_integrity(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            root = Path(folder); (root / "manifest.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(prepared=str(root), output=str(root), seed=42, device="cpu", resume=False)
            data = fixture()
            tiny = ModelConfig(seq_dim=16, topo_dim=4, filters=4, max_length=8)
            embedding = np.arange(24, dtype=np.float32).reshape(6, 4) / 24
            present = np.array([True, True, True, False, False, False]); embedding[~present] = 0
            stack.enter_context(mock.patch.object(cases, "ModelConfig", return_value=tiny))
            stack.enter_context(mock.patch.object(cases, "train_topology", return_value=(embedding, present, "tiny-graph")))
            stack.enter_context(mock.patch.object(cases, "TRAJECTORY_MAX_EPOCHS", 2))
            stack.enter_context(mock.patch.object(cases, "SNAPSHOTS", (1, 2)))
            result = cases.train_case(args, data, root / "trajectory", trajectory=True)
            self.assertEqual(result["epochs_completed"], 2)
            self.assertIsNone(result["config"]["early_stopping_patience"])
            for epoch in (1, 2):
                with np.load(root / "trajectory" / f"validation_hidden_epoch_{epoch}.npz") as z:
                    self.assertEqual(z["features"].shape, (len(data["val_pairs"]), 32))
                    np.testing.assert_array_equal(z["pairs"], data["val_pairs"])
            repeated = cases.train_case(args, data, root / "trajectory", trajectory=True)
            self.assertEqual(repeated["run_signature"], result["run_signature"])
            with (root / "trajectory" / "snapshot_epoch_1.pt").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                cases.train_case(args, data, root / "trajectory", trajectory=True)


if __name__ == "__main__":
    unittest.main()
