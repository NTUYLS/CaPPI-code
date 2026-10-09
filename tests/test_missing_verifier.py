"""Small rejection tests for the independent repair-campaign verifier."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_missing_topology import (independent_checkpoint_epoch, independent_metrics,
                                    mask_counts, select_learning_rate, verify_predictions)


class TestMissingVerifier(unittest.TestCase):
    def test_average_precision_is_not_trapezoidal_pr_area(self):
        result = independent_metrics(np.array([1, 0, 1, 0]), np.array([0.8, 0.7, 0.6, 0.1]))
        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["pr_trapezoid_area"], 19 / 24)
        self.assertNotEqual(result["average_precision"], result["pr_trapezoid_area"])
        self.assertAlmostEqual(result["accuracy"], 0.75)

    def test_nonfinite_or_one_class_probabilities_fail(self):
        with self.assertRaises(ValueError):
            independent_metrics(np.array([0, 1]), np.array([0.1, float("nan")]))
        with self.assertRaises(ValueError):
            independent_metrics(np.array([1, 1]), np.array([0.2, 0.3]))

    def test_validation_checkpoint_uses_declared_min_delta(self):
        history = [{"epoch": 1, "validation_bce": 0.5},
                   {"epoch": 2, "validation_bce": 0.5 - 0.5e-6},
                   {"epoch": 3, "validation_bce": 0.4}]
        self.assertEqual(independent_checkpoint_epoch(history), (3, 0.4))
        self.assertEqual(independent_checkpoint_epoch(history[:2]), (1, 0.5))

    def test_lr_tie_selects_lower_lr_and_rejects_mixed_fold(self):
        trials = [{"family": "residual", "negative": "random", "protocol": "pair",
                   "fold": 0, "model_seed": 20260930, "graph_seed": 20260930,
                   "learning_rate": lr, "best_validation_bce": 0.4} for lr in (0.001, 0.0005)]
        self.assertEqual(select_learning_rate(trials)["learning_rate"], 0.0005)
        trials[1]["fold"] = 1
        with self.assertRaises(ValueError):
            select_learning_rate(trials)

    def test_mask_coverage_distinguishes_both_labels(self):
        data = {"train_pairs": np.array([[0, 1], [1, 2], [0, 2], [2, 3]]),
                "train_labels": np.array([1, 1, 0, 0])}
        result = mask_counts(np.array([False, True, False, False]),
                             np.array([True, True, True, False]), data)
        self.assertEqual(result["positive"]["masked_endpoints"], 2)
        self.assertEqual(result["negative"]["masked_endpoints"], 3)
        self.assertEqual(result["negative"]["effective_missing_endpoints"], 4)

    def test_predictions_reject_changed_labels_and_ids(self):
        fields = ["protein_u", "protein_v", "label", "probability", "prediction", "fold",
                  "model_seed", "graph_seed", "split_hash", "checkpoint_sha256", "partition"]
        data = {"node_ids": np.array([11, 22, 33]), "val_pairs": np.array([[0, 1], [1, 2]]),
                "val_labels": np.array([1, 0], dtype=np.float32), "split_hash": "frozen"}
        identity = {"fold": 0, "model_seed": 5, "graph_seed": 4}
        rows = [[11, 22, 1, 0.8, 1, 0, 5, 4, "frozen", "checkpoint", "validation"],
                [22, 33, 0, 0.2, 0, 0, 5, 4, "frozen", "checkpoint", "validation"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_predictions.csv"
            def write_rows():
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows)
            write_rows()
            metrics, _ = verify_predictions(path, data, "val", identity, "checkpoint")
            self.assertEqual(metrics["accuracy"], 1.0)
            rows[0][2] = 0
            write_rows()
            with self.assertRaises(ValueError):
                verify_predictions(path, data, "val", identity, "checkpoint")
            rows[0][2], rows[0][0] = 1, 99
            write_rows()
            with self.assertRaises(ValueError):
                verify_predictions(path, data, "val", identity, "checkpoint")


if __name__ == "__main__":
    unittest.main()
