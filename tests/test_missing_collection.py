"""Synthetic schema/provenance tests; these fixtures are not research results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_missing_topology import (collect_results, paired_differences,
                                     sha, verification_rows)
from verify_missing_topology import independent_metrics


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class TestMissingCollection(unittest.TestCase):
    def build_fixture(self, directory):
        base = Path(directory)
        root, result = base / "synthetic_runs", base / "synthetic_results"
        root.mkdir()
        actual = Path(__file__).resolve().parents[1] / "runs" / "repair_20261007" / "campaign_contract.json"
        contract = json.loads(actual.read_text(encoding="utf-8"))
        write_json(root / "campaign_contract.json", contract)
        indexed, verified = [], []
        for condition in contract["conditions"]:
            for family in condition["families"]:
                for fold in contract["folds"]:
                    for repeat in contract["repeat_indices"]:
                        folder = root / f'{condition["protocol"]}_{condition["negative"]}_{family}_{fold}_{repeat}'
                        folder.mkdir()
                        job = {"out": str(folder), **{k: condition[k] for k in ("protocol", "negative")},
                               "family": family, "fold": fold, "repeat": repeat,
                               "model_seed": 20260930 + 1000 * repeat + fold,
                               "graph_seed": 20260930 + fold, "learning_rate": .0005}
                        indexed.append(job)
                        labels = np.array([1, 0, 1, 0])
                        scores = np.array([.8, .7, .6, .1]) if repeat != 1 else np.array([.8, .2, .6, .1])
                        metrics = independent_metrics(labels, scores)
                        for partition in ("validation", "test"):
                            with (folder / f"{partition}_predictions.csv").open("w", newline="", encoding="utf-8") as h:
                                w = csv.writer(h)
                                w.writerow(["protein_u", "protein_v", "label", "probability", "prediction"])
                                for i, (label, score) in enumerate(zip(labels, scores)):
                                    w.writerow([i + 10, i + 20, label, score, int(score >= .5)])
                        identity = {"run_signature": "synthetic-signature", "checkpoint_sha256": "synthetic-checkpoint",
                                    "split_hash": f'synthetic-{condition["protocol"]}-{condition["negative"]}-{fold}'}
                        declaration = {"selected_before_test": True, **identity, "selection_rule": "synthetic test fixture",
                                       "repeat": repeat, "learning_rate": .0005}
                        write_json(folder / "selection_frozen.json", declaration)
                        val_sha = sha(folder / "validation_predictions.csv")
                        test_sha = sha(folder / "test_predictions.csv")
                        freeze_sha = sha(folder / "selection_frozen.json")
                        complete = {**identity, "test_evaluated": False, "best_epoch": 1, "epochs_completed": 21,
                                    "best_validation_bce": .4, "validation_metrics": metrics,
                                    "artifact_sha256": {"validation_predictions.csv": val_sha},
                                    "config": {"settings": {"family": family}, "fold": fold,
                                               "model_seed": job["model_seed"], "graph_seed": job["graph_seed"],
                                               "protocol": job["protocol"], "negative_sampling": job["negative"],
                                               "learning_rate": .0005}}
                        write_json(folder / "completed.json", complete)
                        test = {**identity, "graph_hash": f'synthetic-graph-{condition["protocol"]}-{condition["negative"]}-{fold}',
                                "test_evaluated": True, "evaluation_partition": "test", "test_metrics": metrics,
                                "selection_declaration_sha256": freeze_sha,
                                "artifact_sha256": {"test_predictions.csv": test_sha}}
                        write_json(folder / "test_completed.json", test)
                        verified.append({"run_dir": str(folder), "status": "test_evaluated", **identity,
                                         "validation_prediction_sha256": val_sha, "test_prediction_sha256": test_sha,
                                         "selection_freeze_sha256": freeze_sha,
                                         "validation_metrics": metrics, "test_metrics": metrics})
        index = {"runs": indexed, "selected_run_count": len(indexed)}
        write_json(root / "selected_index.json", index)
        report = {"status": "passed", "scope": "selected_test", "verified_runs": len(verified), "runs": verified,
                  "selected_index_sha256": sha(root / "selected_index.json")}
        report_path = base / "synthetic_verification.json"
        write_json(report_path, report)
        return root, result, report_path

    def test_complete_225_run_schema_and_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, result, verification = self.build_fixture(directory)
            summary = collect_results(root, result, verification)
            detail = pd.read_csv(result / "run_metrics.csv")
            seeds = pd.read_csv(result / "seed_summary.csv")
            paired = pd.read_csv(result / "paired_differences.csv")
            self.assertEqual(len(detail), 225)
            self.assertEqual(len(summary), 15)
            self.assertEqual(len(seeds), 45)
            self.assertEqual(len(paired), 240)
            self.assertTrue(np.allclose(paired.test_auroc_delta, 0))
            row = summary.iloc[0]
            expected = detail.loc[(detail.protocol == row.protocol) & (detail.negative == row.negative)
                                  & (detail.family == row.family)].groupby("repeat").test_auroc.mean()
            self.assertAlmostEqual(row.test_auroc_mean, expected.mean())
            self.assertAlmostEqual(row.test_auroc_seed_sd, expected.std(ddof=1))
            self.assertTrue((result / "RESULTS.md").is_file())
            report_text = (result / "RESULTS.md").read_text(encoding="utf-8")
            self.assertIn("与单模态模型的直接比较", report_text)
            self.assertIn("3 次重复的折均值 AUC 均未超过仅序列", report_text)
            self.assertIn("3 次重复的折均值 AUC 均未超过仅拓扑", report_text)
            self.assertEqual(json.loads((result / "collection_status.json").read_text())["status"], "complete_verified")

    def test_final_collection_rejects_incomplete_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root, result, verification = self.build_fixture(directory)
            report = json.loads(verification.read_text())
            report["scope"] = "validation_only_pilot"
            write_json(verification, report)
            with self.assertRaises(ValueError):
                collect_results(root, result, verification)
            self.assertFalse((result / "RESULTS.md").exists())

    def test_final_collection_rejects_changed_prediction_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root, result, verification = self.build_fixture(directory)
            index = json.loads((root / "selected_index.json").read_text())
            path = Path(index["runs"][0]["out"]) / "test_predictions.csv"
            path.write_text(path.read_text() + "10,20,1,0.8,1\n")
            with self.assertRaises(ValueError):
                collect_results(root, result, verification)
            self.assertFalse((result / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
