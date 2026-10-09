"""Verify the 40 feature-control runs against immutable prepared folds.

--check-partial verifies only a snapshot of completed runs and writes no results.
The default requires exactly 2 protocols x 5 folds x 4 controls before exporting.
No classifiers are loaded, trained, tuned, or selected by this script.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, auc,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)

from cappi_data import load_fold
from cappi_graph import GraphConfig, fingerprint
from run_classical_baselines import code_hash, fixed_model_config

HERE = Path(__file__).resolve().parent
PROTOCOLS = ("pair", "protein")
METHODS = ("degree_product", "common_neighbor", "aa20_drw_lr", "aa20_drw_rf")
LEARNED = {"aa20_drw_lr", "aa20_drw_rf"}
LABELS = {"degree_product": "Training-degree product", "common_neighbor": "Training common neighbors",
          "aa20_drw_lr": "AA20 + DRW logistic regression", "aa20_drw_rf": "AA20 + DRW random forest"}
METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "mcc",
           "auroc", "average_precision", "pr_trapezoid_area")
PREDICTION_COLUMNS = {
    "protein_u", "protein_v", "label", "raw_score", "probability", "prediction", "method",
    "protocol", "fold", "seed", "split_hash", "train_array_hash", "val_array_hash",
    "test_array_hash", "checkpoint_sha256",
}
FOLD_CONFIG_KEYS = {"seed", "fold", "split_hash", "split_array_hashes", "prepared_artifact_hashes", "graph_hash"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, probabilistic: bool) -> dict:
    """Independently recompute every exported test metric from round-trip scores."""
    if scores.shape != labels.shape or not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid test-score array")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("The verified benchmark test partitions must contain both classes")
    metrics = {name: None for name in METRICS}
    metrics.update(n=len(labels), positive_n=int(labels.sum()), threshold=None,
                   metric_scope="classification_and_ranking" if probabilistic else "ranking_only",
                   auroc=float(roc_auc_score(labels, scores)),
                   average_precision=float(average_precision_score(labels, scores)))
    if not probabilistic:
        return metrics
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Classifier probabilities outside [0,1]")
    prediction = (scores >= .5).astype(int)
    tn, fp, _, _ = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics.update(accuracy=float(accuracy_score(labels, prediction)),
                   sensitivity=float(recall_score(labels, prediction, zero_division=0)),
                   specificity=float(tn / (tn + fp)),
                   precision=float(precision_score(labels, prediction, zero_division=0)),
                   f1=float(f1_score(labels, prediction, zero_division=0)),
                   mcc=float(matthews_corrcoef(labels, prediction)),
                   pr_trapezoid_area=float(auc(recall, precision)), threshold=.5)
    return metrics


def check_metrics(expected: dict, saved: dict, context: str) -> None:
    if set(expected) != set(saved):
        raise ValueError(f"Metric field mismatch: {context}")
    for key, value in expected.items():
        other = saved[key]
        if value is None or isinstance(value, str):
            matches = other == value
        else:
            matches = isinstance(other, (int, float)) and np.isfinite(other) and np.isclose(value, other, rtol=0, atol=1e-10)
        if not matches:
            raise ValueError(f"Metric mismatch: {context} {key}: recomputed={value!r}, saved={other!r}")


def verify_run(folder: Path, data: dict, expected_code_hash: str) -> tuple[dict, dict]:
    record = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    cfg = record["config"]
    method, protocol, fold = cfg["method"], cfg["protocol"], cfg["fold"]
    if method not in METHODS or protocol not in PROTOCOLS or type(fold) is not int or fold not in range(5):
        raise ValueError(f"Unexpected method/protocol/fold: {folder.name}")
    expected_name = f"{protocol}__random__{method}__fold{fold}"
    if folder.name != expected_name or record["name"] != expected_name:
        raise ValueError(f"Directory/record/config identity mismatch: {folder.name}")
    if any(record[key] != cfg[key] for key in ("method", "protocol", "fold", "split_hash", "split_array_hashes")):
        raise ValueError(f"Top-level record/config mismatch: {folder.name}")
    if not record["test_evaluated"] or cfg["negative_sampling"] != "random":
        raise ValueError(f"Unexpected evaluation scope: {folder.name}")
    if cfg["code_sha256"] != expected_code_hash or cfg["dataset_hash"] != data["manifest"]["dataset_hash"]:
        raise ValueError(f"Code or prepared dataset fingerprint mismatch: {folder.name}")
    if cfg["base_seed"] != data["manifest"]["seed"] or cfg["seed"] != cfg["base_seed"] + fold:
        raise ValueError(f"Seed mismatch: {folder.name}")
    if cfg["graph"] != asdict(GraphConfig()) or cfg["model"] != fixed_model_config(method):
        raise ValueError(f"Unexpected fixed control configuration: {folder.name}")
    split_hashes = {split: array_hash(data[split + "_pairs"], data[split + "_labels"]) for split in ("train", "val", "test")}
    if cfg["split_hash"] != data["split_hash"] or cfg["split_array_hashes"] != split_hashes:
        raise ValueError(f"Prepared membership fingerprint mismatch: {folder.name}")
    relative = f"random/{protocol}/fold_{fold}"
    prepared_hashes = {key: data["manifest"]["files"][f"{relative}/{filename}.npz"] for key, filename in
                      (("train", "train"), ("val", "validation"), ("test", "test"), ("graph", "graph"))}
    if cfg["prepared_artifact_hashes"] != prepared_hashes:
        raise ValueError(f"Prepared file hash mismatch: {folder.name}")
    graph_hash = fingerprint(data["graph_edges"], data["n_nodes"], GraphConfig(), cfg["seed"]) if method in LEARNED else array_hash(data["graph_edges"])
    if cfg["graph_hash"] != graph_hash:
        raise ValueError(f"Training graph identity mismatch: {folder.name}")
    campaign = {key: value for key, value in cfg.items() if key not in FOLD_CONFIG_KEYS}
    if record["run_signature"] != json_hash(cfg) or record["campaign_signature"] != json_hash(campaign):
        raise ValueError(f"Run/campaign signature mismatch: {folder.name}")

    model_file = "model.joblib" if method in LEARNED else "heuristic.json"
    required = {"run_config.json", model_file, "test_predictions.csv", "metrics.json"}
    if set(record["artifacts_sha256"]) != required:
        raise ValueError(f"Artifact inventory mismatch: {folder.name}")
    for filename, expected in record["artifacts_sha256"].items():
        path = folder / filename
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Artifact missing or altered: {path}")
    if record["checkpoint_sha256"] != sha(folder / model_file):
        raise ValueError(f"Model/heuristic hash mismatch: {folder.name}")
    run_config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    if (run_config["run_signature"] != record["run_signature"]
            or run_config["campaign_signature"] != record["campaign_signature"]
            or {key: value for key, value in run_config.items() if key not in {"run_signature", "campaign_signature", "python_executable"}} != cfg):
        raise ValueError(f"run_config.json differs from completed record: {folder.name}")

    with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != PREDICTION_COLUMNS:
            raise ValueError(f"Prediction schema mismatch: {folder.name}")
        predictions = list(reader)
    if len(predictions) != len(data["test_labels"]):
        raise ValueError(f"Test row-count mismatch: {folder.name}")
    endpoints = np.asarray([(int(row["protein_u"]), int(row["protein_v"])) for row in predictions], dtype=np.int64)
    expected_endpoints = data["node_ids"][data["test_pairs"]]
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.int64)
    if not np.array_equal(endpoints, expected_endpoints) or not np.array_equal(labels, data["test_labels"]):
        raise ValueError(f"Test pair order or label mismatch: {folder.name}")
    fixed_columns = {"method": method, "protocol": protocol, "fold": str(fold), "seed": str(cfg["seed"]),
                     "split_hash": data["split_hash"], "checkpoint_sha256": record["checkpoint_sha256"],
                     **{split + "_array_hash": value for split, value in split_hashes.items()}}
    for key, expected in fixed_columns.items():
        if {row[key] for row in predictions} != {expected}:
            raise ValueError(f"Prediction provenance mismatch: {folder.name} {key}")
    probabilistic = method in LEARNED
    score_key = "probability" if probabilistic else "raw_score"
    scores = np.asarray([float(row[score_key]) for row in predictions], dtype=np.float64)
    if probabilistic:
        if any(row["raw_score"] != "" for row in predictions):
            raise ValueError(f"Classifier incorrectly contains heuristic scores: {folder.name}")
        classes = np.asarray([int(row["prediction"]) for row in predictions])
        if not np.array_equal(classes, (scores >= .5).astype(int)):
            raise ValueError(f"Probability threshold/prediction mismatch: {folder.name}")
    else:
        if any(row["probability"] != "" or row["prediction"] != "" for row in predictions):
            raise ValueError(f"Heuristic incorrectly assigned a probability/threshold: {folder.name}")
        # Recompute heuristic scores from the permitted graph as a second check.
        if method == "degree_product":
            degree = np.bincount(data["graph_edges"].reshape(-1), minlength=data["n_nodes"])
            expected_scores = np.prod(degree[data["test_pairs"]], axis=1)
        else:
            adjacency = [set() for _ in range(data["n_nodes"])]
            for u, v in data["graph_edges"]:
                adjacency[u].add(int(v))
                adjacency[v].add(int(u))
            expected_scores = np.asarray([len(adjacency[u] & adjacency[v]) for u, v in data["test_pairs"]])
        if not np.array_equal(scores, expected_scores):
            raise ValueError(f"Heuristic scores differ from training-graph calculation: {folder.name}")
    recalculated = calculate_metrics(labels, scores, probabilistic)
    check_metrics(recalculated, record["test_metrics"], folder.name + " completed.json")
    metrics_file = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    check_metrics(recalculated, metrics_file["test"], folder.name + " metrics.json")
    return {"run": expected_name, "experiment": f"{protocol}__random__{method}",
            "protocol": protocol, "method": method, "fold": fold,
            "campaign_signature": record["campaign_signature"], "run_signature": record["run_signature"],
            "completed_sha256": sha(folder / "completed.json"),
            "prediction_sha256": sha(folder / "test_predictions.csv"), "checkpoint_sha256": record["checkpoint_sha256"],
            "split_hash": data["split_hash"], "test_metrics": recalculated}, campaign


def verify(campaign: Path, prepared: Path, output: Path, check_partial: bool = False) -> dict:
    campaign, prepared, output = campaign.resolve(), prepared.resolve(), output.resolve()
    expected = {f"{protocol}__random__{method}__fold{fold}" for protocol in PROTOCOLS for method in METHODS for fold in range(5)}
    completed = {path.parent.name: path.parent for path in campaign.glob("*/completed.json")}
    extras, missing = sorted(set(completed) - expected), sorted(expected - set(completed))
    if extras:
        raise ValueError(f"Unexpected completed controls: {extras}")
    if missing and not check_partial:
        raise RuntimeError(f"Classical campaign incomplete: {len(missing)} missing of 40; no verified outputs written. First missing: {missing[:4]}")
    if not completed:
        raise RuntimeError("No completed controls to verify")
    data_cache, campaigns, rows = {}, {}, []
    current_code_hash = code_hash()
    for name in sorted(completed):
        # Folder name is not trusted as config; verify_run crosschecks both.
        protocol, _, method, fold_text = name.split("__")
        fold = int(fold_text.removeprefix("fold"))
        if (protocol, fold) not in data_cache:
            data_cache[protocol, fold] = load_fold(prepared, "random", protocol, fold)
        verified, model_campaign = verify_run(completed[name], data_cache[protocol, fold], current_code_hash)
        key = verified["experiment"]
        if key in campaigns and campaigns[key] != model_campaign:
            raise ValueError(f"Mixed control policies across folds: {key}")
        campaigns[key] = model_campaign
        rows.append(verified)
    # Every method must share the same prepared test population within protocol.
    for protocol in PROTOCOLS:
        reference_pairs, seen_test_nodes = set(), set()
        for fold in range(5):
            if (protocol, fold) not in data_cache:
                continue
            data = data_cache[protocol, fold]
            pair_set = set(map(tuple, data["test_pairs"].tolist()))
            if reference_pairs & pair_set:
                raise ValueError(f"Test pair overlap across outer folds: {protocol}")
            reference_pairs |= pair_set
            if protocol == "protein":
                nodes = set(data["test_nodes"].tolist())
                if seen_test_nodes & nodes:
                    raise ValueError("Test proteins overlap across outer folds")
                seen_test_nodes |= nodes
        if not check_partial:
            data = data_cache[protocol, 0]
            if protocol == "pair" and len(reference_pairs) != 2 * data["manifest"]["n_positives"]:
                raise ValueError("Pair test folds do not cover both benchmark pools")
            if protocol == "protein" and len(seen_test_nodes) != data["n_nodes"]:
                raise ValueError("Protein test folds do not cover all node IDs")
    summary = []
    for experiment in sorted(campaigns):
        group = sorted((row for row in rows if row["experiment"] == experiment), key=lambda row: row["fold"])
        if len({row["fold"] for row in group}) != len(group):
            raise ValueError(f"Duplicate fold: {experiment}")
        if not check_partial and [row["fold"] for row in group] != list(range(5)):
            raise ValueError(f"Incomplete summary group: {experiment}")
        protocol, method = group[0]["protocol"], group[0]["method"]
        item = {"experiment": experiment, "label": LABELS[method], "protocol": protocol,
                "method": method, "completed_folds": len(group)}
        for metric in ("auroc", "average_precision", "accuracy", "mcc"):
            values = [row["test_metrics"][metric] for row in group]
            if any(value is None for value in values):
                item[metric + "_mean"] = item[metric + "_sd"] = None
            else:
                item[metric + "_mean"] = float(np.mean(values))
                item[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(item)
    report = {"status": "partial_check_passed" if check_partial else "passed",
              "expected_runs": 40, "verified_runs": len(rows), "missing_runs": missing,
              "outputs_written": not check_partial,
              "prepared_manifest_sha256": sha(prepared / "manifest.json"),
              "training_code_sha256": current_code_hash, "verification_script_sha256": sha(Path(__file__).resolve()),
              "checks": ["exact run inventory and identity", "run/config/campaign signatures", "prepared membership and array/file hashes",
                         "model and prediction content hashes", "test pair order and labels", "per-row protocol/method/fold/seed provenance",
                         "round-trip probability/raw-score parsing", "classifier threshold 0.5", "heuristic scores recalculated from training graph",
                         "all stored test metrics independently recomputed", "outer test disjointness", "five-fold means and sample SD"],
              "heuristic_classification_metrics": "Not defined: JSON null and blank CSV cells for accuracy/MCC mean/SD; no threshold is assigned.",
              "fold_sd_is_not_a_confidence_interval": True, "summary": summary, "runs": rows}
    if not check_partial:
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "verified_classical_summary.csv"
        temporary = summary_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        temporary.replace(summary_path)
        report["verified_summary_sha256"] = sha(summary_path)
        report_path = output / "classical_verification.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(report_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=HERE / "runs" / "classical_controls_v1")
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--check-partial", action="store_true", help="Verify currently completed controls without writing official outputs")
    args = parser.parse_args()
    report = verify(args.campaign, args.prepared, args.output, args.check_partial)
    print(json.dumps({key: report[key] for key in ("status", "expected_runs", "verified_runs", "missing_runs", "outputs_written")}, indent=2))
