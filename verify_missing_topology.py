"""Independent, read-only verification for the missing-topology repair campaign.

No model is trained, tuned, or selected by this module.  Metrics are recomputed
from exported probabilities rather than imported from the trainer.  Validation
and held-out test results retain separate status and scope in every report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch  # Must precede sklearn in the established Windows GPU runtime.
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

from cappi_data import _verify_artifacts


HERE = Path(__file__).resolve().parent
CORE_FILES = ("cappi_data.py", "cappi_graph.py", "cappi_model.py", "legacy_skipgram.py", "run_experiments.py")
REPAIR_FILES = ("run_missing_topology.py", "cappi_missing_topology.py")
TRAINING_ARTIFACTS = {
    "best.pt", "last.pt", "run_config.json", "topology.npz", "parameters.json", "history.json",
    "first_step_gradient_norms.json", "graph_cache_provenance.json", "data_audit.json",
    "mask_audit.json", "mask_trace.npz", "validation_predictions.csv", "validation_metrics.json",
}
METRICS = (
    "accuracy", "sensitivity", "specificity", "precision", "f1", "mcc",
    "auroc", "average_precision", "pr_trapezoid_area",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def run_digest(value: Any) -> str:
    """The JSON identity convention used by the campaign, recomputed locally."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def combined_source_hash(files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for filename in files:
        digest.update(filename.encode())
        digest.update((HERE / filename).read_bytes())
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def independent_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    """Binary metrics at 0.5; AP and trapezoidal PR area remain distinct."""
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    require(labels.shape == probability.shape and labels.ndim == 1 and len(labels) > 0,
            "Invalid probability/label shape or empty partition")
    require(set(labels.tolist()) == {0, 1}, "Partition must contain both binary classes")
    require(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
            "Probabilities must be finite and lie in [0, 1]")
    predicted = (probability >= 0.5).astype(np.int64)
    tn, fp, _, _ = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, probability)
    return {
        "n": int(len(labels)), "positive_n": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "pr_trapezoid_area": float(auc(recall, precision)),
        "threshold": 0.5,
    }


def compare_metrics(expected: dict, actual: dict, context: str) -> None:
    require(set(expected).issubset(actual), f"Missing metric fields: {context}")
    for name, expected_value in expected.items():
        saved_value = actual[name]
        require(isinstance(saved_value, (float, int)) and not isinstance(saved_value, bool)
                and np.isfinite(saved_value)
                and np.isclose(expected_value, saved_value, atol=1e-10, rtol=0),
                f"Metric mismatch: {context} {name}, recomputed={expected_value}, saved={saved_value}")


def verify_predictions(path: Path, data: dict, split: str, identity: dict,
                       checkpoint_sha256: str) -> tuple[dict, dict]:
    """Verify membership, order, labels, threshold and repeated provenance fields."""
    require(split in {"val", "test"}, "Unexpected prediction split")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"protein_u", "protein_v", "label", "probability", "prediction",
                    "fold", "model_seed", "graph_seed", "split_hash", "checkpoint_sha256", "partition"}
        require(required.issubset(columns), f"Prediction columns missing: {path}")
        rows = list(reader)
    require(len(rows) == len(data[split + "_labels"]), f"Prediction row count differs: {path}")
    endpoints = np.asarray([(int(row["protein_u"]), int(row["protein_v"])) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    require(np.array_equal(endpoints, data["node_ids"][data[split + "_pairs"]]),
            f"Prediction ID membership/order differs from frozen prepared fold: {path}")
    require(np.array_equal(labels, data[split + "_labels"]),
            f"Prediction labels differ from frozen prepared fold: {path}")
    fixed = {"fold": str(identity["fold"]), "model_seed": str(identity["model_seed"]),
             "graph_seed": str(identity["graph_seed"]), "partition": "validation" if split == "val" else "test",
             "split_hash": data["split_hash"], "checkpoint_sha256": checkpoint_sha256}
    for column in ("protocol", "negative_sampling", "variant", "candidate", "training_seed"):
        if column in columns and column in identity:
            fixed[column] = str(identity[column])
    for column, expected in fixed.items():
        require({row[column] for row in rows} == {expected}, f"Prediction provenance differs: {path} {column}")
    probability = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    require(np.array_equal(prediction, (probability >= 0.5).astype(np.int64)),
            f"Prediction is inconsistent with the fixed 0.5 threshold: {path}")
    csv_bce = None
    if np.all((probability > 0) & (probability < 1)):
        csv_bce = float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log1p(-probability)))
    return independent_metrics(labels, probability), {
        "prediction_sha256": sha256_file(path), "n": len(rows), "columns": sorted(columns),
        "binary_crossentropy_from_csv": csv_bce,
    }


def verify_prepared_graph(data: dict, identity: dict) -> dict:
    """Independently check target-edge exclusion and the zero topology contract."""
    observed = set(map(tuple, data["graph_edges"].tolist()))
    expected = set(map(tuple, data["train_pairs"][data["train_labels"] == 1].tolist()))
    require(observed == expected, "Topology graph is not exactly the frozen training-positive graph")
    for split in ("val", "test"):
        require(not observed.intersection(map(tuple, data[split + "_pairs"].tolist())),
                f"Held-out {split} pair entered the topology graph")
    require(identity["split_hash"] == data["split_hash"], "Run's prepared split hash changed")
    if "dataset_hash" in identity:
        require(identity["dataset_hash"] == data["manifest"]["dataset_hash"], "Prepared dataset hash changed")
    hashes = {split: array_hash(data[split + "_pairs"], data[split + "_labels"])
              for split in ("train", "val", "test") if split + "_labels" in data}
    if "split_array_hashes" in identity:
        require(identity["split_array_hashes"] == hashes, "Frozen split array hash mismatch")
    return {"split_hash": data["split_hash"], "dataset_hash": data["manifest"]["dataset_hash"],
            "split_array_hashes": hashes, "training_positive_edges": len(observed)}


def load_verification_fold(prepared: Path, negative: str, protocol: str, fold: int,
                           include_test_labels: bool) -> dict:
    """Independent loading: validation-only verification never decodes test labels."""
    manifest = read_json(prepared / "manifest.json")
    _verify_artifacts(prepared, manifest)  # Checks bytes and manifest schema only.
    require(negative in {"random", "degree_matched"} and protocol in {"pair", "protein"},
            "Unexpected negative/protocol")
    require(type(fold) is int and fold in range(manifest["n_folds"]), "Unexpected fold")
    relative = f"{negative}/{protocol}/fold_{fold}"
    with (prepared / "nodes.csv").open(encoding="utf-8", newline="") as handle:
        nodes = list(csv.DictReader(handle))
    require([int(row["index"]) for row in nodes] == list(range(manifest["n_nodes"])),
            "Prepared node index order differs")
    data = {"manifest": manifest, "n_nodes": manifest["n_nodes"],
            "node_ids": np.asarray([int(row["node_id"]) for row in nodes], dtype=np.int64),
            "split_hash": manifest["folds"][relative]["split_hash"]}
    require(len(set(data["node_ids"].tolist())) == data["n_nodes"], "Duplicate prepared node IDs")
    with np.load(prepared / "positives.npz", allow_pickle=False) as archive:
        known = set(map(tuple, archive["pairs"].tolist()))
    for split, filename in (("train", "train"), ("val", "validation"), ("test", "test")):
        with np.load(prepared / relative / (filename + ".npz"), allow_pickle=False) as archive:
            data[split + "_pairs"] = archive["pairs"].copy()
            if split != "test" or include_test_labels:
                data[split + "_labels"] = archive["labels"].copy()
    with np.load(prepared / relative / "graph.npz", allow_pickle=False) as archive:
        for key in ("graph_edges", "train_nodes", "val_nodes", "test_nodes"):
            data[key] = archive[key].copy()
    pair_sets = {}
    for split in ("train", "val", "test"):
        pairs = data[split + "_pairs"]
        require(pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2,
                f"Prepared {split} pair shape/dtype differs")
        require(np.all(pairs[:, 0] < pairs[:, 1]) and np.all(pairs >= 0)
                and np.all(pairs < data["n_nodes"]), f"Prepared {split} pair is not canonical")
        pair_sets[split] = set(map(tuple, pairs.tolist()))
        require(len(pair_sets[split]) == len(pairs), f"Duplicate prepared {split} pairs")
        if split + "_labels" in data:
            labels = data[split + "_labels"]
            require(labels.shape == (len(pairs),) and labels.dtype == np.float32
                    and set(labels.tolist()) == {0.0, 1.0}, f"Invalid prepared {split} labels")
            require(all((tuple(pair) in known) == bool(label) for pair, label in zip(pairs, labels)),
                    f"Prepared {split} labels conflict with known-positive pool")
        if protocol == "protein":
            require(set(pairs.reshape(-1)).issubset(set(data[split + "_nodes"])),
                    f"Prepared {split} pair violates protein membership")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not pair_sets[left].intersection(pair_sets[right]), "Prepared partitions share target pairs")
        if protocol == "protein":
            require(not set(data[left + "_nodes"]).intersection(data[right + "_nodes"]),
                    "Prepared partitions share proteins")
    return data


def independent_graph_hash(data: dict, graph_config: dict, graph_seed: int) -> str:
    """Reproduce the declared graph fingerprint without calling graph training."""
    payload = np.ascontiguousarray(data["graph_edges"], dtype=np.int64).tobytes()
    payload += json.dumps({"config": graph_config, "seed": graph_seed,
                           "n_nodes": data["n_nodes"]}, sort_keys=True).encode()
    payload += (HERE / "cappi_graph.py").read_bytes()
    payload += (HERE / "legacy_skipgram.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()


def verify_topology(path: Path, data: dict, graph_config: dict, graph_seed: int) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"raw_embedding", "embedding", "present", "mean", "std", "fitting_nodes", "graph_hash"}
        require(required.issubset(archive.files), f"Missing topology provenance arrays: {path}")
        arrays = {name: archive[name].copy() for name in required}
    n_nodes, dimension = data["n_nodes"], graph_config["dim"]
    raw, scaled, present = arrays["raw_embedding"], arrays["embedding"], arrays["present"]
    require(raw.shape == scaled.shape == (n_nodes, dimension), "Topology dimensions differ from config")
    require(present.dtype == np.bool_ and present.shape == (n_nodes,), "Invalid topology availability array")
    expected_present = np.bincount(data["graph_edges"].reshape(-1), minlength=n_nodes) > 0
    require(np.array_equal(present, expected_present), "Topology availability is not training graph degree > 0")
    require(np.isfinite(raw).all() and np.isfinite(scaled).all(), "Nonfinite topology array")
    require(np.all(raw[~present] == 0) and np.all(scaled[~present] == 0),
            "Unavailable topology must remain literal zero")
    fitting = np.asarray(data["train_nodes"], dtype=np.int64)
    fitting = fitting[present[fitting]]
    require(np.array_equal(arrays["fitting_nodes"], fitting),
            "Scaler was not fitted on exactly training nodes with permitted topology")
    mean = raw[fitting].mean(0) if len(fitting) else np.zeros(dimension)
    std = raw[fitting].std(0) if len(fitting) else np.ones(dimension)
    require(arrays["mean"].shape == arrays["std"].shape == (dimension,), "Topology scaler dimension differs")
    require(np.allclose(arrays["mean"], mean, atol=1e-7, rtol=1e-6)
            and np.allclose(arrays["std"], std, atol=1e-7, rtol=1e-6),
            "Topology scaler differs from independent train-node recomputation")
    expected_scaled = (raw - mean) / np.maximum(std, 1e-6)
    expected_scaled[~present] = 0
    require(np.allclose(scaled, expected_scaled, atol=2e-6, rtol=1e-6),
            "Scaled topology differs from the declared train-only transform")
    expected_graph_hash = independent_graph_hash(data, graph_config, graph_seed)
    require(str(arrays["graph_hash"]) == expected_graph_hash, "Graph feature hash differs from frozen train graph")
    return {"graph_hash": expected_graph_hash, "present_nodes": int(present.sum()),
            "fitting_nodes": int(len(fitting)), "topology_sha256": sha256_file(path)}


def mask_counts(effective_present: np.ndarray, base_present: np.ndarray, data: dict) -> dict:
    """Count drop-only and effective missing topology separately in both classes."""
    endpoints = data["train_pairs"]
    masked = base_present & ~effective_present
    counts = {}
    for label, name in ((1, "positive"), (0, "negative")):
        pairs = endpoints[data["train_labels"] == label]
        base = base_present[pairs]
        drop = masked[pairs]
        effective = effective_present[pairs]
        counts[name] = {
            "pairs": int(len(pairs)), "available_endpoints": int(base.sum()),
            "masked_endpoints": int(drop.sum()), "masked_any_pairs": int(drop.any(axis=1).sum()),
            "masked_both_pairs": int(drop.all(axis=1).sum()),
            "effective_missing_endpoints": int((~effective).sum()),
            "effective_missing_any_pairs": int((~effective).any(axis=1).sum()),
            "effective_missing_both_pairs": int((~effective).all(axis=1).sum()),
        }
    return counts


def replay_mask_trace(trace_path: Path, data: dict, model_seed: int, probability: float,
                      device: str, epochs_completed: int) -> dict:
    """Replay an isolated node RNG without model weights, labels, or dropout helpers."""
    import torch

    with np.load(trace_path, allow_pickle=False) as archive:
        require({"epochs", "effective_present", "base_present"}.issubset(archive.files),
                "Mask trace is missing required arrays")
        epochs = archive["epochs"].copy()
        effective = archive["effective_present"].copy()
        base = archive["base_present"].copy()
    expected_base = np.bincount(data["graph_edges"].reshape(-1), minlength=data["n_nodes"]) > 0
    require(base.dtype == np.bool_ and np.array_equal(base, expected_base),
            "Mask trace base presence differs from permitted training graph")
    require(effective.dtype == np.bool_ and effective.shape == (epochs_completed, data["n_nodes"]),
            "Mask trace dimensions differ from training epoch count")
    require(np.array_equal(epochs, np.arange(1, epochs_completed + 1)),
            "Mask trace does not cover each training epoch exactly once")
    require(not np.any(effective & ~base[None, :]), "Dropout made unavailable node topology present")
    require(probability in {0.0, 0.5}, "Repair contract allows only topology dropout 0 or 0.5")
    generator = torch.Generator(device=device)
    generator.manual_seed(model_seed + 7919)
    base_tensor = torch.as_tensor(base, dtype=torch.bool, device=device)
    counts = []
    for index, epoch in enumerate(epochs):
        if probability == 0:
            expected = base
        else:
            # Draw for all nodes, then intersect permitted availability. This
            # exactly tests the declared label-independent node-mask schedule.
            keep = torch.rand(base_tensor.shape, device=device, generator=generator) >= probability
            expected = (base_tensor & keep).cpu().numpy()
        require(np.array_equal(effective[index], expected),
                f"Mask trace differs from independent node RNG replay at epoch {epoch}")
        counts.append({"epoch": int(epoch), "counts": mask_counts(effective[index], base, data)})
    totals = {label: {key: sum(row["counts"][label][key] for row in counts)
                      for key in counts[0]["counts"][label]} for label in ("positive", "negative")}
    if probability == 0.5:
        for label in ("positive", "negative"):
            require(totals[label]["available_endpoints"] > 0 and totals[label]["masked_endpoints"] > 0,
                    f"Topology dropout did not cover {label} training endpoints")
    else:
        require(all(totals[label]["masked_endpoints"] == 0 for label in totals),
                "No-dropout control contains hidden topology")
    return {"trace_sha256": sha256_file(trace_path), "node_mask_rng_verified": True,
            "probability": probability, "mask_seed": model_seed + 7919,
            "totals_over_epochs": totals, "epoch_counts": counts}


def independent_checkpoint_epoch(history: list[dict], min_delta: float = 1e-6) -> tuple[int, float]:
    """Reproduce validation-BCE selection without relying on saved best metadata."""
    require(bool(history), "Training history is empty")
    selected_epoch, best = 0, float("inf")
    for index, row in enumerate(history):
        require(row["epoch"] == index + 1, "Training history epochs are missing or duplicated")
        value = float(row["validation_bce"])
        require(np.isfinite(value), "History contains nonfinite validation BCE")
        if value < best - min_delta:
            selected_epoch, best = index + 1, value
    require(selected_epoch > 0, "No finite validation checkpoint")
    return selected_epoch, best


def select_learning_rate(trials: list[dict]) -> dict:
    """Choose minimum best validation BCE; exact ties choose lower LR.

    Input must be independent verified validation-only run records. There is no
    test metric in this selection function and repeats are not new LR trials.
    """
    require(len(trials) == 2, "Each LR selection must contain exactly two trials")
    require({float(row["learning_rate"]) for row in trials} == {0.0005, 0.001},
            "Learning-rate trial grid differs from repair contract")
    fields = ("family", "negative", "protocol", "fold", "model_seed", "graph_seed")
    for key in fields:
        require(len({str(row[key]) for row in trials}) == 1, f"Mixed trial selection group: {key}")
    return min(trials, key=lambda row: (float(row["best_validation_bce"]), float(row["learning_rate"])))


def verify_identity(identity: dict, prepared: Path, expected_selected: dict | None) -> dict:
    require(identity["schema"] == "cappi-missing-topology-run-v1", "Unexpected run schema")
    require(identity["core_code_sha256"] == combined_source_hash(CORE_FILES), "Original core code hash differs")
    require(identity["repair_code_sha256"] == combined_source_hash(REPAIR_FILES), "Repair code hash differs")
    require(identity["prepared_manifest_sha256"] == sha256_file(prepared / "manifest.json"),
            "Prepared manifest checksum differs")
    require(Path(identity["prepared"]).resolve() == prepared, "Run used a different prepared directory")
    settings = identity["settings"]
    family = settings["family"]
    families = {"residual", "topology_dropout", "topology_dropout_bypass", "sequence_only", "topology_only", "concat_matched"}
    require(family in families, "Unexpected repair family")
    probability = 0.5 if family in {"topology_dropout", "topology_dropout_bypass"} else 0.0
    policy = "bypass" if family == "topology_dropout_bypass" else "original"
    require(settings == {"family": family, "learning_rate": identity["learning_rate"],
                         "topology_dropout": probability, "missing_policy": policy},
            "Family/config does not match its predeclared topology policy")
    require(identity["learning_rate"] in {0.0005, 0.001}, "Learning rate is outside the frozen grid")
    fusion = "residual" if family.startswith("topology_dropout") else family
    require(identity["model"] == {"topo_dim": 64, "seq_dim": 256, "filters": 64, "kernel": 7,
                                  "max_length": 1500, "fusion": fusion, "dropout": 0.3,
                                  "pair_dropout": [0.4, 0.4, 0.1]}, "Model contract differs")
    require(identity["graph"] == {"method": "drw", "dim": 64, "walks_per_node": 10,
                                  "walk_length": 20, "window": 5, "epochs": 5,
                                  "negative_samples": 5, "p": 1.0, "q": 0.5}, "Graph contract differs")
    expected_training = {"max_epochs": 300, "patience": 20, "weight_decay": 0.0001,
                         "pair_chunk": 4096, "gradient_clip_norm": 5.0, "optimizer": "AdamW",
                         "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 10},
                         "selection": "minimum validation BCE with improvement > 1e-6", "threshold": 0.5,
                         "test_labels_or_scores_loaded": False, "evaluation_partition": "validation"}
    for key, value in expected_training.items():
        require(identity[key] == value, f"Training contract differs: {key}")
    fold, model_seed, graph_seed = identity["fold"], identity["model_seed"], identity["graph_seed"]
    require(type(fold) is int and fold in range(5), "Invalid campaign fold")
    require(graph_seed == 20260930 + fold, "Graph seed changed across model repeats/families")
    repeat, remainder = divmod(model_seed - 20260930 - fold, 1000)
    require(remainder == 0 and repeat in {0, 1, 2}, "Model seed violates declared repeat formula")
    require(identity["mask_seed"] == model_seed + 7919, "Mask seed differs from declared isolated RNG")
    condition = (identity["protocol"], identity["negative_sampling"])
    require(condition in {("pair", "random"), ("protein", "random"), ("pair", "degree_matched")},
            "Unexpected evaluation condition")
    if condition == ("pair", "degree_matched"):
        require(family in {"residual", "sequence_only", "topology_only"}, "Unexpected degree-matched control")
    summary = {"family": family, "negative": identity["negative_sampling"], "protocol": identity["protocol"],
               "fold": fold, "repeat": repeat, "model_seed": model_seed, "graph_seed": graph_seed,
               "learning_rate": identity["learning_rate"], "topology_dropout": probability, "missing_policy": policy}
    if expected_selected is not None:
        for key, value in summary.items():
            require(expected_selected[key] == value, f"Run does not match indexed/frozen job: {key}")
    return summary


def verify_mask_audit(path: Path, trace_path: Path, data: dict, probability: float,
                      expected_epoch_counts: list[dict]) -> None:
    saved = read_json(path)
    with np.load(trace_path, allow_pickle=False) as archive:
        base, effective = archive["base_present"].copy(), archive["effective_present"].copy()
    require(len(saved) == len(effective) == len(expected_epoch_counts), "Mask audit length differs")
    for index, record in enumerate(saved):
        current = effective[index]
        expected = {"epoch": index + 1, "probability": probability, "n_nodes": len(base),
                    "original_present_nodes": int(base.sum()), "effective_present_nodes": int(current.sum()),
                    "artificially_masked_nodes": int((base & ~current).sum()), "by_label": {}}
        for label, name in ((1, "positive"), (0, "negative")):
            pairs = data["train_pairs"][data["train_labels"] == label]
            original_count, effective_count = base[pairs].sum(1), current[pairs].sum(1)
            expected["by_label"][name] = {
                "n_pairs": len(pairs), "original_present_endpoints": int(original_count.sum()),
                "effective_present_endpoints": int(effective_count.sum()),
                "artificially_masked_endpoints": int((original_count - effective_count).sum()),
                **{f"original_{n}_present": int((original_count == n).sum()) for n in (0, 1, 2)},
                **{f"effective_{n}_present": int((effective_count == n).sum()) for n in (0, 1, 2)},
            }
        require(record == expected, f"Mask coverage audit does not reproduce trace at epoch {index + 1}")


def verify_run(out: Path, prepared: Path, expected_selected: dict | None = None) -> dict:
    """Verify a completed training run, and its test artifact only when present.

    Incomplete training is rejected. Test absence yields ``validation_only``;
    test presence requires the checkpoint-specific selection freeze gate.
    """
    import torch

    out, prepared = Path(out).resolve(), Path(prepared).resolve()
    record = read_json(out / "completed.json")
    require(record["schema"] == "cappi-missing-topology-training-completed-v1", "Unexpected completion schema")
    identity = read_json(out / "run_config.json")
    signature = identity.pop("run_signature")
    require(run_digest(identity) == signature == record["run_signature"], "Run signature differs from configuration")
    require(record["config"] == identity and record["settings"] == identity["settings"],
            "Completion metadata differs from run configuration")
    summary = verify_identity(identity, prepared, expected_selected)
    require(record["name"] == out.name, "Run directory/completion identity differs")
    for key in ("fold", "model_seed", "graph_seed", "split_hash"):
        require(record[key] == identity[key], f"Completion identity differs: {key}")
    require(record["test_evaluated"] is False and record["test_labels_or_scores_loaded"] is False
            and record["evaluation_partition"] == "validation", "Training completion is not validation-only")
    require(set(record["artifact_sha256"]) == TRAINING_ARTIFACTS, "Training artifact inventory differs")
    for filename, expected_hash in record["artifact_sha256"].items():
        require((out / filename).is_file() and sha256_file(out / filename) == expected_hash,
                f"Training artifact missing or changed: {out / filename}")
    checkpoint_hash = sha256_file(out / "best.pt")
    require(record["checkpoint_sha256"] == checkpoint_hash, "Selected checkpoint checksum differs")
    test_exists = (out / "test_completed.json").exists()
    if not test_exists:
        require(not (out / "test_predictions.csv").exists() and not (out / "test_metrics.json").exists(),
                "Incomplete test artifacts must not be reported as validation-only completion")
    data = load_verification_fold(prepared, summary["negative"], summary["protocol"], summary["fold"],
                                  include_test_labels=test_exists)
    prepared_report = verify_prepared_graph(data, identity)
    topology_report = verify_topology(out / "topology.npz", data, identity["graph"], identity["graph_seed"])
    require(record["graph_hash"] == topology_report["graph_hash"], "Completion graph fingerprint differs")
    history = read_json(out / "history.json")
    epoch, best_bce = independent_checkpoint_epoch(history)
    require(record["epochs_completed"] == len(history) <= identity["max_epochs"], "Training epoch count differs")
    require(record["best_epoch"] == epoch and np.isclose(record["best_validation_bce"], best_bce, atol=1e-6, rtol=0),
            "Saved checkpoint was not selected using declared validation BCE policy")
    if len(history) < identity["max_epochs"]:
        require(len(history) - epoch >= identity["patience"], "Training stopped before declared early-stopping patience")
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    require(checkpoint["run_signature"] == signature and checkpoint["graph_hash"] == topology_report["graph_hash"]
            and json.loads(json.dumps(checkpoint["config"], sort_keys=True, default=str)) == identity
            and checkpoint["best_epoch"] == epoch
            and checkpoint["epoch"] + 1 == epoch, "Checkpoint metadata differs from selected run/epoch")
    require(checkpoint["history"] == history[:epoch], "Checkpoint history differs from exported training history")
    require(np.isclose(checkpoint["best"], best_bce, atol=1e-10, rtol=0), "Checkpoint validation BCE metadata differs")
    del checkpoint
    gradient_norms = read_json(out / "first_step_gradient_norms.json")
    require(bool(gradient_norms) and all(value is not None and np.isfinite(value) for value in gradient_norms.values()),
            "Some intended trainable parameters have missing/nonfinite first-step gradients")
    mask_report = replay_mask_trace(out / "mask_trace.npz", data, identity["model_seed"],
                                    summary["topology_dropout"], identity["device"], len(history))
    verify_mask_audit(out / "mask_audit.json", out / "mask_trace.npz", data,
                      summary["topology_dropout"], mask_report["epoch_counts"])
    # Long per-epoch counts are already in the hashed audit; retain totals only.
    mask_report.pop("epoch_counts")
    validation_metrics, val_prediction = verify_predictions(out / "validation_predictions.csv", data, "val",
                                                           identity, checkpoint_hash)
    compare_metrics(validation_metrics, record["validation_metrics"], out.name + " completion validation")
    compare_metrics(validation_metrics, read_json(out / "validation_metrics.json"), out.name + " validation_metrics.json")
    if val_prediction["binary_crossentropy_from_csv"] is not None:
        require(np.isclose(val_prediction["binary_crossentropy_from_csv"], record["best_validation_bce"], atol=1e-6, rtol=0),
                "Selected validation BCE does not reproduce exported checkpoint probabilities")
    result = {**summary, "run_dir": str(out), "out": str(out), "status": "validation_only",
              "run_signature": signature, "checkpoint_sha256": checkpoint_hash,
              "completed_sha256": sha256_file(out / "completed.json"), "split_hash": data["split_hash"],
              "best_validation_bce": record["best_validation_bce"], "best_epoch": epoch,
              "epochs_completed": len(history), "validation_metrics": validation_metrics,
              "validation_bce_from_csv": val_prediction["binary_crossentropy_from_csv"],
              "validation_prediction_sha256": val_prediction["prediction_sha256"],
              "prepared": prepared_report, "topology": topology_report, "mask": mask_report}
    if test_exists:
        freeze_path = out / "selection_frozen.json"
        declaration = read_json(freeze_path)
        require(declaration["selected_before_test"] is True and declaration["run_signature"] == signature
                and declaration["checkpoint_sha256"] == checkpoint_hash,
                "Held-out test lacks matching checkpoint-specific pretest selection declaration")
        require(isinstance(declaration["selection_rule"], str) and bool(declaration["selection_rule"].strip()),
                "Selection rule is absent")
        require(declaration["repeat"] == summary["repeat"] and declaration["learning_rate"] == summary["learning_rate"],
                "Selection declaration changed LR or repeat")
        test_record = read_json(out / "test_completed.json")
        require(test_record["schema"] == "cappi-missing-topology-selected-test-v1"
                and test_record["test_evaluated"] is True and test_record["evaluation_partition"] == "test",
                "Test completion is not selected held-out evaluation")
        for key in ("run_signature", "split_hash", "checkpoint_sha256", "graph_hash", "name"):
            require(test_record[key] == record[key], f"Test provenance differs from training: {key}")
        require(test_record["selection_declaration_sha256"] == sha256_file(freeze_path),
                "Selection freeze changed after test evaluation")
        require(test_record["selection_rule"] == declaration["selection_rule"], "Test selection rule differs")
        require(set(test_record["artifact_sha256"]) == {"test_predictions.csv", "test_metrics.json", "selection_frozen.json"},
                "Selected test artifact inventory differs")
        for filename, expected_hash in test_record["artifact_sha256"].items():
            require(sha256_file(out / filename) == expected_hash, f"Selected test artifact changed: {filename}")
        test_metrics, test_prediction = verify_predictions(out / "test_predictions.csv", data, "test", identity, checkpoint_hash)
        compare_metrics(test_metrics, test_record["test_metrics"], out.name + " test completion")
        compare_metrics(test_metrics, read_json(out / "test_metrics.json"), out.name + " test_metrics.json")
        result.update(status="test_evaluated", test_metrics=test_metrics,
                      test_prediction_sha256=test_prediction["prediction_sha256"],
                      selection_freeze_sha256=sha256_file(freeze_path),
                      test_completed_sha256=sha256_file(out / "test_completed.json"))
    return result


def expected_trial_jobs(campaign: Path, contract: dict) -> list[dict]:
    jobs = []
    for condition in contract["conditions"]:
        for family in condition["families"]:
            for fold in contract["folds"]:
                for learning_rate in contract["validation_grid"]["learning_rate"]:
                    tag = (f'{condition["protocol"]}__{condition["negative"]}__{family}'
                           f'__lr{learning_rate:.4f}').replace(".", "p")
                    policy = contract["families"][family]
                    jobs.append({"protocol": condition["protocol"], "negative": condition["negative"],
                                 "family": family, "fold": fold, "learning_rate": learning_rate,
                                 "repeat": 0, "model_seed": 20260930 + fold,
                                 "graph_seed": 20260930 + fold,
                                 "out": str(campaign / "trials" / f"{tag}__repeat0__fold{fold}"),
                                 "topology_dropout": policy["topology_dropout"], "missing_policy": policy["missing_policy"]})
    return jobs


def path_key(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def compare_job_list(actual: list[dict], expected: list[dict], context: str) -> None:
    require(len(actual) == len(expected), f"Incomplete or excessive run inventory: {context}")
    actual_map = {path_key(job["out"]): job for job in actual}
    expected_map = {path_key(job["out"]): job for job in expected}
    require(len(actual_map) == len(actual) and set(actual_map) == set(expected_map),
            f"Duplicate/unexpected indexed run directories: {context}")
    for key, job in actual_map.items():
        other = expected_map[key]
        require(set(job) == set(other), f"Unexpected indexed job fields: {context}")
        require(all(job[field] == other[field] for field in other if field != "out"),
                f"Indexed job configuration differs: {context} {job['out']}")


def verify_campaign_sources(campaign: Path, prepared: Path) -> dict:
    frozen = read_json(campaign / "frozen_campaign_identity.json")
    require(frozen["contract_sha256"] == sha256_file(campaign / "campaign_contract.json"),
            "Campaign contract changed after freezing")
    expected_sources = set(CORE_FILES + ("cappi_missing_topology.py", "run_missing_topology.py"))
    require(set(frozen["training_sources_sha256"]) == expected_sources, "Frozen training source inventory differs")
    for filename, expected_hash in frozen["training_sources_sha256"].items():
        require(sha256_file(HERE / filename) == expected_hash, f"Training code changed: {filename}")
        require(sha256_file(campaign / "source_snapshot" / filename) == expected_hash,
                f"Frozen training code snapshot differs: {filename}")
    for filename, expected_hash in frozen["protected_files_sha256"].items():
        require(sha256_file(Path(filename)) == expected_hash, f"Protected original artifact changed: {filename}")
    require(frozen["protected_files_sha256"].get(str(prepared / "manifest.json")) == sha256_file(prepared / "manifest.json"),
            "Prepared manifest was not part of the frozen protected originals")
    return {"contract_sha256": frozen["contract_sha256"], "frozen_identity_sha256": sha256_file(campaign / "frozen_campaign_identity.json"),
            "training_sources_sha256": frozen["training_sources_sha256"],
            "protected_original_artifact_count": len(frozen["protected_files_sha256"]),
            "protected_originals_unchanged": True}


def verify_campaign(campaign: Path, prepared: Path) -> dict:
    """Require the full 300-training/225-test campaign and independently verify it.

    No output is written here; an exception prevents the caller from exporting
    a passed report or a mixed/incomplete result summary.
    """
    campaign, prepared = Path(campaign).resolve(), Path(prepared).resolve()
    contract = read_json(campaign / "campaign_contract.json")
    provenance = verify_campaign_sources(campaign, prepared)
    expected_trials = expected_trial_jobs(campaign, contract)
    require(len(expected_trials) == 150 and contract["counts"] == {
        "validation_training_runs": 150, "selected_additional_seed_training_runs": 150,
        "total_training_runs": 300, "selected_test_evaluations": 225}, "Campaign budget/count contract differs")
    trial_index = read_json(campaign / "trial_index.json")
    require(trial_index["count"] == 150, "Trial index count differs")
    compare_job_list(trial_index["runs"], expected_trials, "validation trial grid")
    actual_trial_completed = {path_key(path.parent) for path in (campaign / "trials").glob("*/completed.json")}
    require(actual_trial_completed == {path_key(job["out"]) for job in expected_trials},
            "Validation trial grid is incomplete or has unexpected completed runs")
    verified = {}
    for job in expected_trials:
        result = verify_run(Path(job["out"]), prepared, job)
        verified[path_key(job["out"])] = result
    groups = {}
    for job in expected_trials:
        key = tuple(job[field] for field in ("protocol", "negative", "family", "fold"))
        groups.setdefault(key, []).append(verified[path_key(job["out"])])
    winners = {key: select_learning_rate(trials) for key, trials in groups.items()}
    require(len(winners) == 75, "Validation-selection groups are incomplete")
    registry_path = campaign / "selection_registry.json"
    registry = read_json(registry_path)
    stable = {key: registry[key] for key in ("rule", "choices", "candidate_count", "choice_count", "contract_sha256")}
    require(registry["selection_signature"] == run_digest(stable), "Frozen selection registry signature differs")
    require(registry["test_outcomes_used"] is False and registry["candidate_count"] == 150
            and registry["choice_count"] == 75 and registry["contract_sha256"] == provenance["contract_sha256"]
            and registry["rule"] == contract["validation_grid"]["selection"], "Frozen selection policy differs")
    require(len(registry["choices"]) == 75, "Frozen selection choice inventory differs")
    selected_expected, seen_groups = [], set()
    trial_jobs_by_path = {path_key(job["out"]): job for job in expected_trials}
    for choice in registry["choices"]:
        job = choice["job"]
        key = tuple(job[field] for field in ("protocol", "negative", "family", "fold"))
        require(key in winners and key not in seen_groups, "Selection registry has duplicate/unexpected groups")
        seen_groups.add(key)
        winner = winners[key]
        winner_job = trial_jobs_by_path[path_key(winner["out"])]
        compare_job_list([job], [winner_job], "validation-selected winner")
        require(choice["run_signature"] == winner["run_signature"]
                and choice["checkpoint_sha256"] == winner["checkpoint_sha256"]
                and choice["best_validation_bce"] == winner["best_validation_bce"],
                "Registry winner identity/validation score differs")
        candidate_map = {path_key(candidate["out"]): candidate for candidate in choice["candidates"]}
        candidate_paths = {path_key(trial["out"]) for trial in groups[key]}
        require(len(candidate_map) == 2 and set(candidate_map) == candidate_paths, "Selection candidate inventory differs")
        for candidate_path, candidate in candidate_map.items():
            trial = verified[candidate_path]
            require(candidate["learning_rate"] == trial["learning_rate"]
                    and candidate["best_validation_bce"] == trial["best_validation_bce"]
                    and candidate["checkpoint_sha256"] == trial["checkpoint_sha256"], "Selection candidate metadata differs")
        selected_expected.append(job)
        for repeat in (1, 2):
            repeated = dict(job)
            repeated.update(repeat=repeat, model_seed=20260930 + 1000 * repeat + job["fold"])
            basename = Path(job["out"]).name.replace("__repeat0__", f"__repeat{repeat}__")
            repeated["out"] = str(campaign / "selected_repeats" / basename)
            selected_expected.append(repeated)
    selected_index = read_json(campaign / "selected_index.json")
    require(selected_index["selected_run_count"] == selected_index["expected_count"] == 225
            and selected_index["all_training_complete_before_evaluation"] is True
            and selected_index["selection_signature"] == registry["selection_signature"], "Selected test index differs")
    compare_job_list(selected_index["runs"], selected_expected, "selected test run inventory")
    expected_repeats = [job for job in selected_expected if job["repeat"] > 0]
    actual_repeats = {path_key(path.parent) for path in (campaign / "selected_repeats").glob("*/completed.json")}
    require(actual_repeats == {path_key(job["out"]) for job in expected_repeats},
            "Selected additional-seed grid is incomplete or contains unexpected runs")
    selected_paths = {path_key(job["out"]) for job in selected_expected}
    actual_tests = {path_key(path.parent) for base in ("trials", "selected_repeats")
                    for path in (campaign / base).glob("*/test_completed.json")}
    require(actual_tests == selected_paths, "Tests are missing, excessive, or include unselected learning-rate trials")
    selected_rows = []
    registry_hash = sha256_file(registry_path)
    for job in selected_index["runs"]:
        key = path_key(job["out"])
        row = verified.get(key)
        if row is None:
            row = verify_run(Path(job["out"]), prepared, job)
            verified[key] = row
        require(row["status"] == "test_evaluated", "Selected checkpoint test evaluation is incomplete")
        declaration = read_json(Path(job["out"]) / "selection_frozen.json")
        require(declaration["selection_registry_sha256"] == registry_hash
                and declaration["selection_signature"] == registry["selection_signature"],
                "Run selection freeze is not bound to the frozen campaign registry")
        expected_rule = registry["rule"] if job["repeat"] == 0 else "Repeat-0 validation-selected learning rate; no retuning."
        require(declaration["selection_rule"] == expected_rule, "Repeat selection/retuning policy differs")
        selected_rows.append(row)
    require(len(verified) == 300 and len(selected_rows) == 225, "Complete campaign record counts differ")
    # Same condition and fold must retain identical graph inputs across every
    # family, LR and model seed. These hashes are independent of model seeds.
    shared_graphs, shared_tests = {}, {}
    for row in verified.values():
        key = (row["protocol"], row["negative"], row["fold"])
        graph_identity = (row["topology"]["graph_hash"], row["topology"]["topology_sha256"])
        require(key not in shared_graphs or shared_graphs[key] == graph_identity,
                "Compared models/repeats did not share the same training graph/scaler inputs")
        shared_graphs[key] = graph_identity
        if row["status"] == "test_evaluated":
            test_identity = (row["split_hash"], row["test_metrics"]["n"], row["test_metrics"]["positive_n"])
            require(key not in shared_tests or shared_tests[key] == test_identity,
                    "Compared models/repeats did not use the same held-out population")
            shared_tests[key] = test_identity
    return {"status": "passed", "scope": "selected_test", "verified_runs": len(selected_rows),
            "verified_training_runs": len(verified), "verified_validation_trials": len(expected_trials),
            "verified_test_evaluations": len(selected_rows), "runs": selected_rows,
            "validation_trials": [verified[path_key(job["out"])] for job in expected_trials],
            "selected_runs": [row["out"] for row in selected_rows],
            "selected_index_sha256": sha256_file(campaign / "selected_index.json"),
            "selection_registry_sha256": registry_hash, "selection_signature": registry["selection_signature"],
            "provenance": provenance,
            "metric_definition": "AUROC and trapezoidal PR AUPR; average_precision retained separately; threshold fixed at 0.5",
            "statistical_scope": "Training seeds are repeats. Outer folds share training data and are dependent; no naive 15-sample significance test.",
            "evaluation_limit": contract["evaluation"]["test_status"],
            "selection_freeze_check": "Checkpoint-specific freeze hashes match each test record and the validation-only selection registry."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=HERE / "runs" / "repair_20261007")
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared")
    parser.add_argument("--run", type=Path, help="Verify one completed run, without claiming full campaign completion")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run:
        row = verify_run(args.run, args.prepared)
        report = {"status": "passed", "scope": "single_run", "verified_runs": 1, "runs": [row]}
    else:
        report = verify_campaign(args.campaign, args.prepared)
    if args.output:
        output = args.output.resolve()
        allowed = (HERE / "runs" / "repair_20261007", HERE / "results" / "repair_20261007")
        require(any(base.resolve() in output.parents for base in allowed), "Report output must stay inside the new repair campaign")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key not in {"runs", "validation_trials"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
