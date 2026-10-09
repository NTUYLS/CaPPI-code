"""Reproducible, immutable pair and protein-disjoint CaPPI benchmarks.

Raw files are read-only. Graph inputs contain training positive edges only.
Unobserved pairs are *putative* negatives, not experimentally verified negatives.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "cappi-data-v1"
NEGATIVE_MODES = ("random", "degree_matched")
PROTOCOLS = ("pair", "protein")
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_pairs(pairs: Any) -> np.ndarray:
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return set(map(tuple, np.asarray(pairs).tolist()))


def _integer_id(value: str, location: str) -> int:
    text = value.strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"Non-integer protein ID at {location}: {value!r}")
    result = int(text)
    if not np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max:
        raise ValueError(f"Protein ID exceeds int64 at {location}")
    return result


def _read_raw(root: Path) -> tuple[list[dict], np.ndarray, dict, list[dict]]:
    nodes: list[dict] = []
    id_to_index: dict[int, int] = {}
    normalizations = 0
    with (root / "肝癌 sequence.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"name", "sequence"}.issubset(reader.fieldnames or []):
            raise ValueError("Sequence CSV must contain name,sequence columns")
        for row_number, row in enumerate(reader, 2):
            node_id = _integer_id(row["name"], f"sequence CSV row {row_number}")
            if node_id in id_to_index:
                raise ValueError(f"Duplicate sequence protein ID {node_id} at row {row_number}")
            raw_sequence = row["sequence"]
            if raw_sequence is None:
                raise ValueError(f"Missing sequence for protein {node_id}")
            sequence = "".join(raw_sequence.split()).upper()
            if not sequence or not re.fullmatch(r"[A-Z*]+", sequence):
                raise ValueError(f"Empty or invalid protein sequence for protein {node_id}")
            normalizations += sequence != raw_sequence
            index = len(nodes)
            id_to_index[node_id] = index
            nodes.append({"index": index, "node_id": node_id, "sequence": sequence})
    if len(nodes) < 3:
        raise ValueError("At least three distinct proteins are required")

    seen: set[tuple[int, int]] = set()
    changes: list[dict] = []
    raw_rows = 0
    with (root / "肝癌 interact.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"node1", "node2"}.issubset(reader.fieldnames or []):
            raise ValueError("Interaction CSV must contain node1,node2 columns")
        for row_number, row in enumerate(reader, 2):
            raw_rows += 1
            ids = [_integer_id(row[key], f"interaction CSV row {row_number}")
                   for key in ("node1", "node2")]
            if any(node_id not in id_to_index for node_id in ids):
                raise ValueError(f"Interaction row {row_number} references a missing sequence ID: {ids}")
            pair = tuple(sorted(id_to_index[node_id] for node_id in ids))
            reason = "self_loop" if pair[0] == pair[1] else "duplicate_undirected" if pair in seen else None
            if reason:
                changes.append({"csv_row": row_number, "node1": ids[0], "node2": ids[1], "reason": reason})
            else:
                seen.add(pair)
    if not seen:
        raise ValueError("No positive interactions remain after canonicalization")
    sequence_counts = Counter(row["sequence"] for row in nodes)
    audit = {
        "raw_interaction_rows": raw_rows,
        "canonical_positive_count": len(seen),
        "self_loops_removed": sum(row["reason"] == "self_loop" for row in changes),
        "duplicate_undirected_rows_removed": sum(row["reason"] == "duplicate_undirected" for row in changes),
        "sequence_whitespace_or_case_normalizations": normalizations,
        "distinct_sequences": len(sequence_counts),
        "duplicate_sequence_groups": sum(count > 1 for count in sequence_counts.values()),
        "proteins_in_duplicate_sequence_groups": sum(count for count in sequence_counts.values() if count > 1),
        "sequence_duplicates_policy": "Retained as distinct original IDs; protein-ID disjointness is not sequence-similarity OOD.",
    }
    return nodes, _as_pairs(sorted(seen)), audit, changes


def _degree_summary(degrees: np.ndarray, pairs: np.ndarray) -> dict:
    values = degrees[pairs.reshape(-1)].astype(float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "quantiles_0_25_50_75_100": np.quantile(values, [0, .25, .5, .75, 1]).tolist()}


def _negative_pools(positives: np.ndarray, n_nodes: int, seed: int) -> tuple[dict, dict]:
    # Full-positive degrees are allowed only for the declared benchmark sampling
    # design; these degrees are never exported as model node features.
    n_candidates = n_nodes * (n_nodes - 1) // 2
    if n_candidates > 20_000_000:
        raise ValueError("Candidate space exceeds the bounded in-memory sampler; implement and audit a streaming sampler first")
    all_pairs = np.column_stack(np.triu_indices(n_nodes, k=1)).astype(np.int64)
    positive_codes = positives[:, 0] * n_nodes + positives[:, 1]
    codes = all_pairs[:, 0] * n_nodes + all_pairs[:, 1]
    candidates = all_pairs[~np.isin(codes, positive_codes)]
    n_positive = len(positives)
    if len(candidates) < n_positive:
        raise ValueError(f"Cannot sample 1:1 unique negatives: {n_positive} positives, {len(candidates)} available pairs")
    random_rng = np.random.default_rng(seed + 1001)
    random_pool = candidates[random_rng.choice(len(candidates), n_positive, replace=False)]
    degree = np.bincount(positives.reshape(-1), minlength=n_nodes)
    bins = np.floor(np.log2(degree + 1)).astype(np.int64)
    positive_strata = np.sort(bins[positives], axis=1)
    candidate_strata = np.sort(bins[candidates], axis=1)
    target = Counter(map(tuple, positive_strata.tolist()))
    groups: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(seed + 2001)
    for key in sorted(set(map(tuple, candidate_strata.tolist()))):
        indices = np.flatnonzero(np.all(candidate_strata == key, axis=1))
        groups[key] = rng.permutation(indices)
    cursor = {key: 0 for key in groups}
    selected: list[int] = []
    deficits: list[tuple[tuple[int, int], int]] = []
    strata_audit: list[dict] = []
    # Reserve all available exact matches before allocating scarce fallbacks.
    for key, requested in sorted(target.items()):
        available = len(groups.get(key, []))
        exact = min(requested, available)
        if exact:
            selected.extend(groups[key][:exact].tolist())
            cursor[key] = exact
        deficit = requested - exact
        if deficit:
            deficits.append((key, deficit))
        strata_audit.append({"endpoint_degree_bins": list(key), "requested": requested,
                             "candidate_count": available, "exact_matches": exact, "shortfall": deficit})
    fallback_records: list[dict] = []
    for target_key, deficit in deficits:
        # Nearest available endpoint-degree-bin pair, with deterministic tie order.
        ordered = sorted(groups, key=lambda key: (abs(key[0] - target_key[0]) + abs(key[1] - target_key[1]), key))
        for actual_key in ordered:
            take = min(deficit, len(groups[actual_key]) - cursor[actual_key])
            if not take:
                continue
            start = cursor[actual_key]
            selected.extend(groups[actual_key][start:start + take].tolist())
            cursor[actual_key] += take
            deficit -= take
            fallback_records.append({"requested_bins": list(target_key), "actual_bins": list(actual_key), "count": take})
            if not deficit:
                break
        if deficit:
            raise AssertionError("Negative matching unexpectedly exhausted all candidates")
    matched_pool = candidates[np.asarray(selected, dtype=np.int64)]
    pools = {"random": random_pool, "degree_matched": matched_pool}
    sampling = {
        "ratio_negative_to_positive": 1,
        "label_interpretation": "Putative negatives: unobserved undirected pairs in the supplied positive network; absence is not experimentally verified non-interaction.",
        "excluded": ["every supplied known positive", "self loops", "within-pool duplicates"],
        "candidate_count": len(candidates),
        "degree_definition": "Undirected endpoint degree in the complete canonical supplied positive network.",
        "benchmark_design_disclosure": "The degree-matched pool is pre-specified using the complete positive network, including held-out positives. This is a benchmark sampling design, not a deployable unknown-edge selection procedure. Full-network degrees must not enter graph construction, learned preprocessing, or model inputs.",
        "degree_bins": "floor(log2(degree + 1)); match sorted endpoint-bin pairs without replacement; nearest available bin pairs fill shortfalls.",
        "matching_is_exact": not bool(deficits),
        "exact_bin_matches": n_positive - sum(count for _, count in deficits),
        "fallback_count": sum(count for _, count in deficits),
        "strata": strata_audit,
        "fallback_allocations": fallback_records,
        "positive_endpoint_degree": _degree_summary(degree, positives),
        "pools": {},
    }
    positive_set = _pair_set(positives)
    for mode, pairs in pools.items():
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        pools[mode] = pairs
        assert len(pairs) == n_positive and len(_pair_set(pairs)) == n_positive
        assert not positive_set.intersection(_pair_set(pairs))
        observed = Counter(map(tuple, np.sort(bins[pairs], axis=1).tolist()))
        keys = set(target) | set(observed)
        sampling["pools"][mode] = {
            "count": len(pairs), "endpoint_degree": _degree_summary(degree, pairs),
            "endpoint_bin_pair_total_variation": .5 * sum(abs(target[key] - observed[key]) for key in keys) / n_positive,
            "bin_pair_counts": [{"bins": list(key), "count": observed[key]} for key in sorted(observed)],
        }
    return pools, sampling


def _pair_assignments(n_pairs: int, seed: int, n_folds: int) -> list[dict]:
    outer = np.array_split(np.random.default_rng(seed + 3001).permutation(n_pairs), n_folds)
    assignments = []
    for fold in range(n_folds):
        remaining = np.concatenate([outer[k] for k in range(n_folds) if k != fold])
        shuffled = np.random.default_rng(seed + 4001 + fold).permutation(remaining)
        n_val = max(1, int(round(len(remaining) / 8)))
        if len(remaining) <= n_val:
            raise ValueError("Too few pairs for the requested train/validation/test protocol")
        assignments.append({"train": shuffled[n_val:], "validation": shuffled[:n_val], "test": outer[fold]})
    return assignments


def _node_assignments(n_nodes: int, seed: int, n_folds: int) -> list[dict]:
    outer = np.array_split(np.random.default_rng(seed + 5001).permutation(n_nodes), n_folds)
    assignments = []
    for fold in range(n_folds):
        remaining = np.concatenate([outer[k] for k in range(n_folds) if k != fold])
        shuffled = np.random.default_rng(seed + 6001 + fold).permutation(remaining)
        n_val = max(2, int(round(len(remaining) / 8)))
        if len(remaining) <= n_val:
            raise ValueError("Too few proteins for the requested node-disjoint protocol")
        assignments.append({"train": np.sort(shuffled[n_val:]), "validation": np.sort(shuffled[:n_val]),
                            "test": np.sort(outer[fold])})
    return assignments


def _fold_signature(base: Path, relative: str, files: dict[str, str]) -> str:
    relevant = {name: digest for name, digest in files.items()
                if name in ("nodes.csv", "positives.npz") or name.startswith(relative + "/")}
    return _json_hash(relevant)


def _validate_arrays(splits: dict, graph_edges: np.ndarray, nodes: dict, protocol: str,
                     n_nodes: int, known_positives: set[tuple[int, int]]) -> None:
    split_sets = []
    for split in SPLITS:
        pairs, labels = splits[split]
        if pairs.dtype != np.int64 or pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"Invalid pair array in {split}")
        if labels.dtype != np.float32 or labels.shape != (len(pairs),):
            raise ValueError(f"Invalid label array in {split}")
        if not np.isin(labels, [0., 1.]).all():
            raise ValueError(f"Non-binary label in {split}")
        if len(pairs) and (np.any(pairs[:, 0] >= pairs[:, 1]) or pairs.min() < 0 or pairs.max() >= n_nodes):
            raise ValueError(f"Invalid/noncanonical pair in {split}")
        pair_set = _pair_set(pairs)
        if len(pair_set) != len(pairs):
            raise ValueError(f"Duplicate pair in {split}")
        if not _pair_set(pairs[labels == 1]).issubset(known_positives):
            raise ValueError(f"Unknown positive label in {split}")
        if _pair_set(pairs[labels == 0]) & known_positives:
            raise ValueError(f"Negative conflicts with a known positive in {split}")
        if protocol == "protein" and not set(pairs.reshape(-1)).issubset(set(nodes[split])):
            raise ValueError(f"Protein membership violation in {split}")
        split_sets.append(pair_set)
    if any(split_sets[a] & split_sets[b] for a in range(3) for b in range(a + 1, 3)):
        raise ValueError("Undirected pair leaks across train/validation/test")
    if protocol == "protein" and any(set(nodes[SPLITS[a]]) & set(nodes[SPLITS[b]])
                                     for a in range(3) for b in range(a + 1, 3)):
        raise ValueError("Proteins leak across train/validation/test")
    train_pairs, train_labels = splits["train"]
    if _pair_set(graph_edges) != _pair_set(train_pairs[train_labels == 1]) or len(graph_edges) != int(train_labels.sum()):
        raise ValueError("Graph must contain exactly training positive edges")
    if _pair_set(graph_edges) & (split_sets[1] | split_sets[2]):
        raise ValueError("Held-out pair present in training graph")


def _verify_artifacts(prepared: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible prepared-data schema; create a fresh preparation directory")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Prepared-data file hashes are absent")
    request = {key: manifest.get(key) for key in ("schema_version", "seed", "n_folds", "source_sha256")}
    if _json_hash(request) != manifest.get("request_hash"):
        raise ValueError("Prepared manifest configuration fingerprint is inconsistent")
    if _json_hash({"request": request, "files": manifest["files"]}) != manifest.get("dataset_hash"):
        raise ValueError("Prepared manifest dataset fingerprint is inconsistent")
    expected = {"nodes.csv", "interaction_cleaning.csv", "positives.npz", "negatives_random.npz", "negatives_degree_matched.npz"}
    expected_folds = {f"{mode}/{protocol}/fold_{fold}" for mode in NEGATIVE_MODES for protocol in PROTOCOLS
                      for fold in range(manifest["n_folds"])}
    for relative in expected_folds:
        expected.update(f"{relative}/{filename}.npz" for filename in (*SPLITS, "graph"))
    if set(manifest["files"]) != expected or set(manifest.get("folds", {})) != expected_folds:
        raise ValueError("Prepared manifest artifact inventory is incomplete or incompatible")
    for relative, expected in manifest["files"].items():
        path = (prepared / relative).resolve()
        if not path.is_relative_to(prepared.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Prepared artifact missing, modified, or invalid: {relative}")


def prepare_dataset(root: Path, out: Path, seed: int = 20260930, n_folds: int = 5) -> dict:
    """Prepare and hash immutable benchmarks; refuse incompatible existing outputs.

    With five folds, pair splits approximate 70/10/20 per class. Protein splits
    use the analogous *node* fractions; pair fractions/label ratios vary after
    excluding cross-partition pairs and are recorded, never silently balanced.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    source_files = ("肝癌 sequence.csv", "肝癌 interact.csv")
    sources = {name: {"path": str(root / name), "sha256": sha256_file(root / name)} for name in source_files}
    request = {"schema_version": SCHEMA_VERSION, "seed": int(seed), "n_folds": int(n_folds),
               "source_sha256": {name: item["sha256"] for name, item in sources.items()}}
    request_hash = _json_hash(request)
    if out.exists():
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Refusing an existing incomplete/non-prepared directory: {out}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("request_hash") != request_hash:
            raise ValueError("Existing prepared data has different inputs/configuration; use a fresh output directory")
        _verify_artifacts(out, existing)
        return existing
    nodes, positives, audit, changes = _read_raw(root)
    if len(nodes) < n_folds or len(positives) < n_folds:
        raise ValueError("Each outer fold needs at least one protein and one positive pair")
    pools, sampling = _negative_pools(positives, len(nodes), seed)
    pair_assignments = _pair_assignments(len(positives), seed, n_folds)
    node_assignments = _node_assignments(len(nodes), seed, n_folds)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    files: dict[str, str] = {}

    def record(path: Path) -> None:
        files[path.relative_to(staging).as_posix()] = sha256_file(path)

    try:
        with (staging / "nodes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "node_id", "sequence"])
            writer.writeheader()
            writer.writerows(nodes)
        record(staging / "nodes.csv")
        with (staging / "interaction_cleaning.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["csv_row", "node1", "node2", "reason"])
            writer.writeheader()
            writer.writerows(changes)
        record(staging / "interaction_cleaning.csv")
        np.savez_compressed(staging / "positives.npz", pairs=positives)
        record(staging / "positives.npz")
        folds = {}
        known = _pair_set(positives)
        for mode in NEGATIVE_MODES:
            np.savez_compressed(staging / f"negatives_{mode}.npz", pairs=pools[mode])
            record(staging / f"negatives_{mode}.npz")
            for protocol in PROTOCOLS:
                for fold in range(n_folds):
                    relative = f"{mode}/{protocol}/fold_{fold}"
                    folder = staging / relative
                    folder.mkdir(parents=True)
                    node_partition = node_assignments[fold] if protocol == "protein" else {
                        split: np.arange(len(nodes), dtype=np.int64) for split in SPLITS}
                    splits = {}
                    counts = {}
                    selected_positive, selected_negative = 0, 0
                    for split in SPLITS:
                        if protocol == "pair":
                            index = pair_assignments[fold][split]
                            pos, neg = positives[index], pools[mode][index]
                        else:
                            membership = np.zeros(len(nodes), dtype=bool)
                            membership[node_partition[split]] = True
                            pos = positives[membership[positives].all(axis=1)]
                            neg = pools[mode][membership[pools[mode]].all(axis=1)]
                        selected_positive += len(pos)
                        selected_negative += len(neg)
                        pairs = np.concatenate([pos, neg], axis=0)
                        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
                        permutation = np.random.default_rng(seed + 7001 + fold * 17 + SPLITS.index(split)).permutation(len(pairs))
                        pairs, labels = pairs[permutation], labels[permutation]
                        splits[split] = (pairs, labels)
                        counts[split] = {"pairs": len(pairs), "positives": len(pos), "negatives": len(neg),
                                         "assigned_nodes": len(node_partition[split]),
                                         "observed_nodes": int(len(np.unique(pairs))),
                                         "has_both_classes": bool(len(pos) and len(neg))}
                        np.savez_compressed(folder / f"{split}.npz", pairs=pairs, labels=labels)
                        record(folder / f"{split}.npz")
                    graph_edges = splits["train"][0][splits["train"][1] == 1]
                    graph_edges = graph_edges[np.lexsort((graph_edges[:, 1], graph_edges[:, 0]))]
                    _validate_arrays(splits, graph_edges, node_partition, protocol, len(nodes), known)
                    np.savez_compressed(folder / "graph.npz", graph_edges=graph_edges,
                                        train_nodes=node_partition["train"], val_nodes=node_partition["validation"],
                                        test_nodes=node_partition["test"])
                    record(folder / "graph.npz")
                    fold_info = {"negative_mode": mode, "protocol": protocol, "fold": fold,
                                 "counts": counts, "train_graph_edges": len(graph_edges),
                                 "cross_partition_pairs_excluded": {
                                     "positives": len(positives) - selected_positive,
                                     "negatives": len(pools[mode]) - selected_negative},
                                 "split_hash": _fold_signature(staging, relative, files)}
                    folds[relative] = fold_info
        manifest = {**request, "request_hash": request_hash, "sources": sources,
                    "n_nodes": len(nodes), "n_positives": len(positives), "cleaning": audit,
                    "negative_sampling": sampling,
                    "protocols": {
                        "pair": "Stratified fixed outer pair folds; every positive and every negative pool pair is tested exactly once. Within each outer training set, 1/8 of each class is held out for validation. With five folds this is approximately 70/10/20. Both negative modes use the same positive memberships and the same negative pool-index memberships. All node identities are available: transductive evaluation.",
                        "protein": "Fixed outer node folds, with 1/8 of remaining nodes (at least two) held out for validation. Train/validation/test node IDs are mutually disjoint; only pairs with both endpoints inside a partition are retained. Cross-partition pairs are excluded and counted. Every node is a test node once; some pairs are never tested because their endpoints fall in different outer folds. Negative pools and node assignments are fixed before evaluation. This is ID-disjoint both-protein holdout, not sequence-homology OOD.",
                    },
                    "graph_policy": "Each graph.npz contains exactly the corresponding training positives as canonical E x 2 undirected edges. Message-passing code may expand these edges in both directions; validation/test edges and full-positive degrees must never be added.",
                    "identity_scope": "Original numeric CSV protein IDs are preserved verbatim as integers. This preparation does not assert or certify species, disease association, interaction evidence type, or STRING confidence threshold.",
                    "folds": folds, "files": files,
                    "dataset_hash": _json_hash({"request": request, "files": files})}
        # Recheck inputs before committing so concurrent source edits cannot be hidden.
        if any(sha256_file(root / name) != sources[name]["sha256"] for name in source_files):
            raise ValueError("Raw inputs changed during preparation; no dataset committed")
        _write_json(staging / "manifest.json", manifest)
        # Directory rename refuses to overwrite an existing destination on Windows.
        if out.exists():
            raise ValueError("Preparation destination appeared concurrently; refusing overwrite")
        staging.rename(out)
        return manifest
    except Exception as error:
        # Keep an inspectable failed staging directory; never overwrite raw data.
        if staging.is_dir():
            (staging / "PREPARATION_FAILED.txt").write_text(str(error), encoding="utf-8")
        raise


def load_fold(prepared_dir: Path, negative_mode: str, protocol: str, fold: int) -> dict:
    """Load a verified immutable fold using global zero-based node indices."""
    prepared = Path(prepared_dir).resolve()
    if negative_mode not in NEGATIVE_MODES or protocol not in PROTOCOLS:
        raise ValueError("Unknown negative mode or protocol")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    _verify_artifacts(prepared, manifest)
    if not isinstance(fold, int) or not 0 <= fold < manifest["n_folds"]:
        raise ValueError("Fold index out of range")
    relative = f"{negative_mode}/{protocol}/fold_{fold}"
    folder = prepared / relative
    fold_info = manifest["folds"][relative]
    if _fold_signature(prepared, relative, manifest["files"]) != fold_info["split_hash"]:
        raise ValueError("Fold fingerprint is inconsistent with the manifest")
    with (prepared / "nodes.csv").open(encoding="utf-8", newline="") as handle:
        nodes = list(csv.DictReader(handle))
    if [int(row["index"]) for row in nodes] != list(range(len(nodes))) or len(nodes) != manifest["n_nodes"]:
        raise ValueError("Invalid node index mapping")
    with np.load(prepared / "positives.npz", allow_pickle=False) as data:
        known = _pair_set(data["pairs"])
    if len(known) != manifest["n_positives"]:
        raise ValueError("Positive count is inconsistent with the manifest")
    splits = {}
    for split in SPLITS:
        with np.load(folder / f"{split}.npz", allow_pickle=False) as data:
            splits[split] = (data["pairs"], data["labels"])
    with np.load(folder / "graph.npz", allow_pickle=False) as data:
        graph_edges = data["graph_edges"]
        node_partition = {"train": data["train_nodes"], "validation": data["val_nodes"], "test": data["test_nodes"]}
    _validate_arrays(splits, graph_edges, node_partition, protocol, len(nodes), known)
    return {"train_pairs": splits["train"][0], "train_labels": splits["train"][1],
            "val_pairs": splits["validation"][0], "val_labels": splits["validation"][1],
            "test_pairs": splits["test"][0], "test_labels": splits["test"][1],
            "graph_edges": graph_edges, "n_nodes": len(nodes),
            "node_ids": np.asarray([int(row["node_id"]) for row in nodes], dtype=np.int64),
            "sequences": [row["sequence"] for row in nodes],
            "split_hash": fold_info["split_hash"],
            "train_nodes": node_partition["train"], "val_nodes": node_partition["validation"],
            "test_nodes": node_partition["test"], "manifest": manifest}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    result = prepare_dataset(args.root, args.out, args.seed, args.n_folds)
    print(json.dumps({"prepared_dir": str(args.out.resolve()), "n_nodes": result["n_nodes"],
                      "n_positives": result["n_positives"], "dataset_hash": result["dataset_hash"],
                      "degree_matching_fallback_count": result["negative_sampling"]["fallback_count"]}, indent=2))
