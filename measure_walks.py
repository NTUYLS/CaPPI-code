"""Describe actual train-graph walks without training embeddings or classifiers.

Traversal diagnostics do not measure biological coverage, useful biological
information, interaction validity, or predictive performance.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from cappi_data import load_fold
from cappi_graph import GraphConfig, fingerprint, make_walks

HERE = Path(__file__).resolve().parent
METHOD_LABELS = {
    "drw": "DRW",
    "deepwalk": "Uniform walks (DeepWalk-style, shared SGNS)",
    "node2vec": "node2vec walks (shared SGNS)",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with ties for descriptive Spearman correlation."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def describe_walks(edges: np.ndarray, n_nodes: int, config: GraphConfig, seed: int) -> tuple[dict, str]:
    walks, present = make_walks(edges, n_nodes, config, seed)
    degree = np.bincount(edges.reshape(-1), minlength=n_nodes)
    active = degree > 0
    if not np.array_equal(active, present):
        raise ValueError("Walk node-presence mask differs from train-graph degree")
    if len(walks) != int(active.sum()) * config.walks_per_node or not walks:
        raise ValueError("Unexpected or empty walk corpus")
    adjacency = [set() for _ in range(n_nodes)]
    for u, v in edges:
        adjacency[u].add(int(v))
        adjacency[v].add(int(u))
    lengths, distinct = [], []
    visits = np.zeros(n_nodes, dtype=np.int64)
    starts = np.zeros(n_nodes, dtype=np.int64)
    transitions = revisits = forced_revisit_steps = immediate_returns = 0
    for walk in walks:
        if not walk:
            raise ValueError("Unexpected empty walk")
        lengths.append(len(walk))
        distinct.append(len(set(walk)))
        starts[walk[0]] += 1
        visits[walk[0]] += 1
        visited = {walk[0]}
        for position in range(1, len(walk)):
            previous, node = walk[position - 1], walk[position]
            if node not in adjacency[previous]:
                raise ValueError("Walk contains an edge absent from the training graph")
            no_fresh_neighbor = adjacency[previous].issubset(visited)
            is_revisit = node in visited
            if config.method == "drw" and is_revisit and not no_fresh_neighbor:
                raise ValueError("DRW revisited despite an available unvisited neighbor")
            forced_revisit_steps += no_fresh_neighbor
            revisits += is_revisit
            immediate_returns += position >= 2 and node == walk[position - 2]
            transitions += 1
            visits[node] += 1
            visited.add(node)
    if not np.all(starts[active] == config.walks_per_node) or np.any(starts[~active]):
        raise ValueError("Start-node budget does not match GraphConfig")
    if not np.array_equal(visits > 0, active):
        raise ValueError("Corpus coverage differs from the nonisolated train nodes")
    lengths_array = np.asarray(lengths, dtype=float)
    distinct_array = np.asarray(distinct, dtype=float)
    ratios = distinct_array / lengths_array
    active_degree, active_visits = degree[active], visits[active]
    corpus_sha = hashlib.sha256(json.dumps(walks).encode("utf-8")).hexdigest()
    result = {
        "n_nodes": n_nodes, "train_positive_edges": len(edges),
        "train_active_nodes": int(active.sum()), "train_isolated_nodes": int((~active).sum()),
        "walks_per_active_node": config.walks_per_node, "configured_walk_length": config.walk_length,
        "n_walks": len(walks), "total_node_visits": int(visits.sum()), "total_transitions": transitions,
        "walk_length_mean": float(lengths_array.mean()),
        "walk_distinct_node_count_mean": float(distinct_array.mean()),
        "walk_distinct_node_count_sd": float(distinct_array.std(ddof=1)) if len(walks) > 1 else None,
        "walk_distinct_node_fraction_mean": float(ratios.mean()),
        "walk_distinct_node_fraction_sd": float(ratios.std(ddof=1)) if len(walks) > 1 else None,
        "walk_distinct_node_count_min": int(distinct_array.min()),
        "walk_distinct_node_count_max": int(distinct_array.max()),
        "revisited_destination_steps": int(revisits),
        "revisited_destination_fraction": float(revisits / transitions) if transitions else None,
        "steps_with_no_unvisited_neighbor": int(forced_revisit_steps),
        "no_unvisited_neighbor_fraction": float(forced_revisit_steps / transitions) if transitions else None,
        "drw_fallback_fraction": float(forced_revisit_steps / transitions) if transitions and config.method == "drw" else None,
        "immediate_return_steps": int(immediate_returns),
        "immediate_return_fraction_all_transitions": float(immediate_returns / transitions) if transitions else None,
        "corpus_distinct_nodes": int(np.count_nonzero(visits)),
        "corpus_active_node_fraction": float(np.count_nonzero(visits) / active.sum()),
        "corpus_all_node_fraction": float(np.count_nonzero(visits) / n_nodes),
        "visit_count_degree_pearson_active_nodes": correlation(active_degree, active_visits),
        "visit_count_degree_spearman_active_nodes": correlation(average_ranks(active_degree), average_ranks(active_visits)),
        "walk_corpus_sha256": corpus_sha,
    }
    return result, corpus_sha


def measure(prepared: Path, output: Path, campaign: Path, seed: int = 20260930) -> dict:
    prepared, output, campaign = prepared.resolve(), output.resolve(), campaign.resolve()
    destinations = [output / "walk_diagnostics.csv", output / "walk_diagnostics.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Walk diagnostics already exist; use a fresh --output directory to preserve them")
    rows = []
    for fold in range(5):
        data = load_fold(prepared, "random", "pair", fold)
        if seed != data["manifest"]["seed"]:
            raise ValueError("Diagnostic seed must match this prepared campaign's seed")
        for method in METHOD_LABELS:
            config = GraphConfig(method=method)
            walk_seed = seed + fold
            values, corpus_sha = describe_walks(data["graph_edges"], data["n_nodes"], config, walk_seed)
            graph_hash = fingerprint(data["graph_edges"], data["n_nodes"], config, walk_seed)
            cached_metadata = campaign / "graph_cache" / f"{graph_hash}.json"
            cache_status = "not_available"
            if cached_metadata.is_file():
                metadata = json.loads(cached_metadata.read_text(encoding="utf-8"))
                if metadata["skipgram"]["walks_sha256"] != corpus_sha:
                    raise ValueError("Regenerated corpus does not match the training graph cache")
                cache_status = "walk_corpus_sha256_matches_saved_training_metadata"
            row = {"protocol": "pair", "negative_sampling": "random", "fold": fold,
                   "method": method, "method_label": METHOD_LABELS[method], "seed": walk_seed,
                   "split_hash": data["split_hash"], "graph_fingerprint": graph_hash,
                   "saved_training_corpus_check": cache_status, **values}
            rows.append(row)
            print(f"fold={fold} method={method} distinct/length={values['walk_distinct_node_fraction_mean']:.6f} "
                  f"corpus_nodes={values['corpus_distinct_nodes']} fallback={values['drw_fallback_fraction']}", flush=True)
    summary_metrics = ["walk_distinct_node_count_mean", "walk_distinct_node_fraction_mean", "revisited_destination_fraction",
                       "drw_fallback_fraction", "corpus_distinct_nodes", "corpus_active_node_fraction",
                       "visit_count_degree_pearson_active_nodes", "visit_count_degree_spearman_active_nodes"]
    summary = {}
    for method in METHOD_LABELS:
        method_rows = [row for row in rows if row["method"] == method]
        method_summary = {"n_folds": len(method_rows)}
        for key in summary_metrics:
            values = [row[key] for row in method_rows if row[key] is not None]
            method_summary[key + "_fold_mean"] = float(np.mean(values)) if values else None
            method_summary[key + "_fold_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary[method] = method_summary
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "status": "completed_without_embedding_or_classifier_training",
        "dataset_hash": manifest["dataset_hash"],
        "prepared_manifest_sha256": sha(prepared / "manifest.json"),
        "graph_implementation_sha256": sha(HERE / "cappi_graph.py"),
        "diagnostic_script_sha256": sha(Path(__file__).resolve()),
        "graph_configs": {method: asdict(GraphConfig(method=method)) for method in METHOD_LABELS},
        "definitions": {
            "walk_distinct_node_fraction": "Number of distinct node IDs in one generated walk divided by its actual node-token length; includes the starting node.",
            "drw_fallback_fraction": "For DRW only: transitions whose current node has no unvisited neighbor, divided by all transitions. Other methods have null because they do not use DRW's fallback rule.",
            "revisited_destination_fraction": "Transitions to a node already visited earlier in the same walk, divided by all transitions.",
            "corpus_distinct_nodes": "Distinct IDs anywhere in the complete corpus. Every nonisolated node is deliberately used as a start node, so complete active-node coverage is guaranteed by this budget.",
            "degree_correlations": "Descriptive correlation across nonisolated training nodes between their training-graph degree and total corpus visit counts, including starts; no significance test.",
            "summary_variation": "Means and sample SD across the five predefined training folds; not confidence intervals or independent repeated-seed estimates.",
            "deepwalk_control": "Uniform-walk/DeepWalk-style walk control sharing SGNS with DRW and node2vec; not an exact reproduction of original DeepWalk hierarchical-softmax training.",
        },
        "interpretation_limits": [
            "Traversal diversity does not establish biological coverage, useful information gain, or predictive superiority.",
            "No embedding, classifier, hypothesis test, or hyperparameter search is run by this script.",
            "Only pair/random five-fold training graphs are inspected; validation/test edges are excluded.",
            "Different transition kernels consume random numbers differently; identical seeds do not imply identical walk paths.",
        ],
        "fold_summary": summary, "folds": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    with destinations[0].open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with destinations[1].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--campaign", type=Path, default=HERE / "runs" / "reviewer_v1")
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    report = measure(args.prepared, args.output, args.campaign, args.seed)
    print(json.dumps(report["fold_summary"], indent=2))
