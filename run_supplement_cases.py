"""Supplementary validation-set trajectories and target-withheld rankings.

All original inputs, trained models and primary experiments are read-only.
The 600-epoch trajectory is descriptive, never a model-selection experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch  # Import before sklearn on the established Windows environment.
from sklearn.metrics import average_precision_score, roc_auc_score

from cappi_data import load_fold
from cappi_graph import GraphConfig, train_topology
from cappi_model import CaPPI, ModelConfig, get_parameter_report, tokenize_sequences
from run_experiments import code_hash, save_checkpoint, seed_all, sha, write_json

HERE = Path(__file__).resolve().parent
TARGETS = ("X271_00001", "X271_00015")
SNAPSHOTS = (50, 150, 300, 600)
TRAJECTORY_MAX_EPOCHS = 600
TARGET_MAX_EPOCHS = 300
TARGET_PATIENCE = 20


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def read_fasta(path):
    records = {}; name = None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in records:
                raise ValueError(f"Duplicate FASTA ID: {name}")
            records[name] = ""
        elif line.strip():
            if name is None:
                raise ValueError("FASTA sequence precedes header")
            records[name] += line.strip().upper()
    return records


def map_identifiers(data, fasta_records, known_pairs, source_edges):
    """Join by a unique exact full sequence, never by numeric suffix or row order."""
    reverse = {}
    for identifier, sequence in fasta_records.items():
        reverse.setdefault(sequence, []).append(identifier)
    mappings = []
    for index, sequence in enumerate(data["sequences"]):
        matches = reverse.get(sequence, [])
        if len(matches) != 1:
            raise ValueError(f"Node {data['node_ids'][index]} has {len(matches)} exact FASTA matches")
        mappings.append({"index": index, "node_id": int(data["node_ids"][index]),
                         "source_id": matches[0], "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()})
    if len({r["source_id"] for r in mappings}) != len(mappings):
        raise ValueError("Multiple prepared nodes map to the same source ID")
    source_to_index = {r["source_id"]: r["index"] for r in mappings}
    mapped_edges = {tuple(sorted((source_to_index[a], source_to_index[b])))
                    for a, b in source_edges if a in source_to_index and b in source_to_index and a != b}
    if mapped_edges != set(map(tuple, np.asarray(known_pairs).tolist())):
        raise ValueError("Mapped source links and prepared positive edge set differ")
    targets = {}
    for target in TARGETS:
        matches = [r for r in mappings if r["source_id"].split(".")[-1] == target]
        if len(matches) != 1:
            raise ValueError(f"Target {target} has {len(matches)} unambiguous mapped entries")
        targets[target] = matches[0]
    return mappings, targets


def source_mapping(args, data):
    fasta = Path(args.source_fasta); links = Path(args.source_links)
    with np.load(Path(args.prepared) / "positives.npz", allow_pickle=False) as z:
        positives = z["pairs"].copy()
    edges = []; scores = {}
    for line in links.read_text(encoding="utf-8-sig").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3:
            raise ValueError("Malformed source link row")
        a, b, score = fields[0], fields[1], int(fields[-1])
        edges.append((a, b)); scores[tuple(sorted((a, b)))] = score
    mappings, targets = map_identifiers(data, read_fasta(fasta), positives, edges)
    evidence = {"method": "unique exact full-sequence join; complete positive edge-set equality",
                "sources": {str(p): sha(p) for p in (fasta, links, Path(args.prepared) / "nodes.csv",
                                                     Path(args.prepared) / "positives.npz")},
                "n_mapped_nodes": len(mappings), "positive_edges_matched": len(positives),
                "targets": targets, "numeric_suffix_assumed": False}
    return mappings, targets, positives, scores, evidence


def target_split(base, target, positives):
    """Retain fold-0 non-target membership; remove every target train/val pair."""
    data = dict(base); excluded = {}
    for split in ("train", "val"):
        pairs = base[f"{split}_pairs"]
        keep = ~np.any(pairs == target, axis=1)
        data[f"{split}_pairs"] = pairs[keep].copy()
        data[f"{split}_labels"] = base[f"{split}_labels"][keep].copy()
        excluded[split] = int((~keep).sum())
    data["graph_edges"] = data["train_pairs"][data["train_labels"] == 1].copy()
    for split in ("train", "val"):
        if np.any(data[f"{split}_pairs"] == target):
            raise AssertionError("Target leaked into optimization or validation")
        if len(set(data[f"{split}_labels"])) != 2:
            raise ValueError("Target exclusion leaves an empty or one-class split")
    data["train_nodes"] = np.unique(data["train_pairs"])
    data["val_nodes"] = np.unique(data["val_pairs"])
    others = np.asarray([i for i in range(base["n_nodes"]) if i != target], dtype=np.int64)
    data["test_pairs"] = np.sort(np.column_stack((np.full(len(others), target), others)), axis=1)
    known = set(map(tuple, np.asarray(positives).tolist()))
    data["test_labels"] = np.asarray([int(tuple(pair) in known) for pair in data["test_pairs"]], dtype=np.int64)
    data["test_nodes"] = np.asarray([target], dtype=np.int64)
    graph = set(map(tuple, data["graph_edges"].tolist()))
    for split in ("val", "test"):
        if graph & set(map(tuple, data[f"{split}_pairs"].tolist())):
            raise AssertionError("Held-out edge entered target training graph")
    audit = {"base_split_hash": base["split_hash"], "target_index": int(target),
             "excluded_pairs": excluded, "all_candidates": len(others),
             "training_pair_count": len(data["train_pairs"]), "validation_pair_count": len(data["val_pairs"]),
             "training_positive_edges": len(graph), "target_train_or_val_pair_count": 0,
             "target_graph_degree": 0, "candidate_graph_overlap": 0,
             "policy": "fold-0 non-target membership retained; target absent from train and validation; all other nodes ranked"}
    data["split_hash"] = digest({"audit": audit, "arrays": array_digest(*[data[k] for k in
        ("train_pairs", "train_labels", "val_pairs", "val_labels", "test_pairs", "test_labels", "graph_edges")])})
    audit["split_hash"] = data["split_hash"]
    return data, audit


def symmetric_hidden(model, node_vectors, pairs):
    """32-D ReLU directly before final logit; average both endpoint orders."""
    if model.training:
        raise ValueError("Hidden representation extraction requires eval mode")
    head = model.pair_head
    a, b = node_vectors[pairs[:, 0]], node_vectors[pairs[:, 1]]
    forward = head.classifier[:-1](torch.cat((head.left(a), head.right(b)), dim=-1))
    reverse = head.classifier[:-1](torch.cat((head.left(b), head.right(a)), dim=-1))
    return (forward + reverse) * .5


def train_case(args, data, out, trajectory=False):
    """Same full-batch update and validation scheduler as run_experiments.py."""
    out = Path(out); device = torch.device(args.device)
    cfg = ModelConfig(fusion="residual", max_length=1500, dropout=.3)
    graph_cfg = GraphConfig(method="drw", epochs=5)
    identity = {"core_code_sha256": code_hash(), "supplement_code_sha256": sha(Path(__file__)),
                "prepared_manifest_sha256": sha(Path(args.prepared) / "manifest.json"),
                "split_hash": data["split_hash"], "model": asdict(cfg), "graph": asdict(graph_cfg),
                "seed": args.seed, "max_epochs": TRAJECTORY_MAX_EPOCHS if trajectory else TARGET_MAX_EPOCHS,
                "early_stopping_patience": None if trajectory else TARGET_PATIENCE, "learning_rate": .0005,
                "weight_decay": .0001, "pair_chunk": 4096, "torch_threads": 2,
                "device": args.device, "torch": str(torch.__version__), "numpy": np.__version__,
                "selection": "validation minimum BCE only; no test selection",
                "purpose": "validation-only exploratory trajectory; not primary-model replacement" if trajectory else
                           "exploratory single-target holdout; no external or experimental validation"}
    signature = digest(identity)
    if (out / "training_completed.json").exists():
        result = json.loads((out / "training_completed.json").read_text(encoding="utf-8"))
        if result["run_signature"] != signature:
            raise ValueError("Incompatible completed case")
        for filename, expected in result["artifacts"].items():
            if sha(out / filename) != expected:
                raise ValueError(f"Completed artifact changed: {filename}")
        return result
    if out.exists() and (out / "run_config.json").exists():
        previous = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        if previous["run_signature"] != signature or not args.resume:
            raise ValueError("Existing case needs compatible --resume or fresh output")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", {**identity, "run_signature": signature, "python": sys.executable})
    np.savez_compressed(out / "split_arrays.npz", **{k: data[k] for k in
        ("train_pairs", "train_labels", "val_pairs", "val_labels", "test_pairs", "test_labels", "graph_edges", "node_ids")})
    seed_all(args.seed)
    embedding, present, graph_hash = train_topology(data["graph_edges"], data["n_nodes"], graph_cfg,
        args.seed, str(device), Path(args.output) / "graph_cache")
    fitting = np.asarray(data["train_nodes"], int)
    fitting = fitting[present[fitting]]
    mean = embedding[fitting].mean(0) if len(fitting) else np.zeros(cfg.topo_dim)
    std = embedding[fitting].std(0) if len(fitting) else np.ones(cfg.topo_dim)
    embedding = (embedding - mean) / np.maximum(std, 1e-6); embedding[~present] = 0
    np.savez_compressed(out / "topology.npz", embedding=embedding, present=present, mean=mean, std=std,
                        graph_hash=graph_hash)
    tokens = tokenize_sequences(data["sequences"], cfg.max_length).to(device)
    topology = torch.as_tensor(embedding, dtype=torch.float32, device=device)
    availability = torch.as_tensor(present, dtype=torch.bool, device=device)
    tensors = {k: torch.as_tensor(data[k], dtype=torch.long if "pairs" in k else torch.float32,
                                 device=device) for k in ("train_pairs", "train_labels", "val_pairs", "val_labels")}
    seed_all(args.seed); model = CaPPI(cfg).to(device)
    write_json(out / "parameters.json", get_parameter_report(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=10)
    history = []; best = float("inf"); best_epoch = None; stale = 0; start_epoch = 0
    last = out / "last.pt"; started = time.monotonic()
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck["run_signature"] != signature or ck["graph_hash"] != graph_hash:
            raise ValueError("Resume checkpoint provenance mismatch")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history, best, best_epoch, stale = ck["history"], ck["best"], ck["best_epoch"], ck["stale"]
        start_epoch = ck["epoch"] + 1
        random.setstate(ck["rng_python"]); np.random.set_state(ck["rng_numpy"])
        torch.set_rng_state(ck["rng_torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in ck["rng_cuda"]])
    max_epochs = identity["max_epochs"]
    print(f"START {out.name} train={len(tensors['train_labels'])} val={len(tensors['val_labels'])}", flush=True)
    for epoch in range(start_epoch, max_epochs if trajectory or stale < TARGET_PATIENCE else start_epoch):
        model.train(); optimizer.zero_grad(set_to_none=True)
        vectors = model.encode_nodes(tokens, topology, availability)
        order = torch.randperm(len(tensors["train_pairs"]), device=device)
        losses = []; loss_sum = 0.0
        for i in range(0, len(order), 4096):
            idx = order[i:i + 4096]
            logits = model.score_pairs(vectors, tensors["train_pairs"][idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tensors["train_labels"][idx],
                        reduction="sum") / len(order)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            losses.append(loss); loss_sum += float(loss.detach())
        torch.stack(losses).sum().backward(); del losses, vectors
        if epoch == start_epoch:
            gradients = {name: float(p.grad.norm()) if p.grad is not None else None
                         for name, p in model.named_parameters()}
            if any(x is not None and not np.isfinite(x) for x in gradients.values()):
                raise FloatingPointError("Nonfinite gradient")
            if not (out / "first_step_gradient_norms.json").exists():
                write_json(out / "first_step_gradient_norms.json", gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); model.eval()
        with torch.no_grad():
            vectors = model.encode_nodes(tokens, topology, availability)
            val_logits = torch.cat([model.score_pairs(vectors, tensors["val_pairs"][i:i + 4096])
                                    for i in range(0, len(tensors["val_pairs"]), 4096)])
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(val_logits, tensors["val_labels"]))
            val_ap = float(average_precision_score(data["val_labels"], torch.sigmoid(val_logits).cpu().numpy()))
            if trajectory and epoch + 1 in SNAPSHOTS:
                hidden = torch.cat([symmetric_hidden(model, vectors, tensors["val_pairs"][i:i + 4096])
                                    for i in range(0, len(tensors["val_pairs"]), 4096)]).cpu().numpy()
                np.savez_compressed(out / f"validation_hidden_epoch_{epoch+1}.npz", features=hidden,
                    pairs=data["val_pairs"], labels=data["val_labels"], node_ids=data["node_ids"],
                    epoch=epoch + 1, split_hash=data["split_hash"])
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        scheduler.step(val_loss)
        improved = val_loss < best - 1e-6
        if improved:
            best, best_epoch, stale = val_loss, epoch + 1, 0
        else:
            stale += 1
        history.append({"epoch": epoch + 1, "train_bce": loss_sum, "validation_bce": val_loss,
                        "validation_ap": val_ap, "learning_rate": optimizer.param_groups[0]["lr"]})
        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
              "epoch": epoch, "history": history, "best": best, "best_epoch": best_epoch, "stale": stale,
              "config": identity, "run_signature": signature, "graph_hash": graph_hash,
              "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
              "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if improved:
            save_checkpoint(ck, out / "best.pt")
        if trajectory and epoch + 1 in SNAPSHOTS:
            save_checkpoint(ck, out / f"snapshot_epoch_{epoch+1}.pt")
        if (epoch + 1) % 5 == 0 or (not trajectory and stale >= TARGET_PATIENCE) or epoch + 1 == max_epochs:
            save_checkpoint(ck, last)
        write_json(out / "history.json", history)
        if epoch == start_epoch or (epoch + 1) % 25 == 0 or (not trajectory and stale >= TARGET_PATIENCE):
            print(f"{out.name} epoch={epoch+1} train={loss_sum:.5f} val={val_loss:.5f} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if not trajectory and stale >= TARGET_PATIENCE:
            break
    artifacts = {name: sha(out / name) for name in ("best.pt", "last.pt", "history.json", "split_arrays.npz", "topology.npz")}
    if trajectory:
        for e in SNAPSHOTS:
            for name in (f"snapshot_epoch_{e}.pt", f"validation_hidden_epoch_{e}.npz"):
                artifacts[name] = sha(out / name)
    result = {"run_signature": signature, "split_hash": data["split_hash"], "graph_hash": graph_hash,
              "best_epoch": best_epoch, "epochs_completed": len(history), "best_validation_bce": best,
              "trajectory": trajectory, "elapsed_seconds_this_invocation": time.monotonic() - started,
              "test_used_for_selection": False, "artifacts": artifacts, "config": identity}
    write_json(out / "training_completed.json", result)
    del model, optimizer, scheduler, tokens, topology, tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def rank_target(args, data, mapping, target, scores, out, training):
    out = Path(out); device = torch.device(args.device)
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    if ck["run_signature"] != training["run_signature"]:
        raise ValueError("Ranking checkpoint identity mismatch")
    model = CaPPI(ModelConfig(**ck["config"]["model"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    with np.load(out / "topology.npz", allow_pickle=False) as z:
        embedding, present = z["embedding"].copy(), z["present"].copy()
    index = target["index"]
    if present[index] or np.any(embedding[index] != 0):
        raise AssertionError("Withheld target has nonzero training topology")
    with torch.no_grad():
        vectors = model.encode_nodes(tokenize_sequences(data["sequences"], 1500).to(device),
            torch.as_tensor(embedding, dtype=torch.float32, device=device),
            torch.as_tensor(present, dtype=torch.bool, device=device))
        pair_tensor = torch.as_tensor(data["test_pairs"], dtype=torch.long, device=device)
        predictions = torch.sigmoid(model.score_pairs(vectors, pair_tensor)).cpu().numpy()
    rows = []; checkpoint_hash = sha(out / "best.pt")
    for pair, label, score in zip(data["test_pairs"], data["test_labels"], predictions):
        partner = int(pair[1] if pair[0] == index else pair[0]); source_partner = mapping[partner]["source_id"]
        rows.append({"target_source_id": target["source_id"], "target_node_id": target["node_id"],
                     "partner_source_id": source_partner, "partner_node_id": mapping[partner]["node_id"],
                     "model_score": float(score), "known_association": int(label),
                     "evidence": "Recorded in supplied STRING links" if label else "Unobserved in supplied STRING links",
                     "source_combined_score": scores.get(tuple(sorted((target["source_id"], source_partner)))),
                     "split_hash": data["split_hash"], "checkpoint_sha256": checkpoint_hash, "seed": args.seed})
    rows.sort(key=lambda r: (-r["model_score"], r["partner_source_id"]))
    rows = [{"rank": i + 1, **row} for i, row in enumerate(rows)]
    write_rows(out / "all_candidates.csv", rows); write_rows(out / "top20_candidates.csv", rows[:20])
    labels = data["test_labels"]; hits = sum(row["known_association"] for row in rows[:20])
    result = {"status": "completed", "target": target, "n_candidates": len(rows), "known_associations": int(labels.sum()),
              "auroc": float(roc_auc_score(labels, predictions)) if len(set(labels)) == 2 else None,
              "average_precision": float(average_precision_score(labels, predictions)),
              "hits_at_20_count": hits, "precision_at_20": hits / min(20, len(rows)),
              "recall_at_20": hits / int(labels.sum()) if labels.sum() else None,
              "metrics_scope": "retrospective recovery of supplied database associations among all nonself candidates",
              "limitations": "single target and seed; unknown pairs are not verified negatives; sigmoid score is not calibrated binding probability; no external or experimental validation",
              "checkpoint_sha256": checkpoint_hash, "split_hash": data["split_hash"],
              "all_candidates_sha256": sha(out / "all_candidates.csv"), "top20_sha256": sha(out / "top20_candidates.csv")}
    write_json(out / "ranking_completed.json", result)
    del model, ck, vectors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def plot_trajectory(out, seed):
    out = Path(out); features = []; frames = []
    for epoch in SNAPSHOTS:
        with np.load(out / f"validation_hidden_epoch_{epoch}.npz", allow_pickle=False) as z:
            frames.append({k: z[k].copy() for k in z.files}); features.append(z["features"].copy())
    if any(not np.array_equal(frame["pairs"], frames[0]["pairs"]) or
           not np.array_equal(frame["labels"], frames[0]["labels"]) for frame in frames[1:]):
        raise ValueError("Trajectory snapshots contain different validation samples")
    try:
        import umap
    except ImportError as exc:
        result = {"status": "features_saved_umap_unavailable", "reason": str(exc), "PCA_substituted": False}
        write_json(out / "umap_completed.json", result); return result
    joint = np.concatenate(features, axis=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=.1, n_components=2, metric="euclidean",
                        random_state=seed, transform_seed=seed, n_jobs=1)
    coordinates = reducer.fit_transform(joint)
    np.savez_compressed(out / "joint_umap_coordinates.npz", coordinates=coordinates,
                        epochs=np.repeat(SNAPSHOTS, len(features[0])), labels=np.tile(frames[0]["labels"], 4),
                        pairs=np.tile(frames[0]["pairs"], (4, 1)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    n = len(features[0]); labels = frames[0]["labels"]
    for ax, epoch, start in zip(axes.flat, SNAPSHOTS, range(0, len(coordinates), n)):
        xy = coordinates[start:start + n]
        for label, color, name in ((0, "#18a5a5", "Unobserved pairs"), (1, "#2878c8", "Recorded associations")):
            selected = labels == label
            ax.scatter(xy[selected, 0], xy[selected, 1], s=4, alpha=.5, c=color, label=name, linewidths=0, rasterized=True)
        ax.set_title(f"Epoch {epoch}"); ax.set_xlabel("Joint UMAP 1"); ax.set_ylabel("Joint UMAP 2")
    axes[0, 0].legend(markerscale=2, fontsize=8)
    fig.suptitle("Fixed validation pairs: descriptive representation trajectory")
    fig.savefig(out / "validation_trajectory.png", dpi=300); fig.savefig(out / "validation_trajectory.pdf")
    plt.close(fig)
    result = {"status": "completed", "umap_version": umap.__version__, "epochs": SNAPSHOTS,
              "seed": seed, "n_validation_pairs_per_epoch": n,
              "representation": "32-dimensional penultimate ReLU of pair head; average forward/reverse endpoint order",
              "projection": "one joint unsupervised UMAP fit to concatenated four snapshots; shared axes; no per-panel fit; no feature rescaling",
              "umap_parameters": {"n_neighbors": 15, "min_dist": .1, "metric": "euclidean", "n_jobs": 1},
              "interpretation": "exploratory validation-set visualization only; early stopping disabled for trajectory; no test samples; no choice of model or epoch from cluster separation; not evidence of biological or unseen-protein generalization",
              "artifacts": {name: sha(out / name) for name in ("joint_umap_coordinates.npz", "validation_trajectory.png", "validation_trajectory.pdf")}}
    write_json(out / "umap_completed.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default=str(HERE / "prepared"))
    parser.add_argument("--output", default=str(HERE / "runs/supplement_20261001/cases"))
    parser.add_argument("--source-fasta", default=str(HERE.parent / "1427984.protein.sequences.v12.0.fa"))
    parser.add_argument("--source-links", default=str(HERE.parent / "1427984.protein.links.v12.0.txt"))
    parser.add_argument("--mode", choices=("audit", "trajectory", "targets", "all", "plot"), default="audit")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "scope.json", {"docking": "user_excluded", "tasks": ["validation trajectory", "target-withheld candidate ranking"],
                                  "no_structure_or_docking_services": True})
    base = load_fold(Path(args.prepared), "random", "pair", 0)
    mapping_result = None
    try:
        mapping_result = source_mapping(args, base)
        mapping, targets, positives, source_scores, evidence = mapping_result
        write_json(out / "source_mapping_verified.json", evidence)
        write_rows(out / "source_id_mapping.csv", mapping)
        for name, target in targets.items():
            _, audit = target_split(base, target["index"], positives)
            write_json(out / f"{name}_split_audit.json", audit)
    except (ValueError, FileNotFoundError) as exc:
        write_json(out / "target_mapping_blocked.json", {"status": "blocked", "reason": str(exc),
                    "policy": "No target training without exact sequence mapping and source edge-set reconciliation"})
        print(f"Target mapping unavailable: {exc}", flush=True)
    if args.mode == "audit":
        print(json.dumps({"target_mapping": mapping_result is not None, "umap_installed": importlib.util.find_spec("umap") is not None,
                          "base_split_hash": base["split_hash"], "GPU_training_started": False}), flush=True)
        return
    if args.mode in ("trajectory", "all"):
        train_case(args, base, out / "trajectory_fold0", trajectory=True)
        if args.mode == "trajectory":
            print("GPU_TRAINING_COMPLETE; starting CPU UMAP", flush=True)
            plot_trajectory(out / "trajectory_fold0", args.seed)
    if args.mode == "plot":
        plot_trajectory(out / "trajectory_fold0", args.seed)
    if args.mode in ("targets", "all") and mapping_result is not None:
        for name, target in targets.items():
            data, audit = target_split(base, target["index"], positives)
            folder = out / f"target_{name}"
            training = train_case(args, data, folder)
            write_json(folder / "exclusion_audit.json", audit)
            write_json(folder / "source_mapping_verified.json", evidence)
            rank_target(args, data, mapping, target, source_scores, folder, training)
    if args.mode == "all":
        print("GPU_TRAINING_COMPLETE; starting CPU UMAP", flush=True)
        plot_trajectory(out / "trajectory_fold0", args.seed)


if __name__ == "__main__":
    main()
