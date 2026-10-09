"""Generate new manuscript figures from saved, verified predictions; never train."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, auc, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(r"D:\1.肝癌\CaPPI_revised")
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 8.1, "legend.frameon": False,
    "axes.linewidth": 0.75, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
    "ytick.major.size": 3, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42,
})
BLUE = "#557B9B"
PURPLE = "#807096"
GRAY = "#84888D"
BLACK = "#191919"
GRID = np.linspace(0, 1, 2001)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "generation": "Only new assets; no training, model selection, or edits to source predictions.",
    "root": str(ROOT),
    "metric_definitions": {
        "AUC": "ROC area computed from native per-run prediction thresholds using sklearn roc_auc_score.",
        "AUPR": "Trapezoidal area of the native sklearn precision_recall_curve; sklearn.metrics.auc(recall, precision). Not Average Precision.",
        "Average Precision": "Computed separately only as a verification diagnostic; never relabeled AUPR.",
        "ACC": "Accuracy at the saved fixed classification threshold 0.5.",
    },
    "curve_definition": "Each native ROC/PR curve is linearly interpolated to a fixed 2001-point x-axis grid and curves are averaged. Latest: average five fold curves within each model-seed repeat, then average three repeats. Earlier: average five folds. Legend values average native per-run areas with the same hierarchy; they are not the area under an interpolated mean curve.",
    "precision_recall_endpoints": "No (recall=1, precision=0) endpoint is appended. Native sklearn endpoints are retained through interpolation. PR x-axis is recall and y-axis is precision.",
    "figures": [],
    "unavailable": [{"figure": "new UMAP epoch trajectory", "reason": "No corresponding 10/07 trajectory checkpoints/features were specified or verified. Earlier 600-epoch UMAP belongs to a different fixed-config campaign and is not reused."}],
    "scope_limits": [
        "Latest 10/07 experiments are adaptive same-benchmark comparisons, without a new independent blind or external test set.",
        "Earlier fixed-configuration controls are kept together and are not plotted against latest validation-selected models.",
        "Outer folds share training data. SD is descriptive, not a confidence interval; no significance p-values are computed.",
        "Protein-ID disjointness does not establish sequence-homology-independent generalization.",
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_record(path, **extra):
    return {"path": str(path), "sha256": sha(path), **extra}


def close_metrics(actual, expected, context):
    for key, value in expected.items():
        if value is not None and abs(actual[key] - float(value)) > 1e-9:
            raise ValueError(f"Metric mismatch {context}: {key} {actual[key]} != {value}")


def load_prediction(path, fold, repeat=0, expected_hash=None, expected_metrics=None):
    digest = sha(path)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Prediction SHA256 mismatch: {path}")
    df = pd.read_csv(path, float_precision="round_trip")
    y, p = df["label"].to_numpy(int), df["probability"].to_numpy(float)
    if not np.isfinite(p).all() or not set(np.unique(y)) == {0, 1}:
        raise ValueError(f"Nonfinite scores or nonbinary labels: {path}")
    if len(df[["protein_u", "protein_v"]].drop_duplicates()) != len(df):
        raise ValueError(f"Duplicate test pairs: {path}")
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    metrics = {"AUC": float(roc_auc_score(y, p)), "AUPR": float(auc(recall, precision)),
               "AP": float(average_precision_score(y, p)), "ACC": float(accuracy_score(y, p >= 0.5))}
    close_metrics(metrics, expected_metrics or {}, str(path))
    return {
        "fold": int(fold), "repeat": int(repeat), "roc": np.interp(GRID, fpr, tpr),
        "pr": np.interp(GRID, recall[::-1], precision[::-1]),
        "metrics": metrics, "members": df[["protein_u", "protein_v", "label"]].to_numpy(),
        "source": {"path": str(path), "sha256": digest, "n": len(y), "positive_n": int(y.sum()),
                   "fold": int(fold), "repeat": int(repeat), "recomputed_native_metrics": metrics},
    }


def aggregate(records):
    groups = sorted(set(r["repeat"] for r in records))
    per_repeat = []
    roc, pr = [], []
    for repeat in groups:
        subset = [r for r in records if r["repeat"] == repeat]
        if sorted(r["fold"] for r in subset) != list(range(5)):
            raise ValueError("Five folds are required for every repeat")
        means = {m: float(np.mean([r["metrics"][m] for r in subset])) for m in ("AUC", "AUPR", "AP", "ACC")}
        fold_sd = {m: float(np.std([r["metrics"][m] for r in subset], ddof=1)) for m in means}
        per_repeat.append({"repeat": repeat, "means": means, "sample_fold_sd": fold_sd})
        roc.append(np.mean([r["roc"] for r in subset], axis=0))
        pr.append(np.mean([r["pr"] for r in subset], axis=0))
    mean = {m: float(np.mean([r["means"][m] for r in per_repeat])) for m in per_repeat[0]["means"]}
    if len(groups) == 1:
        sd = per_repeat[0]["sample_fold_sd"]
        sd_definition = "sample SD across five dependent folds"
    else:
        sd = {m: float(np.std([r["means"][m] for r in per_repeat], ddof=1)) for m in mean}
        sd_definition = "sample SD across three model-repeat means, each averaging five folds"
    return {"roc": np.mean(roc, axis=0), "pr": np.mean(pr, axis=0), "mean": mean, "sd": sd,
            "per_repeat": per_repeat, "sd_definition": sd_definition, "n_runs": len(records),
            "sources": [r["source"] for r in records]}


def latest_records(protocol, negative, family):
    index = pd.read_csv(ROOT / "results/repair_20261007/run_metrics.csv", float_precision="round_trip")
    selected = index[(index.protocol == protocol) & (index.negative == negative) & (index.family == family)]
    if len(selected) != 15 or set(selected.verification_status) != {"passed"}:
        raise ValueError(f"Missing/unverified latest runs: {protocol}/{negative}/{family}")
    result = []
    for _, row in selected.sort_values(["repeat", "fold"]).iterrows():
        path = Path(row.run_dir) / "test_predictions.csv"
        result.append(load_prediction(path, row["fold"], row["repeat"], row.test_prediction_sha256,
                                     {"AUC": row.test_auroc, "AUPR": row.test_aupr, "AP": row.test_average_precision, "ACC": row.test_accuracy}))
    return result


def earlier_records(kind):
    if kind in {"residual", "deepwalk", "node2vec"}:
        walk = "drw" if kind == "residual" else kind
        prefix = ROOT / "runs/reviewer_v1"
        name = f"pair__random__{walk}__residual"
        neural = True
    else:
        prefix = ROOT / ("runs/classical_controls_v1" if kind in {"lr", "rf"} else "runs/supplement_20261001/classical")
        name = f"pair__random__aa20_drw_{kind}"
        neural = False
    result = []
    for fold in range(5):
        folder = prefix / f"{name}__fold{fold}"
        complete = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        native = complete["test_metrics"]
        digest = complete["predictions_sha256"] if neural else complete["artifacts_sha256"]["test_predictions.csv"]
        result.append(load_prediction(folder / "test_predictions.csv", fold, 0, digest,
                                     {"AUC": native["auroc"], "AUPR": native["pr_trapezoid_area"], "AP": native["average_precision"], "ACC": native["accuracy"]}))
    return result


def ensure_membership(all_records):
    anchor = {(r["repeat"], r["fold"]): r["members"] for r in all_records[0]}
    for records in all_records[1:]:
        for r in records:
            if not np.array_equal(r["members"], anchor[(r["repeat"], r["fold"])]):
                raise ValueError("Different pair order/labels within a comparison")


def export(fig, stem):
    outputs = []
    for extension in ("png", "svg", "pdf"):
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=350, bbox_inches="tight", pad_inches=0.07)
        outputs.append(source_record(path))
    plt.close(fig)
    return outputs


def curve_figure(stem, series, campaign, protocol, negative, index_sources):
    ensure_membership([s["records"] for s in series])
    fig, axes = plt.subplots(1, 2, figsize=(8.15, 3.63), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.03, wspace=0.10)
    plotted = []
    for spec in series:
        stat = aggregate(spec["records"])
        for idx, key, metric in [(0, "roc", "AUC"), (1, "pr", "AUPR")]:
            axes[idx].plot(GRID, stat[key], color=spec["color"], linestyle=spec["linestyle"],
                           linewidth=1.65 if spec["color"] == BLACK else 1.25,
                           label=f'{spec["label"]} ({metric} = {stat["mean"][metric]:.4f})')
        plotted.append({"model": spec["label"], "legend_AUC": stat["mean"]["AUC"], "legend_AUPR": stat["mean"]["AUPR"],
                        "sd": stat["sd"], "sd_definition": stat["sd_definition"], "per_repeat": stat["per_repeat"],
                        "number_of_runs": stat["n_runs"], "sources": stat["sources"],
                        "interpolated_mean_curve_area": {"ROC": float(auc(GRID, stat["roc"])), "PR": float(auc(GRID, stat["pr"]))}})
    axes[0].plot([0, 1], [0, 1], color="#C4C7CA", linewidth=0.75, linestyle="--", zorder=0)
    for idx, ax in enumerate(axes):
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.2))
        ax.grid(color="#E4E6E8", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xlabel("False positive rate" if idx == 0 else "Recall")
        ax.set_ylabel("True positive rate" if idx == 0 else "Precision")
        ax.set_title("(a)" if idx == 0 else "(b)", loc="left", fontweight="bold", pad=6)
        ax.legend(loc="lower right" if idx == 0 else "lower left", handlelength=2.7, labelspacing=0.43, borderaxespad=0.5)
    outputs = export(fig, stem)
    entry = {"id": stem, "kind": "ROC/PR", "campaign": campaign, "protocol": protocol, "negative_pool": negative,
             "data_partition": "outer test", "sources": [source_record(p) for p in index_sources], "series": plotted,
             "outputs": outputs, "membership_check": "exact pair order and labels match between all plotted models within each fold/repeat"}
    manifest["figures"].append(entry)
    print(stem, json.dumps({s["model"]: {"AUC": s["legend_AUC"], "AUPR": s["legend_AUPR"]} for s in plotted}))


def sensitivity_figure(stem, group, xkey, ykey, xlabel, ylabel):
    path = ROOT / "runs/supplement_20261001/parameter_sensitivity/validation_summary.csv"
    df = pd.read_csv(path, float_precision="round_trip")
    df = df[df.groups.str.split(";").apply(lambda groups: group in groups)].copy()
    if set(df.evaluation_partition) != {"validation"} or set(df.completed_folds) != {5}:
        raise ValueError("Sensitivity data must be validation-only with five completed folds")
    xs, ys = sorted(df[xkey].unique()), sorted(df[ykey].unique())
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.12), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("muted_blue", ["#F4F7FA", "#BDCEDD", BLUE])
    values, labels = {}, ["ACC", "AUC", "AUPR"]
    for idx, (metric, col) in enumerate(zip(labels, ["accuracy_mean", "auroc_mean", "pr_trapezoid_area_mean"])):
        matrix = df.pivot(index=ykey, columns=xkey, values=col).loc[ys, xs].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("Incomplete sensitivity grid")
        values[metric] = matrix.tolist()
        # Each panel has a clearly shown local color scale; labels preserve absolute values.
        lo, hi = float(matrix.min()), float(matrix.max())
        image = axes[idx].imshow(matrix, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        axes[idx].set_xticks(range(len(xs)), [str(int(x)) for x in xs])
        axes[idx].set_yticks(range(len(ys)), [str(int(y)) for y in ys])
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(ylabel if idx == 0 else "")
        axes[idx].set_title(f"({chr(97 + idx)}) {metric}", loc="left", fontweight="bold", pad=7)
        axes[idx].tick_params(length=0)
        for row in range(len(ys)):
            for colidx in range(len(xs)):
                value = matrix[row, colidx]
                color = "white" if (value - lo) / max(hi - lo, 1e-12) > 0.76 else BLACK
                axes[idx].text(colidx, row, f"{value:.4f}", ha="center", va="center", fontsize=8.2, color=color)
        cb = fig.colorbar(image, ax=axes[idx], orientation="horizontal", pad=0.12, fraction=0.065, aspect=24)
        cb.locator = MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=7.5, length=2)
        cb.outline.set_linewidth(0.4)
    outputs = export(fig, stem)
    manifest["figures"].append({"id": stem, "kind": "validation-only sensitivity heatmap", "campaign": "supplement_20261001 fixed-config model sensitivity",
                                "data_partition": "validation; same validation data also select checkpoints, descriptive", "n_folds": 5,
                                "sources": [source_record(path), source_record(path.parent / "verification.json")],
                                "x_variable": xkey, "x_values": [int(x) for x in xs], "y_variable": ykey, "y_values": [int(y) for y in ys],
                                "panel_values": values, "color_scale": "Each panel has a labeled local scale; numerical annotations show absolute values.",
                                "metric_definition": "AUPR uses pr_trapezoid_area_mean; AP is not used for AUPR.",
                                "outputs": outputs})
    print(stem, "validation heatmap complete")


def spec(label, records, color, linestyle="-"):
    return {"label": label, "records": records, "color": color, "linestyle": linestyle}


sensitivity_figure("fig02_drw_validation_sensitivity", "walk_dimension", "walk_length", "dim", "Walk length", "Embedding dimension")
sensitivity_figure("fig03_gcnn_validation_sensitivity", "kernel_length", "max_length", "kernel", "Sequence length (max.)", "Kernel size")

latest_index = [ROOT / "results/repair_20261007/run_metrics.csv", ROOT / "results/repair_20261007/verification.json"]
earlier_index = [ROOT / "results/verified_summary.csv", ROOT / "results/verification.json"]

from generate_figure4_folds import build_figure4
manifest["figures"].append(build_figure4(globals()))

curve_figure("fig06_classical_earlier", [
    spec("Full fusion", earlier_records("residual"), BLACK),
    spec("AdaBoost", earlier_records("adaboost"), GRAY, ":"),
    spec("KNN", earlier_records("knn"), GRAY, "--"),
    spec("LR", earlier_records("lr"), BLUE, ":"),
    spec("RF", earlier_records("rf"), BLUE, "--"),
    spec("SVM", earlier_records("svm"), PURPLE, "--"),
    spec("XGBoost", earlier_records("xgboost"), PURPLE),
], "earlier fixed-config reviewer_v1 + matched classical_controls_v1/supplement_20261001", "pair", "random",
             earlier_index + [ROOT / "results/classical_verification.json", ROOT / "runs/supplement_20261001/classical/verification.json"])

curve_figure("fig07_walk_controls_earlier", [
    spec("DRW", earlier_records("residual"), BLACK),
    spec("Uniform walks", earlier_records("deepwalk"), BLUE, "--"),
    spec("node2vec", earlier_records("node2vec"), PURPLE, ":"),
], "earlier fixed-config reviewer_v1; shared SGNS, uniform/DeepWalk-style not original DeepWalk reproduction", "pair", "random", earlier_index)

curve_figure("fig08_fusion_controls_latest", [
    spec("Full fusion", latest_records("pair", "random", "residual"), BLACK),
    spec("Matched MLP", latest_records("pair", "random", "concat_matched"), PURPLE, "--"),
    spec("Sequence only", latest_records("pair", "random", "sequence_only"), BLUE, ":"),
    spec("Topology only", latest_records("pair", "random", "topology_only"), GRAY, "-."),
], "repair_20261007", "pair", "random", latest_index)

curve_figure("fig10_protein_disjoint_latest", [
    spec("Full fusion", latest_records("protein", "random", "residual"), BLACK),
    spec("Sequence only", latest_records("protein", "random", "sequence_only"), BLUE, "--"),
    spec("Topology masking", latest_records("protein", "random", "topology_dropout"), PURPLE),
    spec("Masking + bypass", latest_records("protein", "random", "topology_dropout_bypass"), GRAY, ":"),
], "repair_20261007", "protein-ID disjoint: both endpoints unseen", "random", latest_index)

manifest["script"] = source_record(Path(__file__))
(OUT / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote figures_manifest.json: 7 figures, PNG/SVG/PDF, all metrics recomputed/checked; no training.")
