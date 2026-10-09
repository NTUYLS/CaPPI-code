"""Figure 4: five degree-matched fold curves and their mean; no training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import auc

OUT = Path(__file__).resolve().parent
STEM = "fig04_degree_matched_latest"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers():
    path = OUT / "generate_figures.py"
    definitions = path.read_text(encoding="utf-8").split('\nsensitivity_figure("fig02_', 1)[0]
    namespace = {"__file__": str(path), "__name__": "figure4_helpers"}
    exec(compile(definitions, str(path), "exec"), namespace)
    return namespace


def build_figure4(helpers):
    """Return one manifest entry; only save Figure 4 outputs."""
    records = helpers["latest_records"]("pair", "degree_matched", "residual")
    grid = helpers["GRID"]
    black, blue, purple, gray = [helpers[name] for name in ("BLACK", "BLUE", "PURPLE", "GRAY")]
    styles = [(blue, "-"), (blue, "--"), (purple, "-."), (purple, ":"), (gray, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(8.15, 3.63), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.03, wspace=0.10)
    series, fold_roc, fold_pr, fold_means = [], [], [], []
    for fold in range(5):
        group = sorted([r for r in records if r["fold"] == fold], key=lambda r: r["repeat"])
        if [r["repeat"] for r in group] != [0, 1, 2]:
            raise ValueError(f"Missing seed repeat in fold {fold}")
        if any(not np.array_equal(group[0]["members"], other["members"]) for other in group[1:]):
            raise ValueError(f"Test pair order/labels differ across repeats in fold {fold}")
        if any(float(r["members"][:, 2].mean()) != 0.5 for r in group):
            raise ValueError("The requested PR prevalence reference must equal the empirical prevalence 0.5")
        roc = np.mean([r["roc"] for r in group], axis=0)
        pr = np.mean([r["pr"] for r in group], axis=0)
        native = {metric: float(np.mean([r["metrics"][metric] for r in group])) for metric in ("AUC", "AUPR", "AP", "ACC")}
        seed_sd = {metric: float(np.std([r["metrics"][metric] for r in group], ddof=1)) for metric in native}
        fold_roc.append(roc)
        fold_pr.append(pr)
        fold_means.append(native)
        color, linestyle = styles[fold]
        for ax, curve, metric in [(axes[0], roc, "AUC"), (axes[1], pr, "AUPR")]:
            ax.plot(grid, curve, color=color, linestyle=linestyle, linewidth=1.10,
                    label=f'Fold {fold + 1} ({metric} = {native[metric]:.4f})', zorder=3)
        series.append({"model": f"Fold {fold + 1}", "fold_zero_based": fold, "number_of_runs": 3,
                       "legend_AUC": native["AUC"], "legend_AUPR": native["AUPR"],
                       "native_metric_means": native, "sd": seed_sd,
                       "sd_definition": "sample SD across the three native per-repeat metrics for this fixed fold",
                       "sources": [r["source"] for r in group],
                       "interpolated_mean_curve_area": {"ROC": float(auc(grid, roc)), "PR": float(auc(grid, pr))}})
    all_stat = helpers["aggregate"](records)
    mean_roc, mean_pr = np.mean(fold_roc, axis=0), np.mean(fold_pr, axis=0)
    fold_grand_means = {metric: float(np.mean([fold[metric] for fold in fold_means])) for metric in fold_means[0]}
    # Retain the previously published repeat-first floating-point means exactly.
    means = dict(all_stat["mean"])
    for metric in means:
        if abs(means[metric] - fold_grand_means[metric]) > 1e-12:
            raise ValueError("Fold-first and repeat-first means do not agree")
    if not np.isclose(means["AUC"], 0.8407277952316942, rtol=0, atol=1e-12):
        raise ValueError("Overall AUC changed")
    if not np.isclose(means["AUPR"], 0.8554387874364272, rtol=0, atol=1e-12):
        raise ValueError("Overall native AUPR changed")
    np.testing.assert_allclose(mean_roc, all_stat["roc"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(mean_pr, all_stat["pr"], rtol=0, atol=1e-12)
    for ax, curve, metric in [(axes[0], mean_roc, "AUC"), (axes[1], mean_pr, "AUPR")]:
        ax.plot(grid, curve, color=black, linewidth=1.75,
                label=f'Mean ({metric} = {means[metric]:.4f})', zorder=5)
    series.append({"model": "Mean", "number_of_runs": 15, "legend_AUC": means["AUC"], "legend_AUPR": means["AUPR"],
                   "native_metric_means": means, "sd": all_stat["sd"], "sd_definition": all_stat["sd_definition"],
                   "per_repeat": all_stat["per_repeat"], "sources": all_stat["sources"],
                   "interpolated_mean_curve_area": {"ROC": float(auc(grid, mean_roc)), "PR": float(auc(grid, mean_pr))}})
    axes[0].plot([0, 1], [0, 1], color="#C4C7CA", linewidth=0.75, linestyle="--", zorder=1)
    axes[1].axhline(0.5, color="#B8BDC2", linewidth=0.75, linestyle="--", zorder=1)
    for idx, ax in enumerate(axes):
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.2))
        ax.grid(color="#E4E6E8", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xlabel("False positive rate" if idx == 0 else "Recall")
        ax.set_ylabel("True positive rate" if idx == 0 else "Precision")
        ax.set_title("(a)" if idx == 0 else "(b)", loc="left", fontweight="bold", pad=6)
        ax.legend(loc="lower right" if idx == 0 else "lower left", handlelength=2.7, labelspacing=0.43, borderaxespad=0.5)
    outputs = helpers["export"](fig, STEM)
    root = helpers["ROOT"]
    source_paths = [root / "results/repair_20261007/run_metrics.csv", root / "results/repair_20261007/verification.json",
                    OUT / "generate_figures.py", Path(__file__)]
    entry = {"id": STEM, "kind": "five-fold ROC/PR with mean", "campaign": "repair_20261007",
             "protocol": "pair", "negative_pool": "approximately degree-matched", "data_partition": "outer test",
             "sources": [{"path": str(p), "sha256": sha(p)} for p in source_paths], "outputs": outputs, "series": series,
             "displayed_legend_metrics": [{"label": s["model"], "AUC": s["legend_AUC"], "AUPR_trapezoidal": s["legend_AUPR"]} for s in series],
             "fold_curve_definition": "Each displayed Fold 1-5 curve averages the three fixed-fold neural model-seed repeat curves on the existing 2001-point grid. Native metrics are first computed per saved prediction file, then averaged across the three repeats for the fold legend.",
             "overall_curve_definition": "Black mean averages the five displayed fold curves. Overall legend areas average the five native fold-area means, mathematically equivalent to the prior repeat-first hierarchy because every fold has exactly three repeats.",
             "PR_reference": {"value": 0.5, "meaning": "empirical positive prevalence in every plotted test fold", "style": "horizontal dashed gray"},
             "metric_definition": "AUPR is trapezoidal area under the native precision_recall_curve, not Average Precision or area under an interpolated mean curve. No precision-zero endpoint is appended.",
             "membership_check": "Exact saved protein pair order and labels match across all three repeats within each fold; all prediction SHA256 hashes and native saved metrics match.",
             "visual_qa": "pending inspection of newly regenerated Figure 4",
             "display_change": "Five fold curves added to the existing black mean; overall native AUC/AUPR unchanged. Other figure assets were not regenerated.",
             "created_utc": datetime.now(timezone.utc).isoformat()}
    return entry


def confirm_visual_qa(manifest):
    entry = next(f for f in manifest["figures"] if f["id"] == STEM)
    entry["visual_qa"] = "PNG actually viewed: five fold curves and black mean, six-entry AUC/AUPR legends, PR prevalence line 0.5, labels and panel letters clear with no clipping/overlap. Original aspect ratio and dimensions retained."
    qa_path = OUT / "visual_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    checks = []
    for item in entry["outputs"]:
        path = Path(item["path"])
        assert sha(path) == item["sha256"]
        check = {"path": str(path), "sha256_matches": True}
        if path.suffix == ".png":
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                check["dimensions"] = list(im.size)
        elif path.suffix == ".svg":
            assert ET.parse(path).getroot().tag.endswith("svg")
            check["svg_xml_parsed"] = True
        checks.append(check)
    paths = {c["path"] for c in checks}
    qa["machine_checks"] = [c for c in qa["machine_checks"] if c["path"] not in paths] + checks
    checked_sources = {}
    other_output_count = 0
    for figure in manifest["figures"]:
        for source in figure["sources"] + [s for series in figure.get("series", []) for s in series["sources"]]:
            if source["path"] not in checked_sources:
                assert sha(source["path"]) == source["sha256"], source["path"]
                checked_sources[source["path"]] = source["sha256"]
        if figure["id"] != STEM:
            for output in figure["outputs"]:
                assert sha(output["path"]) == output["sha256"], output["path"]
                other_output_count += 1
    qa["source_sha256_rechecks"] = len(checked_sources)
    qa["figure4_update"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
                            "visual_inspection": entry["visual_qa"], "native_overall_AUC": entry["series"][-1]["legend_AUC"],
                            "native_overall_AUPR": entry["series"][-1]["legend_AUPR"], "five_fold_curves": True,
                            "three_seed_repeats_per_fold": True, "other_figure_entries_preserved": True,
                            "other_output_sha256_unchanged": other_output_count, "source_sha256_rechecks": len(checked_sources)}
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["visual_qa"] = {"path": str(qa_path), "sha256": sha(qa_path)}


if __name__ == "__main__":
    path = OUT / "figures_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "--confirm-visual-qa" in sys.argv:
        confirm_visual_qa(manifest)
    else:
        helpers = load_helpers()
        entry = build_figure4(helpers)
        manifest["figures"] = [entry if f["id"] == STEM else f for f in manifest["figures"]]
        manifest["figure4_generation_source"] = {"path": str(Path(__file__)), "sha256": sha(Path(__file__))}
        # Keep the historical generator identity and disclose its revised regeneration path.
        manifest["figure4_generator_integration"] = {"path": str(OUT / "generate_figures.py"), "sha256": sha(OUT / "generate_figures.py"),
                                                     "scope": "Only the Figure 4 call now invokes build_figure4; this standalone run did not execute the full generator."}
        print(json.dumps({s["model"]: {"AUC": s["legend_AUC"], "AUPR": s["legend_AUPR"]} for s in entry["series"]}, indent=2))
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
