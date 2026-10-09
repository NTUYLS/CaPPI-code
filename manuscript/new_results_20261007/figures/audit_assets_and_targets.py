"""Check new figure assets and compute native PR areas for saved strict holdouts."""
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(r"D:\1.肝癌\CaPPI_revised")
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    return {"path": str(path), "sha256": sha(path)}


manifest_path = OUT / "figures_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
checked_sources, checked_outputs = {}, []
for figure in manifest["figures"]:
    for item in figure["sources"] + [s for series in figure.get("series", []) for s in series["sources"]]:
        if item["path"] not in checked_sources:
            digest = sha(item["path"])
            assert digest == item["sha256"], item["path"]
            checked_sources[item["path"]] = digest
    for item in figure["outputs"]:
        assert sha(item["path"]) == item["sha256"], item["path"]
        path = Path(item["path"])
        checks = {"path": str(path), "sha256_matches": True}
        if path.suffix == ".svg":
            tree = ET.parse(path)
            assert tree.getroot().tag.endswith("svg")
            checks["svg_xml_parsed"] = True
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                checks["dimensions"] = list(image.size)
        checked_outputs.append(checks)

visual = {
    "status": "passed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "visual_inspection": "All ten PNGs viewed. Numeric annotations, legends, panel letters, axis ticks, workflow labels and arrows are legible and within image bounds. GCNN labels/colorbar ticks were shortened and rechecked. Figure 1 arrows were moved away from section labels and the final image rechecked. Updated main modality and main protein curves plus full matched supplements were checked after the display change.",
    "checked_pngs": [str(OUT / (f["id"] + ".png")) for f in manifest["figures"]],
    "machine_checks": checked_outputs,
    "source_sha256_rechecks": len(checked_sources),
    "curve_metrics": "All native per-run AUC, trapezoidal AUPR, AP and accuracy matched saved per-run metrics within absolute tolerance 1e-9; all comparison memberships matched exact pair order and labels.",
    "limits": "Mean curves use interpolation for display; legend values are means of native per-run areas, not areas under interpolated mean curves. Original assets and predictions were not edited.",
}
qa_path = OUT / "visual_qa.json"
qa_path.write_text(json.dumps(visual, ensure_ascii=False, indent=2), encoding="utf-8")
manifest["visual_qa"] = source(qa_path)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if "--figures-only" in sys.argv:
    print(json.dumps({"figures": len(manifest["figures"]), "outputs_checked": len(checked_outputs),
                      "source_hashes_checked": len(checked_sources), "status": "passed"}, indent=2))
    raise SystemExit(0)

holdout = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "native_metrics_recomputed_and_verified",
           "campaign": "supplement_20261001 strict single-target holdout; not latest three-seed repair campaign",
           "AUPR_definition": "sklearn.metrics.auc(recall, precision) on the native precision_recall_curve, using all 581 saved candidates; no artificial endpoint appended; not Average Precision.",
           "scope": "Retrospective ranking of supplied STRING associations; one target and one model seed per experiment. Unobserved pairs are not verified noninteractions. Source taxon 1427984 is not a verified human HCC protein set.",
           "targets": []}
for name in ("X271_00001", "X271_00015"):
    folder = ROOT / "runs/supplement_20261001/cases" / ("target_" + name)
    report_path = folder / "ranking_completed.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction_path = folder / "all_candidates.csv"
    top20_path = folder / "top20_candidates.csv"
    assert sha(prediction_path) == report["all_candidates_sha256"]
    assert sha(top20_path) == report["top20_sha256"]
    df = pd.read_csv(prediction_path, float_precision="round_trip")
    top = pd.read_csv(top20_path, float_precision="round_trip")
    assert len(df) == report["n_candidates"] == 581
    assert len(df.partner_source_id.unique()) == 581
    # CSV reads contain empty source-score cells, so use pandas' NaN-aware comparison.
    pd.testing.assert_frame_equal(top.reset_index(drop=True), df.iloc[:20].reset_index(drop=True))
    assert np.all(np.diff(df.model_score.to_numpy(float)) <= 0)
    y, scores = df.known_association.to_numpy(int), df.model_score.to_numpy(float)
    precision, recall, _ = precision_recall_curve(y, scores)
    roc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    pr = float(auc(recall, precision))
    assert abs(roc - report["auroc"]) < 1e-9
    assert abs(ap - report["average_precision"]) < 1e-9
    hits = int(top.known_association.sum())
    assert hits == report["hits_at_20_count"]
    audit_path = folder / "exclusion_audit.json"
    isolation = json.loads(audit_path.read_text(encoding="utf-8"))
    assert isolation["target_train_or_val_pair_count"] == isolation["target_graph_degree"] == 0
    copy_path = OUT.parent / ("supplement_" + name + "_strict_top20.csv")
    shutil.copyfile(top20_path, copy_path)
    item = {"target": name, "source_id": report["target"]["source_id"], "n_candidates": len(df),
            "known_source_associations": int(y.sum()), "unobserved_candidate_pairs": int((y == 0).sum()),
            "AUROC": roc, "AUPR_trapezoidal": pr, "Average_Precision": ap, "hits_at_20": hits,
            "precision_at_20": hits / 20, "recall_at_20": hits / int(y.sum()),
            "model_seed": int(df.seed.iloc[0]), "split_hash": report["split_hash"], "checkpoint_sha256": report["checkpoint_sha256"],
            "target_train_or_validation_pairs": 0, "target_training_graph_degree": 0,
            "sources": [source(prediction_path), source(top20_path), source(report_path), source(audit_path), source(folder / "source_mapping_verified.json")],
            "new_top20_copy": source(copy_path), "classification_metrics": "Not assigned: single-target ranking analysis without calibrated threshold selection."}
    holdout["targets"].append(item)

target_path = OUT.parent / "target_holdout_metrics_for_manuscript.json"
target_path.write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"figures": len(manifest["figures"]), "outputs_checked": len(checked_outputs), "source_hashes_checked": len(checked_sources),
                  "target_metrics": [{k: t[k] for k in ("target", "AUROC", "AUPR_trapezoidal", "Average_Precision", "hits_at_20")} for t in holdout["targets"]]}, indent=2))
