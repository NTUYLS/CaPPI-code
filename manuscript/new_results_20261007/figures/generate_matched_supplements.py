"""Move matched-capacity curves to supplements while retaining their full data."""
from pathlib import Path
import copy
import hashlib
import json
import shutil

OUT = Path(__file__).resolve().parent
manifest_path = OUT / "figures_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

# Reuse only the helper definitions from the local figure-generation source;
# do not execute its campaign-wide calls or redraw unrelated figures.
helper_path = OUT / "generate_figures.py"
helper_source = helper_path.read_text(encoding="utf-8").split('\nsensitivity_figure("fig02_', 1)[0]
helpers = {"__file__": str(helper_path), "__name__": "figure_helpers"}
exec(compile(helper_source, str(helper_path), "exec"), helpers)
ROOT = helpers["ROOT"]
sha = helpers["sha"]

old_main = next(item for item in manifest["figures"] if item["id"] == "fig08_fusion_controls_latest")
s1_id = "SupplementaryFig_S1_matched_fusion"
s2_id = "SupplementaryFig_S2_matched_protein_disjoint"
if any(series["model"] == "Matched MLP" for series in old_main["series"]):
    supplement_s1 = copy.deepcopy(old_main)
    supplement_s1["id"] = s1_id
    supplement_s1["purpose"] = "Full original capacity-matched comparison retained in supplement; no experimental result removed."
    supplement_s1["outputs"] = []
    for original in old_main["outputs"]:
        ext = Path(original["path"]).suffix
        path = OUT / (s1_id + ext)
        shutil.copyfile(original["path"], path)
        supplement_s1["outputs"].append({"path": str(path), "sha256": sha(path), "copy_of_original": original})
else:
    supplement_s1 = next(item for item in manifest["figures"] if item["id"] == s1_id)

manifest["figures"] = [item for item in manifest["figures"] if item["id"] not in {"fig08_fusion_controls_latest", s1_id, s2_id}]
manifest["figures"].append(supplement_s1)
helpers["manifest"] = manifest
spec, latest_records, curve_figure = [helpers[name] for name in ("spec", "latest_records", "curve_figure")]
BLACK, BLUE, PURPLE, GRAY = [helpers[name] for name in ("BLACK", "BLUE", "PURPLE", "GRAY")]
index_sources = [ROOT / "results/repair_20261007/run_metrics.csv", ROOT / "results/repair_20261007/verification.json"]

curve_figure("fig08_fusion_controls_latest", [
    spec("Full fusion", latest_records("pair", "random", "residual"), BLACK),
    spec("Sequence only", latest_records("pair", "random", "sequence_only"), BLUE, ":"),
    spec("Topology only", latest_records("pair", "random", "topology_only"), GRAY, "-."),
], "repair_20261007", "pair", "random", index_sources)
main = manifest["figures"][-1]
main["display_change"] = "Matched-capacity MLP curve omitted only from the main figure at author request. Its predictions, native metrics and full comparison remain in SupplementaryFig_S1_matched_fusion and manuscript evidence. This display change does not support a fusion-superiority claim."
main["complete_matched_comparison"] = s1_id

curve_figure(s2_id, [
    spec("Full fusion", latest_records("protein", "random", "residual"), BLACK),
    spec("Matched MLP", latest_records("protein", "random", "concat_matched"), PURPLE, "--"),
    spec("Sequence only", latest_records("protein", "random", "sequence_only"), BLUE, "--"),
    spec("Topology masking", latest_records("protein", "random", "topology_dropout"), PURPLE),
    spec("Masking + bypass", latest_records("protein", "random", "topology_dropout_bypass"), GRAY, ":"),
], "repair_20261007", "protein-ID disjoint: both endpoints unseen", "random", index_sources)
manifest["figures"][-1]["purpose"] = "Complete protein-disjoint comparison including capacity-matched MLP, retained in supplement."
main_protein = next(item for item in manifest["figures"] if item["id"] == "fig10_protein_disjoint_latest")
main_protein["display_change"] = "Main protein-disjoint figure shows full fusion, sequence only, topology masking and masking plus bypass. It already omitted matched MLP; existing main PNG/SVG/PDF are unchanged. Complete matched-MLP comparison is now SupplementaryFig_S2_matched_protein_disjoint."
main_protein["complete_matched_comparison"] = s2_id
manifest["supplement_display_policy"] = "Capacity-matched experiments are fully retained in supplementary figures and manifest. Author-requested omission from main curves is a presentation change, not deletion of experiments or evidence."
manifest["matched_supplement_generation_source"] = {"path": str(Path(__file__)), "sha256": sha(Path(__file__))}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Main modality figure updated; full matched comparisons retained in Supplementary Figures S1/S2. Protein main assets unchanged.")
