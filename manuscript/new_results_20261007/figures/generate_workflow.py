"""Corrected manuscript workflow; only new Figure 1 assets and manifest entry."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

ROOT = Path(r"D:\1.肝癌\CaPPI_revised")
OUT = Path(__file__).resolve().parent
BLUE = "#557B9B"
PURPLE = "#807096"
GRAY = "#84888D"
BLACK = "#202020"
BLUE_FILL = "#EEF3F7"
PURPLE_FILL = "#F1EEF5"
GRAY_FILL = "#F4F5F6"
plt.rcParams.update({"font.family": "Arial", "font.weight": "bold", "font.size": 8.5,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(6.5, 7.35))
fig.subplots_adjust(0.012, 0.012, 0.988, 0.988)
ax.set(xlim=(0, 10), ylim=(0, 14.1))
ax.axis("off")


def box(x, y, w, h, text, color=PURPLE, fill=PURPLE_FILL, icon=None, fs=8.4, dashed=False, lw=1.5):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.07",
                           linewidth=lw, edgecolor=color, facecolor=fill, linestyle="--" if dashed else "-", zorder=3)
    ax.add_patch(patch)
    if icon:
        draw_icon(icon, x + 0.14, y + 0.16, 0.45, color)
        tx = x + 0.66 + (w - 0.77) / 2
    else:
        tx = x + w / 2
    ax.text(tx, y + h / 2, text, ha="center", va="center", fontsize=fs, color=BLACK,
            fontweight="bold", linespacing=1.12, zorder=4)
    return patch


def region(y, h, title):
    box(0.30, y, 9.40, h, "", GRAY, "white", lw=1.75).set_zorder(0.5)
    ax.text(0.54, y + h - 0.26, title, ha="left", va="center", fontsize=10.0, color=BLACK, zorder=4)


def arrow(points, color=BLACK, dashed=False):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.5,
                linestyle="--" if dashed else "-", solid_capstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=9,
                                linewidth=1.5, color=color, linestyle="--" if dashed else "-", zorder=2,
                                shrinkA=0, shrinkB=1.5, connectionstyle="arc3,rad=0"))


def draw_icon(kind, x, y, size, color):
    if kind == "graph":
        points = np.array([[0.05, 0.45], [0.28, 0.78], [0.73, 0.78], [0.90, 0.33], [0.50, 0.05]]) * size + [x, y]
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4), (2, 4)]:
            ax.plot(points[[i, j], 0], points[[i, j], 1], color=color, lw=1.25, zorder=4)
        for px, py in points:
            ax.add_patch(Circle((px, py), size * 0.08, facecolor="white", edgecolor=color, lw=1.25, zorder=5))
    elif kind == "sequence":
        for i in range(4):
            ax.add_patch(Rectangle((x + i * size * 0.22, y + size * (0.26 if i % 2 else 0.43)),
                                   size * 0.17, size * 0.24, facecolor="white", edgecolor=color, lw=1.25, zorder=4))
    elif kind == "conv":
        for i, height in enumerate([0.45, 0.78, 0.58]):
            ax.add_patch(Rectangle((x + size * i * 0.30, y + size * (1 - height) / 2), size * 0.22,
                                   size * height, facecolor="white", edgecolor=color, lw=1.25, zorder=4))
    elif kind == "project":
        for i, ny in enumerate([2, 3]):
            pts = [(x + size * (0.10 + i * 0.70), y + size * (j + 1) / (ny + 1)) for j in range(ny)]
            for px, py in pts:
                ax.add_patch(Circle((px, py), size * 0.06, facecolor="white", edgecolor=color, lw=1.25, zorder=5))
        for j in range(2):
            for k in range(3):
                ax.plot([x + size * 0.10, x + size * 0.80], [y + size * (j + 1) / 3, y + size * (k + 1) / 4],
                        color=color, lw=0.85, zorder=4)
    elif kind == "fusion":
        for yy in [0.23, 0.72]:
            ax.add_patch(Rectangle((x + size * 0.03, y + size * yy), size * 0.27, size * 0.19,
                                   edgecolor=color, facecolor="white", lw=1.25, zorder=4))
        ax.add_patch(Rectangle((x + size * 0.68, y + size * 0.45), size * 0.25, size * 0.20,
                               edgecolor=color, facecolor="white", lw=1.25, zorder=4))
        ax.plot([x + size * 0.30, x + size * 0.52, x + size * 0.68],
                [y + size * 0.82, y + size * 0.55, y + size * 0.55], color=color, lw=1.25, zorder=4)
        ax.plot([x + size * 0.30, x + size * 0.52], [y + size * 0.33, y + size * 0.55], color=color, lw=1.25, zorder=4)
    elif kind == "pair":
        for yy in [0.27, 0.75]:
            ax.add_patch(Circle((x + size * 0.15, y + size * yy), size * 0.08, edgecolor=color, facecolor="white", lw=1.25, zorder=4))
        ax.plot([x + size * 0.26, x + size * 0.77], [y + size * 0.70, y + size * 0.40], color=color, lw=1.25, zorder=4)
        ax.plot([x + size * 0.26, x + size * 0.77], [y + size * 0.30, y + size * 0.60], color=color, lw=1.25, zorder=4)


# Outermost border 2; group borders 1.75; modules 1.5; icon borders 1.25.
box(0.10, 0.10, 9.80, 13.90, "", GRAY, "white", lw=2.0).set_zorder(0)
region(10.80, 3.00, "1  Dataset and fixed partitions")
region(3.90, 6.55, "2  Joint supervised training")
region(0.45, 3.10, "3  Validation selection and final evaluation")

box(0.70, 12.55, 8.60, 0.70, "STRING associations and protein sequences\n582 proteins; 23,707 supplied associations", BLUE, BLUE_FILL, fs=8.8)
box(3.20, 11.72, 3.60, 0.60, "Fixed outer folds\nPair / protein-disjoint", BLUE, BLUE_FILL, fs=8.6)
box(0.75, 10.98, 2.45, 0.53, "Training pairs", BLUE, BLUE_FILL)
box(3.78, 10.98, 2.45, 0.53, "Validation pairs", GRAY, GRAY_FILL)
box(6.80, 10.98, 2.45, 0.53, "Test pairs", GRAY, GRAY_FILL)
arrow([(5.0, 12.55), (5.0, 12.34)])
arrow([(5.0, 11.72), (5.0, 11.63), (1.975, 11.63), (1.975, 11.52)])
arrow([(5.0, 11.72), (5.0, 11.52)])
arrow([(5.0, 11.72), (5.0, 11.63), (8.025, 11.63), (8.025, 11.52)])

box(0.75, 9.05, 2.60, 0.78, "Train-positive\ngraph only", BLUE, BLUE_FILL, icon="graph")
box(6.65, 9.05, 2.60, 0.78, "21-channel\nsequence encoding", BLUE, BLUE_FILL, icon="sequence")
box(0.75, 7.83, 2.60, 0.82, "DRW + SGNS\nFixed 64D vectors", GRAY, GRAY_FILL, icon="graph")
box(6.65, 7.83, 2.60, 0.82, "GCNN + masked\npooling → 256D", PURPLE, PURPLE_FILL, icon="conv")
box(0.75, 6.61, 2.60, 0.82, "Masked topology\nprojection 64D → 256D", PURPLE, PURPLE_FILL, icon="project", fs=7.8)
box(3.80, 6.93, 2.40, 0.79, "Optional arm only\nTopology masking p = 0.5", GRAY, GRAY_FILL, fs=7.9, dashed=True)
box(3.35, 5.30, 3.30, 0.86, "Bidirectional residual\nfusion → 256D", PURPLE, PURPLE_FILL, icon="fusion", fs=8.4)
box(3.35, 4.17, 3.30, 0.82, "Pair MLP: both orders\nMean logits → sigmoid", PURPLE, PURPLE_FILL, icon="pair", fs=8.0)
box(0.75, 4.17, 2.10, 0.82, "Training BCE\nJoint updates", PURPLE, PURPLE_FILL, fs=8.4)

arrow([(0.75, 11.245), (0.48, 11.245), (0.48, 9.44), (0.73, 9.44)])
arrow([(9.30, 12.90), (9.51, 12.90), (9.51, 9.44), (9.27, 9.44)])
arrow([(2.05, 9.05), (2.05, 8.67)])
arrow([(7.95, 9.05), (7.95, 8.67)])
arrow([(2.05, 7.83), (2.05, 7.45)])
arrow([(3.80, 7.32), (3.58, 7.32), (3.58, 7.02), (3.37, 7.02)], PURPLE, dashed=True)
arrow([(2.05, 6.61), (2.05, 5.73), (3.33, 5.73)])
arrow([(7.95, 7.83), (7.95, 5.73), (6.67, 5.73)])
arrow([(5.00, 5.30), (5.00, 5.01)])
arrow([(3.35, 4.58), (2.87, 4.58)])

box(0.75, 1.70, 3.00, 1.08, "Held-out validation\nBCE selects LR\nand checkpoint", GRAY, GRAY_FILL, fs=8.3)
box(4.20, 1.70, 2.20, 1.08, "Freeze selection\nThreshold 0.5", PURPLE, PURPLE_FILL, fs=8.3)
box(6.85, 1.70, 2.40, 1.08, "Held-out test\nAUC / AUPR / MCC", GRAY, GRAY_FILL, fs=8.3)
arrow([(5.00, 4.17), (5.00, 3.71), (0.48, 3.71), (0.48, 2.24), (0.73, 2.24)])
arrow([(3.75, 2.24), (4.18, 2.24)])
arrow([(6.40, 2.24), (6.83, 2.24)])
ax.text(5.0, 0.98, "Validation/test associations never enter the training graph", ha="center", va="center",
        fontsize=8.0, color=BLACK, zorder=4)

outputs = []
for ext in ("png", "svg", "pdf"):
    path = OUT / ("fig01_corrected_workflow." + ext)
    fig.savefig(path, dpi=350, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    outputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
plt.close(fig)
sources = [ROOT / "runs/repair_20261007/campaign_contract.json", ROOT / "runs/repair_20261007/implementation/cappi_model.py",
           ROOT / "cappi_model.py", ROOT / "cappi_missing_topology.py", ROOT / "prepared/manifest.json",
           ROOT / "results/repair_20261007/verification.json"]
source_records = []
for path in sources:
    if path.exists():
        source_records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
entry = {"id": "fig01_corrected_workflow", "kind": "corrected training and evaluation workflow",
         "campaign": "repair_20261007 with earlier fixed-config DRW controls described separately in manuscript",
         "sources": source_records, "outputs": outputs,
         "truth_conditions": ["SGNS vectors fixed at 64D; supervised GCNN/projection/fusion/pair head jointly trainable",
                              "21-channel sequence input and 256D sequence/fused protein features",
                              "Symmetric endpoint-order binary logit mean precedes sigmoid",
                              "Validation BCE selects checkpoint and learning rate before final test",
                              "Topology masking p=0.5 shown only as an optional training arm, not the full-fusion default",
                              "No HCC-specific data claim, QK attention diagram, or tanh output"],
         "visual_style": "Arial bold; blue/purple/gray; white background; black orthogonal arrows 1.5; border hierarchy 2/1.75/1.5/1.25",
         "created_utc": datetime.now(timezone.utc).isoformat()}
manifest_path = OUT / "figures_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["figures"] = [entry] + [item for item in manifest["figures"] if item["id"] != entry["id"]]
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Figure 1 generated (PNG/SVG/PDF); source and output hashes added to manifest.")
