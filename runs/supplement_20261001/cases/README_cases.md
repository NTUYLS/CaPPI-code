# 2026-10-01 supplementary cases

Completed: one 600-epoch descriptive trajectory and two target-withheld ranking runs. Docking is `user_excluded` and is not a pending task.

## Validation trajectory

- Fixed pair/random outer fold 0, seed 20260930, original training settings, with early stopping deliberately disabled only for this descriptive run.
- Four checkpoints at epochs 50, 150, 300 and 600. The same 4,742 validation pairs supply all four feature matrices.
- Representation: 32-dimensional penultimate pair-head ReLU, averaged over forward and reverse endpoint order. Labels are not supplied to UMAP.
- One unsupervised joint UMAP projection across the four matrices, shared axes, n_neighbors=15, min_dist=0.1, Euclidean metric, seed 20260930.
- No test pairs are visualized; no checkpoint or training length is selected from visual separation. These panels do not demonstrate biological or unseen-protein generalization and do not replace the primary checkpoint.
- PNG/PDF: `trajectory_fold0/validation_trajectory.png` and `.pdf`.

The preinstalled UMAP 0.5.12 was incompatible with the existing sklearn 1.3.2 API. Training and feature export had already finished. Rendering was recovered using UMAP 0.5.7 installed only in `plot_deps` with `--no-deps`; the shared environment and the training source were not changed. See `plot_environment.json` for exact versions. To redraw existing features, from `D:\1.肝癌\CaPPI_revised` run:

```powershell
& 'D:\Anaconda\envs\arg_gpu\python.exe' 'runs\supplement_20261001\cases\plot_trajectory_compatible.py'
```

Use this entrypoint for plotting in the current environment; the original `--mode plot` invokes the incompatible preinstalled UMAP unless the isolated `plot_deps` directory is put first on `PYTHONPATH`.

## Target-withheld ranking

All 582 source identifiers were joined to prepared nodes through unique exact full sequences, and all 23,707 positive edges matched the source links after mapping. No ID suffix or row-number assumption was used. See `source_mapping_verified.json` and `source_id_mapping.csv`.

Each target was removed from every training and validation pair before rebuilding topology. Other fold-0 memberships were preserved. Target graph degree and topology input are zero. Checkpoint selection uses validation BCE only. All 581 nonself candidate proteins were then scored.

| Target | Candidate count | Known source associations | AUROC | AP | Hits in top 20 |
|---|---:|---:|---:|---:|---:|
| X271_00001 | 581 | 125 | 0.601842 | 0.264558 | 5 |
| X271_00015 | 581 | 201 | 0.584446 | 0.396026 | 4 |

The two target folders contain `all_candidates.csv`, `top20_candidates.csv`, `ranking_completed.json`, source mapping and exclusion audits, checkpoints, training history, split arrays and topology. Scores are sigmoid model outputs, not calibrated physical-binding probabilities. Labels denote supplied database associations versus unobserved pairs, not experimentally verified interactions versus noninteractions. This is a single-target, single-seed retrospective holdout, not independent external or experimental validation.

## Checks

Eight focused tests passed, including sequence-based mapping, target exclusion, pair symmetry, snapshot membership and an actual tiny training/checkpoint/export test. A second agent reviewed the training loop and scientific scope. `verify_cases.py` independently checks source and checkpoint hashes, snapshots, target isolation, candidate coverage, stored labels and metric recomputation; its result is `verification.json`. Actual PNG inspection is recorded in `visual_qa.json`.

```powershell
& 'D:\Anaconda\envs\arg_gpu\python.exe' 'runs\supplement_20261001\cases\verify_cases.py'
```
