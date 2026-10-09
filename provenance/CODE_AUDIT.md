# CaPPI code release inventory and portability audit

Date: 9 October 2026. Existing scientific sources were read only.

## Verified provenance

- All seven latest training files in `runs/repair_20261007/source_snapshot` are byte-identical to the current working sources. `code_allowlist.json` records every SHA256.
- The latest completed campaign records 300 training runs and 225 selected test evaluations.
- The established graph cache has 20 JSON/NPZ pairs; these are small fitted SGNS representations, not duplicate neural checkpoints. Preserving them allows exact graph representations without requiring the legacy SGNS runtime for each new fit.
- No applicable AGENTS.md was found at the workspace or revised-code roots.
- Existing Chinese README describes the earlier 50 neural / 40 classical campaign and still includes outdated HCC framing. It should be clearly historical or superseded, not used as the current public landing page.

## Code allowlist

`code_allowlist.json` has 25 scientific/collection/plotting/testing entrypoints, 13 synthetic test modules and five legacy original scripts. The seven current training sources must retain their bytes and sibling layout. Keep the five old scripts under `legacy/` and state that they did not generate the revised common-protocol results. Legacy directory names and scripts preserve the author's original naming.

Additional figure source: `results/repair_20261007/figures/gen_fig_repair.py`. It relies on the exact verified campaign/source result directory structure. Current publication generators are retained at `manuscript/new_results_20261007/figures/{generate_figures.py,generate_figure4_folds.py,generate_matched_supplements.py,generate_workflow.py,audit_assets_and_targets.py}`. These original figure generators contain an author-machine ROOT path and absolute source paths in old result indexes; they are archival generation sources, not fully portable entrypoints. Word/PPT-building scripts and document QA are not necessary for training/inference and can be omitted from a code/data repository.

## Observed dependencies

Main scientific runtime: Python 3.10.20; PyTorch 2.11.0+cu128; NumPy 1.26.4; SciPy 1.11.4; scikit-learn 1.3.2; pandas 2.3.3; Matplotlib 3.10.8; joblib 1.3.2. Extra fixed classical controls used XGBoost 2.1.1; trajectory projection used umap-learn 0.5.7. These are observed runtime versions, not a tested cross-platform installation guarantee.

Separate SGNS runtime: Python 3.6.13; gensim 3.1.0; NumPy 1.19.5; SciPy 1.2.1. `legacy_skipgram.py` is isolated and deliberately compatible with this older helper environment. Do not modernize gensim silently: that changes the recorded graph fitting algorithm/runtime.

## Portable execution boundary

- `run_experiments.py` supports `--prepared`, `--root`, `--output`, `--matrix`, `--folds`, `--device` and fresh-output/resume options.
- `cappi_data.py` supports `--root`, `--out`, `--seed` and `--n-folds`; it expects the original Unicode filenames `肝癌 sequence.csv` and `肝癌 interact.csv`.
- `run_missing_topology.py` exposes `--config-file` / `--single-json` and `--evaluate-selected`. A training JSON supplies `prepared`, `output`, `config`, `protocol`, `negative`, `fold`, `model_seed`, `graph_seed`, `device`, `epochs`, `patience`, `resume`, `source_graph_cache` and `no_test=true`. Nested config accepts the six declared families and only initial learning rates 0.0005 or 0.001.
- The missing-topology module requires the exact seven training bytes and the exact saved `prepared/manifest.json` hash. **Copy prepared data byte-for-byte; do not rewrite original source paths in that manifest.** Its fold loader does not dereference the informational absolute raw-source paths. Re-preparing on a different machine changes the manifest SHA even if split membership is identical and therefore is incompatible with this frozen-run guard.
- `run_missing_campaign.py` is an original author-machine orchestration script, not portable unchanged. It has fixed `HERE/runs/repair_20261007` paths, hardcoded CUDA training, verifies original absolute raw-source paths and requires previous verified result exports plus reviewer_v1 graph caches. A release wrapper calling `run_missing_topology.run_repair_fold` with explicit new output paths can preserve original code bytes and selection behavior without those archival machine checks.
- `cappi_graph.py` defaults to `D:\Anaconda\envs\tf\python.exe`, but honors `CAPPI_GENSIM_PYTHON`. Set it explicitly if graph fitting is needed. Saved verified graph caches avoid this helper requirement for the matching represented conditions.
- `run_supplement_neural.py` requires the original core SHA, saved prepared SHA, reviewer_v1 campaign metadata and compatible base checkpoints; use the archived base runs or disclose the dependency. Its `--describe` is read-only.
- `run_supplement_cases.py` supports explicit `--source-fasta`, `--source-links`, `--prepared`, `--output`, `--mode` and `--device`. Supply copied source paths; its defaults expect the old parent directory. `--mode audit` verifies exact full-sequence ID mapping and edge-set equality without training. Target analyses exclude all target-incident train and validation pairs before graph fitting.
- `run_supplement_classical.py` supports explicit prepared/output/reference/cache directories. Its task-local `.supplement_deps` discovery is optional; a normal installed XGBoost 2.1.1 is accepted.
- `collect_missing_topology.py`, `verify_missing_topology.py` and the repair plotter are archival full-run verifiers/collectors. They check checkpoint hashes and run artifact integrity. A repository exporting predictions but not all original checkpoints should provide a separate prediction-level numerical verifier and state the limited scope; do not label that a replay of all original checkpoint provenance.

## Minimal example training JSON

```json
{
  "prepared": "prepared",
  "output": "runs_new/protein_random_sequence_only_lr0p0005_repeat0_fold0",
  "config": {"family": "sequence_only", "learning_rate": 0.0005},
  "protocol": "protein",
  "negative": "random",
  "fold": 0,
  "model_seed": 20260930,
  "graph_seed": 20260930,
  "device": "cuda",
  "epochs": 300,
  "patience": 20,
  "resume": false,
  "source_graph_cache": "runs/reviewer_v1/graph_cache",
  "no_test": true
}
```

Use `python run_missing_topology.py --config-file example.json`. Test evaluation is separately gated by a checkpoint-specific `selection_frozen.json`. Learning-rate selection must compare validation BCE from both declared candidates before recording the chosen checkpoint. Then freeze the selected LR for repeat 1 and 2; the graph seed stays 20260930+fold and model seed is 20260930+1000*repeat+fold.

## Checkpoints and GitHub size

The latest repair campaign contains 600 `.pt` files totaling 7,327,363,784 bytes. They contain both best/last checkpoint and optimizer/resume states. Including these duplicate states in a GitHub upload is unnecessary. A compact inference-only export of `checkpoint['model']` for selected runs is possible, but must be explicitly identified as a transformed export with its original checkpoint SHA, graph representation, fold and run configuration. Original checkpoints remain local. If no weights are exported, README should describe training from the supplied split/cache data and numerical rechecking of saved predictions, not checkpoint inference replay.

Preserve byte hashes across Git checkouts: scientific code is a mixture of original CRLF and LF. `.gitattributes` should disable automatic line-ending conversion for frozen source/manifest/provenance artifacts (e.g. `* -text`) because code and manifest bytes participate in run and graph identities.

## Exclude

No Word/PPT manuscript files, docking/structure outputs, personal EndNote data, source-folder document exports, generated caches of Python packages, `.supplement_deps`, `__pycache__`, task QA renders or transient logs are needed in this code/data bundle. Do not overwrite original code, raw input, prepared splits or results while packaging. No publication/upload/DOI has been performed.
