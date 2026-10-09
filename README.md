# CaPPI — code, data and experimental records

This repository contains the code-and-data release prepared on **9 October 2026** for the revised CaPPI manuscript. Source repository: <https://github.com/NTUYLS/CaPPI-code>. A deposit in a DOI-assigning repository remains pending.

CaPPI combines training-graph random-walk representations, gated convolutional sequence encoding, residual fusion and an endpoint-symmetric pair classifier. The benchmark is the **STRING association network for Candidatus Hepatoplasma crinochetorum Av**, taxonomy ID **1427984**: **582 proteins and 23,707 unique undirected associations**. It is a bacterial association benchmark. The old filenames containing “肝癌” are retained solely to preserve input and audit hashes; they do not identify human hepatocellular carcinoma data.

## Contents

| Path | Contents |
|---|---|
| `cappi_*.py`, `run_*.py`, `collect_*.py`, `verify_*.py` | Revised scientific implementation, experiment runners, aggregation and validation utilities. |
| `run_release.py` | Portable entry point and prediction-based checks added for this distribution; see its command help. |
| `data/raw/` | Original release-labelled STRING links/FASTA and the exact CSV inputs used for preparation. |
| `prepared/` | Frozen canonical pairs, both putative-negative pools, fixed pair/protein partitions and hashed manifests. |
| `runs/repair_20261007/` | Latest campaign selection records, validation/test predictions, histories, topology arrays, source snapshots and audits. |
| `runs/reviewer_v1/` | Earlier fixed-configuration model and walk-policy records. |
| Other `runs/` subfolders | Earlier validation sensitivity, classical controls, strict target rankings and trajectory records. |
| `results/` | Aggregate CSV/JSON results and available scientific plotting outputs. |
| `historical_baselines/` | Explicitly historical PASNVGA/SDNN-PPI reports and retained initial outputs; these were not rerun under the new protocol. |
| `legacy/` | Original component scripts, retained for provenance; not the corrected experimental implementation. |
| `tests/` | Existing implementation and data-boundary checks. |
| `environments/` | Recorded training/isolated-SGNS environments and dependency instructions. |
| `provenance/` | Source-file SHA-256 inventory, dataset identity, package provenance and validation records. |

Approximately 1.2 GB of source code and experimental data are included. Original copied files retain their bytes and SHA-256 hashes. Keep `.gitattributes`: newline conversion would change hashed CSV/JSON and code files.

## Evaluation boundaries

- Graph construction and SGNS fitting use only positive training edges within each fold. The learnable sequence, topology-projection, residual-fusion and classifier modules are optimized jointly.
- Pair-level evaluation is transductive at the protein level. Protein-ID-disjoint evaluation retains within-partition pairs; it does not enforce sequence-homology separation.
- Random and approximately degree-matched pools contain **putative negatives**, not experimentally established non-interactions. Degree matching uses the complete positive reference only for retrospective sampling, not for model topology features.
- The latest campaign contains **300 training runs and 225 selected test evaluations**. It uses five fixed folds and three neural repeats. Latest means/SDs are computed across three five-fold repeat means. Earlier fixed controls and validation sensitivity use fold-based summaries.
- Validation BCE determines learning-rate and checkpoint selection. The benchmark was previously used during development; these are adaptive benchmark comparisons, not a fresh external confirmatory test.
- New AUPR values are native trapezoidal precision–recall areas, not average precision. No synthetic scores are generated to match curves or table values.
- Topology-only and parameter-count-matched controls do not establish a residual-fusion advantage. Missing-topology evaluation exposes a substantial limitation of standard fusion.
- Strict single-target results use one seed for two targets and are retrospective database-association recovery, not independent biological validation.

## Source-data provenance

Retained source filenames are `1427984.protein.links.v12.0.txt` and `1427984.protein.sequences.v12.0.fa`. The original links file contains 47,414 oriented rows corresponding to 23,707 unique undirected pairs. Observed combined scores span 150–999. That observed minimum does **not** establish an original download cutoff of 150. The original retrieval date and evidence-channel filter were not retained. Associations do not by themselves establish direct physical binding or a disease-specific effect.

Source: STRING, <https://string-db.org/>. Preserve STRING attribution and consult the upstream data-use terms when redistributing the source files. No new software license is assigned by this code-and-data release.

## Dependencies and execution

The recorded neural-training environment used Python 3.10.20, PyTorch 2.11.0+cu128, NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2, pandas 2.3.3 and matplotlib 3.10.8. Additional analyses used xgboost 2.1.1, joblib 1.3.2 and umap-learn 0.5.7. These are observed versions, not a claim that a clean installation has been tested on every platform.

The exact legacy SGNS helper used a separate Python 3.6.13 environment with gensim 3.1.0, NumPy 1.19.5 and SciPy 1.2.1. A modern gensim replacement is not claimed to reproduce identical embeddings. Existing cached training-graph arrays are included. To refit the original SGNS implementation, configure `CAPPI_GENSIM_PYTHON` or use the release runner's explicit SGNS interpreter option. See `RUNNING.md` for actual commands and which checks have been executed.

`run_missing_campaign.py` and some plotting/collection scripts are retained author-machine orchestration records. Their original absolute paths are part of archival provenance; they are not advertised as portable one-command pipelines. The release entry point accepts local paths and preserves validation/test-selection boundaries. A short startup, inspection or metric check is **not** a new completed training experiment.

## What is intentionally outside this package

Duplicate best/last optimizer checkpoints account for more than 7 GB in the source workspace and are not included. Trained model binaries (`.pt`, `.joblib`, `.h5`) are omitted; **this is not a pretrained-model distribution**. Saved labels/scores permit independent metric recomputation, and supplied source data/splits support retraining. Full historical checkpoint-byte verification needs the omitted original checkpoints; prediction checks do not claim checkpoint or weight replay.

Installed Python environments, unrelated projects, temporary task files, old manuscript copies and docking materials are excluded. Legacy scripts are archival references; obsolete original-protocol feature exports are not represented as a newly validated reproduction of historical baselines.

## Repository and archival status

The source repository is <https://github.com/NTUYLS/CaPPI-code>. Keep compressed delivery archives outside the repository. Archived records should remain unchanged, and new training outputs should go under the ignored `work/` directory.

Dated provenance records retain the initial local-package audit state. `provenance/PUBLICATION_PREPARATION.json` records preparation of this repository copy; it is not a remote-upload receipt.

GitHub upload alone does not fulfill the editor's requirement for a code archive in a **DOI-assigning repository**. DOI deposition and the corresponding manuscript link remain pending. No code archive DOI has been assigned in this release.
