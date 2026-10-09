# CaPPI revision: code, data and numerical reproduction

This folder contains the revised scientific code, original supplied inputs,
fixed prepared splits, saved graph representations and archived prediction
records. Dataset identity is *Candidatus Hepatoplasma crinochetorum* Av (STRING
taxonomic identifier 1427984); it is not a human hepatocellular carcinoma cohort.
The original Chinese input filenames are preserved only for byte-identical
reproduction and do not describe the organism.

## Environment

The recorded main runtime is Python 3.10.20, PyTorch 2.11.0 with CUDA 12.8,
NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2, pandas 2.3.3 and Matplotlib
3.10.8. Supplementary classical controls additionally used joblib 1.3.2 and
XGBoost 2.1.1. UMAP trajectory figures used umap-learn 0.5.7. Environment JSON
records describe observed versions; identical outputs across platforms are not
guaranteed. Entry points import PyTorch first for the verified Windows runtime.

If a new graph representation must be fitted, SGNS uses a separate Python
3.6.13 environment with gensim 3.1.0, NumPy 1.19.5 and SciPy 1.2.1. Specify that
interpreter with `--sgns-python` or `CAPPI_GENSIM_PYTHON`. The archived verified
graph caches allow matching graph inputs to be reused without refitting SGNS.
Do not silently replace gensim 3.1.0 with a new SGNS implementation.

## Inspect data and recheck archived numerical results

Run commands from the release folder in the main scientific environment:

```powershell
python run_release.py --help
python run_release.py inspect --report checks/data.json
python run_release.py verify --report checks/prediction_metrics.json
```

The first audit validates all 20 saved folds and artifact hashes. The second
verifies all 225 selected test runs, 450 validation/test prediction CSV files,
their exact saved membership/labels and reported metrics. It also recomputes
all 15 summaries: first average five folds within each model repeat, then take
the mean and sample SD of three repeat means. AUPR is the trapezoidal PR area;
Average Precision is independently retained. The test threshold is 0.5.

This verification uses saved probabilities; it does not rerun checkpoint
inference. The 7.33 GB collection of duplicate optimizer/resume checkpoints is
not included. Original checkpoint identifiers remain in the provenance files.
Saved results come from an adaptively reused benchmark, not a new blind or
independent external test.

## Fit a new model with the unchanged scientific implementation

Keep the supplied `prepared` directory and manifest byte-for-byte. Absolute
raw-source paths in that manifest are historical metadata and are not accessed
by the portable fold loader. Regenerating the manifest on another machine
changes its byte hash and violates the frozen campaign guard even when fold
membership is unchanged. Raw data and construction code are provided for audit,
but a freshly prepared dataset requires a separately declared experiment.

Inspect the resolved configuration without training:

```powershell
python run_release.py train --protocol protein --family sequence_only --fold 0 --learning-rate 0.0005 --output runs_new/seq_lr0005 --show-config
```

For a complete learning-rate choice, fit both declared candidates using only
training/validation data, freeze their choice, then evaluate the chosen fit:

```powershell
python run_release.py train --protocol protein --family sequence_only --fold 0 --learning-rate 0.0005 --output runs_new/seq_lr0005 --device cuda
python run_release.py train --protocol protein --family sequence_only --fold 0 --learning-rate 0.001 --output runs_new/seq_lr0010 --device cuda
python run_release.py select --candidates runs_new/seq_lr0005 runs_new/seq_lr0010
python run_release.py evaluate --run runs_new/SELECTED_RUN_FROM_SELECT_OUTPUT --device cuda
```

Replace the final placeholder with the actual chosen directory printed by
`select`. Default budget is 300 epochs, early-stopping patience 20. Each fit
chooses its checkpoint using the minimum validation BCE. The LR choice uses
the minimum validation BCE across candidates, breaking ties toward the lower
LR. No test labels/probabilities enter fit or selection; the evaluator requires
a checkpoint-specific pre-test frozen declaration. Fresh outputs are required;
`--resume` resumes only a compatible new fit.

The wrapper exposes all six declared families, both protocols and both saved
negative pools; the reported latest campaign evaluated random negatives under
pair/protein protocols and degree-matched negatives under pair protocol only.
Families are `residual`, `topology_dropout`, `topology_dropout_bypass`,
`sequence_only`, `topology_only` and `concat_matched`. Training dropout variants
use their predeclared probability 0.5, with no stochastic masking at evaluation.

For the reported repeat design, graph seed is 20260930+fold and model seed is
20260930+1000*repeat+fold. Choose LR once at repeat 0 and freeze it for repeat
1/2. Train the chosen LR with `--repeat 1` or `--repeat 2`, then use:

```powershell
python run_release.py freeze-repeat --run runs_new/REPEAT1_RUN --selected runs_new/SELECTED_REPEAT0_RUN
python run_release.py evaluate --run runs_new/REPEAT1_RUN --device cuda
```

Replace placeholders with the actual new run paths. `freeze-repeat` verifies
the selected repeat-0 declaration, fixed LR, same graph/fold/model settings and
declared model-seed offset; it does not retune the learning rate.

The complete original author-machine orchestration and its contract are
retained in `run_missing_campaign.py` / `runs/repair_20261007`, but that
orchestrator is not portable unchanged: it additionally checks original
absolute workspace paths and hardware. The wrapper is an additive release
interface; none of the frozen scientific sources were rewritten.

## Other analyses

- `run_experiments.py --help`: earlier fixed neural/walk-control campaign.
- `run_supplement_neural.py --describe`: validation-only sensitivity grid.
  Reusing its baseline checkpoint requires the separately retained original
  reviewer_v1 base checkpoint; archived numerical sensitivity results are
  provided even when these optimizer checkpoints are excluded.
- `run_classical_baselines.py --help` and `run_supplement_classical.py --help`:
  fixed classical feature controls, with explicit cache/output arguments.
- `run_supplement_cases.py --mode audit --source-fasta data/raw/1427984.protein.sequences.v12.0.fa --source-links data/raw/1427984.protein.links.v12.0.txt`:
  exact sequence-based source identifier and edge-set audit; no training.
  Use a new `--output` directory for new audit records. `--mode targets` fits
  target-withheld analyses; all target train/validation associations are
  excluded before graph fitting.
- `python run_tests.py --cpu-only --report checks/contracts.json --log checks/contracts.log`:
  synthetic implementation contract tests; not a substitute for biological
  validation or rerunning scientific experiments.

Original PASNVGA/SDNN-PPI values are retained historical manuscript results,
not reimplemented/retrained comparisons under the revised protocol. The old
five component scripts are separated under `legacy/` for provenance and are
reference-only code, not the revised training entry points; their original
component feature CSV inputs are not included. Original publication figure
generators under `manuscript/new_results_20261007/figures` are retained as
archival code. They contain original machine/source paths and are not portable
unchanged; the portable verifier checks all source curve probabilities/metrics
independently. Word/PPT manuscript, docking and task
rendering files are outside this code/data release.

## Interpretation and publication status

STRING positive labels denote database-supported associations, not independently
validated direct physical interactions. Missing pairs are putative negatives.
Degree-matched sampling uses the complete known positive network only as a
declared retrospective benchmark design; complete degrees do not enter fitted
graph features. Pair splits are transductive; protein splits separate IDs but
do not enforce sequence-homology independence. External organism/cohort
validation and HCC-specific evaluation have not been performed.

Source repository: <https://github.com/NTUYLS/CaPPI-code>. A GitHub URL alone
is not the DOI deposit requested by the editor. After freezing the intended
repository version, archive that version in a DOI-assigning repository and add
its actual DOI to Code Availability. A code archive DOI remains pending.
Dated provenance audits preserve their initial local-package status; they do
not serve as a remote-publication receipt.
