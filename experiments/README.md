# TA-DVFG experiment package

This directory turns the research plan into a reproducible experiment matrix.
It uses the existing training engines:

- `main experiment/ta_dvfg_hgb_reliability.py`
- `multi-modalities/ta_dvfg_hgb_multimodal.py`

No model logic is duplicated. The runner adds job generation, isolated outputs,
logs, saved commands, resume markers, aggregation, statistical tests, and
ablation plots.

## 1. Validate the environment

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\validate_setup.py
```

The HGB-derived data are expected under `data_hgb/hgb/{acm,dblp,imdb}`.
The runner enables Python UTF-8 mode for the Windows HGB text files.

## 2. Inspect or smoke-test the plan

```powershell
# List every core job.
.\.venv\Scripts\python.exe experiments\run_experiments.py core --list

# Print commands only.
.\.venv\Scripts\python.exe experiments\run_experiments.py core --dry-run

# Two epochs, one seed, CPU, first ACM job only.
.\.venv\Scripts\python.exe experiments\run_experiments.py core --quick --datasets ACM --max-jobs 1
```

Smoke-test outputs are isolated under `results/smoke/` and never overwrite
production results.

## 3. Run the experiments

```powershell
# Six main jobs: ACM/DBLP/IMDB, main + hard noisy, five seeds, nine methods.
.\scripts\run_core.ps1

# Required ablations from the RP.
.\scripts\run_ablations.ps1

# Every core, ablation, extension, and appendix job.
.\scripts\run_all_experiments.ps1

# Aggregate tables, paired tests, and ablation figures after runs finish.
.\scripts\run_analysis.ps1

# Active-party-only validation evaluator, replayed from cached local predictors.
.\scripts\run_active_party_protocol.ps1

# Deployment-aware topology objectives: active, local_mean, and joint frontier.
.\scripts\run_deployment_objective.ps1 -StrictCache
```

Direct runner examples:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiments.py core --device cuda
.\.venv\Scripts\python.exe experiments\run_experiments.py edge_budget noise_ratio --device cuda
.\.venv\Scripts\python.exe experiments\run_experiments.py all --datasets ACM DBLP --device cuda
```

Completed jobs are skipped by default. Use `--force` to rerun them. Use
`--continue-on-error` for unattended batches.

## Experiment suites

| Suite | Jobs | Purpose |
|---|---:|---|
| `core` | 6 | ACM/DBLP/IMDB main and hard, 13 reviewer-safe methods |
| `reviewer_safe` | 4 | Exact ACM/DBLP reviewer-safe output bundles |
| `reviewer_safe_shuffled` | 2 | ACM main/hard with seed-shuffled party positions |
| `topology_objective` | 2 | pair vs complementarity vs graph-level validation |
| `edge_budget` | 24 | budget `{5,10,15,20}` × min-gain `{0,.001,.005}` |
| `consensus` | 28 | one-factor-at-a-time steps, self-weight, and weighting |
| `noise_ratio` | 8 | useful parties `{3,4,6,9}` out of 15 |
| `multimodal_core` | 6 | mixed-modality extension on ACM/DBLP/IMDB main and hard |
| `modality_ablation` | 12 | ATTR/TEXT/STRUCT/HYBRID/TEXT_STRUCT/mixed on ACM |
| `central_baselines` | 4 | optional centralized references |
| `protocol_sensitivity` | 4 | custom stratified-split sensitivity |
| `readout_ablation` | 12 | active vote vs mean party vs best party |
| `active_party_evaluator` | 8 | shared vs active-party validation evaluator |
| `active_party_protocol` | 4 | shuffled cached ACM/DBLP main+hard under an active label-holder |
| `strict_label_training` | 2 | active-party-only labels with passive logit-gradient returns |
| `split_validation` | 4 | shared vs disjoint topology/model-selection validation |
| `decentralized_readout` | 8 | global active vote vs party-local deployment readouts |
| `active_party_identity` | 10 | evaluator-ID invariance over five party IDs |
| `validation_label_budget` | 10 | topology labels from 10% to 100% |
| `topology_frequency` | 8 | one-shot and 10/20/50-epoch topology updates |
| `party_scalability` | 5 | ACM Hard from 5 to 30 parties |
| `strong_setting` | 4 | strict active-party labels, no-global metrics, and missing-party inference |
| `deployment_objective` | 12 | strict active-party deployment frontier: active/local/joint topology objectives |

The consensus suite deliberately uses one-factor-at-a-time changes around the
paper default. This covers every requested value without conflating three
simultaneous changes or multiplying the run count with a redundant Cartesian
product.

The edge-budget suite uses `max_degree=3`. With 15 parties, the paper default
`max_degree=2` permits at most 15 undirected edges and would silently collapse
the requested budget-20 point onto budget 15.

## Output layout

Every job is self-contained:

```text
results/<suite>/<job>/
  job.json
  command.txt
  run.log
  metrics.csv
  metrics_summary.csv
  metrics_config.json
  figures/
  _SUCCESS
```

`_FAILED` is written instead when the engine exits non-zero.

## 4. Aggregate, test, and plot

```powershell
.\.venv\Scripts\python.exe experiments\aggregate_results.py
.\.venv\Scripts\python.exe experiments\statistical_tests.py
.\.venv\Scripts\python.exe experiments\make_ablation_plots.py
```

The combined outputs are written to `results/combined/`:

- `raw_results.csv`
- `summary_results.csv`
- `paper_core_table.csv`
- `paper_communication_table.csv`
- `paper_topology_objective_table.csv`
- `paired_significance.csv`
- `figures/*.png`

Reviewer-safe jobs use the requested filenames directly:

- `acm_main_reviewer_safe_5seeds.csv`
- `acm_hard_reviewer_safe_5seeds.csv`
- `dblp_main_reviewer_safe_5seeds.csv`
- `dblp_hard_reviewer_safe_5seeds.csv`

Their automatically generated summaries append `_summary.csv`.

Every newly generated metrics CSV separates inference communication into:

- `peer_to_peer_comm`: bidirectional prediction exchange over collaboration
  edges. The legacy `inference_comm` column remains an alias for this value.
- `global_readout_comm`: prediction collection required by global diagnostic
  votes or the explicitly evaluated active-party readout.
- `total_comm`: peer exchange plus global-readout collection.
- `protocol_type`: `global_diagnostic`, `peer_to_peer_topology`,
  `local_reference`, or `centralized_reference`.

Global Uniform Vote, Global Reliability Vote, and Global Top-k Reliability are
diagnostic readouts rather than peer-to-peer topology baselines. Global Top-k
Reliability has zero peer-edge exchange but nonzero centralized prediction
collection.

The shuffled reviewer-defense rerun writes:

- `acm_main_reviewer_safe_shuffled_5seeds.csv`
- `acm_hard_reviewer_safe_shuffled_5seeds.csv`

These rows also store the party permutation, useful-party mask, view names,
selected topology edges, final reliabilities, and useful/useful,
useful/noisy, noisy/noisy edge counts.

## Active-party evaluator protocol

This reviewer-facing suite keeps local predictor training unchanged, but only
the designated active evaluator accesses validation labels. Passive parties
submit validation probabilities; the evaluator computes party reliability,
Global Top-k ranking, candidate topology scores, and graph-level edge gains.
It reuses the seed-shuffled `results/core_cache_v2` predictions and therefore
does not retrain local models.

```powershell
# Fast check: ACM Hard, one cached seed.
.\.venv\Scripts\python.exe experiments\run_experiments.py active_party_protocol `
  --match acm_hard --seeds 42 --device cuda --strict-cache

# Required Hard settings only, all five seeds.
.\scripts\run_active_party_protocol.ps1 -HardOnly

# Full ACM/DBLP Main + Hard suite, all five seeds.
.\scripts\run_active_party_protocol.ps1
```

The four jobs write the requested files inside their job directories:

- `acm_hard_active_evaluator.csv` and `_summary.csv`
- `acm_main_active_evaluator.csv` and `_summary.csv`
- `dblp_hard_active_evaluator.csv` and `_summary.csv`
- `dblp_main_active_evaluator.csv` and `_summary.csv`

Run `experiments/final_analysis.py` afterward to generate
`paper_active_party_protocol.csv`,
`active_party_protocol_paired_significance.csv`, and
`active_party_protocol.png`.

## Deployment-aware topology objective suite

This is the main reviewer-facing upgrade for the active/global versus local
post-consensus gap.  It keeps the strict active-party label-holder protocol:
passive parties do not receive labels, validation is split into topology and
model-selection subsets, and cached strict local predictions are replayed.

```powershell
# Smoke test: one seed, two epochs/cache override, first ACM deployment job.
.\.venv\Scripts\python.exe experiments\run_experiments.py deployment_objective `
  --quick --datasets ACM --max-jobs 1 --device cpu --force

# Formal replay from strict cache, five seeds on ACM Hard and DBLP Hard.
.\scripts\run_deployment_objective.ps1 -Device cuda -StrictCache

# Force rerun topology replay and regenerate CSV/plot.
.\scripts\run_deployment_objective.ps1 -Device cuda -StrictCache -Force
```

Final outputs:

- `results/deployment_objective/deployment_objective_results.csv`
- `results/deployment_objective/deployment_objective_summary.csv`
- `results/deployment_objective/figures/deployment_tradeoff.pdf`
- `results/deployment_objective/figures/deployment_tradeoff.png`

The plotted frontier contains Full Mesh, Best Matched Sparse,
Global Top-k Reliability, TA-DVFG-active, TA-DVFG-local, and
TA-DVFG-joint at lambda 0.25/0.5/0.75.

## Reviewer-resistance protocol suite

```powershell
# Cached sweeps only; does not retrain local models.
.\scripts\run_reviewer_protocols.ps1 -Stage cached

# Label-siloed training and party-count scalability.
.\scripts\run_reviewer_protocols.ps1 -Stage training

# Everything above.
.\scripts\run_reviewer_protocols.ps1 -Stage all
```

`active_party_logit_gradient` keeps train and validation labels at the active
party. Passive parties submit train logits and receive gradients only with
respect to those logits. This establishes label ownership but does not claim
cryptographic privacy or resistance to logit/gradient leakage.

## Strong active-party label-holder suite

```powershell
# Uses existing strict_label_cache_v1 caches and refuses to retrain if missing.
.\scripts\run_strong_setting.ps1 -Device cuda -StrictCache
```

This suite runs only the reviewer-critical methods:
`active_local_only`, `global_topk_reliability`, `full_mesh`,
`best_matched_sparse`, and `adaptive_graph_val` on ACM/DBLP Hard with seeds
42--46. It writes:

- `results/strong_setting/strict_active_party_results.csv`
- `results/strong_setting/strict_active_party_summary.csv`
- `results/strong_setting/no_global_readout_results.csv`
- `results/strong_setting/missing_party_robustness_results.csv`
- `results/strong_setting/strong_setting_summary.csv`

The protocol is `active_party_only + prediction_split + split_val`; passive
label access is false. Missing-party inference randomly removes non-active
parties at test time only.

## Cached topology, edge-budget, and consensus ablations

The `topology_objective`, `edge_budget`, and `consensus` suites share the same
fair, seed-shuffled local prediction trajectories as cached core. Across all
54 jobs, only four local configurations are needed: ACM/DBLP times main/hard.
With five seeds this means 20 local training trajectories, followed by
topology/readout-only replay.

```powershell
# Safe one-seed smoke test; outputs and cache are isolated automatically.
.\scripts\run_cached_ablations.ps1 -Stage all -Device cuda -Seeds "42" -Epochs 5 -NoPlot

# Recommended: build any missing caches, then run all three suites.
.\scripts\run_cached_ablations.ps1 -Stage all -Device cuda -NoPlot

# Expensive stage only.
.\scripts\run_cached_ablations.ps1 -Stage build-cache -Device cuda

# Strict replay only; fails instead of retraining when a cache is absent.
.\scripts\run_cached_ablations.ps1 -Stage sweep -Device cuda -NoPlot
```

The caches are stored under `results/core_cache_v2/`, allowing completed
cached-core trajectories to be reused directly. The ordinary
`run_experiments.py` and `run_ablations.ps1` entry points also build a missing
cache once and reuse it for later jobs. Add `--strict-cache` to the direct
runner when accidental retraining must be forbidden.

Quick tests use `results/smoke_cache/`, and other non-300-epoch overrides use
`results/cache_overrides/epochsN/`, so test runs cannot overwrite the
300-epoch production cache.

## Cached multimodal experiments

Multimodal core and modality ablation jobs now use the same validated
prediction-cache backend as the standard experiments. Each
dataset/setting/modality/seed trains local parties once; every topology and
readout method reuses those exact per-epoch predictions.

```powershell
# Six mixed-modality core jobs.
.\scripts\run_cached_multimodal.ps1 -Mode core -Device cuda

# Twelve ACM modality-ablation jobs.
.\scripts\run_cached_multimodal.ps1 -Mode ablation -Device cuda

# Both groups.
.\scripts\run_cached_multimodal.ps1 -Mode all -Device cuda
```

Results remain under `results/multimodal_core/` and
`results/modality_ablation/`. Reusable caches are stored under
`results/multimodal_cache_v3/`. Cache namespaces include the exact modality
configuration, so ATTR-only, mixed, and TEXT_STRUCT experiments cannot
accidentally share incompatible predictions.

The significance script performs paired t-tests, Wilcoxon signed-rank tests,
95% confidence intervals, paired Cohen's `d_z`, and Holm correction. Its default
comparisons are TA-DVFG against local reliability, fixed ring, and full mesh.

## Cached local predictions and min-edges sweep

To train each local party model once and reuse exactly the same per-epoch
predictions for `adaptive_min_edges={0,1,3,5}`:

```powershell
.\scripts\run_cached_minedges_sweep.ps1 -Device cuda -Stage all
```

Run the stages separately when desired:

```powershell
# Expensive stage: one local training trajectory per dataset/seed.
.\scripts\run_cached_minedges_sweep.ps1 -Device cuda -Stage build-cache

# Cheap stage: no GCN training; topology/readout evaluation only.
.\scripts\run_cached_minedges_sweep.ps1 -Device cuda -Stage sweep
```

Outputs:

- `results/minedge_sweep/acm_hard_minedges_sweep.csv`
- `results/minedge_sweep/acm_hard_minedges_sweep_summary.csv`
- `results/minedge_sweep/dblp_hard_minedges_sweep.csv`
- `results/minedge_sweep/dblp_hard_minedges_sweep_summary.csv`

Direct engine cache flags are also available:

```powershell
# Build cache while evaluating the selected methods.
python "main experiment\ta_dvfg_hgb_reliability.py" --dataset ACM `
  --graph_views PAP,PSP,KNN --view_setting hard --useful_parties 3 `
  --shuffle_party_positions --seeds 42 --cache_predictions

# Strict topology-only replay: fails rather than retraining if cache is absent.
python "main experiment\ta_dvfg_hgb_reliability.py" --dataset ACM `
  --graph_views PAP,PSP,KNN --view_setting hard --useful_parties 3 `
  --shuffle_party_positions --seeds 42 --methods adaptive_graph_val `
  --adaptive_min_edges 3 --reuse_prediction_cache `
  --skip_local_training_if_cache_exists
```

The cache stores final train/validation/test/all-node probabilities and
per-epoch validation/test probabilities. The latter preserve the existing
`Test@BestVal` semantics during topology-only replay.

### Rebuild all six core experiments deterministically

The recommended fair rebuild uses a stable SHA-256 meta-path sampling seed,
seed-shuffled party positions, and one shared local prediction trajectory per
dataset/setting/seed:

```powershell
.\scripts\run_cached_core.ps1 -Device cuda -Seeds "42,43,44,45,46" -Epochs 300
```

Outputs are written under `results/core_cached_v2/`; reusable caches are under
`results/core_cache_v2/`. Rerunning the command reuses valid caches and skips
completed CSVs. Use `-Force` only when you intentionally want to regenerate the
six output tables.

To combine a fixed reviewer-safe baseline summary with the cached sweep, while
validating every seed's local-training config:

```powershell
.\.venv\Scripts\python.exe experiments\merge_cached_baselines.py `
  --baseline-summary results\reviewer_safe_shuffled\acm_hard\acm_hard_reviewer_safe_shuffled_5seeds_summary.csv `
  --baseline-config results\reviewer_safe_shuffled\acm_hard\acm_hard_reviewer_safe_shuffled_5seeds_config.json `
  --sweep-summary results\minedge_sweep\acm_hard_minedges_sweep_summary.csv `
  --cache-dir results\cache `
  --output results\minedge_sweep\acm_hard_combined_with_baselines.csv
```

The merge is refused if dataset, split, party assignment, local model, feature
construction, or training hyperparameters differ.

## Protocol notes

- `main` means 15 parties with 6 useful and 9 noisy views.
- `hard_noisy` means 15 parties with 3 useful and 12 noisy views.
- Both use the engine's `view_setting=hard`; only `useful_parties` changes.
- Results are HGB-derived controlled VFGL benchmarks, not official HGB
  leaderboard submissions.
- The current IMDB engine converts multi-label targets to a controlled
  single-label task. Keep this caveat visible in the paper.
- Multimodal features are simulated private modalities derived from each
  party's local feature block or graph view; they are not a real
  image-text-audio benchmark.
