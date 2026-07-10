# TA-DVFG

TA-DVFG studies sparse prediction-topology learning for vertical graph views. The current paper frames the method as collaboration without representation alignment: parties keep local representations private, exchange aligned prediction probabilities, and select a sparse communication graph using validation reliability and graph-level utility.

## Repository Structure

```text
main experiment/              Core HGB TA-DVFG implementation
experiments/                  Experiment runners, audits, aggregation scripts
experiments/movielens/        MovieLens real-world experiment suite
models/                       MovieLens party models
scripts/                      PowerShell and shell entry points
data_hgb/                     HGB-derived local data
data/                         MovieLens and other local data
results/                      HGB experiment outputs and summaries
outputs/                      Closure audits, MovieLens outputs, paper packages
figures/                      Paper-facing figures copied from final outputs
docs/                         Control documents and repository manifest
paper/                        Local paper-control notes; current paper source is Overleaf
```

## Current Paper Source

The latest Overleaf project contains `main.tex`, `supplementary.tex`, `ReproducibilityChecklist.tex`, and `CameraReady2027.tex`.

- `main.tex` is the real main paper compile root.
- `supplementary.tex` is a standalone supplement.
- `ReproducibilityChecklist.tex` is standalone/input-guarded but is not included by current `main.tex`.
- `CameraReady2027.tex` is an AAAI formatting-instructions template and does not wrap `main.tex`.

## Core Method Code

- HGB and cached topology experiments: `main experiment/ta_dvfg_hgb_reliability.py`
- General experiment matrix: `experiments/run_experiments.py`
- Cached core: `experiments/run_cached_core.py`
- Cached ablations: `experiments/run_cached_ablations.py`
- MovieLens suite: `experiments/movielens/run_movielens_tadvfg.py`

## Important Evidence

The current paper evidence is retained in place. Key locations include:

- HGB cached core tables: `results/core_cached_v2/`
- HGB reusable prediction caches: `results/core_cache_v2/`
- Mechanism and communication analyses: `results/combined/`, `results/topology_objective/`, `results/edge_budget/`, `results/consensus/`, `results/deployment_objective/`
- Reviewer protocol audits: `results/active_party_protocol/`, `results/validation_label_budget/`, `results/topology_frequency/`, `results/strict_label_training/`, `results/party_scalability/`, `results/strong_setting/`
- K sensitivity and nested weak-party scaling: `results/k_sensitivity_runs/`, `results/nested_weak_scaling/`
- Minimum-link audit: `results/minedge_sweep/`, `outputs/minimum_link_audit/`
- MovieLens final evidence: `outputs/movielens/`, `outputs/movielens_main15/`, `outputs/movielens_hard15/`, `outputs/movielens_final_package/`
- Stronger-information alignment references: `outputs/alignment_references_acmhard5/`

## Reproduction

Use `REPRODUCE.md` for the paper experiment map, canonical entry scripts, expected outputs, and caveats. This cleanup did not rerun experiments or regenerate evidence.
