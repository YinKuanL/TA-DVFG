# Repository Manifest

Last updated: 2026-07-10

This manifest records the important retained paths after the cleanup pass. Ambiguous paths were kept rather than archived.

| Path | Classification | Role |
|---|---|---|
| `README.md` | KEEP_REPRODUCIBILITY | Repository overview and current structure |
| `REPRODUCE.md` | KEEP_REPRODUCIBILITY | Paper experiment-to-evidence map |
| `AGENTS.md` | KEEP_CONTROL | Root repository operating instructions |
| `docs/` | KEEP_CONTROL | Task and paper-control documentation |
| `paper/AGENTS.md` | KEEP_CONTROL | Paper editing constraints |
| `experiments/AGENTS.md` | KEEP_CONTROL | Experiment integrity constraints |
| `main experiment/ta_dvfg_hgb_reliability.py` | KEEP_CORE | Core HGB TA-DVFG implementation |
| `main experiment/` | KEEP_CORE | Legacy core implementation directory and HGB evidence helpers |
| `experiments/` | KEEP_EXPERIMENT | Runners, audits, aggregation scripts, and paper experiment matrix |
| `experiments/movielens/` | KEEP_EXPERIMENT | MovieLens experiment suite |
| `models/` | KEEP_CORE | MovieLens party model definitions |
| `scripts/` | KEEP_REPRODUCIBILITY | Canonical PowerShell and shell wrappers |
| `tests/` | KEEP_REPRODUCIBILITY | Static/unit checks retained for maintainability |
| `requirements-experiments.txt` | KEEP_REPRODUCIBILITY | Minimal experiment dependency record |
| `data_hgb/` | KEEP_LOCAL_NOT_GIT | HGB-derived local data used by retained experiment scripts; ignored for normal Git |
| `data/` | KEEP_LOCAL_NOT_GIT | MovieLens/local data used by retained experiment scripts; ignored for normal Git |
| `data_acm/` | KEEP_LOCAL_NOT_GIT | Large local ACM data artifact; retained locally and ignored |
| `acm/` | KEEP_LOCAL_NOT_GIT | Local ACM artifact directory; retained locally and ignored |
| `results/core_cached_v2/` | KEEP_EVIDENCE | HGB five-seed cached-core outputs |
| `results/core_cache_v2/` | KEEP_LOCAL_NOT_GIT | Reusable HGB prediction caches for cached replay; ignored for normal Git |
| `results/combined/` | KEEP_EVIDENCE | Aggregated paper tables and communication summaries |
| `results/topology_objective/` | KEEP_EVIDENCE | Mechanism/topology-objective outputs |
| `results/edge_budget/` | KEEP_EVIDENCE | Edge-budget sensitivity evidence |
| `results/consensus/` | KEEP_EVIDENCE | Consensus-depth and readout ablation evidence |
| `results/deployment_objective/` | KEEP_EVIDENCE | Deployment frontier outputs and figures |
| `results/active_party_protocol/` | KEEP_EVIDENCE | Active-party evaluator protocol evidence |
| `results/validation_label_budget/` | KEEP_EVIDENCE | Label-budget sensitivity evidence |
| `results/topology_frequency/` | KEEP_EVIDENCE | Topology-refresh frequency evidence |
| `results/strict_label_training/` | KEEP_EVIDENCE | Active-party-only label training evidence |
| `results/party_scalability/` | KEEP_EVIDENCE | Party-count scalability evidence |
| `results/strong_setting/` | KEEP_EVIDENCE | Strict active-party label-holder outputs |
| `results/minedge_sweep/` | KEEP_EVIDENCE | Minimum-link sweep evidence |
| `results/k_sensitivity_runs/` | KEEP_EVIDENCE | K sensitivity runs and manifest |
| `results/nested_weak_scaling/` | KEEP_EVIDENCE | Weak-party scaling evidence |
| `outputs/config_audit/` | KEEP_EVIDENCE | Paper configuration audit |
| `outputs/ac_review_closure/` | KEEP_EVIDENCE | Claim and protocol closure reports |
| `outputs/minimum_link_audit/` | KEEP_EVIDENCE | Minimum-link audit report and summaries |
| `outputs/movielens/` | KEEP_EVIDENCE | Canonical MovieLens five-party outputs and combined figures |
| `outputs/movielens_main15/` | KEEP_EVIDENCE | Canonical MovieLens 15-party Main outputs |
| `outputs/movielens_hard15/` | KEEP_EVIDENCE | Canonical MovieLens 15-party Hard outputs |
| `outputs/movielens_final_package/` | KEEP_EVIDENCE | Final MovieLens package and claim audit |
| `outputs/movielens_*_leakage_safe_20260705/` | KEEP_EVIDENCE | Timestamped leakage-safe rerun evidence |
| `outputs/alignment_references_acmhard5/` | KEEP_EVIDENCE | ACM Hard stronger-information reference study |
| `figures/` | KEEP_EVIDENCE | Paper-facing final figures |
| `multi-modalities/` | KEEP_LOCAL_NOT_GIT | Older multimodal extension; retained locally but ignored for normal Git |
| `github_upload_TA_DVFG/` | KEEP_LOCAL_NOT_GIT | Nested packaging clone used for Git recovery; retained locally but ignored |
| `supplement_experiment_code/` | KEEP_LOCAL_NOT_GIT | Supplement code package; retained locally but ignored |
| `.codex/` | KEEP_CONTROL | Local Codex configuration examples and workspace metadata |

## Archived By This Pass

The archive manifest is written outside the repository at:

`C:\Users\Yin Kuan\OneDrive\獢\Durham_CS\Research\TA-DVFG_archive_20260710\ARCHIVE_MANIFEST.md`

Only clearly generated, stale, duplicate, or superseded paths were moved. No `.tex`, `.bib`, CSV, figure, experiment source, data file, result summary, or Overleaf file was edited.

## Git Payload Policy

Normal Git tracking should include source code, scripts, tests, documentation, small CSV/JSON/MD evidence, and final small figures. Large downloaded data, processed datasets, prediction caches, checkpoints, temporary outputs, and duplicate packaging clones are retained locally or in the external archive but ignored by `.gitignore`.
