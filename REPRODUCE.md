# Reproduce Current Paper Evidence

This file records the static reproduction map for the current paper. Commands are copied from local experiment documentation where available. Experiments were not rerun during cleanup.

## Paper Compile Roots

| Target | Compile file | Status |
|---|---|---|
| Submission paper | Overleaf `main.tex` | Real current main root |
| Camera-ready paper | Overleaf `main.tex`, with venue metadata adjusted if needed | `CameraReady2027.tex` is only the AAAI instruction template |
| Supplement | Overleaf `supplementary.tex` | Standalone supplement |
| Reproducibility checklist | Overleaf `ReproducibilityChecklist.tex` | Standalone/input-guarded; not included by current `main.tex` |

## Experiment Families

| Family | Claim role | Entry script or wrapper | Expected evidence | Status |
|---|---|---|---|---|
| HGB cached core | Main six controlled HGB-derived settings and Table 1 comparisons | `.\scripts\run_cached_core.ps1 -Device cuda -Seeds "42,43,44,45,46" -Epochs 300` | `results/core_cached_v2/`, `results/core_cache_v2/` | COMPLETE, static evidence retained |
| HGB aggregation and statistics | Paper tables, communication, paired tests | `.\scripts\run_analysis.ps1`; direct scripts `experiments\aggregate_results.py`, `experiments\statistical_tests.py`, `experiments\make_ablation_plots.py` | `results/combined/`, `results/final_analysis/`, `results/figures/` | COMPLETE/UNCERTAIN where exact final aggregation provenance is not explicit |
| Topology objective, edge budget, consensus | Mechanism decomposition and supplement ablations | `.\scripts\run_cached_ablations.ps1 -Stage all -Device cuda -NoPlot` | `results/topology_objective/`, `results/edge_budget/`, `results/consensus/`, paper figures | COMPLETE, static evidence retained |
| Active-party protocol | Label-holder and no-passive-label-access caveat | `.\scripts\run_active_party_protocol.ps1`; hard-only variant `.\scripts\run_active_party_protocol.ps1 -HardOnly` | `results/active_party_protocol/` and final analysis outputs | COMPLETE, static evidence retained |
| Deployment objective | Active/local/joint deployment frontier | `.\scripts\run_deployment_objective.ps1 -Device cuda -StrictCache` | `results/deployment_objective/`, `figures/deployment_tradeoff.png` | COMPLETE, static evidence retained |
| Reviewer protocols | Label budget, topology frequency, strict-label training, scalability | `.\scripts\run_reviewer_protocols.ps1 -Stage all` | `results/validation_label_budget/`, `results/topology_frequency/`, `results/strict_label_training/`, `results/party_scalability/` | COMPLETE, static evidence retained |
| Strong active-party label-holder | Strict label-holder and missing-party checks | `.\scripts\run_strong_setting.ps1 -Device cuda -StrictCache` | `results/strong_setting/` | COMPLETE, static evidence retained |
| Minimum-link sweep | `m=0` / edge-count audit | `.\scripts\run_cached_minedges_sweep.ps1 -Device cuda -Stage all` | `results/minedge_sweep/`, `outputs/minimum_link_audit/` | COMPLETE, static evidence retained |
| K sensitivity | Consensus-depth sensitivity | `experiments\run_k_sensitivity.py`; exact executed commands are recorded in `results/k_sensitivity_run_manifest.json` | `results/k_sensitivity_runs/` | COMPLETE, command wrapper UNCERTAIN |
| Nested weak-party scaling | Weak-party scaling supplement result | `experiments\run_nested_weak_scaling.py` | `results/nested_weak_scaling/` | COMPLETE, exact command UNCERTAIN |
| MovieLens five-party | Real-world AUC/communication evidence | `& $PY experiments\movielens\run_movielens_tadvfg.py --data-dir data\movielens\ml-1m --output-dir outputs\movielens --seeds 42,43,44,45,46 --num-parties 5 --setting five --local-epochs 8` | `outputs/movielens/` | COMPLETE, leakage-safe rerun retained |
| MovieLens 15-party Main | Real-world scaling evidence | `& $PY experiments\movielens\run_movielens_tadvfg.py --data-dir data\movielens\ml-1m --output-dir outputs\movielens_main15 --seeds 42,43,44,45,46 --setting main --local-epochs 8` | `outputs/movielens_main15/` | COMPLETE, leakage-safe rerun retained |
| MovieLens 15-party Hard | Real-world stress evidence | `& $PY experiments\movielens\run_movielens_tadvfg.py --data-dir data\movielens\ml-1m --output-dir outputs\movielens_hard15 --seeds 42,43,44,45,46 --setting hard --local-epochs 8` | `outputs/movielens_hard15/` | COMPLETE, leakage-safe rerun retained |
| MovieLens figures/package | Paper-facing MovieLens figures and values | `& $PY experiments\movielens\plot_movielens_results.py --output-dir outputs\movielens`; `experiments\movielens\package_final_results.py` | `outputs/movielens_final_package/`, `figures/movielens_*.png` | COMPLETE, static evidence retained |
| Alignment reference | Stronger-information reference study, not deployable baseline | `experiments\alignment_references.py`, `experiments\package_acmhard5_alignment.py` | `outputs/alignment_references_acmhard5/`, `figures/alignment_acmhard5_*.pdf` | COMPLETE, command details UNCERTAIN |

## Protocol Caveats To Preserve

- Oracle Matched is a post-hoc/test-mean diagnostic and must not be described as deployable.
- Stronger-information alignment references are reference studies, not same-information baselines.
- Active-party settings keep labels at the active party; they do not claim cryptographic privacy.
- Passive parties may exchange logits or receive logit gradients only in the strict-label-training variant described by the paper.
- MovieLens uses chronological splits and train-only feature statistics in the current implementation.
- Statistical claims require the paired statistics already present in retained result files; this cleanup did not recompute them.

## Static Validation Scope

This repository cleanup verified paths and structure only. Static reproducibility structure verified. Experiments were not rerun.

## Data and Local Caches

Large datasets, processed data, prediction caches, and model checkpoints are intentionally not part of the normal Git payload. They remain local when present and are ignored by `.gitignore`.

Current local-only data/cache roots include `data/`, `data_hgb/`, `data_acm/`, `results/*cache*/`, and `outputs/**/cache/`. Exact public download or regeneration commands are recorded only where existing project documentation already provides them; otherwise the data source is marked UNCERTAIN rather than invented.

Compact evidence intended for Git consists of source code, configuration metadata, per-seed CSV/JSON summaries, final tables, final paper figures, and documentation.
