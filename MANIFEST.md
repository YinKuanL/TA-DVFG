# Manifest

## Core

- `main experiment/ta_dvfg_hgb_reliability.py`
- `models/movielens_parties.py`

## Experiment Runners

- `experiments/run_cached_core.py`
- `experiments/run_cached_ablations.py`
- `experiments/run_cached_minedges_sweep.py`
- `experiments/run_k_sensitivity.py`
- `experiments/run_nested_weak_scaling.py`
- `experiments/alignment_references.py`
- `experiments/run_experiments.py`
- `experiments/experiment_plan.py`

## Analysis and Audit

- `experiments/final_analysis.py`
- `experiments/final_claims_audit.py`
- `experiments/config_audit.py`
- `experiments/statistical_tests.py`
- `experiments/deployment_objective_analysis.py`
- `experiments/strong_setting_analysis.py`
- `experiments/aggregate_results.py`
- `experiments/merge_cached_baselines.py`
- `experiments/make_ablation_plots.py`
- `experiments/make_readable_paper_figures.py`

## MovieLens

- `experiments/movielens/prepare_movielens.py`
- `experiments/movielens/run_movielens_tadvfg.py`
- `experiments/movielens/extended_movielens_suite.py`
- `experiments/movielens/aggregate_15party.py`
- `experiments/movielens/plot_movielens_results.py`
- `experiments/movielens/package_final_results.py`
- `experiments/movielens/README.md`

## Tests

- `tests/test_experiment_plan.py`
- `tests/test_reviewer_defense.py`

## Metadata

- `metadata/config_manifest.json`
- `metadata/paper_values.csv`
- `metadata/EXPERIMENT_STATUS.md`
- `metadata/headline_run_config_audit.csv`
- `metadata/headline_run_config_audit.json`
- `metadata/CONFIG_AUDIT_SUMMARY.md`
- `metadata/k_sensitivity_summary.csv`
- `metadata/k_sensitivity_raw.csv`
