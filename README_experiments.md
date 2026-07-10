# TA-DVFG Round 2 Closure Experiments

This repo now includes cached-prediction closure scripts for the reviewer-facing Round 2 tasks.

## Bundle Contract

The runnable scripts consume `.npz` bundles with:

- `val_probs`: `[N,V,C]` or binary `[N,V]`
- `test_probs`: `[N,T,C]` or binary `[N,T]`
- `y_val`: `[V]`
- `y_test`: `[T]`
- `dataset`: scalar string
- `setting`: scalar string
- `seed`: scalar int
- `metric`: `accuracy` or `roc_auc`

Use `experiments/export_bundle_template.py` as the adapter point if the canonical cache format differs.

## Runnable Cached-Prediction Tasks

Each script accepts `bundles_dir`, `--out`, and `--smoke`.

```powershell
python experiments/task1_steps0_exact.py --smoke --out outputs/round2_closure_smoke
python experiments/task2_exchange_decomposition.py --smoke --out outputs/round2_closure_smoke
python experiments/task3_subset_baseline.py --smoke --out outputs/round2_closure_smoke
python experiments/task4_degree_weight_ablation.py --smoke --out outputs/round2_closure_smoke
python experiments/task8_no_collaboration.py --smoke --out outputs/round2_closure_smoke
```

For real bundles:

```powershell
python experiments/task1_steps0_exact.py path/to/bundles --out outputs/round2_closure
```

Each task writes `*_per_seed.csv` and `*_paired_stats.csv`.

## Review-Closure Tasks

`experiments/task5_audit_baselines.py` writes a grep-based implementation audit to `outputs/round2_closure/task5_baseline_audit.md` and a companion CSV.

`experiments/task5_rerun_baselines.py`, `experiments/task6_strict_matched_controls.py`, and `experiments/task7_validation_reuse.py` are adapter-backed runners. They intentionally require `ProjectAdapter` methods to be filled after the audit identifies the canonical engine commands.

## MovieLens and Leakage Utilities

```powershell
python experiments/task9_movielens_stats.py path/to/movielens.csv --reference TA-DVFG --baseline BestSingle
python experiments/task10_label_leakage.py path/to/leakage_probe.npz
```

Task 9 expects `seed, method, auc, communication`. Task 10 expects `y`, `grad`, and optionally `logits`.
