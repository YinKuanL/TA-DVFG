# Minimal package content map

This map describes the anonymous, dataset-free AAAI reproducibility package.
The package supports integrity checks and regeneration from bundled small
CSV/JSON evidence. It is not a claim that every full experiment can be rerun
without separately obtained public datasets and excluded prediction caches.

## Validation levels

| Level | Command | Training | Bundled support |
|---|---|---:|---|
| Integrity | `python scripts/verify_package.py` | No | Complete |
| Reported-value checks | `python scripts/verify_reported_values.py` | No | Complete for the checks implemented by the script |
| Tables/figures | `python scripts/regenerate_artifacts.py` | No | Complete for the selected regenerated artifacts |
| Unit/smoke tests | `python -m pytest tests/unit tests/smoke -q` | No | Complete |
| Full experiments | Commands in `REPRODUCE.md` | Yes | Documented only; not run during package construction |

All regenerated files are written under `artifacts/regenerated/`. Raw and
summary evidence is read-only input to the regeneration workflow.

## Code and configuration

| Claim family | Entry points | Explicit configuration |
|---|---|---|
| HGB core, topology, communication, readout | `core/ta_dvfg_hgb_reliability.py`, `experiments/run_cached_core.py`, `experiments/experiment_plan.py` | `configs/hgb/defaults.json`, `configs/headline/*/metrics_config.json` |
| Modality, label budget/location, party count, topology frequency and controls | `experiments/experiment_plan.py`, `experiments/run_cached_ablations.py`, `experiments/run_experiments.py` | `configs/hgb/defaults.json`, `configs/robustness/control_sensitivity.json` |
| K sensitivity | `experiments/run_k_sensitivity.py` | `configs/robustness/k_sensitivity.json` |
| Nested weak-party scaling | `experiments/run_nested_weak_scaling.py` | `configs/robustness/nested_weak_scaling.json` |
| MovieLens | `experiments/movielens/run_movielens_tadvfg.py`, aggregation and plotting scripts in the same directory | `configs/movielens/defaults.json` |
| Deployment objectives | `experiments/deployment_objective_analysis.py` | `configs/mechanisms/deployment_objectives.json` |
| Prediction-space alignment references | `experiments/alignment_references.py` | `configs/alignment/acm_hard.json` |
| Same-state exchange and provenance audits | `analysis/mechanisms/`, `analysis/provenance/` | Arguments documented by each entry point |

The historical audit implementations are package-local under the corresponding
`_historical/` directories. Their large replay bundles are intentionally not
included. The package builder rewrites all available experiment runners to use
the package-local `core/` engine and bundled headline configurations.

## Evidence map

| Area | Seed-level/raw evidence | Derived evidence | Status |
|---|---|---|---|
| HGB core and ablations | `artifacts/raw/hgb/seed_results.csv` | `artifacts/summaries/hgb/` | Complete for cached claim suites in the compact evidence file |
| HGB headline accuracy/communication | `artifacts/raw/hgb/paper_main_accuracy.csv`, `paper_communication_efficiency.csv` | matching files in `artifacts/summaries/hgb/tables/` | Complete for fixed-value verification |
| K sensitivity | `artifacts/raw/k_sensitivity/k_sensitivity_raw.csv` and `metadata/k_sensitivity_raw.csv` | `artifacts/summaries/k_sensitivity/` and `metadata/k_sensitivity_summary.csv` | Complete |
| MovieLens | `artifacts/raw/movielens/*per_seed.csv` and topology JSON | bundled summary CSVs | Complete for five-party and fifteen-party reported checks |
| Same-state and provenance mechanisms | `artifacts/raw/mechanisms/task*_per_seed.csv` plus audits | `artifacts/summaries/mechanisms/` | Complete for bundled statistical/value checks; replay bundles excluded |
| Alignment references | `artifacts/raw/alignment/` | bundled summary/statistics CSVs | Partial: the available ACM CSV contains one seed; DBLP hard five-seed evidence is included as supporting material |
| Nested weak-party scaling | none | none | Missing: runner and config are included, but no small raw/summary output was available for packaging |
| Multimodal HGB and modality ablation | compact seed evidence in `artifacts/raw/hgb/seed_results.csv` | bundled HGB tables | Partial: the experiment plan and results are included, but the separately referenced historical multimodal engine is absent from the repository |
| Subset and degree audits | `artifacts/raw/mechanisms/task3_*` and `task4_*` | bundled reports/statistics | Complete code/evidence: the exact historical combined generator is package-local; large replay bundles are excluded |

The partial and missing rows are intentionally not reconstructed from manuscript
text. No reported values are synthesized, altered, or rounded by the builder.

## Exclusions

The package excludes manuscript sources, PDFs, citation metadata, datasets,
downloaded data, notebooks, model checkpoints, prediction caches, generated
Python caches, logs, credentials, repository history, workstation paths, and
identity-bearing URLs or metadata. These exclusions are checked against the
manifest by `scripts/verify_package.py`.

## Integrity files

`MANIFEST.txt` records SHA-256, byte size, and package-relative path for every
payload file. `SHA256SUMS.txt` records every payload hash plus the manifest hash.
The ZIP SHA-256 is stored beside the archive in `ZIP_SHA256.txt` so that it does
not create a self-referential checksum.
