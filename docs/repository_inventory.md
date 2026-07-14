# TA-DVFG Repository Inventory

Inventory date: 2026-07-14. Starting branch: `main`. Starting commit:
`febe05ea8079d3a4f70673dd62625480b79099f7`. Work branch:
`repo/supplement-reproducibility-cleanup`.

## Scope and preservation record

This inventory was completed before any repository file was moved or rewritten.
The initial worktree already contained modifications to all three protected paper
sources and untracked paper PDFs, TeX drafts, package-builder material, a package
tree, and `github_upload_TA_DVFG/`. Those changes predate this cleanup and are not
staged or altered by it. Initial protected-file SHA-256 values are:

| File | Initial SHA-256 |
|---|---|
| `paper/source/main.tex` | `39688fe7880a6f89cfbb5f91a4834228aff5e05e2f9bd039dadee5b873b654e4` |
| `paper/source/supplementary.tex` | `c6fbcc33b03ac532fe2673fc423ab3c37e0b53beaac88fccef109dadd463f632` |
| `paper/source/ReproducibilityChecklist.tex` | `e7c205b1e48075199f5a14927cfbb5283580fb0fc1d457431547c844338fbe` |
| `paper/source/references.bib` | `1889747b93e6d3624dd3ebab321fd327cca0f4674462b9f2a31e19f51f88adc0` |

## Concise tree before changes

```text
TA-DVFG/
|-- paper/source/               canonical TeX, bibliography, style, paper figures
|-- main experiment/            monolithic HGB engine and historical run logs
|-- models/                     MovieLens party models
|-- experiments/                HGB, MovieLens, alignment, audit, analysis runners
|-- tests/                      existing experiment-plan/reviewer-defense tests
|-- metadata/                   tracked configuration and paper-value audits
|-- results/                    ignored HGB raw/summary results and large caches
|-- outputs/                    ignored MovieLens/alignment/mechanism evidence
|-- data/, data_hgb/, data_acm/ ignored datasets and replay bundles
|-- multi-modalities/           ignored duplicate HGB dataset material
|-- dist/                       untracked prior package builds
|-- github_upload_TA_DVFG/      untracked historical repository snapshot
|-- TA-DVFG_AAAI27_Code_Package_Builder/  untracked historical builder/audit
|-- docs/                       paper synchronization note
|-- .github/workflows/          paper-review automation
|-- README.md, REPRODUCE.md, MANIFEST.md, requirements.txt
`-- untracked paper PDFs and alternate TeX drafts
```

Approximate pre-change sizes were `results/`: 1,734 files / 13.5 GiB,
`outputs/`: 380 files / 6.9 GiB, datasets: 3.5 GiB, and existing `dist/`: 98 MiB.
Large caches are retained locally but are not required for Level-1 verification.

## File-level source inventory

The table records all tracked research-code and documentation files. `Exec` means
the file has a direct CLI. `Pkg` means it is required in the reproducibility ZIP.
Inputs and outputs are repository-relative. A dash means not applicable.

| Class | Path | Purpose / supplement support | Inputs -> outputs | Exec | Pkg | Historical / duplicate / move risk |
|---|---|---|---|:---:|:---:|---|
| A | `paper/source/main.tex` | Canonical main paper; all headline claims | TeX assets -> paper PDF | yes | no | Protected; dirty before cleanup; never move |
| A | `paper/source/supplementary.tex` | Primary specification for all requested coverage | TeX assets -> supplement PDF | yes | no | Protected; dirty before cleanup; never move |
| A | `paper/source/ReproducibilityChecklist.tex` | Canonical AAAI checklist | TeX assets -> checklist PDF | yes | no | Protected; dirty before cleanup; never move |
| A | `paper/source/references.bib` | Canonical bibliography | citations -> bibliography | no | no | Protected; never move |
| A | `paper/source/aaai2027.sty`, `paper/source/aaai2027.bst` | AAAI template/style | TeX -> formatted PDF | no | no | Protected template files |
| A/K | `paper/source/figures/*` | Existing paper figures | cached summaries -> paper assets | no | no | Protected; do not regenerate in place |
| B/C/D | `main experiment/ta_dvfg_hgb_reliability.py` | HGB loading, party construction/training, cache replay, reliability, consensus, readouts, topology selection, baselines, label location, communication, Test@BestVal | HGB/caches/config -> seed CSV, summaries, figures, caches | yes | yes | Public path used by every runner; moving breaks imports and recorded commands |
| C/G | `models/movielens_parties.py` | Five heterogeneous MovieLens predictor families and training/prediction helpers | train-only features -> probabilities | no | yes | Imported by MovieLens runner; explicit supplement path |
| D/F | `experiments/run_cached_core.py` | ACM/DBLP/IMDB Main/Hard cached-core orchestration | HGB caches/config -> per-seed CSV | yes | yes | Uses canonical experiment-plan defaults |
| D/F/H/J | `experiments/experiment_plan.py` | Audited suite/job manifests for core, baselines, ablations, deployment, robustness, scalability | suite name -> deterministic job list | no | yes | Canonical defaults override older engine CLI defaults |
| D | `experiments/run_experiments.py` | Generic job executor and provenance recorder | job plan -> commands/logs/CSV | yes | yes | Do not change command ordering |
| D/H | `experiments/run_cached_ablations.py` | Cache reuse for planned sensitivity suites | caches/job plan -> result trees | yes | yes | Full execution is Level 3 |
| H/J | `experiments/run_cached_minedges_sweep.py` | `m` and selector sanity sweep | caches -> raw/summary CSV | yes | yes | Required for `m=0` audit |
| H/J | `experiments/run_k_sensitivity.py` | Independent K=1/2/3 replay, reachability, communication, figures/tables | cached trajectories -> raw/summary/table/figures | yes | yes | Exact supplement-claimed path |
| H | `experiments/run_nested_weak_scaling.py` | Nested 3/5/8/13/18-party ACM/DBLP comparison | 18-party caches -> seed/summary CSV, plots, manifest | yes | yes | Exact supplement-claimed path; full result CSV was absent initially |
| G | `experiments/movielens/prepare_movielens.py` | Optional GroupLens download/extraction | network -> MovieLens files | yes | yes | Dataset download prohibited in this cleanup |
| G | `experiments/movielens/run_movielens_tadvfg.py` | Train-only preparation, five/15-party runs, AUC and communication | MovieLens/caches -> per-seed/summary/topology JSON | yes | yes | Exact supplement-claimed path |
| G/K | `experiments/movielens/plot_movielens_results.py` | Base MovieLens figures from existing CSV | MovieLens CSV -> PNG | yes | yes | Safe Level-1 generation |
| G/K | `experiments/movielens/extended_movielens_suite.py` | Architecture audit, paired deltas, Pareto figures/tables | existing MovieLens CSV -> CSV/PNG/TeX | yes | yes | Safe cached-output analysis |
| G/K | `experiments/movielens/aggregate_15party.py` | Combine Main/Hard 15-party outputs | two result dirs -> combined CSV/PNG | yes | yes | No retraining |
| G/K | `experiments/movielens/package_final_results.py` | Historical MovieLens-only packaging | output dirs -> focused package | yes | no | Duplicates new repository-wide builder |
| G | `experiments/movielens/README.md` | MovieLens protocol and commands | - | no | yes | Contains a personal interpreter path; must not enter final ZIP unchanged |
| I | `experiments/alignment_references.py` | Fresh hidden export, latent fusion references, equivalence audit, statistics/figures | fresh frozen predictor cache -> raw/summary/stats/figures | yes | yes | Kept separate from cached-core evidence |
| G | `experiments/deployment_objective_analysis.py` | Active/local/joint metric and trade-off summaries | deployment result CSV -> tables/figures | yes | yes | Analysis-only |
| K/L | `experiments/final_analysis.py` | Primary HGB aggregation, communication, statistics, tables, figures | result tree -> `results/final_analysis/` | yes | yes | Main cached-output regeneration entry point |
| K/L | `experiments/statistical_tests.py` | Paired t/Wilcoxon/CI/Cohen dz/Holm statistics | raw paired CSV -> statistics CSV | yes | yes | Confirmatory/exploratory distinction is documentary |
| K | `experiments/make_readable_paper_figures.py` | Selected edge-budget/objective figures | summary CSV -> PDF/PNG | yes | yes | Generates outside protected paper tree by default |
| K | `experiments/make_ablation_plots.py` | Legacy ablation plots | summary CSV -> PNG | yes | yes | Overlaps `final_analysis.py`; behavior not consolidated |
| K/L | `experiments/aggregate_results.py` | Legacy result aggregation | result CSV -> summaries | yes | yes | Partly duplicates `final_analysis.py`; retained |
| L | `experiments/config_audit.py` | Paper-default audit | saved configs -> audit CSV/MD | yes | yes | Detects engine/canonical default distinction |
| L | `experiments/final_claims_audit.py` | Paper-value and configuration audit | code/results -> metadata | yes | yes | Tracked metadata was stale at inventory time |
| L | `experiments/merge_cached_baselines.py` | Merge cache-compatible baseline outputs | config/cache CSV -> merged CSV | yes | yes | Preserves cached trajectories |
| H/L | `experiments/strong_setting_analysis.py` | Strict/readout/dropout robustness analysis | result tree -> summaries | yes | yes | Not a primary supplement table |
| M | `metadata/paper_values.csv` | Tracked paper-value reference | paper audit -> CSV | no | yes | Values protected |
| M | `metadata/headline_run_config_audit.csv`, `metadata/headline_run_config_audit.json` | 90-row HGB configuration evidence | saved configs -> audit | no | yes | Required evidence; no value edits |
| M | `metadata/k_sensitivity_raw.csv`, `metadata/k_sensitivity_summary.csv` | Tracked K sensitivity evidence | cached replay -> CSV | no | yes | Duplicates `results/k_sensitivity_*`; tracked copy is package source |
| M | `metadata/config_manifest.json` | Historical generated configuration manifest | audit -> JSON | no | no | Contains a personal absolute path and stale engine defaults |
| M | `metadata/CONFIG_AUDIT_SUMMARY.md` | PASS summary for 90 rows | audit CSV -> Markdown | no | yes | Current supplement support |
| M/N | `metadata/EXPERIMENT_STATUS.md` | 2026-07-04 status snapshot | audit observations -> Markdown | no | no | Historical and contradicted by later alignment/nested caches and supplement |
| K | `tests/test_experiment_plan.py` | Suite/default completeness regression tests | code -> test result | yes | yes | Existing tests |
| K | `tests/test_reviewer_defense.py` | Cache, topology, protocol, communication/readout tests | synthetic arrays -> test result | yes | yes | Existing tests |
| N | `MANIFEST.md` | Old tracked source manifest | repository -> Markdown | no | yes | Incomplete for later mechanism evidence |
| N | `README.md`, `REPRODUCE.md` | Paper-sync-centric documentation | - | no | yes | Must be rewritten for reviewers |
| N | `apply_selective_reviewer_edits.py`, `.github/workflows/*`, `.accept-evidence-push-trigger*` | Historical paper-review automation | paper/workflow state -> paper edits | yes | no | Out of reproducibility-package scope |
| N | `docs/PAPER_SYNC_PROTOCOL.md`, `paper/source/SYNC_MANIFEST.md`, `paper/AGENTS.md` | Paper synchronization governance | - | no | no | Retain; package exclusion |

## Evidence and generated-file inventory

Every file under the following rows inherits the row metadata. This family-level
notation is necessary because the ignored evidence trees contain more than 2,100
result/output files and many seed/job repetitions. Package inclusion is narrowed
later by the provenance manifest; no directory is deleted.

| Class | Current path / file family | Purpose / section | Inputs -> outputs | Exec | Pkg | Historical / duplicate / move risk |
|---|---|---|---|:---:|:---:|---|
| L | `results/reviewer_safe_shuffled/**/metrics*.csv`, `*_config.json` | HGB Main/Hard primary and comparison rows | cached probabilities -> seed metrics | no | selected | Primary raw/summary evidence; directory commands refer to current paths |
| L | `main experiment/core/*/*.csv`, `*_config.json` | Historical HGB headline runs including IMDB | local predictions -> seed metrics | no | selected | Primary configuration provenance despite location under source tree |
| L | `results/{active_party_evaluator,active_party_protocol,strict_label_training}/**` | Protocol robustness | caches -> seed/summary metrics | no | selected | Logs excluded; CSV/JSON retained |
| L | `results/{deployment_objective_runs,topology_objective,decentralized_readout}/**` | Active/local/joint deployments | caches -> metrics | no | selected | Objective analyses |
| L | `results/{validation_label_budget,topology_frequency,party_scalability,edge_budget}/**` | Efficiency/scalability/sweeps | caches -> metrics | no | selected | CSV/JSON evidence; cache/log files excluded |
| L | `results/{consensus,readout_ablation,protocol_sensitivity,noise_ratio,modality_ablation}/**` | Robustness/sensitivity | caches -> metrics | no | selected | Exploratory evidence |
| L | `data/round2_bundles_paper_eval/*.npz`, `data/round2_bundles/*.npz` | Best-validation trajectory replay | cached HGB trajectories -> mechanism replay | no | selected small bundles only | Paper-eval bundles are large; package feasibility assessed by builder |
| F/L | `results/{core_cache_v2,multimodal_cache_v3,nested_weak_cache,scalability_cache_v1,strict_label_cache_v1}/**/*.pt` | Local prediction caches | trained local models -> trajectory cache | no | no | Large (about 13.8 GiB total results); excluded unless indispensable |
| M/K | `results/final_analysis/data/*.csv`, `tables/*.csv`, `statistics/*.csv`, `figures/*` | HGB combined raw data, paper tables, tests, plots | result tree -> derived artifacts | no | selected | Derived, reproducible from included raw CSV |
| L/M | `results/k_sensitivity_*` | K=1/2/3 raw, summary, tables, manifest | cached replay -> CSV/TeX/MD | no | selected | Duplicated by tracked metadata for raw/summary |
| F/L/M | `outputs/movielens_leakage_safe_20260705/*.csv`, `*.json`, `*.png` | Five-party MovieLens | caches -> raw/summary/figures | no | selected | Primary leakage-safe output family |
| F/L/M | `outputs/{movielens_main15_leakage_safe_20260705,movielens_hard15_leakage_safe_20260705}/**` | 15-party Main/Hard MovieLens | caches -> raw/summary | no | selected | `.pt` caches excluded; CSV/JSON included |
| M/K | `outputs/movielens_final_package_leakage_safe_20260705/**` | Curated MovieLens raw/summary/appendix/main figures | above outputs -> package | no | selected | Script copies duplicate tracked sources; those copies excluded |
| I/L/M | `outputs/alignment_references/*.csv`, `*.json`, `*.md`, `*.png` | Alignment study raw/summary/equivalence/failed diagnostics | fresh cache -> evidence | no | selected | One-seed historical run; superseded for headline by ACM-hard-five tree |
| I/L/M | `outputs/alignment_references_acmhard5/*` and `fresh_within_suite_cache/*.pt` | ACM Hard five-seed latent study | fresh frozen predictors -> figures/table support | one plot script | selected | Raw per-seed CSV is not in this directory initially; some evidence remains in prior package tree |
| E/L/M | `outputs/round2_closure/task1*`, `task2*`, `task3*`, `task4*` | Provenance, K=0 interventions, subset and degree audits | replay bundles -> CSV/MD | one stranded runner | selected | Core mechanism code is incorrectly stored below ignored outputs |
| E | `outputs/round2_closure/step_c/*.py`, `step_c5/*.py` | Test@BestVal and topology provenance replay/audit | caches/bundles -> CSV/MD | yes | yes after organization | Must be copied, not moved, to stable analysis entry points |
| E | `outputs/round2_closure/task1_2_runner.py` | Same-state/fixed-trajectory/reselected K=0 mechanism audit | bundles -> task1/task2 CSV/MD | yes | yes after organization | Must be copied to stable analysis path without behavior edits |
| N | `outputs/reproducibility*`, `outputs/reviewer_closure_20260711`, `outputs/ac_review_closure` | Historical audit/reviewer reports | prior repository states -> reports | mixed | no | Superseded; retain locally |
| N/P | `outputs/vertibench_satellite`, `freebase_vertical*`, `imdb_vertical_replacement*` | Satellite/failed/replacement studies outside current supplement | datasets -> reports/results | mixed | no | Not claimed by latest canonical supplement |
| O | `**/__pycache__`, `*.pyc`, `*.log`, `.pytest_cache`, `dist/**` | Generated caches, logs, package builds | execution -> temporary/generated files | no | no | Ignore/exclude; do not use as provenance |
| N | `github_upload_TA_DVFG/**` | Historical upload snapshot | repository copy -> upload tree | mixed | no | Duplicates tracked code and metadata |
| N | `TA-DVFG_AAAI27_Code_Package_Builder/**` | Prior untracked builder and audit | repository -> earlier package | yes | no | Useful audit reference; replaced by tracked builder |
| N | `TA_DVFG_*revised.tex`, root paper PDFs | Alternate paper drafts/builds | TeX -> PDF | mixed | no | Untracked historical/generated; protected from cleanup |
| O | `data/**`, `data_hgb/**`, `data_acm/**`, `multi-modalities/**` | Downloaded/processed datasets | external datasets -> local storage | no | no | Dataset licenses/size preclude packaging |
| P | `paper/source/figures/method_overview.png` | Main-paper overview figure referenced by dirty `main.tex` | unknown -> PNG | no | unresolved | Referenced but absent anywhere in the repository at inventory time |

## Initial findings

1. All three supplement-explicit Python paths exist.
2. Core defaults in the engine are historical (`m=15`, `tau=0`), while the
   canonical job plan and all 90 audited headline configurations resolve to
   `m=5`, `tau=0.001`, `B=15`, `d_max=2`, `K=1`, `eta=0.85`.
3. Mechanism replay implementations exist but are ignored because they live
   under `outputs/round2_closure/`; stable, packageable entry points are needed.
4. Full nested weak-party caches exist for ACM/DBLP seeds 42--46, but the initial
   tree has no nested weak scaling raw/summary CSV matching the supplement table.
5. The five-seed ACM alignment cache exists. Evidence is split across output
   families and prior package material, so provenance must be consolidated.
6. `metadata/config_manifest.json` and the MovieLens README contain personal
   absolute paths and cannot be copied into the final package unchanged.
7. No `LICENSE` or `CITATION.cff` exists initially.
8. The dirty main paper references `paper/source/figures/method_overview.png`,
   which is absent. This cleanup does not create or alter paper figures.

