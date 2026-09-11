# Result Provenance

This map records the current evidence chain. Paper values are authoritative and are not copied back into raw result files. Paths under ignored bulk output directories are source evidence for package construction, not Git-tracked claims.

## Curated Public Results

| Artifact | Source | Notes |
|---|---|---|
| `results/tables/hgb_main_results.csv` | Manuscript `tab:hgb-main` values and the checked README table | Values copied without recomputation or rounding changes. |
| `results/tables/movielens_results.csv` | Manuscript MovieLens table values and verified communication totals from tracked metadata | The public table reports the manuscript ROC-AUC values and scalar counts used for the README. |
| `results/tables/joint_deployment_results.csv` | Manuscript deployment-objective table and supplementary communication decomposition | Includes active/local accuracy, selected links, peer/readout/control scalar counts, and total scalars. |
| `results/tables/k_sensitivity_summary.csv` | Copied from tracked `metadata/k_sensitivity_summary.csv` | Compact sensitivity summary already present in the clean package. |
| `results/figures/method_overview.png` | Copied unchanged from `figures/ta_dvfg_overview.png`; originally copied from the local paper-source checkout's `paper/source/figures/method_overview.png` | Manuscript method overview. |
| `results/figures/movielens_auc_communication.png` | Copied unchanged from `figures/ta_dvfg_movielens_auc_communication.png`; source `paper/source/figures/movielens_pareto_auc_communication.png` | AUC-communication trade-off panel of the manuscript MovieLens evidence figure. |
| `results/figures/movielens_party_val_test_auc.png` | Copied unchanged from `figures/ta_dvfg_movielens_party_val_test_auc.png`; source `paper/source/figures/movielens_party_val_test_auc.png` | Individual predictor validation/test panel of the manuscript MovieLens evidence figure. |
| `results/figures/topology_objective_ablation.pdf` | Copied unchanged from `figures/ta_dvfg_topology_objective_ablation.pdf`; source `paper/source/figures/topology_objective_ablation.pdf` | Topology-objective panel of the manuscript mechanism/deployment evidence. |
| `results/figures/deployment_objective_tradeoff.png` | Copied unchanged from `figures/ta_dvfg_deployment_objective_tradeoff.png`; source `paper/source/figures/deployment_tradeoff.png` | Active/local deployment-objective trade-off panel. |

No dataset, checkpoint, cache, generated result directory, package archive, or manuscript PDF is added by the public-results cleanup.

## Main Paper

| Artifact | Source/config | Raw input | Summary / generator | Expected artifact | Seeds / metric / reported values | Validation |
|---|---|---|---|---|---|---|
| Table `tab:hgb-main` | HGB engine + `experiment_plan.py::reviewer_safe_jobs`; headline audited defaults | `results/reviewer_safe_shuffled/**/metrics.csv`, `main experiment/core/*/*.csv` | `experiments/final_analysis.py`; `results/final_analysis/tables/paper_main_*.csv` | paper table in `main.tex` | 42--46; Test@BestVal and Macro-F1; exact values parsed by verifier | `python scripts/verify_reported_values.py` |
| Table `tab:active-only-boundary` | mechanism replay / default configs | `outputs/round2_closure/task3/task3_per_seed.csv`, task4 CSV | task3/task4 reports | paper table | 42--46; active accuracy, communication | reported-value verifier |
| Table `tab:movielens-main` | MovieLens leakage-safe configs | `outputs/movielens_final_package_leakage_safe_20260705/raw_results/*.csv` | `extended_movielens_suite.py`, `aggregate_15party.py` | paper table | 42--46; ROC-AUC and scalars communicated | reported-value verifier |
| Figure `fig:movielens-evidence` | same | same raw MovieLens CSV | `extended_movielens_suite.py` | `paper/source/figures/movielens_pareto_auc_communication.png`, `movielens_party_val_test_auc.png` | paired seed AUC/communication and party val/test AUC | regenerate to staging and hash/semantic check |
| Figure `fig:mechanism-deployment` | deployment objective + mechanism evidence | `results/deployment_objective_runs/**/metrics.csv`, mechanism task CSV | `deployment_objective_analysis.py`, `final_analysis.py` | `topology_objective_ablation.pdf`, `deployment_tradeoff.png` | 42--46; active/local objective tradeoff | regenerate to staging and compare values |
| Figure `fig:overview` | paper source figure asset | not numerical | manual manuscript figure export | `paper/source/figures/method_overview.png` | not numerical | source asset located in the local paper-source checkout |

## Supplement Tables and Figures

| Label | Script / configuration | Raw input | Summary | Expected output / reported values | Validation |
|---|---|---|---|---|---|
| `tab:supp-complete` | HGB engine, comparison methods, audited headline configs | reviewer-safe/core seed CSV | final-analysis main and communication tables | inline table: HGB Main/Hard methods | reported-value verifier |
| `tab:supp-exchange-decomposition` | `outputs/round2_closure/task1_2_runner.py` | paper-eval replay NPZ bundles | task2 per-seed/stats/report | inline table; same-state active/local deltas | replay consistency CSV + verifier |
| `tab:supp-subset-baselines` | mechanism task3 generator (not stably tracked initially) | replay bundles | task3 per-seed/stats/report | inline Top-Reliability-q / Greedy-Subset-q | task3 consistency checks + verifier |
| `tab:supp-degree-decomposition` | mechanism task4 generator (not stably tracked initially) | replay bundles | task4 per-seed/stats/report | inline degree same-state/reselection | task4 consistency checks + verifier |
| `tab:supp-objective-acm`, `tab:supp-objective-dblp` | `deployment_objective_analysis.py`; objective jobs including lambda .25/.5/.75 | deployment objective metrics CSV | deployment summary CSV | inline active/local/joint metrics | reported-value verifier |
| `tab:supp-weak-scaling` | `run_nested_weak_scaling.py`; weak 0/2/5/10/15 | `results/nested_weak_cache/{acm,dblp}/*.pt` | expected raw/summary/paired-delta CSV | inline means and link counts | raw/summary missing initially |
| `tab:supp-movielens-interfaces` | `models/movielens_parties.py`, architecture audit | MovieLens train data / party specs | party architecture audit CSV | inline five-interface descriptions | static verifier |
| `tab:supp-movielens-summary` | MovieLens runner | leakage-safe per-seed CSV | summary/Pareto CSV | inline AUC/communication | reported-value verifier |
| `tab:supp-alignment-acmhard5` | `alignment_references.py` | five fresh frozen ACM Hard caches | alignment per-seed/summary/stats | inline accuracy/F1/CI/p/W-T-L | equivalence and value verifier |
| `fig:supp-alignment-diagnostics` | `outputs/alignment_references_acmhard5/make_alignment_acmhard5_figures.py` | alignment per-seed CSV | same | two protected PDF figures | regenerate to staging |
| `tab:supp-alignment-boundaries` | alignment script metadata | communication/parameter-count CSV | boundary report | inline payload/parameter table | static/value verifier |
| `tab:supp-active-label-audits` | active-party and strict-label jobs | protocol result CSV | final-analysis protocol tables | inline ACM/DBLP results | reported-value verifier |
| `tab:supp-k-sensitivity` | `run_k_sensitivity.py`; K=1/2/3 | `metadata/k_sensitivity_raw.csv` | `metadata/k_sensitivity_summary.csv`, results TeX/CSV | inline K table | reported-value verifier |
| `tab:supp-communication` | `estimate_communication` + final analysis | ACM Main seed CSV | paper communication table | inline 136125, 45375, 2041875, 152460 totals | formula and value tests |
| `fig:supp-label-budget` | final analysis label-budget/frequency plotting | corresponding result metrics | final-analysis tables | protected PNG | regenerate to staging |
| `tab:supp-scalability` | party scalability jobs | `results/party_scalability/**/metrics.csv` | final-analysis scalability table | inline N=5/10/15/20/30 values | reported-value verifier |
| `tab:supp-minlink-audit` | cached min-edge sweep plus first-edge audit | min-edge metrics / replay bundles | min-edge table | inline edge counts and first-edge gain ranges | partial until generator stabilized |
| `fig:supp-edge-budget` | edge budget jobs; `make_readable_paper_figures.py` | edge budget metrics CSV | final-analysis edge table | protected ACM/DBLP edge PDFs | regenerate to staging |

## Configuration and Selection Provenance

- HGB headline values: saved `metrics_config.json` is primary; job/command text is secondary; engine CLI defaults are not evidence for historical headline runs.
- Best epoch: validation improves strictly (`>`); the test prediction from that same epoch is reported. Ties retain the earlier best epoch.
- Best topology: replayed best-validation state is authoritative. A historical `selected_edges` field stored final/last-refresh edges and is not topology truth.
- MovieLens: validation selects reliability/topology; test labels are evaluated only after the state is fixed.
- Alignment: fresh hidden-export caches are separate from HGB cached-core caches.
