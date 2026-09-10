# Result Provenance

This map records the current evidence chain. Paper values are authoritative and
are not copied back into raw result files. Paths under ignored `results/` and
`outputs/` are source evidence for package construction, not Git-tracked claims.

## Main paper

| Artifact | Source/config | Raw input | Summary / generator | Expected artifact | Seeds / metric / reported values | Validation |
|---|---|---|---|---|---|---|
| Table `tab:hgb-main` | HGB engine + `experiment_plan.py::reviewer_safe_jobs`; headline audited defaults | `results/reviewer_safe_shuffled/**/metrics.csv`, `main experiment/core/*/*.csv` | `experiments/final_analysis.py`; `results/final_analysis/tables/paper_main_*.csv` | paper table in `main.tex` | 42--46; Test@BestVal and Macro-F1; exact values parsed by verifier | `python scripts/verify_reported_values.py` |
| Table `tab:active-only-boundary` | mechanism replay / default configs | `outputs/round2_closure/task3/task3_per_seed.csv`, task4 CSV | task3/task4 reports | paper table | 42--46; active accuracy, communication | reported-value verifier |
| Table `tab:movielens-main` | MovieLens leakage-safe configs | `outputs/movielens_final_package_leakage_safe_20260705/raw_results/*.csv` | `extended_movielens_suite.py`, `aggregate_15party.py` | paper table | 42--46; ROC-AUC and scalars communicated | reported-value verifier |
| Figure `fig:movielens-evidence` | same | same raw MovieLens CSV | `extended_movielens_suite.py` | `paper/source/figures/movielens_pareto_auc_communication.png`, `movielens_party_val_test_auc.png` | paired seed AUC/communication and party val/test AUC | regenerate to staging and hash/semantic check |
| Figure `fig:mechanism-deployment` | deployment objective + mechanism evidence | `results/deployment_objective_runs/**/metrics.csv`, mechanism task CSV | `deployment_objective_analysis.py`, `final_analysis.py` | `topology_objective_ablation.pdf`, `deployment_tradeoff.png` | 42--46; active/local objective tradeoff | regenerate to staging and compare values |
| Figure `fig:overview` | unknown | missing | unknown | `paper/source/figures/method_overview.png` | not numerical | **missing at inventory time** |

## README staging figures and tables

The `staging/results-readme` branch curates a compact public README from the current manuscript source of truth without publishing the manuscript PDF.

| README artifact | Source |
|---|---|
| HGB main table | Manuscript-provided `tab:hgb-main` values copied without recomputation or rounding changes |
| Joint deployment table | Manuscript-provided active/local/link values copied without recomputation or rounding changes |
| MovieLens table | Manuscript-provided MovieLens ROC-AUC and communication values copied without recomputation or rounding changes |
| `figures/ta_dvfg_movielens_pareto.png` | Copied from `paper/source/figures/movielens_pareto_auc_communication.png` |
| `figures/ta_dvfg_deployment_tradeoff.png` | Copied from `paper/source/figures/deployment_tradeoff.png` |
| `figures/ta_dvfg_topology_objective_ablation.pdf` | Copied from `paper/source/figures/topology_objective_ablation.pdf` |

No dataset, checkpoint, cache, generated result directory, package archive, or manuscript PDF is added by the README staging branch.

## Supplement tables and figures

| Label | Script / configuration | Raw input | Summary | Expected output / reported values | Validation |
|---|---|---|---|---|---|
| `tab:supp-complete` | HGB engine, comparison methods, audited headline configs | reviewer-safe/core seed CSV | final-analysis main and communication tables | inline table: HGB Main/Hard methods | reported-value verifier |
| `tab:supp-exchange-decomposition` | `outputs/round2_closure/task1_2_runner.py` | paper-eval replay NPZ bundles | task2 per-seed/stats/report | inline table; same-state active/local deltas | replay consistency CSV + verifier |
| `tab:supp-subset-baselines` | mechanism task3 generator (not stably tracked initially) | replay bundles | task3 per-seed/stats/report | inline Top-Reliability-q / Greedy-Subset-q | task3 consistency checks + verifier |
| `tab:supp-degree-decomposition` | mechanism task4 generator (not stably tracked initially) | replay bundles | task4 per-seed/stats/report | inline degree same-state/reselection | task4 consistency checks + verifier |
| `tab:supp-objective-acm`, `tab:supp-objective-dblp` | `deployment_objective_analysis.py`; objective jobs including lambda .25/.5/.75 | deployment objective metrics CSV | deployment summary CSV | inline active/local/joint metrics | reported-value verifier |
| `tab:supp-weak-scaling` | `run_nested_weak_scaling.py`; weak 0/2/5/10/15 | `results/nested_weak_cache/{acm,dblp}/*.pt` | expected raw/summary/paired-delta CSV | inline means and link counts | **raw/summary missing initially** |
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

## Configuration and selection provenance

- HGB headline values: saved `metrics_config.json` is primary; job/command text is
  secondary; engine CLI defaults are not evidence for historical headline runs.
- Best epoch: validation improves strictly (`>`); the test prediction from that
  same epoch is reported. Ties retain the earlier best epoch.
- Best topology: replayed best-validation state is authoritative. A historical
  `selected_edges` field stored final/last-refresh edges and is not topology truth.
- MovieLens: validation selects reliability/topology; test labels are evaluated
  only after the state is fixed.
- Alignment: fresh hidden-export caches are separate from HGB cached-core caches.

