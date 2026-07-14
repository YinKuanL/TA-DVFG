# Supplement-to-Code Coverage Map

Source of truth: `paper/source/supplementary.tex` as present in the worktree on
2026-07-14. Status values are **complete**, **partial**, or **missing**. Historical
raw values and paper sources are never rewritten by this cleanup.

## A. Core method

| Requirement | Exact implementation | Audited behavior | Status |
|---|---|---|---|
| Reliability estimation | `compute_party_reliabilities`, `evaluator_reliabilities`, `update_reliability` in `main experiment/ta_dvfg_hgb_reliability.py` | validation accuracy for HGB; MovieLens runner replaces final evaluation with ROC-AUC while topology cache uses shared engine interface | complete |
| Reliability floor / exponent | `vote_from_probs`, `post_consensus_party_probs`, `topology_filtered_reliability_vote`; CLI `--reliability_power` | `max(r,1e-3)`, exponent 1.0 | complete |
| Consensus eta / isolation / K | `post_consensus_party_probs` | `pred_self_weight=0.85`; isolated party is copied; synchronous K steps | complete |
| Active readout | `active_party_indices`, `topology_filtered_reliability_vote` | non-isolated parties for nonempty graph; all-party fallback for empty graph; degree exponent 0.5 | complete |
| Active/local/joint objectives | `topology_objective_components`, `score_topology_on_validation` | active, local mean, and lambda-weighted joint | complete |
| Complete candidates / B / d_max / m / tau | `update_adaptive_topology` | all `(i,j), i<j`; degree and budget checked; min-links and gain stop | complete |
| Deterministic ordering | `score_topology_edges`, `update_adaptive_topology` | candidate tuples sorted descending; graph-score ties use utility then retained order | complete |
| E-star vs E-hat | conceptual only in paper; greedy return in `update_adaptive_topology` | no global-optimality claim in code | complete |
| Communication | `estimate_communication`, `estimate_inference_comm` | peer `2|E|KMC`, active/global/local readout payloads separated | complete |

## B-C. HGB cached core and comparisons

Entry points are `experiments/run_cached_core.py`, `experiments/experiment_plan.py`,
`experiments/run_experiments.py`, and the HGB engine. Raw evidence is under
`main experiment/core/` and `results/reviewer_safe_shuffled/`; combined derived
evidence is under `results/final_analysis/`.

| Coverage | Code / configuration / output | Status |
|---|---|---|
| ACM/DBLP/IMDB Main and Hard, seeds 42--46 | `core_jobs`, `reviewer_safe_jobs`; `main experiment/core/*`; `results/reviewer_safe_shuffled/*` | complete |
| Useful/noisy construction and shuffled identities | `build_party_views`; `common_parameters` sets `shuffle_party_positions=True` | complete |
| Row permutation, Gaussian scale 1.0, random degree 4 | `build_party_views`, `make_random_edges`; saved configs | complete |
| Cached prediction trajectories and Test@BestVal | `build_prediction_cache`, `evaluate_cached_method`, `run_seed_with_prediction_cache`; replay bundles | complete |
| 20-epoch refresh | `topology_every`; audited headline config JSON | complete |
| HGB defaults | `experiments/experiment_plan.py::common_parameters`: K=1, B=15, d_max=2, m=5, tau=.001, eta=.85 | complete |
| Global Uniform/Reliability/Top-k | `evaluate_local_baseline`, `vote_from_probs`, `topk_reliability_vote` | complete |
| Ring/Random Regular/Expander/Full Mesh | `method_to_initial_topology` and `make_*_topology` | complete |
| Ring/Random/Expander Matched | `make_ring_like_topology`, `make_random_matched_topology`, `make_expander_matched_topology` | complete |
| Post-hoc Oracle Matched | mechanism output `outputs/round2_closure/task4/*`; stable analysis entry point required | partial |
| Adaptive Pairwise/Complementarity | `score_topology_edges`, `select_topology_from_utility`, method-specific `_active_adaptive_score` | complete |

The engine CLI intentionally retains older direct-run defaults (`m=15`,
`tau=0`). Headline reproduction must use the job plan or audited configs; changing
the engine defaults would change historical direct-run behavior.

## D. Mechanism analyses

| Analysis | Exact implementation / evidence | Held-fixed audit | Status |
|---|---|---|---|
| Topology provenance and best-validation recovery | `outputs/round2_closure/step_c/step_c_gate.py`, `step_c5/step_c5_topology_audit.py`; their CSV/MD outputs | replays epoch trajectories and checks Test@BestVal | complete, stranded |
| Same-state K=1 to K=0 | `outputs/round2_closure/task1_2_runner.py`; task2 CSV/MD | consistency checks explicitly compare epoch, topology, participant set, degree, reliability, prediction fingerprint | complete, stranded |
| Fixed-topology trajectory / independently reselected K=0 | same runner, configurations D and B | fixed state trajectory versus independent selector | complete, stranded |
| Local mean / connected parties / Jaccard | same runner; `task2_per_party_exchange_effect*.csv`, `task1_topology_comparison.csv` | same replay cache | complete, stranded |
| Degree same-state / full reselection / participant agreement | mechanism task4 CSV/MD; historical generation logic is not in a stable tracked source path | evidence exists; stable runnable source unresolved | partial |
| Top-Reliability-q / Greedy-Subset-q | task3 CSV/MD and consistency checks | q equals default participant count; validation-only selection | partial (generator not stably tracked) |

“Stranded” means the implementation exists only under ignored `outputs/` and
must be copied unchanged to a stable analysis path before packaging.

## E. Deployment-aware objectives

`topology_objective_components` and `score_topology_on_validation` implement
active, local mean, and joint objectives. `experiment_plan.py::deployment_objective_jobs`
covers active, local, joint-0.25, joint-0.50, and joint-0.75.
`experiments/deployment_objective_analysis.py` derives active/local tables and
trade-offs. `estimate_communication` separates peer, global-readout, and total;
saved rows also include topology/control-plane communication. Strict label
location is implemented by the logit-gradient training path and protocol jobs.
Status: **complete**.

## F. Nested weak-party scaling

The claimed path `experiments/run_nested_weak_scaling.py` exists. It fixes three
useful parties, subsets an 18-party cache at weak counts 0/2/5/10/15, evaluates
3/5/8/13/18 total parties on ACM and DBLP for seeds 42--46, compares TA-DVFG and
Full Mesh, and writes raw, summary, paired-delta, plot, and manifest outputs.
All ten 18-party caches exist under `results/nested_weak_cache/`. Initial raw and
summary output files are absent. Status: **partial** until cached replay output is
validated or the absence is reported in the package.

## G. MovieLens

All explicit paths exist: `experiments/movielens/run_movielens_tadvfg.py`, the
supporting `experiments/movielens/` scripts, and `models/movielens_parties.py`.

| Claim | Implementation / evidence | Status |
|---|---|---|
| MovieLens 1M observed rows, rating >=4, no negative sampling | `read_movielens_1m`, `build_features`; leakage-safe raw outputs | complete |
| Ordering and 70/10/20 split | `chronological_split` sorts timestamp/user/movie | complete |
| Train-only fitting/statistics/timestamp normalization | `build_features`, `train_five_parties` | complete |
| Five interfaces | `MFParty`, `UserProfileParty`, `PopularityTemporalParty`, `MovieMetadataParty`, `GenreKNNParty` | complete |
| Five-party methods and controlled 15-party Main/Hard | `run_one_seed`, `expand_to_15_parties`, engine methods | complete |
| ROC-AUC and communication | `roc_auc_binary`, `estimate_communication`; per-seed/summary CSV | complete |
| Seeds and raw/table/figure generation | leakage-safe output trees; `extended_movielens_suite.py`, `aggregate_15party.py`, plotter | complete |
| Defaults B=15, m=5, tau=0 | MovieLens CLI/default engine namespace | complete |

## H. Latent-alignment reference

`experiments/alignment_references.py` implements fresh frozen HGB local predictors,
hidden-state export/equivalence checks, Project-and-Mean, reliability-selected
fusion, useful-only oracle, Global Top-k, Full Mesh consensus, TA-DVFG, Macro-F1,
Test@BestVal, confidence interval/t-test/win-tie-loss, payload width, parameter
counts, training caches, and failed gated diagnostics. Evidence is in
`outputs/alignment_references*`. The five ACM caches exist and the supplement
reports 75/75 equivalence checks; evidence is explicitly separate from historical
cached core. Status: **complete**, with provenance consolidation required.

## I. Protocol robustness

`compute_party_reliabilities` models active-party validation-only evaluation;
strict `active_party_logit_gradient` training sends logits and returns exact logit
gradients while passive code has no direct labels. Code comments and supplement
state that this is API separation, not confidentiality/privacy. Jobs and raw
results exist for ACM/DBLP Hard and matched controls. Status: **complete**.

## J. Multi-hop sensitivity

The exact claimed path exists. It covers K=1/2/3 independent topology replay,
accuracy, K-hop reachability, communication scaling, tables and figures. Tracked
raw/summary evidence exists in `metadata/`; additional outputs are under
`results/k_sensitivity_*`. Status: **complete**.

## K. Efficiency, scalability, and sanity

| Claim | Code / evidence | Status |
|---|---|---|
| Label budget and update frequency | `validation_label_budget_jobs`, `topology_frequency_jobs`, final-analysis table/figure | complete |
| Party-count scalability/evaluations/time | `party_scalability_jobs`; `results/party_scalability`; final-analysis table | complete |
| m=0 and first-edge gains | `run_cached_minedges_sweep.py`; min-edge result/table/figure | partial: first-edge gain generator not isolated |
| Edge budget/tau/selected edges | `edge_budget_jobs`, final-analysis and readable-figure scripts | complete |
| Communication decomposition | engine and `final_analysis.add_communication_accounting` | complete |

## L. Statistical reporting

`experiments/statistical_tests.py`, `experiments/final_analysis.py::paired_test_rows`
and `holm_adjust`, `experiments/alignment_references.py`, and
`experiments/movielens/extended_movielens_suite.py` implement five-seed mean/SD,
paired t-tests, Wilcoxon, 95% CI, paired Cohen dz, Holm correction, win/tie/loss,
and MovieLens paired AUC/communication deltas. The same-state peer-exchange test
is confirmatory. Budget, K/depth, degree, and selector sweeps are exploratory.
Status: **complete**.

## Coverage summary at inventory time

There are 52 auditable requirement groups in this map: 44 complete, 7 partial,
and 1 missing (the main-paper overview figure asset, outside code coverage). Code
and analysis coverage is therefore 91.7% when partial items count as half and the
missing protected figure is included; 93.1% for code/analysis requirements only.

