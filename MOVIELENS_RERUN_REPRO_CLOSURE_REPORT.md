# MovieLens Rerun Reproducibility Closure Report

Date: 2026-07-05

Final status: FAIL/BLOCKED, not submission-grade PASS.

The complete MovieLens experimental rerun and canonical MovieLens artifact refresh were completed, but final submission-grade closure is blocked by local LaTeX compilation failures and by stale MovieLens numbers remaining in noncanonical historical drafts.

## Protocol Audit

Audit status: PASS for the current MovieLens implementation.

File evidence:

- `experiments/movielens/run_movielens_tadvfg.py:76-103` reads `ratings.dat`, `users.dat`, and `movies.dat` from MovieLens 1M.
- `experiments/movielens/run_movielens_tadvfg.py:137-143` sorts rating rows by `timestamp`, `user_id`, `movie_id` and splits 70/10/20.
- `experiments/movielens/run_movielens_tadvfg.py:163` defines labels as `rating >= 4`.
- `experiments/movielens/run_movielens_tadvfg.py:186-199` computes popularity counts, means, positive rates, and timestamp mean/std from `train = ratings.loc[train_idx]`.
- `experiments/movielens/run_movielens_tadvfg.py:204-215` applies train-derived timestamp normalization to any frame.
- `experiments/movielens/run_movielens_tadvfg.py:239-263` trains the five local parties on `train_idx` only and evaluates validation/test separately.
- `experiments/movielens/run_movielens_tadvfg.py:311-384` builds one aligned cache over the same ordered observed rating rows for all parties.

Task construction:

- Dataset: GroupLens MovieLens 1M, local archive `data/movielens/ml-1m.zip`.
- Task: binary prediction over observed rating rows.
- Positive label: rating >= 4.
- Negative label: observed ratings 1--3.
- No negative sampling or unobserved user-item negative insertion was found in the current pipeline.

## Preprocessing Leakage Audit

Status: PASS for the audited leakage-sensitive paths.

Train-only:

- Matrix factorization and local predictors: trained in `train_five_parties` on `train_batches` and `y_train`.
- Popularity/count/mean/positive-rate statistics: computed from `train`.
- Timestamp normalization: `train_ts_mean` and `train_ts_std` computed from `train`.

Validation/test boundaries:

- Validation labels are used for party reliability and topology selection.
- Test labels are used for final metrics only.

## Syntax Check

PASS.

Command recorded in `MOVIELENS_RERUN_COMMANDS.txt`.

## Rerun Coverage

Old outputs were archived under:

- `outputs/movielens_archive_pre_leakage_safe_20260705/`

New timestamped outputs:

- `outputs/movielens_leakage_safe_20260705/`
- `outputs/movielens_main15_leakage_safe_20260705/`
- `outputs/movielens_hard15_leakage_safe_20260705/`
- `outputs/movielens_final_package_leakage_safe_20260705/`

Canonical outputs refreshed:

- `outputs/movielens/`
- `outputs/movielens_main15/`
- `outputs/movielens_hard15/`
- `outputs/movielens_final_package/`

Seeds:

- 42, 43, 44, 45, 46 for five-party, 15-party Main, and 15-party Hard.

## New MovieLens Results

Five-party summary from `outputs/movielens/movielens_tadvfg_summary.csv`:

| Method | AUC | Total comm. | Links |
|---|---:|---:|---:|
| Best Single | 0.7402 | 0 | 0.0 |
| Global Top-k | 0.7487 | 800,168 | 0.0 |
| Adaptive Pairwise | 0.7510 | 5,201,092 | 4.0 |
| Full Mesh | 0.7497 | 10,002,100 | 10.0 |
| TA-DVFG | 0.7516 | 3,200,672 | 2.4 |

15-party summary from `outputs/movielens/movielens_15party_summary.csv`:

| Setting | Method | AUC | Links | Total comm. |
|---|---|---:|---:|---:|
| Main | TA-DVFG | 0.7516 | 5.2 | 6,481,360.8 |
| Main | Full Mesh | 0.7457 | 105.0 | 90,018,900 |
| Main | Adaptive Pairwise | 0.7460 | 14.6 | 17,603,696 |
| Hard | TA-DVFG | 0.7498 | 5.0 | 6,401,344 |
| Hard | Full Mesh | 0.7435 | 105.0 | 90,018,900 |
| Hard | Adaptive Pairwise | 0.7440 | 14.2 | 17,123,595.2 |

Full Mesh verification:

- `outputs/movielens/movielens_15party_per_seed.csv` has 10 Full Mesh rows.
- All 10 rows have `final_edges = 105`.

Paired statistics from `outputs/movielens/movielens_paired_statistics.csv`:

- TA-DVFG vs Full Mesh AUC delta: +0.001905; paired t-test p = 0.0258.
- TA-DVFG vs Adaptive Pairwise AUC delta: +0.000660; paired t-test p = 0.2646.
- TA-DVFG vs Global Top-k AUC delta: +0.002932; paired t-test p = 0.0161.

## Changed Claims

Updated canonical main source:

- `main_aaai27_alignment_submission_ready.tex`

Updated canonical supplement source:

- `Collaboration_without_Representation_Alignment_supplement_fixed.tex`

Old five-party AUC values were replaced:

- Best Single: 0.7421 -> 0.7402
- Global Top-k: 0.7479 -> 0.7487
- Adaptive Pairwise: 0.7508 -> 0.7510
- Full Mesh: 0.7495 -> 0.7497
- TA-DVFG: 0.7512 -> 0.7516

Old 15-party values were replaced:

- Main TA-DVFG: 0.7511, 5.4 links -> 0.7516, 5.2 links
- Hard TA-DVFG: 0.7491, 5.0 links -> 0.7498, 5.0 links
- Main Full Mesh: 0.7455 -> 0.7457
- Hard Full Mesh: 0.7427 -> 0.7435
- Adaptive Pairwise links: 14.2/14.0 -> 14.6/14.2

The canonical files no longer contain the searched stale MovieLens numeric literals. Historical noncanonical `.tex` drafts still contain old values.

## Changed Figures And Tables

Regenerated under `outputs/movielens/`:

- `movielens_tadvfg_per_seed.csv`
- `movielens_tadvfg_summary.csv`
- `movielens_party_performance.csv`
- `movielens_paired_statistics.csv`
- `movielens_pareto_summary.csv`
- `movielens_pareto_per_seed_deltas.csv`
- `movielens_15party_per_seed.csv`
- `movielens_15party_summary.csv`
- `movielens_15party_topologies.json`
- `movielens_accuracy_auc_bar.png`
- `movielens_comm_tradeoff.png`
- `movielens_party_reliability.png`
- `movielens_party_val_test_auc.png`
- `movielens_pareto_auc_communication.png`
- `movielens_15party_main_hard.png`
- `movielens_15party_comm_tradeoff.png`
- `movielens_15party_edge_composition.png`
- `movielens_topology_example.png`

## Exact Readout Formula

Implementation evidence:

- `main experiment/ta_dvfg_hgb_reliability.py:973-980`: global reliability vote uses `max(r_i, 1e-3)` and normalizes by the sum.
- `main experiment/ta_dvfg_hgb_reliability.py:1018-1042`: topology-filtered readout uses active non-isolated parties, reliability power, degree power, and normalized weights.
- `experiments/movielens/run_movielens_tadvfg.py:469-472`: MovieLens wrapper sets `vote_weighting="topology_reliability"`, `reliability_power=1.0`, `topology_degree_power=0.5`, and `final_reliability_floor=0.0`.

For consensus, each party averages neighbor predictions using

`w_ij = max(r_j, 1e-3) / sum_{l in N_i(E)} max(r_l, 1e-3)`.

For active readout, if `E` has at least one edge,

`S(E) = { i : deg_E(i) > 0 }`.

If `E` is edgeless, `S(E) = {1,...,N}`.

The active party receives no special additional term. Isolated parties are excluded whenever the selected graph has at least one edge.

The implemented readout is:

`Q(E) = sum_{i in S(E)} a_i P_i^(K)(E)`,

where

`a_i = [max(r_i, 1e-3)^1.0 * (deg_E(i)+1)^0.5] / sum_{j in S(E)} [max(r_j, 1e-3)^1.0 * (deg_E(j)+1)^0.5]`.

The same `topology_filtered_reliability_vote` is used for topology scoring, validation readout, and test readout when `vote_weighting="topology_reliability"` and the graph has at least one edge.

## Tie-Breaking Audit

Implementation evidence:

- `main experiment/ta_dvfg_hgb_reliability.py:1472-1474`: utility candidates are `(utility, i, j)` and sorted with `reverse=True`, so utility desc, then larger `i`, then larger `j`.
- `main experiment/ta_dvfg_hgb_reliability.py:1528-1543` and `2016-2048`: graph-validation greedy selection scans that deterministic order.
- Validation-score ties select the candidate with higher utility via `util > best_util`.
- Exact utility ties retain the first candidate encountered in the already sorted deterministic order.

Therefore, the old generic "fixed lexicographic ordering of undirected edge pairs" statement is incomplete.

## Claim Map

Machine-readable claim map:

- `movielens_claim_map.csv`

## Compilation And Visual Check

Status: BLOCKED.

Main paper compile:

- Attempted with bundled latex helper.
- Tectonic was available, but main paper was deemed unsuitable for Tectonic due to bibliography tooling.
- TeX Live/MacTeX was not detected: missing `latexmk`, `pdflatex`, and `kpsewhich`.

Supplement compile:

- Tectonic attempted compilation.
- Failed because `aaai2027.sty` was not found.
- TeX Live/MacTeX fallback unavailable.

No compiled PDFs were produced, so unresolved references, overfull boxes, and visual layout cannot be marked PASS.

## Remaining Blockers

1. Install or provide a complete LaTeX environment and `aaai2027.sty`, then compile main and supplement.
2. Decide whether historical noncanonical TeX drafts should be globally scrubbed or archived. They still contain stale MovieLens values.
3. The pending supplement file `TA_DVFG_supplement_reproducibility_upgraded_PENDING_MOVIELENS_RERUN.tex` still contains the older generic gamma formula and should not be used as canonical unless updated.

## Final Checklist

- [x] Latest train-only timestamp normalization is used.
- [x] Complete MovieLens five-party suite rerun for seeds 42--46.
- [x] Complete 15-party Main/Hard suite rerun for seeds 42--46.
- [x] CSV/JSON outputs regenerated for canonical MovieLens paths.
- [x] Paired statistics regenerated.
- [x] MovieLens figures regenerated.
- [x] Canonical main/supplement MovieLens values updated.
- [x] Exact readout formula audited and canonical supplement corrected.
- [x] Degree/readout power 0.5 specified correctly in canonical supplement.
- [x] Reliability weight epsilon and exponent specified correctly in canonical supplement.
- [x] Exact tie-breaking audited.
- [x] Dataset/task construction audited.
- [x] No preprocessing leakage found in current audited code.
- [ ] Every historical paper draft scrubbed of stale MovieLens values.
- [ ] Main and supplement compile successfully.

Final decision: FAIL/BLOCKED, not SUBMISSION-GRADE PASS.
