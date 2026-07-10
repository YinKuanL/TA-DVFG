# Minimum-Link Audit

## 1. Implementation audit

- Selector implementation: `main experiment\ta_dvfg_hgb_reliability.py`.
- Scoring function: `score_topology_on_validation` at line 1261.
- Live training path: `update_adaptive_topology` at line 1484; cached replay path: `update_adaptive_topology_from_cache` at line 1997.
- Both paths initialize `adj` as an all-zero graph, compute `current_val = val_score(adj)` (lines 1523 and 2034), then greedily test feasible one-edge additions.
- With `adaptive_min_edges = 0`, `must_fill = edge_count(adj) < min_edges` is false from the first iteration (lines 1543 and 2054), so the first edge is accepted only when `best_val >= current_val + adaptive_min_gain` (lines 1544 and 2055).
- The threshold stop is implemented by `if not must_fill and not improves: break` (lines 1545--1546 and 2056--2057).
- Therefore an empty topology is allowed, and the selector can stop at any edge count from 0 through `adaptive_edge_budget`, subject to the max-degree constraint and feasible candidates.
- I found no other cached-selector code path that forces communication when `adaptive_min_edges = 0`; defaults can force edges only if the caller leaves `adaptive_min_edges` at the legacy parser default rather than passing the paper setting.

## 2. Existing evidence found

- Existing sweep script: `experiments/run_cached_minedges_sweep.py`.
- Existing outputs: `results/minedge_sweep/acm_hard_minedges_sweep.csv` and `results/minedge_sweep/dblp_hard_minedges_sweep.csv`.
- Existing config records `min_edges = [0, 1, 3, 5]`, seeds `42,43,44,45,46`, cached predictions from `results/core_cache_v2`, `adaptive_edge_budget = 15`, `adaptive_min_gain = 0.001`, `max_degree = 2`, and `pred_consensus_steps = 1`.

## 3. New experiments run

- No local predictors were retrained and no full benchmark suite was rerun.
- I replayed the existing cached selector on the final cached epoch to recover first-edge gains, final topology validation scores, and stopping reasons.

## 4. Exact results

| dataset | setting | adaptive_min_edges | test_at_best_val_mean | test_at_best_val_std | final_edges_mean | final_edges_std | empty_topology_seeds | final_topology_val_score_mean | final_topology_val_score_std | topology_update_time_mean | topology_update_time_std | threshold not met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | Hard | 0 | 0.884349 | 0.014661 | 2.600000 | 0.547723 | 0 | 0.892384 | 0.031911 | 13.658055 | 5.209307 | 5 |
| ACM | Hard | 1 | 0.884349 | 0.014661 | 2.600000 | 0.547723 | 0 | 0.892384 | 0.031911 | 27.931940 | 2.431828 | 5 |
| ACM | Hard | 3 | 0.879407 | 0.020688 | 3.400000 | 0.547723 | 0 | 0.892384 | 0.031954 | 33.524846 | 1.813619 | 5 |
| ACM | Hard | 5 | 0.855354 | 0.029086 | 5.000000 | 0.000000 | 0 | 0.883113 | 0.030271 | 40.633433 | 1.612121 | 5 |
| DBLP | Hard | 0 | 0.902211 | 0.009927 | 3.000000 | 0.707107 | 0 | 0.911358 | 0.012172 | 25.534335 | 1.190363 | 5 |
| DBLP | Hard | 1 | 0.902211 | 0.009927 | 3.000000 | 0.707107 | 0 | 0.911358 | 0.012172 | 25.212926 | 2.082055 | 5 |
| DBLP | Hard | 3 | 0.903685 | 0.009927 | 3.200000 | 0.447214 | 0 | 0.911358 | 0.012172 | 31.039715 | 1.134819 | 5 |
| DBLP | Hard | 5 | 0.902457 | 0.011413 | 5.000000 | 0.000000 | 0 | 0.908395 | 0.011627 | 39.390309 | 0.923033 | 5 |

m=0 per-seed selected edge counts:

| dataset | seed | final_edges | first_edge_gain | stopping_reason |
| --- | --- | --- | --- | --- |
| ACM | 42 | 3 | 0.048013 | threshold not met |
| ACM | 43 | 3 | 0.067881 | threshold not met |
| ACM | 44 | 2 | 0.140728 | threshold not met |
| ACM | 45 | 2 | 0.182119 | threshold not met |
| ACM | 46 | 3 | 0.087748 | threshold not met |
| DBLP | 42 | 3 | 0.040741 | threshold not met |
| DBLP | 43 | 3 | 0.037037 | threshold not met |
| DBLP | 44 | 4 | 0.035802 | threshold not met |
| DBLP | 45 | 3 | 0.028395 | threshold not met |
| DBLP | 46 | 2 | 0.032099 | threshold not met |

## 5. Reviewer-risk interpretation

- ACM Hard: with m=0, mean Test@BestVal is 0.8843 and mean selected edges is 2.60; with m=5, mean Test@BestVal is 0.8554 and mean selected edges is 5.00.
- DBLP Hard: with m=0, mean Test@BestVal is 0.9022 and mean selected edges is 3.00; with m=5, mean Test@BestVal is 0.9025 and mean selected edges is 5.00.
- Across the audited seeds, m=0 never selected an empty topology.
- The minimum-link setting behaves more like a stabilizing/reporting prior for a fixed sparse operating point than a necessary condition for collaboration to appear.
- The statement that B is an upper bound remains accurate for the implemented selector, but the paper should avoid implying that m=5 is not also a small minimum quota in the default configuration.

## 6. Recommended paper action

Keep the headline m=5 result unchanged, add a supplement sensitivity table/plot for m in {0,1,3,5}, and add one clarifying sentence in the main text or supplement.

## 7. Suggested sentence

In a cached minimum-link sensitivity audit on ACM/DBLP Hard, setting m=0 still selected nonempty topologies in all five seeds, indicating that communication links emerge from the validation objective rather than solely from the minimum-link prior; m=5 fixes a conservative sparse operating point for the main comparison.

## Output files

- `minimum_link_summary.csv`
- `minimum_link_per_seed.csv`
- `first_edge_gain.csv`
- `minimum_link_test_edges.png`
