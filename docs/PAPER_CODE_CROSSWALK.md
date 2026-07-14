# Paper-Code Crosswalk

| Paper concept | Canonical code | Configuration / evidence |
|---|---|---|
| Independent local HGB predictors | `main experiment/ta_dvfg_hgb_reliability.py::LocalGCN`, `build_party_views` | `configs/hgb/defaults.json` |
| Prediction reliability and consensus | `compute_party_reliabilities`, `post_consensus_party_probs` | HGB/MovieLens defaults |
| Sparse topology selection | `score_topology_edges`, `update_adaptive_topology` | HGB selector manifest; saved config audit |
| Topology-aware active readout | `active_party_indices`, `topology_filtered_reliability_vote` | reliability/degree exponents in HGB defaults |
| Deployment objectives | `topology_objective_components`, `score_topology_on_validation` | `configs/mechanisms/deployment_objectives.json` |
| Communication | `estimate_communication` | peer/readout/control fields in result CSV |
| HGB primary comparisons | `run_cached_core.py`, `experiment_plan.py` | reviewer-safe/core raw CSV and final-analysis tables |
| Same-state mechanism | `analysis/mechanisms/run_peer_exchange_audit.py` | round-2 replay bundles and task CSV |
| Topology provenance | `analysis/provenance/*` | step-C/step-C5 audit CSV and reports |
| Nested weak scaling | `run_nested_weak_scaling.py` | nested weak cache and robustness manifest |
| MovieLens | `experiments/movielens/`, `models/movielens_parties.py` | leakage-safe raw/final package |
| Latent alignment | `alignment_references.py` | fresh ACM Hard five-seed cache/evidence |
| Protocol robustness | HGB label/evaluator/training protocol functions and jobs | active/strict label result trees |
| K sensitivity | `run_k_sensitivity.py` | tracked raw/summary CSV |
| Statistical reporting | `statistical_tests.py`, `final_analysis.py`, MovieLens/alignment stats helpers | statistics CSV and reports |

For table/figure-level provenance, use `docs/RESULT_PROVENANCE.md`. For every
supplement requirement and current completeness status, use
`docs/SUPPLEMENT_TO_CODE_MAP.md`.

