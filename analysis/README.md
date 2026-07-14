# Analysis Entry Points

The research engine remains at its historical public path to preserve imports and
recorded commands. This directory gives stable names to analysis-only workflows
without changing their implementations.

| Area | Stable command | Historical implementation / inputs |
|---|---|---|
| Test@BestVal replay | `python analysis/provenance/run_testatbestval_replay.py` | `outputs/round2_closure/step_c/step_c_gate.py` |
| Best-validation topology provenance | `python analysis/provenance/run_topology_provenance.py` | `outputs/round2_closure/step_c5/step_c5_topology_audit.py` |
| Same-state peer-exchange mechanism | `python analysis/mechanisms/run_peer_exchange_audit.py` | `outputs/round2_closure/task1_2_runner.py` |
| HGB tables/figures/statistics | `python experiments/final_analysis.py` | ignored `results/` seed CSV/JSON |
| Deployment objectives | `python experiments/deployment_objective_analysis.py` | objective result CSV |
| Statistical comparisons | `python experiments/statistical_tests.py` | paired seed CSV |
| MovieLens tables/figures | `python experiments/movielens/extended_movielens_suite.py` | leakage-safe output CSV |

The three wrappers search first for a package-local `_historical/` copy and then
for the original ignored workspace path. They do not reinterpret arguments or
modify candidate ordering, scoring, cached values, or output contents.

