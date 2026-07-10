# MovieLens TA-DVFG Final Result Package

This folder contains the completed, reusable MovieLens experiment artifacts
generated from five seeds: 42, 43, 44, 45, 46.

## Recommended Main-Paper Placement

Use these in the main paper:

1. `main_text/movielens_table_a_five_party.csv`
   - Main 5-party real-world table.
   - Shows ROC-AUC, Accuracy, F1, NDCG@10, selected links, and communication.

2. `main_text/movielens_pareto_auc_communication.png`
   - Main AUC-communication Pareto figure.
   - Best visual evidence that TA-DVFG improves the AUC/communication tradeoff.

3. `main_text/movielens_15party_main_hard.png`
   - Main 15-party noisy-view result.
   - Shows sparse topology benefit under Main/Hard settings.

4. `main_text/movielens_15party_comm_tradeoff.png`
   - Communication evidence for 15-party setting.

5. `main_text/movielens_party_val_test_auc.png`
   - Compact evidence that party quality heterogeneity exists.

## Recommended Appendix Placement

Use these in appendix/supplement:

- `appendix/party_architecture_audit.csv`
- `appendix/party_architectures.csv`
- `appendix/movielens_party_generalization_gap.csv`
- `appendix/movielens_paired_statistics.csv`
- `appendix/movielens_pareto_summary.csv`
- `appendix/movielens_pareto_per_seed_deltas.csv`
- `appendix/movielens_15party_edge_composition.png`
- `appendix/movielens_topology_example.png`
- `raw_results/*.csv`
- `raw_results/*.json`

## Main Findings

Five-party MovieLens:

- TA-DVFG ROC-AUC: 0.7516.
- Full Mesh ROC-AUC: 0.7497.
- Adaptive Pairwise ROC-AUC: 0.7510.
- TA-DVFG uses 3.20M total prediction communication versus 10.00M for Full Mesh.
- Mean communication reduction versus Full Mesh: 68.0%.
- Mean communication reduction versus Adaptive Pairwise: 38.5%.

15-party MovieLens:

- Main: TA-DVFG AUC 0.7516 with 5.2 links; Full Mesh AUC 0.7457 with 105 links.
- Hard: TA-DVFG AUC 0.7498 with 5.0 links; Full Mesh AUC 0.7435 with 105 links.
- Full Mesh edge count was verified as 105 in both 15-party settings.

## Claims Supported By Completed Runs

Supported:

- Prediction-level collaboration improves over the best single party.
- TA-DVFG gives a better AUC-communication tradeoff than Full Mesh.
- Sparse topology is especially valuable in the 15-party noisy-view setting.
- Party quality heterogeneity exists under chronological MovieLens splitting.

Partially supported:

- Model heterogeneity: current completed 5-party result has heterogeneous views and hidden dimensions, but not the full LightGCN/GCN/GraphSAGE/GAT/MLP architecture ladder.

Not yet completed:

- Heterogeneity ladder H0-H3.
- Representation-alignment baselines: Project-and-Mean, Project-and-Concat, Gated Fusion.
- Noisy-party scaling with +0/+2/+5/+10/+15 additional weak parties.
- Sampled-negative ranking protocol beyond the current observed-test NDCG@10.

## Exact Commands Used For Completed Results

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$PY='C:\Users\Yin Kuan\AppData\Local\Programs\Python\Python314\python.exe'

& $PY experiments\movielens\run_movielens_tadvfg.py `
  --data-dir data\movielens\ml-1m `
  --output-dir outputs\movielens `
  --seeds 42,43,44,45,46 `
  --num-parties 5 --setting five --local-epochs 8

& $PY experiments\movielens\extended_movielens_suite.py `
  --output-dir outputs\movielens

& $PY experiments\movielens\run_movielens_tadvfg.py `
  --data-dir data\movielens\ml-1m `
  --output-dir outputs\movielens_main15 `
  --seeds 42,43,44,45,46 `
  --setting main --local-epochs 8 `
  --methods single_party_best,local_uniform_vote,local_reliability_vote,adaptive_graph_val,topk_reliability_vote,fixed_ring,random_matched,full_mesh,adaptive_pair,adaptive_complementarity,expander_matched `
  --adaptive-edge-budget 15 --adaptive-min-edges 5 --matched-edge-count 5

& $PY experiments\movielens\run_movielens_tadvfg.py `
  --data-dir data\movielens\ml-1m `
  --output-dir outputs\movielens_hard15 `
  --seeds 42,43,44,45,46 `
  --setting hard --local-epochs 8 `
  --methods single_party_best,local_uniform_vote,local_reliability_vote,adaptive_graph_val,topk_reliability_vote,fixed_ring,random_matched,full_mesh,adaptive_pair,adaptive_complementarity,expander_matched `
  --adaptive-edge-budget 15 --adaptive-min-edges 5 --matched-edge-count 5

& $PY experiments\movielens\aggregate_15party.py `
  --main-dir outputs\movielens_main15 `
  --hard-dir outputs\movielens_hard15 `
  --output-dir outputs\movielens
```
