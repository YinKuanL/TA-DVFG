"""Package completed MovieLens experiment artifacts into one usable folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MAIN_TEXT = [
    "movielens_table_a_five_party.csv",
    "movielens_pareto_auc_communication.png",
    "movielens_15party_main_hard.png",
    "movielens_15party_comm_tradeoff.png",
    "movielens_party_val_test_auc.png",
]

APPENDIX = [
    "party_architecture_audit.csv",
    "party_architectures.csv",
    "movielens_party_generalization_gap.csv",
    "movielens_paired_statistics.csv",
    "movielens_pareto_summary.csv",
    "movielens_pareto_per_seed_deltas.csv",
    "movielens_15party_edge_composition.png",
    "movielens_topology_example.png",
    "movielens_accuracy_auc_bar.png",
    "movielens_comm_tradeoff.png",
    "movielens_party_reliability.png",
]

RAW_RESULTS = [
    "movielens_party_performance.csv",
    "movielens_tadvfg_per_seed.csv",
    "movielens_tadvfg_summary.csv",
    "movielens_selected_topologies.json",
    "movielens_15party_per_seed.csv",
    "movielens_15party_summary.csv",
    "movielens_15party_topologies.json",
    "movielens_alignment_comparison_table.csv",
]

FINAL_AUDIT = [
    "config_manifest.json",
    "paper_values.csv",
    "EXPERIMENT_STATUS.md",
    "paper_patch_macros.tex",
]

SCRIPTS = [
    "experiments/movielens/run_movielens_tadvfg.py",
    "experiments/movielens/extended_movielens_suite.py",
    "experiments/movielens/aggregate_15party.py",
    "experiments/movielens/plot_movielens_results.py",
    "experiments/movielens/prepare_movielens.py",
    "experiments/run_nested_weak_scaling.py",
    "experiments/final_claims_audit.py",
    "models/movielens_parties.py",
]


README = """# MovieLens TA-DVFG Final Result Package

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
$env:PYTHONPATH=(Resolve-Path '.venv\\Lib\\site-packages').Path
$PY='C:\\Users\\Yin Kuan\\AppData\\Local\\Programs\\Python\\Python314\\python.exe'

& $PY experiments\\movielens\\run_movielens_tadvfg.py `
  --data-dir data\\movielens\\ml-1m `
  --output-dir outputs\\movielens `
  --seeds 42,43,44,45,46 `
  --num-parties 5 --setting five --local-epochs 8

& $PY experiments\\movielens\\extended_movielens_suite.py `
  --output-dir outputs\\movielens

& $PY experiments\\movielens\\run_movielens_tadvfg.py `
  --data-dir data\\movielens\\ml-1m `
  --output-dir outputs\\movielens_main15 `
  --seeds 42,43,44,45,46 `
  --setting main --local-epochs 8 `
  --methods single_party_best,local_uniform_vote,local_reliability_vote,adaptive_graph_val,topk_reliability_vote,fixed_ring,random_matched,full_mesh,adaptive_pair,adaptive_complementarity,expander_matched `
  --adaptive-edge-budget 15 --adaptive-min-edges 5 --matched-edge-count 5

& $PY experiments\\movielens\\run_movielens_tadvfg.py `
  --data-dir data\\movielens\\ml-1m `
  --output-dir outputs\\movielens_hard15 `
  --seeds 42,43,44,45,46 `
  --setting hard --local-epochs 8 `
  --methods single_party_best,local_uniform_vote,local_reliability_vote,adaptive_graph_val,topk_reliability_vote,fixed_ring,random_matched,full_mesh,adaptive_pair,adaptive_complementarity,expander_matched `
  --adaptive-edge-budget 15 --adaptive-min-edges 5 --matched-edge-count 5

& $PY experiments\\movielens\\aggregate_15party.py `
  --main-dir outputs\\movielens_main15 `
  --hard-dir outputs\\movielens_hard15 `
  --output-dir outputs\\movielens
```
"""


def copy_existing(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/movielens"))
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/movielens_final_package"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.package_dir.mkdir(parents=True, exist_ok=True)
    for name in MAIN_TEXT:
        copy_existing(args.source_dir / name, args.package_dir / "main_text" / name)
    for name in APPENDIX:
        copy_existing(args.source_dir / name, args.package_dir / "appendix" / name)
    for name in RAW_RESULTS:
        copy_existing(args.source_dir / name, args.package_dir / "raw_results" / name)
    audit_source = args.source_dir.parent / "final_claims_audit"
    for name in FINAL_AUDIT:
        copy_existing(audit_source / name, args.package_dir / "final_claims_audit" / name)
    for name in SCRIPTS:
        copy_existing(Path(name), args.package_dir / "scripts" / name)
    (args.package_dir / "README_USE_THIS.md").write_text(README, encoding="utf-8")
    print(f"Packaged MovieLens artifacts under {args.package_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
