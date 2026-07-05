"""Merge fixed baseline summaries with a cached adaptive min-edges sweep.

The utility refuses to merge unless the baseline training config matches every
prediction cache used by the sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_KEYS = (
    "dataset",
    "target_node",
    "data_dir",
    "ignore_dataset_split",
    "graph_views",
    "graph_k",
    "max_metapath_edges",
    "random_graph_degree",
    "num_parties",
    "useful_parties",
    "view_setting",
    "distractor_feature_noise",
    "shuffle_party_positions",
    "party_shuffle_seed_offset",
    "train_ratio",
    "val_ratio",
    "epochs",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "cache")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--exclude-baseline-methods", nargs="+", default=["adaptive_graph_val"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize(key: str, value: object) -> object:
    if key == "data_dir":
        return str(Path(str(value)).resolve())
    return value


def validate_config(baseline_config: Dict[str, object], cache: Dict[str, object], path: Path) -> str:
    cache_config = cache.get("training_config")
    if not isinstance(cache_config, dict):
        raise ValueError(f"{path} has no training_config")
    mismatches = []
    for key in TRAINING_KEYS:
        baseline_value = normalize(key, baseline_config.get(key))
        cache_value = normalize(key, cache_config.get(key))
        if baseline_value != cache_value:
            mismatches.append(f"{key}: baseline={baseline_value!r}, cache={cache_value!r}")
    if mismatches:
        raise ValueError(f"Baseline/cache mismatch for {path}:\n  - " + "\n  - ".join(mismatches))
    return str(cache["cache_fingerprint"])


def main() -> int:
    args = parse_args()
    baseline_config = json.loads(args.baseline_config.read_text(encoding="utf-8"))
    dataset = str(baseline_config["dataset"]).lower()
    setting = str(baseline_config["view_setting"]).lower()
    useful = int(baseline_config["useful_parties"])
    fingerprints: List[str] = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        path = args.cache_dir / f"{dataset}_{setting}_useful{useful}_seed{seed}_party_preds.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache required for fair merge: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        fingerprints.append(validate_config(baseline_config, cache, path))

    baseline = pd.read_csv(args.baseline_summary)
    baseline = baseline[~baseline["method"].isin(args.exclude_baseline_methods)].copy()
    sweep = pd.read_csv(args.sweep_summary).rename(
        columns={
            "macro_f1_at_best_val_mean": "macro_f1_mean",
            "macro_f1_at_best_val_std": "macro_f1_std",
        }
    )
    baseline.insert(0, "source", "fixed_baseline")
    sweep.insert(0, "source", "cached_adaptive_sweep")
    columns = [
        "source",
        "dataset",
        "method",
        "adaptive_min_edges",
        "test_at_best_val_mean",
        "test_at_best_val_std",
        "macro_f1_mean",
        "macro_f1_std",
        "inference_comm_mean",
        "inference_comm_std",
        "peer_to_peer_comm_mean",
        "peer_to_peer_comm_std",
        "global_readout_comm_mean",
        "global_readout_comm_std",
        "total_comm_mean",
        "total_comm_std",
        "protocol_type",
        "final_edges_mean",
        "final_edges_std",
        "useful_useful_edges_mean",
        "useful_noisy_edges_mean",
        "noisy_noisy_edges_mean",
        "topology_eval_count_mean",
        "topology_update_comm_mean",
        "topology_update_time_mean",
    ]
    combined = pd.concat([baseline, sweep], ignore_index=True, sort=False)
    combined = combined[[column for column in columns if column in combined.columns]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    provenance = {
        "baseline_summary": str(args.baseline_summary.resolve()),
        "baseline_config": str(args.baseline_config.resolve()),
        "sweep_summary": str(args.sweep_summary.resolve()),
        "cache_fingerprints": fingerprints,
        "validation": "Local-training/data config matched for every seed.",
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.provenance.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
