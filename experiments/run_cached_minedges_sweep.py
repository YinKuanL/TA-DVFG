"""Build local-prediction caches once and sweep adaptive_min_edges cheaply.

Examples:
    python experiments/run_cached_minedges_sweep.py --stage all --device cuda
    python experiments/run_cached_minedges_sweep.py --stage build-cache --datasets ACM
    python experiments/run_cached_minedges_sweep.py --stage sweep --datasets ACM
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
DATASET_VIEWS: Dict[str, str] = {
    "ACM": "PAP,PSP,KNN",
    "DBLP": "APA,APCPA,APTPA,KNN",
}
METRICS = [
    "test_at_best_val",
    "macro_f1_at_best_val",
    "inference_comm",
    "final_edges",
    "useful_useful_edges",
    "useful_noisy_edges",
    "noisy_noisy_edges",
    "topology_eval_count",
    "topology_update_comm",
    "topology_update_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cached adaptive_graph_val min-edges sweep.",
    )
    parser.add_argument("--stage", choices=["all", "build-cache", "sweep"], default="all")
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP"], default=["ACM", "DBLP"])
    parser.add_argument("--min-edges", nargs="+", type=int, default=[0, 1, 3, 5])
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "cache")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "minedge_sweep")
    parser.add_argument("--no-shuffle-party-positions", action="store_true")
    parser.add_argument("--party-shuffle-seed-offset", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def common_command(args: argparse.Namespace, dataset: str) -> List[str]:
    command = [
        str(args.python.resolve()),
        str(ENGINE),
        "--dataset",
        dataset,
        "--graph_views",
        DATASET_VIEWS[dataset],
        "--num_parties",
        "15",
        "--view_setting",
        "hard",
        "--useful_parties",
        "3",
        "--epochs",
        str(args.epochs),
        "--seeds",
        args.seeds,
        "--device",
        args.device,
        "--data_dir",
        str(REPO_ROOT / "data_hgb"),
        "--cache_dir",
        str(args.cache_dir.resolve()),
        "--party_shuffle_seed_offset",
        str(args.party_shuffle_seed_offset),
        "--adaptive_edge_budget",
        "15",
        "--adaptive_min_gain",
        "0.001",
        "--max_degree",
        "2",
        "--pred_consensus_steps",
        "1",
        "--pred_self_weight",
        "0.85",
        "--vote_weighting",
        "topology_reliability",
    ]
    if not args.no_shuffle_party_positions:
        command.append("--shuffle_party_positions")
    return command


def run(command: List[str]) -> None:
    print("\n" + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_cache(args: argparse.Namespace, dataset: str) -> None:
    command = common_command(args, dataset)
    command += [
        "--cache_predictions",
        "--cache_only",
        "--methods",
        "adaptive_graph_val",
    ]
    run(command)


def part_path(args: argparse.Namespace, dataset: str, min_edges: int) -> Path:
    return args.output_dir / "parts" / f"{dataset.lower()}_hard_minedges{min_edges}.csv"


def evaluate_min_edges(args: argparse.Namespace, dataset: str, min_edges: int) -> Path:
    output = part_path(args, dataset, min_edges)
    if output.exists() and not args.force:
        print(f"Reusing completed sweep part {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = common_command(args, dataset)
    command += [
        "--methods",
        "adaptive_graph_val",
        "--reuse_prediction_cache",
        "--skip_local_training_if_cache_exists",
        "--adaptive_min_edges",
        str(min_edges),
        "--save_csv",
        str(output),
    ]
    run(command)
    return output


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    available = [metric for metric in METRICS if metric in raw.columns]
    summary = raw.groupby(["dataset", "adaptive_min_edges", "method"], dropna=False)[available].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def combine_dataset(args: argparse.Namespace, dataset: str, paths: List[Path]) -> None:
    frames = [pd.read_csv(path) for path in paths]
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.sort_values(["adaptive_min_edges", "seed", "method"]).reset_index(drop=True)
    summary = summarize(raw)
    raw_path = args.output_dir / f"{dataset.lower()}_hard_minedges_sweep.csv"
    summary_path = args.output_dir / f"{dataset.lower()}_hard_minedges_sweep_summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    if any(value < 0 for value in args.min_edges):
        raise ValueError("--min-edges values must be non-negative")

    manifest = {
        "stage": args.stage,
        "datasets": args.datasets,
        "min_edges": args.min_edges,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "cache_dir": str(args.cache_dir.resolve()),
        "shuffle_party_positions": not args.no_shuffle_party_positions,
        "party_shuffle_seed_offset": args.party_shuffle_seed_offset,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sweep_config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if args.stage in {"all", "build-cache"}:
        for dataset in args.datasets:
            build_cache(args, dataset)
    if args.stage in {"all", "sweep"}:
        for dataset in args.datasets:
            paths = [evaluate_min_edges(args, dataset, value) for value in args.min_edges]
            combine_dataset(args, dataset, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
