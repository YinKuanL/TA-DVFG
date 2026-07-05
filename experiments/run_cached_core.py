"""Rebuild the six fair core experiments with shared local prediction caches."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
DATASET_VIEWS: Dict[str, str] = {
    "ACM": "PAP,PSP,KNN",
    "DBLP": "APA,APCPA,APTPA,KNN",
    "IMDB": "MAM,MDM,KNN",
}
SETTINGS = (("main", 6), ("hard_noisy", 3))
METHODS = ",".join(
    [
        "local_uniform_vote",
        "local_reliability_vote",
        "topk_reliability_vote",
        "fixed_ring",
        "fixed_ring_matched",
        "random_regular",
        "random_matched",
        "expander",
        "expander_matched",
        "full_mesh",
        "adaptive_pair",
        "adaptive_complementarity",
        "adaptive_graph_val",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run ACM/DBLP/IMDB main+hard using one shared local trajectory per seed.",
    )
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP", "IMDB"], default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "core_cache_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "core_cached_v2")
    parser.add_argument("--party-shuffle-seed-offset", type=int, default=2026)
    parser.add_argument("--no-shuffle-party-positions", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def output_path(args: argparse.Namespace, dataset: str, setting: str) -> Path:
    name = f"{dataset.lower()}_{setting}_cached_5seeds.csv"
    return args.output_dir / f"{dataset.lower()}_{setting}" / name


def build_command(args: argparse.Namespace, dataset: str, setting: str, useful: int) -> List[str]:
    output = output_path(args, dataset, setting)
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
        str(useful),
        "--epochs",
        str(args.epochs),
        "--seeds",
        args.seeds,
        "--methods",
        METHODS,
        "--device",
        args.device,
        "--data_dir",
        str(REPO_ROOT / "data_hgb"),
        "--cache_dir",
        str(args.cache_dir.resolve()),
        "--cache_predictions",
        "--reuse_prediction_cache",
        "--party_shuffle_seed_offset",
        str(args.party_shuffle_seed_offset),
        "--adaptive_edge_budget",
        "15",
        "--adaptive_min_edges",
        "5",
        "--adaptive_min_gain",
        "0.001",
        "--max_degree",
        "2",
        "--matched_edge_count",
        "5",
        "--pred_consensus_steps",
        "1",
        "--pred_self_weight",
        "0.85",
        "--vote_weighting",
        "topology_reliability",
        "--save_csv",
        str(output.resolve()),
    ]
    if not args.no_shuffle_party_positions:
        command.append("--shuffle_party_positions")
    if args.plot:
        command += ["--plot", "--plot_dir", str((output.parent / "figures").resolve())]
    return command


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    for dataset in args.datasets:
        for setting, useful in SETTINGS:
            output = output_path(args, dataset, setting)
            if output.exists() and not args.force:
                print(f"Skipping completed output {output}")
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            command = build_command(args, dataset, setting, useful)
            print("\n" + subprocess.list2cmdline(command))
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
