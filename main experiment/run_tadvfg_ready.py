"""
Ready-to-run TA-DVFG reliability experiment runner
=================================================

This runner uses the current recommended settings by default:
  - topology_reliability voting
  - adaptive_edge_budget = 15
  - adaptive_min_edges = 5
  - adaptive_min_gain = 0.001
  - max_degree = 2
  - pred_consensus_steps = 1
  - 5 seeds: 42,43,44,45,46

Usage examples:
  # Main setting: 15 parties, 6 useful, 9 noisy
  python run_tadvfg_ready.py --mode main --datasets ACM DBLP

  # Hard/noisy setting: 15 parties, 3 useful, 12 noisy
  python run_tadvfg_ready.py --mode hard --datasets ACM DBLP

  # Run both main and hard settings for all datasets
  python run_tadvfg_ready.py --mode both

  # Show commands without running
  python run_tadvfg_ready.py --mode both --dry_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_METHODS = [
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

DATASET_VIEWS: Dict[str, str] = {
    "ACM": "PAP,PSP,KNN",
    "DBLP": "APA,APCPA,APTPA,KNN",
    "IMDB": "MAM,MDM,KNN",
}


def run_cmd(cmd: Sequence[str], dry_run: bool = False) -> None:
    printable = " ".join(str(x) for x in cmd)
    print("\n" + "=" * 110)
    print(printable)
    print("=" * 110)
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def make_cmd(args: argparse.Namespace, dataset: str, useful_parties: int, setting_name: str) -> List[str]:
    views = DATASET_VIEWS[dataset]
    methods = ",".join(args.methods)
    csv_name = f"{dataset.lower()}_{setting_name}_min{args.adaptive_min_edges}_{args.seed_tag}.csv"
    plot_dir = f"figures_{dataset.lower()}_{setting_name}_min{args.adaptive_min_edges}_{args.seed_tag}"

    cmd = [
        sys.executable,
        str(args.script),
        "--dataset", dataset,
        "--graph_views", views,
        "--num_parties", str(args.num_parties),
        "--view_setting", "hard",
        "--useful_parties", str(useful_parties),
        "--epochs", str(args.epochs),
        "--seeds", args.seeds,
        "--methods", methods,
        "--adaptive_edge_budget", str(args.adaptive_edge_budget),
        "--adaptive_min_edges", str(args.adaptive_min_edges),
        "--adaptive_min_gain", str(args.adaptive_min_gain),
        "--max_degree", str(args.max_degree),
        "--pred_consensus_steps", str(args.pred_consensus_steps),
        "--pred_self_weight", str(args.pred_self_weight),
        "--vote_weighting", args.vote_weighting,
        "--reliability_power", str(args.reliability_power),
        "--topology_degree_power", str(args.topology_degree_power),
        "--final_reliability_floor", str(args.final_reliability_floor),
        "--topk_reliability_k", str(args.topk_reliability_k),
        "--matched_edge_count", str(args.matched_edge_count),
        "--readout_mode", args.readout_mode,
        "--validation_evaluator", args.validation_evaluator,
        "--active_party_id", str(args.active_party_id),
        "--device", args.device,
        "--save_csv", csv_name,
    ]
    if args.shuffle_party_positions:
        cmd += ["--shuffle_party_positions", "--party_shuffle_seed_offset", str(args.party_shuffle_seed_offset)]
    if args.plot:
        cmd += ["--plot", "--plot_dir", plot_dir]
    return cmd


def selected_settings(args: argparse.Namespace) -> List[Tuple[str, int]]:
    if args.mode == "main":
        return [("main", args.main_useful_parties)]
    if args.mode == "hard":
        return [("hard_noisy", args.hard_useful_parties)]
    if args.mode == "both":
        return [("main", args.main_useful_parties), ("hard_noisy", args.hard_useful_parties)]
    if args.mode == "quick":
        # Quick smoke test: one seed, low epochs, fewer methods.
        args.epochs = min(args.epochs, 5)
        args.seeds = "42"
        args.methods = ["local_uniform_vote", "fixed_ring", "adaptive_graph_val"]
        return [("quick", args.main_useful_parties)]
    raise ValueError(args.mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="main", choices=["quick", "main", "hard", "both"])
    parser.add_argument("--script", type=Path, default=Path("ta_dvfg_hgb_reliability.py"))
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"], choices=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_parties", type=int, default=15)
    parser.add_argument("--main_useful_parties", type=int, default=6)
    parser.add_argument("--hard_useful_parties", type=int, default=3)

    parser.add_argument("--adaptive_edge_budget", type=int, default=15)
    parser.add_argument("--adaptive_min_edges", type=int, default=5)
    parser.add_argument("--adaptive_min_gain", type=float, default=0.001)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--pred_consensus_steps", type=int, default=1)
    parser.add_argument("--pred_self_weight", type=float, default=0.85)
    parser.add_argument("--vote_weighting", type=str, default="topology_reliability", choices=["uniform", "reliability", "topology_reliability"])
    parser.add_argument("--reliability_power", type=float, default=1.0)
    parser.add_argument("--topology_degree_power", type=float, default=0.5)
    parser.add_argument("--final_reliability_floor", type=float, default=0.0)
    parser.add_argument("--topk_reliability_k", type=int, default=5)
    parser.add_argument("--matched_edge_count", type=int, default=5)
    parser.add_argument("--readout_mode", choices=["active_vote", "mean_party", "best_party"], default="active_vote")
    parser.add_argument("--validation_evaluator", choices=["all_parties", "active_party"], default="all_parties")
    parser.add_argument("--active_party_id", type=int, default=0)
    parser.add_argument("--shuffle_party_positions", action="store_true")
    parser.add_argument("--party_shuffle_seed_offset", type=int, default=2026)

    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.datasets = [d.upper() for d in args.datasets]
    args.seed_tag = "5seeds" if "," in args.seeds else f"seed{args.seeds}"
    return args


def main() -> None:
    args = parse_args()

    if not args.script.exists():
        raise FileNotFoundError(
            f"Cannot find {args.script}. Put this runner in the same folder as ta_dvfg_hgb_reliability.py, "
            "or pass --script path/to/ta_dvfg_hgb_reliability.py"
        )

    for setting_name, useful in selected_settings(args):
        for dataset in args.datasets:
            cmd = make_cmd(args, dataset, useful, setting_name)
            run_cmd(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
