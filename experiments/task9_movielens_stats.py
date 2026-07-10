"""Task 9: MovieLens per-seed delta and paired stats from CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from closure_core import paired_stats, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path, help="CSV with seed, method, auc, communication columns.")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "round2_closure")
    parser.add_argument("--reference", default="TA-DVFG")
    parser.add_argument("--baseline", default="BestSingle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"seed", "method", "auc", "communication"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{args.input} is missing columns: {sorted(missing)}")
    ref = frame[frame["method"] == args.reference][["seed", "auc", "communication"]]
    base = frame[frame["method"] == args.baseline][["seed", "auc", "communication"]]
    paired = ref.merge(base, on="seed", suffixes=("_reference", "_baseline"))
    rows = []
    for row in paired.to_dict("records"):
        rows.append(
            {
                "seed": row["seed"],
                "reference": args.reference,
                "baseline": args.baseline,
                "auc_delta": row["auc_reference"] - row["auc_baseline"],
                "communication_delta": row["communication_reference"] - row["communication_baseline"],
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "task9_movielens_stats_per_seed.csv")
    stats_row = {
        "reference": args.reference,
        "baseline": args.baseline,
        **paired_stats(paired["auc_reference"].to_numpy(), paired["auc_baseline"].to_numpy()),
    }
    write_csv([stats_row], args.out / "task9_movielens_stats_paired_stats.csv")
    print(f"Wrote MovieLens closure stats to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
