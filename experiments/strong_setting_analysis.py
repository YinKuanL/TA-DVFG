"""Aggregate the reviewer-critical strong-setting experiments.

This script intentionally keeps the all-supervised core outside the primary
tables.  The strict active-party label-holder runs are the main evidence; old
all-supervised results remain an oracle/supplementary reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


METHOD_DISPLAY = {
    "active_local_only": "Active Local Only",
    "global_topk_reliability": "Global Top-k Reliability",
    "full_mesh": "Full Mesh",
    "best_matched_sparse": "Best Matched Sparse",
    "adaptive_graph_val": "TA-DVFG",
}

REQUIRED_COLUMNS = [
    "dataset",
    "setting",
    "seed",
    "method",
    "label_protocol",
    "training_protocol",
    "validation_protocol",
    "active_party_id",
    "passive_label_access",
    "validation_evaluator",
    "test_at_best_val",
    "active_readout_acc",
    "local_post_consensus_mean_acc",
    "active_party_post_consensus_acc",
    "dropout_rate",
    "dropout_seed",
    "remaining_parties",
    "test_acc_after_dropout",
    "local_post_consensus_acc_after_dropout",
    "num_edges",
    "peer_comm",
    "readout_comm",
    "total_comm",
    "control_plane_comm",
    "topology_eval_count",
    "topology_update_time",
]


def _read_inputs(input_root: Path) -> pd.DataFrame:
    paths = sorted(
        path
        for path in input_root.rglob("*.csv")
        if not path.name.endswith("_summary.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No strong-setting CSV files found under {input_root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        frame["source_file"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["topology_evaluator"] = result["topology_evaluator"].replace(
        {"active_party_id": "active_party"}
    )
    result["method_display"] = result["method"].map(METHOD_DISPLAY).fillna(result["method"])
    return result


def _summary(frame: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    metrics = [
        "test_at_best_val",
        "active_readout_acc",
        "local_post_consensus_mean_acc",
        "active_party_post_consensus_acc",
        "test_acc_after_dropout",
        "local_post_consensus_acc_after_dropout",
        "num_edges",
        "peer_comm",
        "readout_comm",
        "total_comm",
        "control_plane_comm",
        "topology_eval_count",
        "topology_update_time",
        "remaining_parties",
    ]
    grouped = frame.groupby(list(group_columns), dropna=False)
    rows = grouped[metrics].agg(["mean", "std"]).reset_index()
    rows.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in rows.columns
    ]
    counts = grouped.size().rename("runs").reset_index()
    return rows.merge(counts, on=list(group_columns), how="left")


def _print_gaps(strict_summary: pd.DataFrame) -> None:
    print("\n=== Strong-setting gaps (active readout, percentage points) ===")
    for dataset, frame in strict_summary.groupby("dataset"):
        values = frame.set_index("method")["active_readout_acc_mean"].to_dict()
        ta = values.get("adaptive_graph_val")
        if ta is None:
            continue
        print(f"{dataset}:")
        for baseline, label in (
            ("full_mesh", "Full Mesh"),
            ("best_matched_sparse", "Best Matched Sparse"),
            ("global_topk_reliability", "Global Top-k Reliability"),
        ):
            if baseline in values:
                print(f"  TA-DVFG - {label}: {(ta - values[baseline]) * 100:+.2f} pp")


def run(input_root: Path, output_dir: Path) -> None:
    raw = _read_inputs(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict = raw[raw["dropout_protocol"] == "none"].copy()
    dropout = raw[raw["dropout_protocol"] == "random_party_dropout"].copy()

    strict.to_csv(output_dir / "strict_active_party_results.csv", index=False)
    strict_summary = _summary(
        strict,
        [
            "dataset",
            "setting",
            "method",
            "method_display",
            "label_protocol",
            "training_protocol",
            "validation_protocol",
            "passive_label_access",
        ],
    )
    strict_summary.to_csv(output_dir / "strict_active_party_summary.csv", index=False)

    no_global_columns = [
        "dataset",
        "setting",
        "seed",
        "method",
        "method_display",
        "active_readout_acc",
        "local_post_consensus_mean_acc",
        "active_party_post_consensus_acc",
        "num_edges",
        "peer_comm",
        "readout_comm",
        "total_comm",
    ]
    strict[no_global_columns].to_csv(
        output_dir / "no_global_readout_results.csv",
        index=False,
    )
    dropout.to_csv(output_dir / "missing_party_robustness_results.csv", index=False)

    combined_summary = _summary(
        raw,
        [
            "dataset",
            "setting",
            "method",
            "method_display",
            "dropout_protocol",
            "dropout_rate",
            "label_protocol",
            "training_protocol",
            "validation_protocol",
        ],
    )
    combined_summary.to_csv(output_dir / "strong_setting_summary.csv", index=False)
    _print_gaps(strict_summary)
    print(f"\nSaved strong-setting outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/strong_setting_runs/strong_setting"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/strong_setting"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input_root.resolve(), arguments.output_dir.resolve())
