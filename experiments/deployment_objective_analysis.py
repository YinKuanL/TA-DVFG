"""Aggregate deployment-aware topology-objective experiments.

The raw engine keeps ``method=adaptive_graph_val`` for every TA-DVFG replay.
This analysis script materializes paper-facing method names such as
``TA-DVFG-local`` and ``TA-DVFG-joint-0.5`` from the topology objective columns.
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
}

REQUIRED_COLUMNS = [
    "dataset",
    "setting",
    "seed",
    "method",
    "topology_objective",
    "joint_lambda",
    "active_readout_acc",
    "local_post_consensus_mean_acc",
    "active_party_post_consensus_acc",
    "local_worst_acc",
    "local_best_acc",
    "num_edges",
    "peer_comm",
    "readout_comm",
    "total_comm",
    "control_plane_comm",
    "topology_eval_count",
    "topology_update_time",
]

OUTPUT_COLUMNS = [
    "dataset",
    "setting",
    "seed",
    "method",
    "source_method",
    "topology_objective",
    "joint_lambda",
    "active_readout_acc",
    "local_post_consensus_mean_acc",
    "active_party_post_consensus_acc",
    "local_worst_acc",
    "local_best_acc",
    "num_edges",
    "peer_comm",
    "readout_comm",
    "total_comm",
    "control_plane_comm",
    "topology_eval_count",
    "topology_update_time",
]

PLOT_ORDER = [
    "Full Mesh",
    "Best Matched Sparse",
    "TA-DVFG-active",
    "TA-DVFG-local",
    "TA-DVFG-joint-0.25",
    "TA-DVFG-joint-0.5",
    "TA-DVFG-joint-0.75",
    "Global Top-k Reliability",
]


def _lambda_text(value: object) -> str:
    return f"{float(value):g}"


def _deployment_method(row: pd.Series) -> str:
    source = str(row["method"])
    if source == "adaptive_graph_val":
        objective = str(row["topology_objective"])
        if objective == "local_mean":
            return "TA-DVFG-local"
        if objective == "joint":
            return f"TA-DVFG-joint-{_lambda_text(row['joint_lambda'])}"
        return "TA-DVFG-active"
    return METHOD_DISPLAY.get(source, source)


def _read_inputs(input_root: Path) -> pd.DataFrame:
    paths = sorted(
        path
        for path in input_root.rglob("*.csv")
        if not path.name.endswith("_summary.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No deployment-objective CSV files found under {input_root}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        frame["source_file"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["source_method"] = raw["method"]
    raw["method"] = raw.apply(_deployment_method, axis=1)
    raw["joint_lambda"] = raw["joint_lambda"].astype(float)
    return raw


def _summary(frame: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    metrics = [
        "active_readout_acc",
        "local_post_consensus_mean_acc",
        "active_party_post_consensus_acc",
        "local_worst_acc",
        "local_best_acc",
        "num_edges",
        "peer_comm",
        "readout_comm",
        "total_comm",
        "control_plane_comm",
        "topology_eval_count",
        "topology_update_time",
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


def _plot_tradeoff(summary: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["ACM", "DBLP"]
    colors = {
        "Full Mesh": "#4c78a8",
        "Best Matched Sparse": "#f58518",
        "TA-DVFG-active": "#54a24b",
        "TA-DVFG-local": "#b279a2",
        "TA-DVFG-joint-0.25": "#e45756",
        "TA-DVFG-joint-0.5": "#72b7b2",
        "TA-DVFG-joint-0.75": "#ff9da6",
        "Global Top-k Reliability": "#9d755d",
    }
    markers = {
        "Full Mesh": "s",
        "Best Matched Sparse": "D",
        "Global Top-k Reliability": "X",
    }
    labels = {
        "Full Mesh": "Full Mesh",
        "Best Matched Sparse": "Matched",
        "TA-DVFG-active": "TA-active",
        "TA-DVFG-local": "TA-local",
        "TA-DVFG-joint-0.25": "TA-joint-.25",
        "TA-DVFG-joint-0.5": "TA-joint-.5",
        "TA-DVFG-joint-0.75": "TA-joint-.75",
        "Global Top-k Reliability": "Top-k",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), sharex=False, sharey=False)
    handles_by_method = {}
    for panel_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        subset = summary[summary["dataset"] == dataset].copy()
        ax.set_title(f"{dataset} Hard", fontsize=12, pad=5)
        ax.set_xlabel("Active readout acc. (%)", fontsize=11)
        if panel_idx == 0:
            ax.set_ylabel("Local post-consensus mean acc. (%)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        for method in PLOT_ORDER:
            row = subset[subset["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            x = float(row["active_readout_acc_mean"]) * 100.0
            y = float(row["local_post_consensus_mean_acc_mean"]) * 100.0
            handle = ax.scatter(
                x,
                y,
                s=72,
                marker=markers.get(method, "o"),
                color=colors.get(method),
                edgecolors="black",
                linewidths=0.35,
                label=labels.get(method, method),
            )
            handles_by_method.setdefault(method, handle)
        x_pad = max(
            1.0,
            0.06
            * (subset["active_readout_acc_mean"].max() - subset["active_readout_acc_mean"].min())
            * 100.0,
        )
        y_pad = max(
            0.6,
            0.10
            * (
                subset["local_post_consensus_mean_acc_mean"].max()
                - subset["local_post_consensus_mean_acc_mean"].min()
            )
            * 100.0,
        )
        ax.set_xlim(subset["active_readout_acc_mean"].min() * 100.0 - x_pad, subset["active_readout_acc_mean"].max() * 100.0 + x_pad)
        ax.set_ylim(subset["local_post_consensus_mean_acc_mean"].min() * 100.0 - y_pad, subset["local_post_consensus_mean_acc_mean"].max() * 100.0 + y_pad)
    ordered_handles = [handles_by_method[m] for m in PLOT_ORDER if m in handles_by_method]
    ordered_labels = [labels[m] for m in PLOT_ORDER if m in handles_by_method]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=4,
        fontsize=8.4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.72, bottom=0.20, left=0.11, right=0.99, wspace=0.22)
    fig.savefig(figures_dir / "deployment_tradeoff.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(figures_dir / "deployment_tradeoff.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _print_frontier(summary: pd.DataFrame) -> None:
    print("\n=== Deployment objective frontier ===")
    for dataset, frame in summary.groupby("dataset"):
        print(f"{dataset}:")
        best_active = frame.sort_values("active_readout_acc_mean", ascending=False).iloc[0]
        best_local = frame.sort_values("local_post_consensus_mean_acc_mean", ascending=False).iloc[0]
        print(
            f"  best active: {best_active['method']} "
            f"{best_active['active_readout_acc_mean'] * 100:.2f}%"
        )
        print(
            f"  best local:  {best_local['method']} "
            f"{best_local['local_post_consensus_mean_acc_mean'] * 100:.2f}%"
        )


def run(input_root: Path, output_dir: Path) -> None:
    raw = _read_inputs(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = raw[OUTPUT_COLUMNS].copy()
    results.to_csv(output_dir / "deployment_objective_results.csv", index=False)

    summary = _summary(
        results,
        [
            "dataset",
            "setting",
            "method",
            "source_method",
            "topology_objective",
            "joint_lambda",
        ],
    )
    summary.to_csv(output_dir / "deployment_objective_summary.csv", index=False)
    _plot_tradeoff(summary, output_dir / "figures")
    _print_frontier(summary)
    print(f"\nSaved deployment-objective outputs to {output_dir}")
    print(f"Saved trade-off plot to {output_dir / 'figures' / 'deployment_tradeoff.pdf'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/deployment_objective_runs/deployment_objective"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/deployment_objective"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input_root.resolve(), arguments.output_dir.resolve())
