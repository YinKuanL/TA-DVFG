"""Regenerate paper figures that must remain readable at one-column width."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_many(paths: Iterable[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, target_dir / path.name)


def _plot_edge_budget_one(
    frame: pd.DataFrame,
    dataset: str,
    metric: str,
    metric_std: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    subset = frame[frame["dataset"] == dataset].copy()
    if subset.empty:
        raise ValueError(f"No edge-budget rows found for {dataset}")
    subset["param_adaptive_edge_budget"] = subset["param_adaptive_edge_budget"].astype(int)
    subset["param_adaptive_min_gain"] = subset["param_adaptive_min_gain"].astype(float)

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    markers = ["o", "s", "^"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, (gain, rows) in enumerate(sorted(subset.groupby("param_adaptive_min_gain"))):
        rows = rows.sort_values("param_adaptive_edge_budget")
        y = rows[metric].astype(float)
        yerr = rows[metric_std].astype(float)
        if "acc" in metric or "test" in metric or "best" in metric:
            y = y * 100.0
            yerr = yerr * 100.0
        ax.errorbar(
            rows["param_adaptive_edge_budget"],
            y,
            yerr=yerr,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.8,
            markersize=5.0,
            capsize=2.5,
            label=f"$\\tau={gain:g}$",
        )
    ax.set_xlabel("Edge budget", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted(subset["param_adaptive_edge_budget"].unique()))
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    ax.legend(
        fontsize=8.2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.36),
        ncol=3,
        title="Min gain",
        title_fontsize=8.2,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_edge_budget_figures(edge_budget_csv: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(edge_budget_csv)
    outputs = [
        (
            "ACM",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "Test@BestVal (%)",
            "ACM accuracy vs. edge budget",
            "edge_budget_acm_accuracy.pdf",
        ),
        (
            "DBLP",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "Test@BestVal (%)",
            "DBLP accuracy vs. edge budget",
            "edge_budget_dblp_accuracy.pdf",
        ),
        (
            "ACM",
            "final_edges_mean",
            "final_edges_std",
            "Selected edges",
            "ACM selected edges vs. budget",
            "edge_budget_acm_edges.pdf",
        ),
        (
            "DBLP",
            "final_edges_mean",
            "final_edges_std",
            "Selected edges",
            "DBLP selected edges vs. budget",
            "edge_budget_dblp_edges.pdf",
        ),
    ]
    written: list[Path] = []
    for dataset, metric, metric_std, ylabel, title, filename in outputs:
        path = output_dir / filename
        _plot_edge_budget_one(frame, dataset, metric, metric_std, ylabel, title, path)
        written.append(path)
    return written


def make_topology_objective_figure(topology_csv: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    frame = pd.read_csv(topology_csv)
    label_map = {
        "adaptive_pair": "Pairwise",
        "adaptive_complementarity": "Complement",
        "adaptive_graph_val": "TA-DVFG",
    }
    color_map = {
        "adaptive_pair": "#4c78a8",
        "adaptive_complementarity": "#72b7b2",
        "adaptive_graph_val": "#c43c39",
    }
    method_order = ["adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"]
    datasets = ["ACM", "DBLP"]

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    x = np.arange(len(datasets))
    width = 0.23
    for offset, method in enumerate(method_order):
        rows = (
            frame[frame["method"] == method]
            .set_index("dataset")
            .reindex(datasets)
            .reset_index()
        )
        positions = x + (offset - 1) * width
        means = rows["test_at_best_val_mean"].astype(float) * 100.0
        stds = rows["test_at_best_val_std"].astype(float) * 100.0
        ax.bar(
            positions,
            means,
            width=width,
            color=color_map[method],
            label=label_map[method],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.9,
            capsize=2.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dataset} Hard" for dataset in datasets], fontsize=10)
    ax.set_ylabel("Test@BestVal (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(70, 94)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=0.75,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    output_path = output_dir / "topology_objective_ablation.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [output_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-budget-csv",
        type=Path,
        default=REPO_ROOT / "results" / "final_analysis" / "tables" / "paper_edge_budget.csv",
    )
    parser.add_argument(
        "--topology-objective-csv",
        type=Path,
        default=REPO_ROOT / "results" / "final_analysis" / "tables" / "paper_topology_objective.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "final_analysis" / "figures",
    )
    parser.add_argument(
        "--overleaf-figures",
        type=Path,
        default=REPO_ROOT / "overleaf_tadvfg_aaai2026" / "figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = make_edge_budget_figures(args.edge_budget_csv.resolve(), args.output_dir.resolve())
    written.extend(
        make_topology_objective_figure(
            args.topology_objective_csv.resolve(),
            args.output_dir.resolve(),
        )
    )
    _copy_many(written, args.overleaf_figures.resolve())
    _copy_many([path.with_suffix(".png") for path in written], args.overleaf_figures.resolve())
    print("Wrote readable paper figures:")
    for path in written:
        print(f"  {path}")
    print(f"Copied PDFs/PNGs to {args.overleaf_figures.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
