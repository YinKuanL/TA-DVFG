"""Generate publication-style figures for MovieLens TA-DVFG outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/movielens"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt

    summary = pd.read_csv(args.output_dir / "movielens_tadvfg_summary.csv")
    per_seed = pd.read_csv(args.output_dir / "movielens_tadvfg_per_seed.csv")
    party = pd.read_csv(args.output_dir / "movielens_party_performance.csv")

    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.18, summary["accuracy_mean"], width=0.36, label="Accuracy")
    ax.bar(x + 0.18, summary["auc_mean"], width=0.36, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("MovieLens 1M binary preference prediction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "movielens_accuracy_auc_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(summary["total_comm_mean"], summary["auc_mean"], s=70)
    for _, row in summary.iterrows():
        ax.annotate(row["method"], (row["total_comm_mean"], row["auc_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total prediction communication")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("MovieLens communication tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "movielens_comm_tradeoff.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    party_summary = party.groupby("party", as_index=False)["val_accuracy"].mean()
    ax.bar(party_summary["party"], party_summary["val_accuracy"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("MovieLens party reliability")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(args.output_dir / "movielens_party_reliability.png", dpi=220)
    plt.close(fig)

    topology_path = args.output_dir / "movielens_selected_topologies.json"
    if topology_path.exists():
        topologies = json.loads(topology_path.read_text(encoding="utf-8"))
        key = next((k for k in topologies if k.endswith(":adaptive_graph_val")), next(iter(topologies), None))
        if key:
            item = topologies[key]
            views = item.get("party_views", [])
            n = len(views)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            fig, ax = plt.subplots(figsize=(5.4, 5.4))
            ax.axis("off")
            for i, j in item.get("edges", []):
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], linewidth=2.0)
            ax.scatter(pos[:, 0], pos[:, 1], s=900, zorder=3)
            for i in range(n):
                label = f"P{i + 1}"
                ax.text(pos[i, 0], pos[i, 1], label, ha="center", va="center", fontsize=9, zorder=4)
            ax.set_title(f"TA-DVFG selected topology ({key})")
            fig.tight_layout()
            fig.savefig(args.output_dir / "movielens_topology_example.png", dpi=220)
            plt.close(fig)

    print(f"Saved MovieLens figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

