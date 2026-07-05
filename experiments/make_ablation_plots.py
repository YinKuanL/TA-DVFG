"""Create paper-ready plots from aggregated ablation results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "combined" / "summary_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "combined" / "figures",
    )
    return parser.parse_args()


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_edge_budget(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame[frame["suite"] == "edge_budget"].copy()
    if frame.empty:
        return
    frame["budget"] = pd.to_numeric(frame["param_adaptive_edge_budget"])
    frame["min_gain"] = pd.to_numeric(frame["param_adaptive_min_gain"])
    for dataset, group in frame.groupby("dataset"):
        plt.figure(figsize=(6.2, 4.2))
        for min_gain, curve in group.groupby("min_gain"):
            curve = curve.sort_values("budget")
            plt.errorbar(
                curve["budget"],
                curve["test_at_best_val_mean"],
                yerr=curve["test_at_best_val_std"],
                marker="o",
                capsize=4,
                label=f"min gain={min_gain:g}",
            )
        plt.xlabel("Adaptive edge budget")
        plt.ylabel("Test@BestVal")
        plt.title(f"{dataset}: budget / early-stop ablation")
        plt.legend()
        save_figure(output_dir / f"edge_budget_{dataset.lower()}.png")


def plot_noise_ratio(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame[frame["suite"] == "noise_ratio"].copy()
    if frame.empty:
        return
    frame["useful"] = pd.to_numeric(frame["param_useful_parties"])
    for dataset, group in frame.groupby("dataset"):
        plt.figure(figsize=(6.2, 4.2))
        for method, curve in group.groupby("method"):
            curve = curve.sort_values("useful")
            plt.plot(curve["useful"], curve["test_at_best_val_mean"], marker="o", label=method)
        plt.xlabel("Useful parties (out of 15)")
        plt.ylabel("Test@BestVal")
        plt.title(f"{dataset}: useful/noisy-party robustness")
        plt.legend(fontsize=8)
        save_figure(output_dir / f"noise_ratio_{dataset.lower()}.png")


def plot_modality(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame[frame["suite"] == "modality_ablation"].copy()
    if frame.empty:
        return
    frame["modality"] = frame["param_modalities"].astype(str)
    for setting, group in frame.groupby("setting"):
        group = group.sort_values("modality")
        plt.figure(figsize=(7.2, 4.2))
        plt.bar(group["modality"], group["test_at_best_val_mean"], yerr=group["test_at_best_val_std"], capsize=4)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Test@BestVal")
        plt.title(f"ACM {setting}: modality ablation")
        save_figure(output_dir / f"modality_acm_{setting}.png")


def plot_consensus(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame[frame["suite"] == "consensus"].copy()
    if frame.empty:
        return
    for (dataset, setting), group in frame.groupby(["dataset", "setting"]):
        labels = group["job_id"].str.rsplit("_", n=1).str[-1]
        order = group.assign(label=labels).sort_values("label")
        plt.figure(figsize=(7.2, 4.2))
        plt.bar(order["label"], order["test_at_best_val_mean"], yerr=order["test_at_best_val_std"], capsize=4)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Test@BestVal")
        plt.title(f"{dataset} {setting}: consensus ablation")
        save_figure(output_dir / f"consensus_{dataset.lower()}_{setting}.png")


def plot_readout(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame[frame["suite"] == "readout_ablation"].copy()
    if frame.empty:
        return
    for (dataset, setting), group in frame.groupby(["dataset", "setting"]):
        group = group.sort_values("param_readout_mode")
        plt.figure(figsize=(6.2, 4.2))
        plt.bar(
            group["param_readout_mode"],
            group["test_at_best_val_mean"],
            yerr=group["test_at_best_val_std"],
            capsize=4,
        )
        plt.ylabel("Test@BestVal")
        plt.title(f"{dataset} {setting}: final readout")
        save_figure(output_dir / f"readout_{dataset.lower()}_{setting}.png")


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for plotter in (plot_edge_budget, plot_noise_ratio, plot_modality, plot_consensus, plot_readout):
        plotter(frame, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
