"""Regenerate a small reviewer-facing artifact set from bundled CSV evidence.

This script never trains a model and never writes into ``artifacts/raw`` or
``artifacts/summaries``.  Its default destination is safe to delete and rebuild.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def copy_csv(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def regenerate_hgb(root: Path, output: Path) -> None:
    raw = root / "artifacts" / "raw" / "hgb"
    for name in ("paper_main_accuracy.csv", "paper_communication_efficiency.csv"):
        copy_csv(raw / name, output / "hgb" / name)


def regenerate_k(root: Path, output: Path) -> None:
    experiments = root / "experiments"
    sys.path.insert(0, str(experiments))
    try:
        import run_k_sensitivity as k_sensitivity
    finally:
        sys.path.pop(0)
    summary = pd.read_csv(root / "metadata" / "k_sensitivity_summary.csv")
    destination = output / "k_sensitivity"
    destination.mkdir(parents=True, exist_ok=True)
    k_sensitivity.write_tables(summary, destination)
    k_sensitivity.write_figures(summary, destination)


def regenerate_movielens(root: Path, output: Path) -> None:
    raw = root / "artifacts" / "raw" / "movielens"
    destination = output / "movielens"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("movielens_tadvfg_summary.csv", "movielens_15party_summary.csv"):
        copy_csv(raw / name, destination / name)

    summary = pd.read_csv(raw / "movielens_tadvfg_summary.csv")
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.scatter(summary["total_comm_mean"], summary["auc_mean"], color="#2455a4")
    for row in summary.itertuples(index=False):
        axis.annotate(str(row.method), (row.total_comm_mean, row.auc_mean), fontsize=7)
    axis.set_xlabel("Total communication")
    axis.set_ylabel("AUC")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "movielens_auc_communication.png", dpi=160)
    plt.close(fig)


def regenerate_mechanisms(root: Path, output: Path) -> None:
    source = root / "artifacts" / "raw" / "mechanisms" / "task2_paired_statistics.csv"
    destination = output / "mechanisms" / "same_state_exchange.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("comparison") == "A_vs_C_same_state_exchange":
                rows.append(row)
    if not rows:
        raise ValueError("same-state exchange rows are missing")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "artifacts" / "regenerated").resolve()
    if output == root or root in output.parents and output.name in {"raw", "summaries"}:
        raise ValueError("refusing to write into source evidence")
    regenerate_hgb(root, output)
    regenerate_k(root, output)
    regenerate_movielens(root, output)
    regenerate_mechanisms(root, output)
    print(f"Regenerated reviewer artifacts in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
