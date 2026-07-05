"""Paired significance tests for the paper's main TA-DVFG comparisons."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "combined" / "raw_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "combined" / "paired_significance.csv",
    )
    parser.add_argument("--suite", default="core")
    parser.add_argument("--reference", default="adaptive_graph_val")
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["local_reliability_vote", "fixed_ring", "full_mesh"],
    )
    parser.add_argument("--metric", default="test_at_best_val")
    return parser.parse_args()


def paired_ci(differences: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    n = len(differences)
    mean = float(np.mean(differences))
    if n < 2:
        return mean, mean
    sem = stats.sem(differences)
    critical = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    return float(mean - critical * sem), float(mean + critical * sem)


def safe_wilcoxon(differences: np.ndarray) -> float:
    if np.allclose(differences, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(differences, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def holm_adjust(p_values: Iterable[float]) -> List[float]:
    values = list(p_values)
    finite = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [float("nan")] * len(values)
    running = 0.0
    m = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def compare_group(
    group: pd.DataFrame,
    group_values: Dict[str, object],
    reference: str,
    baseline: str,
    metric: str,
) -> Dict[str, object] | None:
    ref = group[group["method"] == reference][["seed", metric]].rename(columns={metric: "reference"})
    base = group[group["method"] == baseline][["seed", metric]].rename(columns={metric: "baseline"})
    paired = ref.merge(base, on="seed", how="inner").dropna()
    if paired.empty:
        return None

    differences = paired["reference"].to_numpy(float) - paired["baseline"].to_numpy(float)
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    ci_low, ci_high = paired_ci(differences)
    t_p = float(stats.ttest_rel(paired["reference"], paired["baseline"]).pvalue) if len(differences) > 1 else float("nan")
    effect_dz = mean_difference / std_difference if std_difference > 0 else float("inf")
    return {
        **group_values,
        "reference_method": reference,
        "baseline_method": baseline,
        "metric": metric,
        "n_pairs": len(differences),
        "reference_mean": float(paired["reference"].mean()),
        "baseline_mean": float(paired["baseline"].mean()),
        "mean_difference": mean_difference,
        "mean_difference_pp": 100.0 * mean_difference,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_p": t_p,
        "wilcoxon_p": safe_wilcoxon(differences),
        "cohen_dz": effect_dz,
        "seeds": ",".join(str(value) for value in paired["seed"].tolist()),
    }


def main() -> int:
    args = parse_args()
    raw = pd.read_csv(args.input)
    raw = raw[raw["suite"] == args.suite].copy()
    if raw.empty:
        print(f"No rows for suite={args.suite} in {args.input}")
        return 1
    if args.metric not in raw.columns:
        raise ValueError(f"Metric '{args.metric}' is not present in {args.input}")

    group_columns = ["job_id", "dataset", "setting"]
    rows: List[Dict[str, object]] = []
    for keys, group in raw.groupby(group_columns, dropna=False):
        group_values = dict(zip(group_columns, keys))
        for baseline in args.comparisons:
            row = compare_group(group, group_values, args.reference, baseline, args.metric)
            if row is not None:
                rows.append(row)

    if not rows:
        print("No paired comparisons had both methods and matching seeds.")
        return 1

    output = pd.DataFrame(rows)
    output["paired_t_p_holm"] = holm_adjust(output["paired_t_p"].tolist())
    output["wilcoxon_p_holm"] = holm_adjust(output["wilcoxon_p"].tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} paired tests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
