"""Task 10: simple label-leakage probes over saved gradient/logit NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from closure_core import score_predictions, stats_rank_auc, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("npz", type=Path, help="NPZ with y, grad, and optional logits arrays.")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "round2_closure")
    return parser.parse_args()


def _as_class_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], values.shape[1], -1).mean(axis=-1)
    return values


def macro_auc(y: np.ndarray, scores: np.ndarray) -> float:
    scores = _as_class_scores(scores)
    aucs = []
    for cls in range(scores.shape[1]):
        labels = (y == cls).astype(int)
        if len(np.unique(labels)) == 2:
            aucs.append(stats_rank_auc(labels, scores[:, cls]))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def main() -> int:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    if "y" not in data or "grad" not in data:
        raise SystemExit(f"{args.npz} must contain y and grad arrays")
    y = np.asarray(data["y"]).astype(int)
    grad = _as_class_scores(np.asarray(data["grad"]))
    attacks = {
        "argmin_gradient": -grad,
        "negative_gradient": -grad,
    }
    if "logits" in data:
        logits = _as_class_scores(np.asarray(data["logits"]))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        softmax = exp / exp.sum(axis=1, keepdims=True)
        attacks["softmax_minus_gradient"] = softmax - grad
    rows = []
    for name, scores in attacks.items():
        rows.append(
            {
                "attack": name,
                "reconstruction_accuracy": score_predictions(scores, y, "accuracy"),
                "macro_auc": macro_auc(y, scores),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "task10_label_leakage.csv")
    print(f"Wrote label-leakage probe results to {args.out / 'task10_label_leakage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
