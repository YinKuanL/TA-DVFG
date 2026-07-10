"""Cached-prediction closure utilities for TA-DVFG Round 2 experiments."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from scipy import stats as scipy_stats
except ModuleNotFoundError:  # pragma: no cover - exercised in lean runtimes
    scipy_stats = None


Edge = tuple[int, int]
EdgeSet = list[Edge]


@dataclass(frozen=True)
class Bundle:
    val_probs: np.ndarray
    test_probs: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    dataset: str
    setting: str
    seed: int
    metric: str = "accuracy"

    @property
    def n_parties(self) -> int:
        return int(self.val_probs.shape[0])


def _scalar(value: object, default: object = "") -> object:
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 2:
        probs = np.stack([1.0 - probs, probs], axis=-1)
    if probs.ndim != 3:
        raise ValueError(f"Expected probabilities with shape [N,S,C] or binary [N,S], got {probs.shape}")
    sums = probs.sum(axis=-1, keepdims=True)
    sums = np.where(sums <= 0.0, 1.0, sums)
    return probs / sums


def load_bundle(path: str | Path) -> Bundle:
    data = np.load(path, allow_pickle=True)
    required = ["val_probs", "test_probs", "y_val", "y_test"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required NPZ keys: {', '.join(missing)}")
    metric = str(_scalar(data["metric"] if "metric" in data else "accuracy", "accuracy"))
    return Bundle(
        val_probs=_normalize_probs(data["val_probs"]),
        test_probs=_normalize_probs(data["test_probs"]),
        y_val=np.asarray(data["y_val"]).astype(int),
        y_test=np.asarray(data["y_test"]).astype(int),
        dataset=str(_scalar(data["dataset"] if "dataset" in data else Path(path).stem, Path(path).stem)),
        setting=str(_scalar(data["setting"] if "setting" in data else "unknown", "unknown")),
        seed=int(_scalar(data["seed"] if "seed" in data else 0, 0)),
        metric=metric,
    )


def labels_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs).argmax(axis=-1)


def score_predictions(probs: np.ndarray, labels: np.ndarray, metric: str = "accuracy") -> float:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 3:
        probs = _normalize_probs(probs).mean(axis=0)
    elif probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=-1)
    elif probs.ndim != 2:
        raise ValueError(f"Expected scores with shape [S,C], [S], or [N,S,C], got {probs.shape}")
    if metric == "roc_auc":
        if probs.shape[-1] == 2:
            return float(stats_rank_auc(labels, probs[:, 1]))
        scores = []
        for cls in range(probs.shape[-1]):
            binary = (labels == cls).astype(int)
            if len(np.unique(binary)) == 2:
                scores.append(stats_rank_auc(binary, probs[:, cls]))
        return float(np.mean(scores)) if scores else float("nan")
    return float(np.mean(labels_from_probs(probs) == labels))


def stats_rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = rankdata(scores)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    auc = (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def rankdata(values: np.ndarray) -> np.ndarray:
    if scipy_stats is not None:
        return scipy_stats.rankdata(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _sem(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _normal_two_sided_p(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def _paired_t_p(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    if scipy_stats is not None:
        return float(scipy_stats.ttest_rel(left, right).pvalue)
    diff = left - right
    sem = _sem(diff)
    if sem == 0.0:
        return 1.0 if np.allclose(diff, 0.0) else 0.0
    return _normal_two_sided_p(float(np.mean(diff) / sem))


def _wilcoxon_p(diff: np.ndarray) -> float:
    if len(diff) < 2:
        return float("nan")
    if np.allclose(diff, 0.0):
        return 1.0
    if scipy_stats is not None:
        try:
            return float(scipy_stats.wilcoxon(diff, zero_method="wilcox").pvalue)
        except ValueError:
            return float("nan")
    nonzero = diff[~np.isclose(diff, 0.0)]
    ranks = rankdata(np.abs(nonzero))
    signed = float(np.sum(np.sign(nonzero) * ranks))
    variance = len(nonzero) * (len(nonzero) + 1) * (2 * len(nonzero) + 1) / 6.0
    return _normal_two_sided_p(signed / math.sqrt(variance)) if variance > 0.0 else float("nan")


def _t_critical_975(df: int) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(0.975, df=df))
    lookup = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    return lookup.get(df, 1.96)


def party_reliability(bundle: Bundle) -> np.ndarray:
    return np.asarray([score_predictions(p, bundle.y_val, bundle.metric) for p in bundle.val_probs], dtype=float)


def _edge_degrees(edges: Iterable[Edge], n_parties: int) -> np.ndarray:
    degrees = np.zeros(n_parties, dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    return degrees


def _mean_for_parties(probs: np.ndarray, parties: Sequence[int], weights: np.ndarray | None = None) -> np.ndarray:
    if not parties:
        parties = list(range(probs.shape[0]))
    selected = probs[np.asarray(parties, dtype=int)]
    if weights is None:
        return selected.mean(axis=0)
    weights = np.asarray(weights, dtype=float)
    if np.all(weights <= 0.0):
        weights = np.ones_like(weights)
    return np.average(selected, axis=0, weights=weights)


def readout_probs(bundle: Bundle, edges: Sequence[Edge], *, split: str = "test", degree_weighted: bool = True) -> np.ndarray:
    probs = bundle.test_probs if split == "test" else bundle.val_probs
    degrees = _edge_degrees(edges, bundle.n_parties)
    parties = [i for i, degree in enumerate(degrees) if degree > 0]
    if not parties:
        return local_readout(bundle, split=split)
    reliability = party_reliability(bundle)[parties]
    weights = reliability * np.sqrt(degrees[parties] + 1.0) if degree_weighted else reliability
    return _mean_for_parties(probs, parties, weights)


def active_readout(bundle: Bundle, edges: Sequence[Edge], degree_weighted: bool = True) -> np.ndarray:
    return readout_probs(bundle, edges, split="test", degree_weighted=degree_weighted)


def local_readout(bundle: Bundle, *, split: str = "test") -> np.ndarray:
    probs = bundle.test_probs if split == "test" else bundle.val_probs
    return probs.mean(axis=0)


def select_topology(
    bundle: Bundle,
    K: int,
    B: int = 15,
    d_max: int = 2,
    m: int = 5,
    tau: float = 0.001,
    eta: float = 0.85,
) -> EdgeSet:
    if K <= 0 or B <= 0:
        return []
    target_edges = min(int(K), int(B), bundle.n_parties * (bundle.n_parties - 1) // 2)
    reliabilities = party_reliability(bundle)
    reliable = reliabilities >= eta
    if reliable.sum() < 2:
        reliable = np.ones(bundle.n_parties, dtype=bool)
    candidates: list[Edge] = [
        (i, j)
        for i in range(bundle.n_parties)
        for j in range(i + 1, bundle.n_parties)
        if reliable[i] or reliable[j]
    ]
    selected: EdgeSet = []
    degrees = np.zeros(bundle.n_parties, dtype=int)
    current = score_predictions(local_readout(bundle, split="val"), bundle.y_val, bundle.metric)
    min_edges = min(int(m), target_edges)
    while len(selected) < target_edges:
        best: tuple[float, float, Edge] | None = None
        for edge in candidates:
            if edge in selected:
                continue
            i, j = edge
            if degrees[i] >= d_max or degrees[j] >= d_max:
                continue
            trial = selected + [edge]
            score = score_predictions(readout_probs(bundle, trial, split="val"), bundle.y_val, bundle.metric)
            gain = score - current
            tie = reliabilities[i] + reliabilities[j]
            candidate = (gain, tie, edge)
            if best is None or (-candidate[0], -candidate[1], candidate[2]) < (-best[0], -best[1], best[2]):
                best = candidate
        if best is None:
            break
        gain, _tie, edge = best
        if len(selected) >= min_edges and gain < tau:
            break
        selected.append(edge)
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
        current += gain
    return selected


def paired_stats(a: Sequence[float], b: Sequence[float]) -> dict[str, object]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"Paired inputs must have same shape, got {left.shape} and {right.shape}")
    diff = left - right
    n = int(diff.size)
    mean = float(np.mean(diff)) if n else float("nan")
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = _sem(diff)
        critical = _t_critical_975(n - 1)
        t_p = _paired_t_p(left, right)
        ci_low = float(mean - critical * sem)
        ci_high = float(mean + critical * sem)
        w_p = _wilcoxon_p(diff)
    else:
        t_p = float("nan")
        w_p = float("nan")
        ci_low = mean
        ci_high = mean
    return {
        "n_pairs": n,
        "mean_a": float(np.mean(left)) if n else float("nan"),
        "mean_b": float(np.mean(right)) if n else float("nan"),
        "mean_diff": mean,
        "mean_diff_pp": 100.0 * mean if math.isfinite(mean) else float("nan"),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "cohen_dz": mean / sd if sd > 0.0 else float("inf"),
        "wins": int(np.sum(diff > 0.0)),
        "ties": int(np.sum(np.isclose(diff, 0.0))),
        "losses": int(np.sum(diff < 0.0)),
    }


def write_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
