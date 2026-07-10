"""Shared CLI and reporting helpers for Round 2 closure task scripts."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from closure_core import Bundle, Edge, load_bundle, paired_stats, score_predictions, select_topology, write_csv


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bundles_dir", nargs="?", type=Path, help="Directory containing Round 2 NPZ bundles.")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "round2_closure")
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--smoke", action="store_true", help="Run on deterministic synthetic bundles.")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--B", type=int, default=15)
    parser.add_argument("--d-max", type=int, default=2)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--eta", type=float, default=0.85)


def synthetic_bundles() -> list[Bundle]:
    bundles: list[Bundle] = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        y_val = rng.integers(0, 3, size=36)
        y_test = rng.integers(0, 3, size=48)
        val = rng.random((6, y_val.size, 3))
        test = rng.random((6, y_test.size, 3))
        for party in range(3):
            val[party, np.arange(y_val.size), y_val] += 1.8 - 0.25 * party
            test[party, np.arange(y_test.size), y_test] += 1.5 - 0.2 * party
        val /= val.sum(axis=-1, keepdims=True)
        test /= test.sum(axis=-1, keepdims=True)
        bundles.append(
            Bundle(
                val_probs=val,
                test_probs=test,
                y_val=y_val,
                y_test=y_test,
                dataset="SMOKE",
                setting="synthetic",
                seed=seed,
                metric="accuracy",
            )
        )
    return bundles


def load_bundles(args: argparse.Namespace) -> list[Bundle]:
    if args.smoke:
        return synthetic_bundles()
    if args.bundles_dir is None:
        raise SystemExit("bundles_dir is required unless --smoke is used")
    paths = sorted(args.bundles_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No bundles matched {args.pattern!r} under {args.bundles_dir}")
    return [load_bundle(path) for path in paths]


def evaluate_method(bundle: Bundle, method: str, probs: np.ndarray, edges: Sequence[Edge] = ()) -> dict[str, object]:
    return {
        "dataset": bundle.dataset,
        "setting": bundle.setting,
        "seed": bundle.seed,
        "method": method,
        "metric": bundle.metric,
        "score": score_predictions(probs, bundle.y_test, bundle.metric),
        "n_edges": len(edges),
        "edges": list(map(list, edges)),
    }


def topology(bundle: Bundle, args: argparse.Namespace, K: int | None = None) -> list[Edge]:
    return select_topology(
        bundle,
        args.K if K is None else K,
        B=args.B,
        d_max=args.d_max,
        m=args.m,
        tau=args.tau,
        eta=args.eta,
    )


def write_task_outputs(
    rows: list[dict[str, object]],
    out_dir: Path,
    stem: str,
    comparisons: Sequence[tuple[str, str]],
) -> None:
    per_seed = out_dir / f"{stem}_per_seed.csv"
    write_csv(rows, per_seed)
    stats_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["setting"])), []).append(row)
    for (dataset, setting), group in groups.items():
        by_method: dict[str, dict[int, float]] = {}
        for row in group:
            by_method.setdefault(str(row["method"]), {})[int(row["seed"])] = float(row["score"])
        for a_name, b_name in comparisons:
            common = sorted(set(by_method.get(a_name, {})) & set(by_method.get(b_name, {})))
            if not common:
                continue
            stats_row = paired_stats(
                [by_method[a_name][seed] for seed in common],
                [by_method[b_name][seed] for seed in common],
            )
            stats_rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "method_a": a_name,
                    "method_b": b_name,
                    "seeds": ",".join(map(str, common)),
                    **stats_row,
                }
            )
    write_csv(stats_rows, out_dir / f"{stem}_paired_stats.csv")
    print(f"[SMOKE TEST] wrote {per_seed}" if "SMOKE" in {str(r["dataset"]) for r in rows} else f"Wrote {per_seed}")


def run_task(
    description: str,
    stem: str,
    evaluator: Callable[[Bundle, argparse.Namespace], list[dict[str, object]]],
    comparisons: Sequence[tuple[str, str]],
) -> int:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    bundles = load_bundles(args)
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        rows.extend(evaluator(bundle, args))
    args.out.mkdir(parents=True, exist_ok=True)
    write_task_outputs(rows, args.out, stem, comparisons)
    return 0


def write_smoke_npz_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="tadvfg_round2_smoke_"))
    for bundle in synthetic_bundles():
        np.savez(
            tmp / f"{bundle.dataset}_{bundle.setting}_{bundle.seed}.npz",
            val_probs=bundle.val_probs,
            test_probs=bundle.test_probs,
            y_val=bundle.y_val,
            y_test=bundle.y_test,
            dataset=bundle.dataset,
            setting=bundle.setting,
            seed=bundle.seed,
            metric=bundle.metric,
        )
    return tmp
