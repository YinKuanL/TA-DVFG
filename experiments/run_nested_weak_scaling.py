"""Nested weak-party scaling replay for TA-DVFG.

This implements the reviewer-requested protocol: train/load one full local
prediction cache with 3 useful parties and 15 weak parties, then evaluate
nested subsets containing the same 3 useful parties plus w weak parties.

The subset replay reuses the existing cached-prediction topology selector and
consensus code. No test label is used for topology selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
DATASET_VIEWS = {
    "ACM": "PAP,PSP,KNN",
    "DBLP": "APA,APCPA,APTPA,KNN",
}
METHODS = [
    "single_party_best",
    "topk_reliability_vote",
    "random_matched",
    "adaptive_pair",
    "adaptive_complementarity",
    "full_mesh",
    "adaptive_graph_val",
]


def load_engine():
    spec = importlib.util.spec_from_file_location("tadvfg_hgb_engine_nested", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TA-DVFG engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = load_engine()


def read_git_hash() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("ref:"):
        ref = text.split(" ", 1)[1]
        ref_path = REPO_ROOT / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8", errors="replace").strip()
    return text


def parse_ints(text: str) -> List[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def base_args(args: argparse.Namespace, dataset: str, seed: int) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.dataset = dataset
    ns.target_node = "auto"
    ns.data_dir = str((REPO_ROOT / "data_hgb").resolve())
    ns.ignore_dataset_split = False
    ns.device = args.device
    ns.num_parties = 18
    ns.graph_views = DATASET_VIEWS[dataset]
    ns.graph_k = 10
    ns.max_metapath_edges = 300000
    ns.random_graph_degree = 4
    ns.view_setting = "hard"
    ns.useful_parties = 3
    ns.distractor_feature_noise = 1.0
    ns.shuffle_party_positions = False
    ns.party_shuffle_seed_offset = args.party_shuffle_seed_offset
    ns.train_ratio = 0.6
    ns.val_ratio = 0.2
    ns.hidden_dim = args.hidden_dim
    ns.dropout = 0.5
    ns.epochs = args.epochs
    ns.lr = 0.01
    ns.weight_decay = 5e-4
    ns.lambda_kd = 0.0
    ns.pred_consensus_steps = 1
    ns.pred_self_weight = 0.85
    ns.consensus_mode = "standard"
    ns.consensus_reliability_margin = 0.0
    ns.vote_weighting = "topology_reliability"
    ns.reliability_power = 1.0
    ns.topology_degree_power = 0.5
    ns.final_reliability_floor = 0.0
    ns.topk_reliability_k = 5
    ns.readout_mode = "active_vote"
    ns.validation_evaluator = "all_parties"
    ns.active_party_id = 0
    ns.label_protocol = "all_supervised"
    ns.training_protocol = "supervised"
    ns.validation_protocol = "shared_val"
    ns.local_training_protocol = "all_parties_supervised"
    ns.validation_split_mode = "shared"
    ns.topology_val_fraction = 1.0
    ns.validation_split_seed_offset = 4096
    ns.passive_label_access = True
    ns.dropout_protocol = "none"
    ns.dropout_rate = 0.0
    ns.dropout_seed = 0
    ns.setting_name = "nested_weak_scaling"
    ns.topology_every = 1
    ns.topology_max_updates = 0
    ns.max_degree = 2
    ns.matched_edge_count = 5
    ns.adaptive_score = "graph_val"
    ns._active_adaptive_score = "graph_val"
    ns.topology_objective = "active"
    ns.joint_lambda = 0.5
    ns.adaptive_edge_budget = 15
    ns.adaptive_min_edges = 5
    ns.adaptive_min_gain = 0.001
    ns.adaptive_candidate_edges = 0
    ns.pair_acc_weight = 0.45
    ns.pair_gain_weight = 0.20
    ns.reliability_weight = 0.25
    ns.corrective_weight = 0.30
    ns.both_wrong_weight = 0.20
    ns.diversity_weight = 0.0
    ns.disagreement_weight = 0.05
    ns.methods = ",".join(METHODS)
    ns.seed = seed
    ns.seeds = str(seed)
    ns.save_csv = ""
    ns.cache_predictions = True
    ns.reuse_prediction_cache = True
    ns.skip_local_training_if_cache_exists = False
    ns.cache_only = False
    ns.cache_dir = str((args.cache_dir / dataset.lower()).resolve())
    ns.cache_namespace = "nested_weak18"
    ns.plot = False
    ns.plot_dir = ""
    ns.verbose = args.verbose
    ns.log_every = 20
    ns.local_training_labels = "all_parties_supervised_simulation"
    ns.validation_label_protocol = "all_parties"
    ns.topology_evaluator = "all_parties"
    ns.party_permutation_by_seed = {}
    ns.useful_party_mask_by_seed = {}
    ns.resolved_topk_reliability_k_by_seed = {}
    return ns


def load_or_build_full_cache(args: argparse.Namespace, dataset: str, seed: int) -> tuple[Dict[str, object], Path]:
    ns = base_args(args, dataset, seed)
    expected = ENGINE.prediction_cache_training_config(ns, seed)
    path = ENGINE.prediction_cache_path(ns, seed)
    if path.exists() and not args.force_rebuild_cache:
        cache = ENGINE.load_prediction_cache(path, expected)
        return cache, path

    ENGINE.set_seed(seed)
    x, y, graph_edges, split, target = ENGINE.load_hgb_dataset(ns, seed)
    train_idx, val_idx, test_idx = split
    xs, adjs, view_names, useful_mask, permutation = ENGINE.build_party_views(x, graph_edges, ns, seed)
    cache = ENGINE.train_local_prediction_cache(
        seed,
        ns,
        x,
        y,
        xs,
        adjs,
        train_idx,
        val_idx,
        test_idx,
        target,
        useful_mask,
        view_names,
        permutation,
    )
    ENGINE.save_prediction_cache(cache, path)
    return cache, path


def subset_cache(
    cache: Dict[str, object],
    weak_count: int,
    seed: int,
    shuffle_offset: int,
) -> tuple[Dict[str, object], List[int], List[int]]:
    useful = [0, 1, 2]
    weak = list(range(3, 18))
    selected = useful + weak[:weak_count]
    rng = np.random.default_rng(seed + shuffle_offset + weak_count * 1009)
    order = rng.permutation(len(selected)).tolist()
    selected_shuffled = [selected[i] for i in order]

    def party_slice(value):
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 18:
            return value[selected_shuffled].clone()
        return value

    sub = dict(cache)
    for key in [
        "probs_train",
        "probs_val",
        "probs_test",
        "probs_all",
        "probs_val_epochs",
        "probs_test_epochs",
    ]:
        value = cache.get(key)
        if isinstance(value, torch.Tensor):
            if key.endswith("_epochs"):
                sub[key] = value[:, selected_shuffled].clone()
            elif value.ndim >= 1 and value.shape[0] == 18:
                sub[key] = value[selected_shuffled].clone()

    sub["party_reliabilities"] = party_slice(cache["party_reliabilities"])
    sub["party_permutation"] = selected_shuffled
    sub["useful_party_mask"] = [idx in useful for idx in selected_shuffled]
    sub["view_names"] = [cache["view_names"][idx] for idx in selected_shuffled]
    sub["num_parties"] = len(selected_shuffled)
    sub["training_config"] = dict(cache["training_config"])
    sub["training_config"]["nested_weak_count"] = weak_count
    sub["training_config"]["nested_selected_parties"] = selected_shuffled
    sub["cache_fingerprint"] = ENGINE.stable_fingerprint(
        {
            "parent_cache_fingerprint": cache["cache_fingerprint"],
            "weak_count": weak_count,
            "selected_parties": selected_shuffled,
        }
    )
    return sub, selected, selected_shuffled


def eval_args(args: argparse.Namespace, dataset: str, seed: int, n: int) -> argparse.Namespace:
    ns = base_args(args, dataset, seed)
    ns.num_parties = n
    max_sparse_edges = max(0, n * ns.max_degree // 2)
    ns.matched_edge_count = min(5, n * (n - 1) // 2)
    ns.adaptive_edge_budget = min(15, max_sparse_edges)
    ns.adaptive_min_edges = min(5, ns.adaptive_edge_budget)
    ns.topk_reliability_k = min(5, n)
    ns.setting_name = "nested_weak_scaling"
    return ns


def evaluate_level(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    weak_count: int,
    cache: Dict[str, object],
    parent_path: Path,
) -> List[Dict[str, object]]:
    sub, selected, selected_shuffled = subset_cache(
        cache,
        weak_count,
        seed,
        args.party_shuffle_seed_offset,
    )
    ns = eval_args(args, dataset, seed, len(selected_shuffled))
    pseudo_path = parent_path.with_name(parent_path.stem + f"_w{weak_count}_subset.pt")
    rows: List[Dict[str, object]] = []
    for method in METHODS:
        if method == "topk_reliability_vote":
            ns._resolved_topk_reliability_k = ns.topk_reliability_k
        result = ENGINE.evaluate_cached_method(method, seed, ns, sub, pseudo_path)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "weak_count": weak_count,
                "num_parties": len(selected_shuffled),
                "method": method,
                "test_at_best_val": result.test_at_best_val,
                "macro_f1": result.macro_f1_at_best_val,
                "selected_links": result.final_edges,
                "peer_communication": result.peer_to_peer_comm,
                "readout_communication": result.global_readout_comm,
                "topology_control_communication": result.topology_update_comm,
                "total_communication": result.total_comm,
                "candidate_evaluations": result.topology_eval_count,
                "topology_time_seconds": result.topology_update_time,
                "selected_edges": result.selected_edges,
                "useful_party_mask": result.useful_party_mask,
                "party_view_names": result.party_view_names,
                "party_permutation": json.dumps(selected_shuffled),
                "parent_cache": str(parent_path),
                "parent_cache_fingerprint": str(cache["cache_fingerprint"]),
                "subset_cache_fingerprint": str(sub["cache_fingerprint"]),
            }
        )
    if hasattr(ns, "_resolved_topk_reliability_k"):
        delattr(ns, "_resolved_topk_reliability_k")
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_at_best_val",
        "macro_f1",
        "selected_links",
        "peer_communication",
        "readout_communication",
        "topology_control_communication",
        "total_communication",
        "candidate_evaluations",
        "topology_time_seconds",
    ]
    rows = []
    for (dataset, weak_count, method), group in raw.groupby(["dataset", "weak_count", "method"]):
        row = {"dataset": dataset, "weak_count": weak_count, "method": method, "seeds": group["seed"].nunique()}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, weak_count), group in raw.groupby(["dataset", "weak_count"]):
        pivot = group.pivot_table(index="seed", columns="method", values=["test_at_best_val", "total_communication"])
        if "adaptive_graph_val" not in pivot["test_at_best_val"].columns:
            continue
        for baseline in ["full_mesh", "adaptive_pair", "topk_reliability_vote"]:
            if baseline not in pivot["test_at_best_val"].columns:
                continue
            diff = pivot["test_at_best_val"]["adaptive_graph_val"] - pivot["test_at_best_val"][baseline]
            comm_base = pivot["total_communication"][baseline].replace(0, np.nan)
            comm_reduction = (pivot["total_communication"][baseline] - pivot["total_communication"]["adaptive_graph_val"]) / comm_base
            row = {
                "dataset": dataset,
                "weak_count": weak_count,
                "comparison": f"TA-DVFG vs {baseline}",
                "test_at_best_val_diff_mean": diff.mean(),
                "test_at_best_val_diff_std": diff.std(ddof=0),
                "communication_reduction_mean": comm_reduction.mean(),
                "communication_reduction_std": comm_reduction.std(ddof=0),
            }
            try:
                from scipy import stats

                row["paired_t_p"] = float(stats.ttest_rel(
                    pivot["test_at_best_val"]["adaptive_graph_val"],
                    pivot["test_at_best_val"][baseline],
                ).pvalue)
                row["wilcoxon_p"] = float(stats.wilcoxon(diff).pvalue)
            except Exception:
                row["paired_t_p"] = float("nan")
                row["wilcoxon_p"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def make_plots(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    label = {
        "single_party_best": "Best Single",
        "topk_reliability_vote": "Global Top-k",
        "random_matched": "Random Matched",
        "adaptive_pair": "Adaptive Pairwise",
        "adaptive_complementarity": "Adaptive Comp.",
        "full_mesh": "Full Mesh",
        "adaptive_graph_val": "TA-DVFG",
    }
    keep = ["single_party_best", "topk_reliability_vote", "adaptive_pair", "full_mesh", "adaptive_graph_val"]
    for dataset, data in summary.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method in keep:
            m = data[data["method"] == method].sort_values("weak_count")
            if m.empty:
                continue
            ax.errorbar(
                m["weak_count"],
                m["test_at_best_val_mean"],
                yerr=m["test_at_best_val_std"],
                marker="o",
                capsize=3,
                label=label[method],
            )
        ax.set_xlabel("Added weak parties")
        ax.set_ylabel("Test@BestVal accuracy")
        ax.set_title(f"{dataset} Hard nested weak-party scaling")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset.lower()}_weak_scaling_performance.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method in ["random_matched", "adaptive_pair", "full_mesh", "adaptive_graph_val"]:
            m = data[data["method"] == method].sort_values("weak_count")
            if m.empty:
                continue
            ax.errorbar(
                m["weak_count"],
                m["selected_links_mean"],
                yerr=m["selected_links_std"],
                marker="o",
                capsize=3,
                label=label[method],
            )
        ax.set_xlabel("Added weak parties")
        ax.set_ylabel("Selected links")
        ax.set_title(f"{dataset} Hard selected links")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset.lower()}_weak_scaling_links_comm.pdf")
        plt.close(fig)


def write_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": read_git_hash(),
        "command": "python experiments/run_nested_weak_scaling.py",
        "datasets": args.datasets,
        "seeds": parse_ints(args.seeds),
        "weak_counts": parse_ints(args.weak_counts),
        "methods": METHODS,
        "cache_protocol": "train/load 18-party hard cache with 3 useful + 15 weak, then replay nested subsets",
        "test_label_rule": "test labels are used only inside existing post-hoc evaluation, never for topology selection",
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP"], default=["ACM", "DBLP"])
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--weak-counts", default="0,2,5,10,15")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "nested_weak_cache")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "nested_weak_scaling")
    parser.add_argument("--party-shuffle-seed-offset", type=int, default=2026)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ENGINE.DEVICE = ENGINE.resolve_device(args.device)
    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        for seed in parse_ints(args.seeds):
            cache, parent_path = load_or_build_full_cache(args, dataset, seed)
            if int(cache["num_parties"]) != 18:
                raise AssertionError(f"Expected 18-party parent cache, got {cache['num_parties']}")
            for weak_count in parse_ints(args.weak_counts):
                if weak_count > 15:
                    raise ValueError("--weak-counts cannot exceed 15")
                rows.extend(evaluate_level(args, dataset, seed, weak_count, cache, parent_path))
                print(f"completed dataset={dataset} seed={seed} weak_count={weak_count}")

    raw = pd.DataFrame(rows)
    raw_path = args.output_dir / "raw_weak_scaling.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw)
    summary.to_csv(args.output_dir / "weak_scaling_summary.csv", index=False)
    deltas = paired_deltas(raw)
    deltas.to_csv(args.output_dir / "weak_scaling_paired_deltas.csv", index=False)
    if not args.no_plots:
        make_plots(summary, args.output_dir)
    write_manifest(args, args.output_dir)
    print(f"Wrote nested weak-party scaling outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

