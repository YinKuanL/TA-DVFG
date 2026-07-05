"""Extended MovieLens analyses for the model-heterogeneity claim.

This script is intentionally analysis-first. It preserves existing MovieLens
outputs and writes additional CSV/figure artifacts needed for the next paper
claim: model-heterogeneous predictors collaborate through prediction space
without hidden-representation alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.movielens_parties import (  # noqa: E402
    GenreKNNParty,
    MFParty,
    MovieMetadataParty,
    PopularityTemporalParty,
    UserProfileParty,
    party_specs,
)


METHOD_LABELS = {
    "single_party_best": "Best Single",
    "local_uniform_vote": "Uniform Vote",
    "local_reliability_vote": "Reliability Vote",
    "topk_reliability_vote": "Global Top-k",
    "random_matched": "Random Matched",
    "fixed_ring": "Ring",
    "adaptive_pair": "Adaptive Pairwise",
    "adaptive_complementarity": "Adaptive Complementarity",
    "full_mesh": "Full Mesh",
    "adaptive_graph_val": "TA-DVFG",
}


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def ci95(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        m = float(arr.mean()) if arr.size else float("nan")
        return m, 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    mean = float(diff.mean())
    std = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    out = {
        "mean_diff": mean,
        "std_diff": std,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "cohens_dz": mean / std if std > 0 else float("nan"),
    }
    try:
        from scipy import stats

        t = stats.ttest_rel(a, b)
        out["paired_t_p"] = float(t.pvalue)
        try:
            w = stats.wilcoxon(diff)
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    except Exception:
        out["paired_t_p"] = float("nan")
        out["wilcoxon_p"] = float("nan")
    return out


def load_outputs(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    party = pd.read_csv(output_dir / "movielens_party_performance.csv")
    per_seed = pd.read_csv(output_dir / "movielens_tadvfg_per_seed.csv")
    summary = pd.read_csv(output_dir / "movielens_tadvfg_summary.csv")
    return party, per_seed, summary


def infer_feature_dims() -> Tuple[int, int, int]:
    # MovieLens 1M users: normalized age + gender + 21 occupations.
    user_dim = 23
    # 18 genres + normalized release year.
    movie_dim = 19
    # user/movie activity, means, like rates, timestamp.
    pair_dim = 7
    return user_dim, movie_dim, pair_dim


def parameter_counts(num_users: int = 6040, num_movies: int = 3952) -> Dict[str, int]:
    user_dim, movie_dim, pair_dim = infer_feature_dims()
    models = {
        "rating_interaction_mf": MFParty(num_users, num_movies, 64),
        "user_profile_mlp": UserProfileParty(num_movies, user_dim, 32),
        "movie_metadata_mlp": MovieMetadataParty(num_users, movie_dim, 128),
        "genre_knn_mlp": GenreKNNParty(num_users, movie_dim, 48),
        "popularity_temporal_mlp": PopularityTemporalParty(pair_dim, 16),
    }
    return {
        name: int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        for name, model in models.items()
    }


def write_architecture_audit(output_dir: Path, party: pd.DataFrame) -> pd.DataFrame:
    counts = parameter_counts()
    specs = {spec.name: spec for spec in party_specs(5, "five")}
    graph_based = {
        "rating_interaction_mf": True,
        "user_profile_mlp": False,
        "movie_metadata_mlp": False,
        "genre_knn_mlp": True,
        "popularity_temporal_mlp": False,
    }
    model_family = {
        "rating_interaction_mf": "matrix factorization",
        "user_profile_mlp": "MLP + embedding",
        "movie_metadata_mlp": "metadata MLP + embedding",
        "genre_knn_mlp": "KNN-smoothed MLP",
        "popularity_temporal_mlp": "MLP",
    }
    rows = []
    for name, group in party.groupby("party"):
        row = {
            "party": name,
            "private_view": specs[name].private_view if name in specs else "",
            "model_family": model_family.get(name, ""),
            "hidden_dim": specs[name].hidden_dim if name in specs else np.nan,
            "graph_based": bool(graph_based.get(name, False)),
            "num_parameters": counts.get(name, np.nan),
        }
        for metric in ["val_accuracy", "val_auc", "test_accuracy", "test_auc", "test_f1"]:
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(group[metric])
        rows.append(row)
    audit = pd.DataFrame(rows).sort_values("test_auc_mean", ascending=False)
    audit.to_csv(output_dir / "party_architecture_audit.csv", index=False)
    audit[["party", "private_view", "model_family", "hidden_dim", "graph_based", "num_parameters"]].to_csv(
        output_dir / "party_architectures.csv",
        index=False,
    )
    return audit


def write_party_gap(output_dir: Path, party: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in party.groupby("party"):
        val_m, _ = mean_std(group["val_auc"])
        test_m, test_s = mean_std(group["test_auc"])
        rows.append({
            "party": name,
            "val_auc_mean": val_m,
            "test_auc_mean": test_m,
            "val_test_gap": val_m - test_m,
            "test_auc_std": test_s,
        })
    out = pd.DataFrame(rows).sort_values("test_auc_mean", ascending=False)
    out.to_csv(output_dir / "movielens_party_generalization_gap.csv", index=False)
    return out


def plot_party_auc(output_dir: Path, party: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    })
    names = []
    val_m, val_s, test_m, test_s = [], [], [], []
    for name, group in party.groupby("party"):
        names.append(name)
        m, s = mean_std(group["val_auc"])
        val_m.append(m)
        val_s.append(s)
        m, s = mean_std(group["test_auc"])
        test_m.append(m)
        test_s.append(s)
    order = np.argsort(test_m)[::-1]
    names = [names[i] for i in order]
    val_m = [val_m[i] for i in order]
    val_s = [val_s[i] for i in order]
    test_m = [test_m[i] for i in order]
    test_s = [test_s[i] for i in order]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, val_m, width=0.36, yerr=val_s, capsize=3, label="Validation ROC-AUC")
    ax.bar(x + 0.18, test_m, width=0.36, yerr=test_s, capsize=3, label="Test ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.6, 0.78)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("MovieLens party reliability and generalization")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "movielens_party_val_test_auc.png", dpi=220)
    plt.close(fig)


def write_pareto_and_stats(output_dir: Path, per_seed: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in per_seed.groupby("method"):
        rows.append({
            "method": method,
            "Method": METHOD_LABELS.get(method, method),
            "auc_mean": group["auc"].mean(),
            "auc_std": group["auc"].std(ddof=0),
            "total_comm_mean": group["total_comm"].mean(),
            "total_comm_std": group["total_comm"].std(ddof=0),
            "selected_edges_mean": group["final_edges"].mean(),
        })
    pareto = pd.DataFrame(rows).sort_values(["total_comm_mean", "auc_mean"])
    is_pareto = []
    for _, row in pareto.iterrows():
        dominated = (
            (pareto["total_comm_mean"] <= row["total_comm_mean"])
            & (pareto["auc_mean"] >= row["auc_mean"])
            & (
                (pareto["total_comm_mean"] < row["total_comm_mean"])
                | (pareto["auc_mean"] > row["auc_mean"])
            )
        ).any()
        is_pareto.append(not bool(dominated))
    pareto["pareto_optimal"] = is_pareto
    pareto.to_csv(output_dir / "movielens_pareto_summary.csv", index=False)

    base = per_seed.pivot_table(index="seed", columns="method", values=["auc", "total_comm", "final_edges"])
    stat_rows = []
    for baseline in ["full_mesh", "adaptive_pair", "topk_reliability_vote"]:
        if baseline not in base["auc"].columns or "adaptive_graph_val" not in base["auc"].columns:
            continue
        auc_ta = base["auc"]["adaptive_graph_val"].to_numpy()
        auc_b = base["auc"][baseline].to_numpy()
        comm_ta = base["total_comm"]["adaptive_graph_val"].to_numpy()
        comm_b = base["total_comm"][baseline].to_numpy()
        stats = paired_stats(auc_ta, auc_b)
        stats.update({
            "comparison": f"TA-DVFG vs {METHOD_LABELS.get(baseline, baseline)}",
            "baseline": baseline,
            "auc_diff_mean": stats["mean_diff"],
            "communication_reduction_mean": float(np.mean((comm_b - comm_ta) / np.maximum(comm_b, 1))),
            "communication_reduction_std": float(np.std((comm_b - comm_ta) / np.maximum(comm_b, 1), ddof=0)),
        })
        stat_rows.append(stats)
    stats_frame = pd.DataFrame(stat_rows)
    stats_frame.to_csv(output_dir / "movielens_paired_statistics.csv", index=False)

    per_seed_rows = []
    for seed, row in base.iterrows():
        if "adaptive_graph_val" not in base["auc"].columns:
            continue
        for baseline in ["full_mesh", "adaptive_pair"]:
            if baseline not in base["auc"].columns:
                continue
            per_seed_rows.append({
                "seed": seed,
                "comparison": f"TA-DVFG vs {METHOD_LABELS.get(baseline, baseline)}",
                "auc_diff": row[("auc", "adaptive_graph_val")] - row[("auc", baseline)],
                "communication_reduction": (
                    row[("total_comm", baseline)] - row[("total_comm", "adaptive_graph_val")]
                ) / max(row[("total_comm", baseline)], 1),
            })
    pd.DataFrame(per_seed_rows).to_csv(output_dir / "movielens_pareto_per_seed_deltas.csv", index=False)
    return pareto


def plot_pareto(output_dir: Path, pareto: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
    })
    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    key_offsets = {
        "Best Single": (8, 8),
        "Global Top-k": (8, 8),
        "TA-DVFG": (8, 8),
        "Full Mesh": (8, 8),
    }
    for _, row in pareto.iterrows():
        color = "#d62728" if row["Method"] == "TA-DVFG" else ("#1f77b4" if row["pareto_optimal"] else "#777777")
        comm_m = row["total_comm_mean"] / 1_000_000
        comm_std_m = row["total_comm_std"] / 1_000_000
        ax.errorbar(
            comm_m,
            row["auc_mean"],
            xerr=comm_std_m,
            yerr=row["auc_std"],
            fmt="o",
            capsize=3,
            color=color,
        )
        if row["Method"] in key_offsets:
            ax.annotate(
                row["Method"],
                (comm_m, row["auc_mean"]),
                xytext=key_offsets[row["Method"]],
                textcoords="offset points",
                fontsize=10,
            )
    ax.set_xlim(-0.25, 10.9)
    ax.set_xlabel("Total prediction communication (M scalars)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("MovieLens AUC-communication Pareto")
    ax.grid(alpha=0.25)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=7, label="TA-DVFG"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="#1f77b4", markersize=7, label="Pareto reference"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#777777", markersize=7, label="Other baseline"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "movielens_pareto_auc_communication.png", dpi=240)
    plt.close(fig)


def write_paper_tables(output_dir: Path, per_seed: pd.DataFrame, pareto: pd.DataFrame) -> None:
    wanted = [
        "single_party_best",
        "local_uniform_vote",
        "local_reliability_vote",
        "topk_reliability_vote",
        "adaptive_pair",
        "full_mesh",
        "adaptive_graph_val",
    ]
    rows = []
    for method in wanted:
        group = per_seed[per_seed["method"] == method]
        if group.empty:
            continue
        rows.append({
            "Method": METHOD_LABELS[method],
            "ROC-AUC": f"{group['auc'].mean():.4f} +/- {group['auc'].std(ddof=0):.4f}",
            "Accuracy": f"{group['test_at_best_val'].mean():.4f} +/- {group['test_at_best_val'].std(ddof=0):.4f}",
            "F1": f"{group['f1'].mean():.4f} +/- {group['f1'].std(ddof=0):.4f}",
            "NDCG@10": f"{group['ndcg_at_10'].mean():.4f} +/- {group['ndcg_at_10'].std(ddof=0):.4f}",
            "Links": f"{group['final_edges'].mean():.1f}",
            "Total communication": f"{group['total_comm'].mean():.0f}",
        })
    pd.DataFrame(rows).to_csv(output_dir / "movielens_table_a_five_party.csv", index=False)

    alignment_rows = [
        {
            "Method": "Reliability Vote",
            "ROC-AUC": float(per_seed.loc[per_seed["method"] == "local_reliability_vote", "auc"].mean()),
            "Communication": float(per_seed.loc[per_seed["method"] == "local_reliability_vote", "total_comm"].mean()),
            "Extra parameters": 0,
            "Hidden exchange?": "No",
            "Alignment training?": "No",
        },
        {
            "Method": "Full Mesh",
            "ROC-AUC": float(per_seed.loc[per_seed["method"] == "full_mesh", "auc"].mean()),
            "Communication": float(per_seed.loc[per_seed["method"] == "full_mesh", "total_comm"].mean()),
            "Extra parameters": 0,
            "Hidden exchange?": "No",
            "Alignment training?": "No",
        },
        {
            "Method": "TA-DVFG",
            "ROC-AUC": float(per_seed.loc[per_seed["method"] == "adaptive_graph_val", "auc"].mean()),
            "Communication": float(per_seed.loc[per_seed["method"] == "adaptive_graph_val", "total_comm"].mean()),
            "Extra parameters": 0,
            "Hidden exchange?": "No",
            "Alignment training?": "No",
        },
    ]
    pd.DataFrame(alignment_rows).to_csv(output_dir / "movielens_alignment_comparison_table.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "movielens")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    party, per_seed, summary = load_outputs(args.output_dir)
    audit = write_architecture_audit(args.output_dir, party)
    gap = write_party_gap(args.output_dir, party)
    pareto = write_pareto_and_stats(args.output_dir, per_seed, summary)
    write_paper_tables(args.output_dir, per_seed, pareto)
    if not args.no_plots:
        plot_party_auc(args.output_dir, party)
        plot_pareto(args.output_dir, pareto)

    print("MovieLens audit summary")
    print(audit[["party", "model_family", "hidden_dim", "num_parameters", "test_auc_mean", "test_accuracy_mean"]].to_string(index=False))
    print("\nPareto summary")
    print(pareto[["Method", "auc_mean", "total_comm_mean", "selected_edges_mean", "pareto_optimal"]].to_string(index=False))
    print(f"\nWrote extended MovieLens artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
