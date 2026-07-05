"""Aggregate MovieLens 15-party Main/Hard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--main-dir", type=Path, default=Path("outputs/movielens_main15"))
    parser.add_argument("--hard-dir", type=Path, default=Path("outputs/movielens_hard15"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/movielens"))
    return parser.parse_args()


def load_setting(path: Path, setting: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "movielens_tadvfg_per_seed.csv")
    if "setting" in frame.columns:
        frame["setting"] = setting
    else:
        frame.insert(0, "setting", setting)
    return frame


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.concat(
        [load_setting(args.main_dir, "main"), load_setting(args.hard_dir, "hard")],
        ignore_index=True,
    )
    per_seed.to_csv(args.output_dir / "movielens_15party_per_seed.csv", index=False)

    rows = []
    for (setting, method), group in per_seed.groupby(["setting", "method"]):
        rows.append({
            "setting": setting,
            "method": method,
            "auc_mean": group["auc"].mean(),
            "auc_std": group["auc"].std(ddof=0),
            "accuracy_mean": group["test_at_best_val"].mean(),
            "accuracy_std": group["test_at_best_val"].std(ddof=0),
            "f1_mean": group["f1"].mean(),
            "f1_std": group["f1"].std(ddof=0),
            "ndcg_at_10_mean": group["ndcg_at_10"].mean(),
            "selected_links_mean": group["final_edges"].mean(),
            "peer_comm_mean": group["peer_to_peer_comm"].mean(),
            "readout_comm_mean": group["global_readout_comm"].mean(),
            "total_comm_mean": group["total_comm"].mean(),
            "topology_eval_count_mean": group["topology_eval_count"].mean(),
            "topology_update_time_mean": group["topology_update_time"].mean(),
            "useful_useful_edges_mean": group["useful_useful_edges"].mean(),
            "useful_noisy_edges_mean": group["useful_noisy_edges"].mean(),
            "noisy_noisy_edges_mean": group["noisy_noisy_edges"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "movielens_15party_summary.csv", index=False)
    make_figures(summary, args.output_dir)

    topologies = {}
    for setting, path in [("main", args.main_dir), ("hard", args.hard_dir)]:
        source = path / "movielens_selected_topologies.json"
        if source.exists():
            payload = json.loads(source.read_text(encoding="utf-8"))
            for key, value in payload.items():
                topologies[f"{setting}:{key}"] = value
    (args.output_dir / "movielens_15party_topologies.json").write_text(
        json.dumps(topologies, indent=2), encoding="utf-8"
    )

    full_mesh = per_seed[per_seed["method"] == "full_mesh"]
    if not (full_mesh["final_edges"] == 105).all():
        raise AssertionError("15-party Full Mesh should have exactly 105 undirected edges.")

    print("15-party MovieLens summary")
    print(summary[summary["method"].isin(["adaptive_graph_val", "full_mesh", "adaptive_pair", "topk_reliability_vote"])][
        ["setting", "method", "auc_mean", "accuracy_mean", "selected_links_mean", "total_comm_mean"]
    ].to_string(index=False))
    return 0


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping 15-party figures: {exc}")
        return
    labels = {
        "adaptive_graph_val": "TA-DVFG",
        "full_mesh": "Full Mesh",
        "adaptive_pair": "Adaptive Pairwise",
        "topk_reliability_vote": "Global Top-k",
        "random_matched": "Random Matched",
    }
    keep = list(labels)
    sub = summary[summary["method"].isin(keep)].copy()
    sub["label"] = sub["method"].map(labels)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.36
    settings = ["main", "hard"]
    x = range(len(settings))
    for offset, method in enumerate(["adaptive_graph_val", "full_mesh", "adaptive_pair", "topk_reliability_vote"]):
        vals = [
            float(sub[(sub["setting"] == setting) & (sub["method"] == method)]["auc_mean"].iloc[0])
            for setting in settings
        ]
        ax.bar([v + (offset - 1.5) * width / 1.7 for v in x], vals, width / 1.7, label=labels[method])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Main", "Hard"])
    ax.set_ylim(0.72, 0.76)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("MovieLens 15-party Main/Hard")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "movielens_15party_main_hard.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for _, row in sub.iterrows():
        ax.scatter(float(row["total_comm_mean"]), float(row["auc_mean"]), s=70)
        ax.annotate(f"{row['label']} ({row['setting']})", (float(row["total_comm_mean"]), float(row["auc_mean"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total prediction communication")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("15-party communication tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "movielens_15party_comm_tradeoff.png", dpi=240)
    plt.close(fig)

    edge = summary[summary["method"] == "adaptive_graph_val"].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = range(len(edge))
    ax.bar(x, edge["useful_useful_edges_mean"], label="Useful-useful")
    ax.bar(x, edge["useful_noisy_edges_mean"], bottom=edge["useful_useful_edges_mean"], label="Useful-weak")
    bottom = edge["useful_useful_edges_mean"] + edge["useful_noisy_edges_mean"]
    ax.bar(x, edge["noisy_noisy_edges_mean"], bottom=bottom, label="Weak-weak")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v).title() for v in edge["setting"]])
    ax.set_ylabel("Mean selected links")
    ax.set_title("TA-DVFG 15-party edge composition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "movielens_15party_edge_composition.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
