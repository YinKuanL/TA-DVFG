"""Run and package the HGB consensus-depth sensitivity study.

The runner is deliberately cache-only: local HGB predictors must already exist
in ``results/core_cache_v2`` and are reused for every consensus depth.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
DATASET_VIEWS = {
    "ACM": "PAP,PSP,KNN",
    "DBLP": "APA,APCPA,APTPA,KNN",
}
DATASET_TARGETS = {
    "ACM": 0.8553541898727417,
    "DBLP": 0.9024569869041443,
}
METHODS_BY_K = {
    1: [
        "topk_reliability_vote",
        "full_mesh",
        "adaptive_pair",
        "adaptive_complementarity",
        "adaptive_graph_val",
    ],
    2: ["full_mesh", "adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"],
    3: ["full_mesh", "adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"],
}
METHOD_LABELS = {
    "topk_reliability_vote": "Global Top-k",
    "full_mesh": "Full Mesh",
    "adaptive_pair": "Adaptive Pairwise",
    "adaptive_complementarity": "Adaptive Complementarity",
    "adaptive_graph_val": "TA-DVFG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP"], default=["ACM", "DBLP"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "core_cache_v2")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results" / "k_sensitivity_runs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_path(run_dir: Path, dataset: str, k: int) -> Path:
    return run_dir / f"{dataset.lower()}_hard_k{k}" / f"{dataset.lower()}_hard_k{k}.csv"


def build_command(args: argparse.Namespace, dataset: str, k: int) -> List[str]:
    out = output_path(args.run_dir, dataset, k)
    methods = ",".join(METHODS_BY_K[k])
    return [
        str(args.python.resolve()),
        str(ENGINE),
        "--dataset",
        dataset,
        "--graph_views",
        DATASET_VIEWS[dataset],
        "--num_parties",
        "15",
        "--view_setting",
        "hard",
        "--useful_parties",
        "3",
        "--epochs",
        str(args.epochs),
        "--seeds",
        args.seeds,
        "--methods",
        methods,
        "--device",
        args.device,
        "--data_dir",
        str(REPO_ROOT / "data_hgb"),
        "--cache_dir",
        str(args.cache_dir.resolve()),
        "--cache_predictions",
        "--reuse_prediction_cache",
        "--skip_local_training_if_cache_exists",
        "--shuffle_party_positions",
        "--party_shuffle_seed_offset",
        "2026",
        "--adaptive_edge_budget",
        "15",
        "--adaptive_min_edges",
        "5",
        "--adaptive_min_gain",
        "0.001",
        "--max_degree",
        "2",
        "--matched_edge_count",
        "5",
        "--pred_consensus_steps",
        str(k),
        "--pred_self_weight",
        "0.85",
        "--vote_weighting",
        "topology_reliability",
        "--topology_every",
        "20",
        "--save_csv",
        str(out.resolve()),
    ]


def run_command(command: List[str], cwd: Path, log_path: Path) -> float:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(subprocess.list2cmdline(command) + "\n\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def parse_edges(value: object) -> List[Tuple[int, int]]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except json.JSONDecodeError:
            raw = ast.literal_eval(str(value))
    edges = []
    for item in raw:
        if len(item) != 2:
            continue
        a, b = int(item[0]), int(item[1])
        if a != b:
            edges.append((min(a, b), max(a, b)))
    return sorted(set(edges))


def graph_metrics(edges: List[Tuple[int, int]], n: int, k: int) -> Dict[str, float]:
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = [False] * n
    comp_sizes = []
    diameter = math.nan
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        comp_sizes.append(len(nodes))
        if len(nodes) > 1:
            local_diam = 0
            node_set = set(nodes)
            for src in nodes:
                dist = {src: 0}
                q: deque[int] = deque([src])
                while q:
                    cur = q.popleft()
                    for nb in adj[cur]:
                        if nb in node_set and nb not in dist:
                            dist[nb] = dist[cur] + 1
                            q.append(nb)
                local_diam = max(local_diam, max(dist.values()))
            diameter = local_diam if math.isnan(diameter) else max(diameter, local_diam)

    def reachable_within(src: int, depth: int) -> int:
        if depth <= 0:
            return 1
        dist = {src: 0}
        q: deque[int] = deque([src])
        while q:
            cur = q.popleft()
            if dist[cur] >= depth:
                continue
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return len(dist)

    reach1 = [reachable_within(i, 1) for i in range(n)]
    reachk = [reachable_within(i, k) for i in range(n)]
    non_isolated = [size for size in comp_sizes if size > 1]
    return {
        "connected_components": float(len(comp_sizes)),
        "largest_component_size": float(max(comp_sizes) if comp_sizes else 0),
        "mean_component_size_nonisolated": float(np.mean(non_isolated)) if non_isolated else 0.0,
        "graph_diameter": float(diameter) if not math.isnan(diameter) else math.nan,
        "reachable_1hop": float(np.mean(reach1)),
        "reachable_Khop": float(np.mean(reachk)),
        "multihop_gain_fraction": float(np.mean([b > a for a, b in zip(reach1, reachk)])),
    }


def read_and_enrich(csv_path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["K"] = k
    frame["method_label"] = frame["method"].map(METHOD_LABELS).fillna(frame["method"])
    metric_rows = []
    for _, row in frame.iterrows():
        edges = parse_edges(row.get("selected_edges", "[]"))
        metric_rows.append(graph_metrics(edges, n=15, k=k))
    metrics = pd.DataFrame(metric_rows)
    out = pd.concat([frame.reset_index(drop=True), metrics], axis=1)
    if "peer_to_peer_comm" not in out.columns and "peer_comm" in out.columns:
        out["peer_to_peer_comm"] = out["peer_comm"]
    if "total_comm" not in out.columns:
        out["total_comm"] = out.get("inference_comm", 0)
    return out


def mean_std(group: pd.core.groupby.generic.DataFrameGroupBy, column: str) -> pd.DataFrame:
    return group[column].agg(["mean", "std"]).rename(columns={"mean": f"{column}_mean", "std": f"{column}_std"})


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group = raw.groupby(["dataset", "method", "method_label", "K"], dropna=False)
    pieces = [
        mean_std(group, "test_at_best_val"),
        mean_std(group, "active_readout_acc"),
        mean_std(group, "local_post_consensus_mean_acc"),
        mean_std(group, "final_edges"),
        group["total_comm"].mean().rename("comm_mean"),
        group["topology_update_time"].mean().rename("topology_time_mean"),
        group["reachable_1hop"].mean().rename("reachable_1hop_mean"),
        group["reachable_Khop"].mean().rename("reachable_Khop_mean"),
        group["multihop_gain_fraction"].mean().rename("multihop_gain_fraction_mean"),
    ]
    summary = pd.concat(pieces, axis=1).reset_index()
    summary = summary.rename(
        columns={
            "test_at_best_val_mean": "test_mean",
            "test_at_best_val_std": "test_std",
            "active_readout_acc_mean": "active_mean",
            "active_readout_acc_std": "active_std",
            "local_post_consensus_mean_acc_mean": "local_mean",
            "local_post_consensus_mean_acc_std": "local_std",
            "final_edges_mean": "edges_mean",
            "final_edges_std": "edges_std",
        }
    )
    return summary


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    table_dir = output_dir
    rows = []
    for (dataset, method, label), group in summary.groupby(["dataset", "method", "method_label"]):
        row = {"Dataset": dataset, "Method": label}
        for k in [1, 2, 3]:
            hit = group[group["K"] == k]
            if hit.empty:
                row[f"K={k}"] = "--"
            else:
                r = hit.iloc[0]
                row[f"K={k}"] = f"{100*r['test_mean']:.2f} $\\pm$ {100*r['test_std']:.2f}"
        rows.append(row)
    perf = pd.DataFrame(rows)
    perf.to_csv(table_dir / "k_sensitivity_table.csv", index=False)
    with (table_dir / "k_sensitivity_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llccc}\n\\toprule\n")
        handle.write("Dataset & Method & K=1 & K=2 & K=3 \\\\\n\\midrule\n")
        for _, row in perf.iterrows():
            handle.write(
                f"{latex_escape(row['Dataset'])} & {latex_escape(row['Method'])} & "
                f"{row['K=1']} & {row['K=2']} & {row['K=3']} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")

    ta = summary[summary["method"] == "adaptive_graph_val"].copy()
    ta_rows = []
    for _, row in ta.iterrows():
        ta_rows.append(
            {
                "Dataset": row["dataset"],
                "K": int(row["K"]),
                "Acc.": f"{100*row['test_mean']:.2f} $\\pm$ {100*row['test_std']:.2f}",
                "Links": f"{row['edges_mean']:.1f}",
                "1-hop reach": f"{row['reachable_1hop_mean']:.2f}",
                "K-hop reach": f"{row['reachable_Khop_mean']:.2f}",
                "Multi-hop gain %": f"{100*row['multihop_gain_fraction_mean']:.1f}",
            }
        )
    ta_table = pd.DataFrame(ta_rows)
    ta_table.to_csv(table_dir / "k_sensitivity_tadvfg_topology_table.csv", index=False)
    with (table_dir / "k_sensitivity_tadvfg_topology_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        handle.write("Dataset & K & Acc. & Links & 1-hop reach & K-hop reach & Multi-hop gain \\\\\n\\midrule\n")
        for _, row in ta_table.iterrows():
            handle.write(
                f"{latex_escape(row['Dataset'])} & {row['K']} & {row['Acc.']} & {row['Links']} & "
                f"{row['1-hop reach']} & {row['K-hop reach']} & {row['Multi-hop gain %']}\\% \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def write_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["adaptive_graph_val", "adaptive_pair", "adaptive_complementarity", "full_mesh"]
    labels = {m: METHOD_LABELS[m] for m in methods}
    for dataset, frame in summary[summary["method"].isin(methods)].groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for method in methods:
            sub = frame[frame["method"] == method].sort_values("K")
            ax.errorbar(sub["K"], 100 * sub["test_mean"], yerr=100 * sub["test_std"], marker="o", capsize=3, label=labels[method])
        ax.set_xlabel("Consensus depth K")
        ax.set_ylabel("Test@BestVal (%)")
        ax.set_title(f"{dataset} Hard K-sensitivity")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dataset.lower()}_k_accuracy.png", dpi=240)
        plt.close(fig)

    ta = summary[summary["method"] == "adaptive_graph_val"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for dataset, sub in ta.groupby("dataset"):
        sub = sub.sort_values("K")
        ax.plot(sub["K"], sub["reachable_Khop_mean"], marker="o", label=dataset)
    ax.set_xlabel("Consensus depth K")
    ax.set_ylabel("Avg. K-hop reachable parties")
    ax.set_title("TA-DVFG multi-hop reachability")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "k_reachability.png", dpi=240)
    plt.close(fig)


def write_analysis(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, commands: List[Dict[str, object]]) -> None:
    ta = summary[summary["method"] == "adaptive_graph_val"].copy()
    lines = [
        "# K-Sensitivity Analysis",
        "",
        "This study reuses cached HGB local-prediction trajectories and changes only `pred_consensus_steps`.",
        "TA-DVFG topology selection is rerun separately for every dataset and K.",
        "",
        "## Regression Gate",
    ]
    for dataset in ["ACM", "DBLP"]:
        actual = ta[(ta["dataset"] == dataset) & (ta["K"] == 1)]["test_mean"].iloc[0]
        target = DATASET_TARGETS[dataset]
        lines.append(f"- {dataset} Hard K=1 TA-DVFG: {actual:.6f}; cached paper target {target:.6f}; delta {actual - target:+.6f}.")
    lines += ["", "## Results"]
    for dataset, frame in ta.groupby("dataset"):
        ordered = frame.sort_values("K")
        vals = ", ".join(f"K={int(r.K)}: {100*r.test_mean:.2f}%" for _, r in ordered.iterrows())
        reach = ", ".join(f"K={int(r.K)}: {r.reachable_Khop_mean:.2f}" for _, r in ordered.iterrows())
        gain = ", ".join(f"K={int(r.K)}: {100*r.multihop_gain_fraction_mean:.1f}%" for _, r in ordered.iterrows())
        lines.append(f"- {dataset}: TA-DVFG Test@BestVal is {vals}.")
        lines.append(f"- {dataset}: average K-hop reach is {reach}; strict multi-hop expansion fractions are {gain}.")
    lines += [
        "",
        "## Interpretation Checklist",
        "",
        "1. Stability from K=1 to K=3: see `k_sensitivity_summary.csv`; treat changes larger than the seed-level standard deviation as meaningful.",
        "2. Predictive effect of K>1: use the per-dataset rows above, not cherry-picked seeds.",
        "3. Genuine multi-hop topology: supported when K-hop reach exceeds 1-hop reach and the strict expansion fraction is positive.",
        "4. Whole-graph TA-DVFG versus Pairwise/Complementarity: compare rows in `k_sensitivity_summary.csv` for each K.",
        "5. Accuracy-communication trade-off: peer communication scales with K, so K>1 should be justified only if accuracy or reachability evidence is needed.",
        "",
        "## Candidate Statement Audit",
    ]
    supports_reach = bool((ta[ta["K"] > 1]["reachable_Khop_mean"] > ta[ta["K"] > 1]["reachable_1hop_mean"]).all())
    stable = bool(ta.groupby("dataset")["test_mean"].apply(lambda s: s.max() - s.min() <= 0.02).all())
    if supports_reach and stable:
        lines.append(
            "Supported with qualification: although the headline experiments use one consensus step, the same learned topology supports multi-hop propagation. Across K in {1,2,3}, TA-DVFG remains stable while additional consensus steps expand the set of reachable peer predictions."
        )
    else:
        lines.append(
            "Not fully supported as written. Use the summary table to qualify either the stability or the multi-hop reachability clause."
        )
    lines += ["", "## Exact Commands"]
    for item in commands:
        lines.append(f"- {item['dataset']} K={item['K']}: `{item['command']}`")
    (output_dir / "k_sensitivity_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
    args.run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    commands: List[Dict[str, object]] = []
    timings: List[Dict[str, object]] = []

    for dataset in args.datasets:
        for k in args.ks:
            command = build_command(args, dataset, k)
            out = output_path(args.run_dir, dataset, k)
            out.parent.mkdir(parents=True, exist_ok=True)
            commands.append({"dataset": dataset, "K": k, "command": subprocess.list2cmdline(command)})
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            if out.exists() and not args.force:
                print(f"Skipping existing {out}")
                timings.append({"dataset": dataset, "K": k, "seconds": 0.0, "status": "skipped_existing"})
                continue
            print(f"Running {dataset} Hard K={k}")
            seconds = run_command(command, REPO_ROOT, out.parent / "run.log")
            timings.append({"dataset": dataset, "K": k, "seconds": seconds, "status": "success"})

        if not args.dry_run and 1 in args.ks:
            k1 = read_and_enrich(output_path(args.run_dir, dataset, 1), 1)
            actual = float(k1[k1["method"] == "adaptive_graph_val"]["test_at_best_val"].mean())
            target = DATASET_TARGETS[dataset]
            if abs(actual - target) > args.tolerance:
                raise RuntimeError(
                    f"K=1 regression failed for {dataset}: got {actual:.6f}, target {target:.6f}, "
                    f"tolerance {args.tolerance:.6f}. Stopping before trusting K>1."
                )

    if args.dry_run:
        return 0

    frames = []
    for dataset in args.datasets:
        for k in args.ks:
            frames.append(read_and_enrich(output_path(args.run_dir, dataset, k), k))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.output_dir / "k_sensitivity_raw.csv", index=False)

    summary = summarize(raw)
    wanted = [
        "dataset",
        "method",
        "method_label",
        "K",
        "test_mean",
        "test_std",
        "active_mean",
        "active_std",
        "local_mean",
        "local_std",
        "edges_mean",
        "edges_std",
        "comm_mean",
        "topology_time_mean",
        "reachable_1hop_mean",
        "reachable_Khop_mean",
        "multihop_gain_fraction_mean",
    ]
    summary[wanted].to_csv(args.output_dir / "k_sensitivity_summary.csv", index=False)
    write_tables(summary, args.output_dir)
    write_figures(summary, args.output_dir)
    write_analysis(raw, summary, args.output_dir, commands)
    (args.output_dir / "k_sensitivity_run_manifest.json").write_text(
        json.dumps({"commands": commands, "timings": timings}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote K-sensitivity artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
