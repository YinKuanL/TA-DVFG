"""Combine completed job CSVs into analysis-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from final_analysis import add_communication_accounting


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-incomplete", action="store_true")
    return parser.parse_args()


def flatten_parameters(parameters: Dict[str, object]) -> Dict[str, object]:
    return {f"param_{key}": value for key, value in parameters.items()}


def load_jobs(results_root: Path, include_incomplete: bool) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for manifest_path in sorted(results_root.glob("*/*/job.json")):
        job_dir = manifest_path.parent
        if not include_incomplete and not (job_dir / "_SUCCESS").exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics_path = job_dir / manifest.get("output_filename", "metrics.csv")
            frame = pd.read_csv(metrics_path)
        except Exception as exc:
            print(f"warning: skipping {job_dir}: {exc}")
            continue

        frame.insert(0, "job_id", manifest["job_id"])
        frame.insert(1, "suite", manifest["suite"])
        frame.insert(2, "setting", manifest["setting"])
        frame.insert(3, "engine", manifest["engine"])
        for key, value in flatten_parameters(manifest.get("parameters", {})).items():
            frame[key] = value
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["job_id", "suite", "setting", "dataset", "method"]
    metric_columns = [
        "best_val",
        "test_at_best_val",
        "macro_f1_at_best_val",
        "best_observed_test",
        "final_test",
        "train_comm",
        "inference_comm",
        "peer_to_peer_comm",
        "global_readout_comm",
        "total_comm",
        "final_edges",
        "topology_update_comm",
        "topology_eval_count",
        "topology_update_time",
        "useful_useful_edges",
        "useful_noisy_edges",
        "noisy_noisy_edges",
    ]
    available_metrics = [column for column in metric_columns if column in raw.columns]
    summary = raw.groupby(group_columns, dropna=False)[available_metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    if "protocol_type" in raw.columns:
        protocols = (
            raw.groupby(group_columns, dropna=False)["protocol_type"]
            .first()
            .reset_index()
        )
        summary = summary.merge(protocols, on=group_columns, how="left")

    parameter_columns = [column for column in raw.columns if column.startswith("param_")]
    if parameter_columns:
        params = raw.groupby("job_id", dropna=False)[parameter_columns].first().reset_index()
        summary = summary.merge(params, on="job_id", how="left")
    return summary


def write_paper_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    core = summary[summary["suite"] == "core"].copy()
    if not core.empty:
        columns = [
            "dataset",
            "setting",
            "method",
            "protocol_type",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "macro_f1_at_best_val_mean",
            "macro_f1_at_best_val_std",
            "inference_comm_mean",
            "peer_to_peer_comm_mean",
            "global_readout_comm_mean",
            "total_comm_mean",
            "final_edges_mean",
            "topology_update_comm_mean",
            "topology_eval_count_mean",
            "topology_update_time_mean",
        ]
        core[[column for column in columns if column in core.columns]].to_csv(
            output_dir / "paper_core_table.csv", index=False
        )

    topology = summary[summary["suite"] == "topology_objective"].copy()
    if not topology.empty:
        columns = [
            "dataset",
            "method",
            "protocol_type",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "final_edges_mean",
        ]
        topology[[column for column in columns if column in topology.columns]].to_csv(
            output_dir / "paper_topology_objective_table.csv", index=False
        )

    communication = core[
        core["method"].isin(["fixed_ring", "full_mesh", "adaptive_graph_val"])
    ].copy()
    if not communication.empty:
        columns = [
            "dataset",
            "setting",
            "method",
            "test_at_best_val_mean",
            "inference_comm_mean",
            "peer_to_peer_comm_mean",
            "global_readout_comm_mean",
            "total_comm_mean",
            "final_edges_mean",
            "topology_update_comm_mean",
            "topology_eval_count_mean",
            "topology_update_time_mean",
        ]
        communication[[column for column in columns if column in communication.columns]].to_csv(
            output_dir / "paper_communication_table.csv", index=False
        )

    reviewer = summary[summary["suite"] == "reviewer_safe"].copy()
    if not reviewer.empty:
        columns = [
            "dataset",
            "setting",
            "method",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "macro_f1_at_best_val_mean",
            "macro_f1_at_best_val_std",
            "inference_comm_mean",
            "final_edges_mean",
            "topology_update_comm_mean",
            "topology_eval_count_mean",
            "topology_update_time_mean",
        ]
        reviewer[[column for column in columns if column in reviewer.columns]].to_csv(
            output_dir / "paper_reviewer_safe_table.csv", index=False
        )

        overhead = reviewer[
            reviewer["method"].isin(["adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"])
        ].copy()
        columns = [
            "dataset",
            "setting",
            "method",
            "topology_update_comm_mean",
            "topology_update_comm_std",
            "topology_eval_count_mean",
            "topology_eval_count_std",
            "topology_update_time_mean",
            "topology_update_time_std",
            "inference_comm_mean",
        ]
        overhead[[column for column in columns if column in overhead.columns]].to_csv(
            output_dir / "paper_topology_overhead_table.csv", index=False
        )

    readout = summary[summary["suite"] == "readout_ablation"].copy()
    if not readout.empty:
        columns = [
            "dataset",
            "setting",
            "param_readout_mode",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "macro_f1_at_best_val_mean",
            "macro_f1_at_best_val_std",
        ]
        readout[[column for column in columns if column in readout.columns]].to_csv(
            output_dir / "paper_readout_ablation_table.csv", index=False
        )

    evaluator = summary[summary["suite"] == "active_party_evaluator"].copy()
    if not evaluator.empty:
        columns = [
            "dataset",
            "setting",
            "param_validation_evaluator",
            "param_active_party_id",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "topology_update_comm_mean",
            "topology_update_time_mean",
        ]
        evaluator[[column for column in columns if column in evaluator.columns]].to_csv(
            output_dir / "paper_active_party_evaluator_table.csv", index=False
        )


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or results_root / "combined").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_jobs(results_root, args.include_incomplete)
    if raw.empty:
        print(f"No completed metrics.csv files found under {results_root}")
        return 1
    raw = add_communication_accounting(raw)

    summary = summarize(raw)
    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_results.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_paper_tables(summary, output_dir)

    print(f"Wrote {len(raw)} seed-method rows to {raw_path}")
    print(f"Wrote {len(summary)} aggregate rows to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
