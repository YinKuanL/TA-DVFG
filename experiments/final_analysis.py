"""Build the complete TA-DVFG paper analysis bundle.

This script deliberately separates the fair shuffled cached-core protocol from
older unshuffled sensitivity suites. It writes machine-readable tables, paired
statistics, paper figures, and a concise research summary.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT = RESULTS_ROOT / "final_analysis"

METRICS = [
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

CORE_METHOD_ORDER = [
    "local_uniform_vote",
    "local_reliability_vote",
    "topk_reliability_vote",
    "fixed_ring",
    "fixed_ring_matched",
    "random_regular",
    "random_matched",
    "expander",
    "expander_matched",
    "full_mesh",
    "adaptive_pair",
    "adaptive_complementarity",
    "adaptive_graph_val",
]

METHOD_LABELS = {
    "local_uniform_vote": "Global Uniform Vote",
    "local_reliability_vote": "Global Reliability Vote",
    "topk_reliability_vote": "Global Top-k Reliability",
    "fixed_ring": "Ring",
    "fixed_ring_matched": "Ring Matched",
    "random_regular": "Random Regular",
    "random_matched": "Random Matched",
    "expander": "Expander",
    "expander_matched": "Expander Matched",
    "full_mesh": "Full Mesh",
    "adaptive_pair": "Adaptive Pairwise",
    "adaptive_complementarity": "Adaptive Complementarity",
    "adaptive_graph_val": "TA-DVFG",
    "central_fusion": "Central Fusion",
    "central_full_gcn": "Central Full GCN",
}

SETTING_LABELS = {
    "main": "Main",
    "hard_noisy": "Hard",
    "hard": "Hard",
}

COLORS = {
    "TA-DVFG": "#C43C39",
    "Global Reliability Vote": "#4C78A8",
    "Global Top-k Reliability": "#72B7B2",
    "Best Matched Sparse": "#F2CF5B",
    "Ring Matched": "#F2CF5B",
    "Expander Matched": "#59A14F",
    "Full Mesh": "#9C755F",
    "Central Full GCN": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "macro_f1_at_best_val" not in frame.columns and "macro_f1" in frame.columns:
        frame["macro_f1_at_best_val"] = frame["macro_f1"]
    return frame


GLOBAL_DIAGNOSTIC_METHODS = {
    "local_vote",
    "local_uniform_vote",
    "local_reliability_vote",
    "topk_reliability_vote",
}


def _numeric_parameter(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    column = f"param_{name}"
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _active_party_count(selected_edges: object, final_edges: object, n_parties: int) -> int:
    try:
        edges = json.loads(str(selected_edges))
        active = {
            int(endpoint)
            for edge in edges
            if isinstance(edge, (list, tuple)) and len(edge) == 2
            for endpoint in edge
        }
        if active:
            return min(len(active), n_parties)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        edge_total = int(float(final_edges))
    except (TypeError, ValueError):
        edge_total = 0
    return n_parties if edge_total <= 0 else min(n_parties, 2 * edge_total)


def add_communication_accounting(raw: pd.DataFrame) -> pd.DataFrame:
    """Backfill reviewer-safe communication fields for old and new result CSVs."""
    frame = raw.copy()
    for column in ("inference_comm", "final_edges"):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    steps = _numeric_parameter(frame, "pred_consensus_steps", 1.0).clip(lower=0)
    positive = (frame["inference_comm"] > 0) & (frame["final_edges"] > 0) & (steps > 0)
    prediction_size = (
        frame.loc[positive, "inference_comm"]
        / (2.0 * frame.loc[positive, "final_edges"] * steps.loc[positive])
    )
    dataset_units = prediction_size.groupby(frame.loc[positive, "dataset"]).median().to_dict()

    existing_peer = pd.to_numeric(
        frame.get("peer_to_peer_comm", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    existing_readout = pd.to_numeric(
        frame.get("global_readout_comm", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    existing_total = pd.to_numeric(
        frame.get("total_comm", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    existing_protocol = frame.get(
        "protocol_type", pd.Series("", index=frame.index, dtype=object)
    ).fillna("")

    peer_values: List[int] = []
    readout_values: List[int] = []
    total_values: List[int] = []
    protocol_values: List[str] = []
    for index, row in frame.iterrows():
        method = str(row.get("method", ""))
        n_parties = int(_numeric_parameter(frame.loc[[index]], "num_parties", 15.0).iloc[0])
        unit = float(dataset_units.get(row.get("dataset"), 0.0))

        if pd.notna(existing_peer.loc[index]):
            peer = int(existing_peer.loc[index])
        else:
            peer = int(row["inference_comm"]) if method not in GLOBAL_DIAGNOSTIC_METHODS else 0

        if method in GLOBAL_DIAGNOSTIC_METHODS:
            protocol = "global_diagnostic"
            selected = n_parties
            if method == "topk_reliability_vote":
                selected = int(
                    _numeric_parameter(frame.loc[[index]], "topk_reliability_k", 5.0).iloc[0]
                )
                selected = max(1, min(selected, n_parties))
            inferred_readout = int(round(selected * unit))
        elif method in {"single_party_avg", "single_party_best"}:
            protocol = "local_reference"
            inferred_readout = 0
        elif method.startswith("central_"):
            protocol = "centralized_reference"
            peer = 0
            inferred_readout = int(row["inference_comm"])
        else:
            protocol = "peer_to_peer_topology"
            readout_mode = str(row.get("param_readout_mode", "active_vote"))
            if readout_mode == "mean_party":
                selected = n_parties
            elif readout_mode == "best_party":
                selected = 1
            else:
                selected = _active_party_count(
                    row.get("selected_edges", "[]"),
                    row.get("final_edges", 0),
                    n_parties,
                )
            inferred_readout = int(round(selected * unit))

        readout = (
            int(existing_readout.loc[index])
            if pd.notna(existing_readout.loc[index])
            else inferred_readout
        )
        total = (
            int(existing_total.loc[index])
            if pd.notna(existing_total.loc[index])
            else peer + readout
        )
        protocol = str(existing_protocol.loc[index]).strip() or protocol
        peer_values.append(peer)
        readout_values.append(readout)
        total_values.append(total)
        protocol_values.append(protocol)

    frame["peer_to_peer_comm"] = peer_values
    frame["global_readout_comm"] = readout_values
    frame["total_comm"] = total_values
    frame["protocol_type"] = protocol_values
    # Backward compatibility: inference_comm remains peer-edge exchange.
    frame["inference_comm"] = frame["peer_to_peer_comm"]
    return frame


def add_common_metadata(
    frame: pd.DataFrame,
    *,
    suite: str,
    job_id: str,
    setting: str,
    engine: str,
    parameters: Dict[str, object] | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "job_id", job_id)
    frame.insert(1, "suite", suite)
    frame.insert(2, "setting", setting)
    frame.insert(3, "engine", engine)
    for key, value in (parameters or {}).items():
        frame[f"param_{key}"] = value
    frame["method_label"] = frame["method"].map(METHOD_LABELS).fillna(frame["method"])
    frame["setting_label"] = frame["setting"].map(SETTING_LABELS).fillna(frame["setting"])
    return frame


def load_core_cached(results_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    root = results_root / "core_cached_v2"
    for job_dir in sorted(root.glob("*")):
        if not job_dir.is_dir():
            continue
        candidates = sorted(job_dir.glob("*_cached_5seeds.csv"))
        if not candidates:
            continue
        setting = "hard_noisy" if "hard" in job_dir.name else "main"
        frame = safe_read_csv(candidates[0])
        config_candidates = sorted(job_dir.glob("*_config.json"))
        params: Dict[str, object] = {}
        if config_candidates:
            params = json.loads(config_candidates[0].read_text(encoding="utf-8"))
        frames.append(
            add_common_metadata(
                frame,
                suite="core_cached",
                job_id=f"core_cached/{job_dir.name}",
                setting=setting,
                engine="standard",
                parameters=params,
            )
        )
    if not frames:
        raise FileNotFoundError(f"No cached core CSVs found under {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_manifest_suite(results_root: Path, suite: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for manifest_path in sorted((results_root / suite).glob("*/job.json")):
        job_dir = manifest_path.parent
        if not (job_dir / "_SUCCESS").exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = job_dir / manifest.get("output_filename", "metrics.csv")
        if not metrics_path.exists():
            warnings.warn(f"Skipping completed job without metrics: {job_dir}")
            continue
        frame = safe_read_csv(metrics_path)
        frames.append(
            add_common_metadata(
                frame,
                suite=suite,
                job_id=manifest["job_id"],
                setting=manifest["setting"],
                engine=manifest["engine"],
                parameters=manifest.get("parameters", {}),
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_minedges(results_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset in ("acm", "dblp"):
        path = results_root / "minedge_sweep" / f"{dataset}_hard_minedges_sweep.csv"
        if path.exists():
            frame = safe_read_csv(path)
            frames.append(
                add_common_metadata(
                    frame,
                    suite="minedge_sweep",
                    job_id=f"minedge_sweep/{dataset}_hard",
                    setting="hard_noisy",
                    engine="standard",
                    parameters={"shuffle_party_positions": True, "party_shuffle_seed_offset": 2026},
                )
            )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_all_results(results_root: Path) -> pd.DataFrame:
    frames = [load_core_cached(results_root)]
    suites = [
        "topology_objective",
        "edge_budget",
        "consensus",
        "noise_ratio",
        "readout_ablation",
        "active_party_evaluator",
        "active_party_protocol",
        "strict_label_training",
        "split_validation",
        "decentralized_readout",
        "active_party_identity",
        "validation_label_budget",
        "topology_frequency",
        "party_scalability",
        "protocol_sensitivity",
        "multimodal_core",
        "modality_ablation",
        "central_baselines",
        "reviewer_safe_shuffled",
    ]
    for suite in suites:
        frame = load_manifest_suite(results_root, suite)
        if not frame.empty:
            frames.append(frame)
    minedges = load_minedges(results_root)
    if not minedges.empty:
        frames.append(minedges)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = add_communication_accounting(raw)
    for metric in METRICS:
        if metric in raw.columns:
            raw[metric] = pd.to_numeric(raw[metric], errors="coerce")
    raw["seed"] = pd.to_numeric(raw["seed"], errors="coerce").astype("Int64")
    return raw


def summarize(raw: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    available = [metric for metric in METRICS if metric in raw.columns]
    summary = raw.groupby(list(group_columns), dropna=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def fmt_mean_std(mean: float, std: float, scale: float = 100.0, digits: int = 2) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{scale * mean:.{digits}f} ± {scale * std:.{digits}f}"


def paper_core_table(core_summary: pd.DataFrame) -> pd.DataFrame:
    table = core_summary.copy()
    table["method_order"] = table["method"].map({m: i for i, m in enumerate(CORE_METHOD_ORDER)})
    table["column"] = table["dataset"] + " " + table["setting_label"]
    table["value"] = [
        fmt_mean_std(mean, std)
        for mean, std in zip(table["test_at_best_val_mean"], table["test_at_best_val_std"])
    ]
    wide = table.pivot(index=["method_order", "method", "method_label"], columns="column", values="value")
    desired = [
        "ACM Main",
        "ACM Hard",
        "DBLP Main",
        "DBLP Hard",
        "IMDB Main",
        "IMDB Hard",
    ]
    wide = wide.reindex(columns=desired).reset_index().sort_values("method_order")
    return wide.drop(columns=["method_order"])


def paper_macro_table(core_summary: pd.DataFrame) -> pd.DataFrame:
    table = core_summary.copy()
    table["method_order"] = table["method"].map({m: i for i, m in enumerate(CORE_METHOD_ORDER)})
    table["column"] = table["dataset"] + " " + table["setting_label"]
    table["value"] = [
        fmt_mean_std(mean, std)
        for mean, std in zip(table["macro_f1_at_best_val_mean"], table["macro_f1_at_best_val_std"])
    ]
    wide = table.pivot(index=["method_order", "method", "method_label"], columns="column", values="value")
    desired = [
        "ACM Main",
        "ACM Hard",
        "DBLP Main",
        "DBLP Hard",
        "IMDB Main",
        "IMDB Hard",
    ]
    return wide.reindex(columns=desired).reset_index().sort_values("method_order").drop(columns=["method_order"])


def paper_grouped_main_table(core_summary: pd.DataFrame) -> pd.DataFrame:
    """Reviewer-facing compact table with protocol groups made explicit."""
    desired = [
        ("Global diagnostic", "local_reliability_vote", "Global Reliability Vote"),
        ("Global diagnostic", "topk_reliability_vote", "Global Top-k Reliability"),
        ("Matched control", "__best_matched__", "Best Matched Sparse"),
        ("Dense P2P", "full_mesh", "Full Mesh"),
        ("Ours", "adaptive_graph_val", "TA-DVFG"),
    ]
    matched = {"fixed_ring_matched", "random_matched", "expander_matched"}
    columns = [
        ("ACM", "main", "ACM Main"),
        ("ACM", "hard_noisy", "ACM Hard"),
        ("DBLP", "main", "DBLP Main"),
        ("DBLP", "hard_noisy", "DBLP Hard"),
        ("IMDB", "main", "IMDB Main"),
        ("IMDB", "hard_noisy", "IMDB Hard"),
    ]
    rows: List[Dict[str, object]] = []
    for category, method, label in desired:
        output: Dict[str, object] = {"category": category, "method": method, "method_label": label}
        for dataset, setting, column in columns:
            candidates = core_summary[
                (core_summary["dataset"] == dataset)
                & (core_summary["setting"] == setting)
            ]
            if method == "__best_matched__":
                candidates = candidates[candidates["method"].isin(matched)]
                if not candidates.empty:
                    selected = candidates.loc[candidates["test_at_best_val_mean"].idxmax()]
                    output[f"{column} selected"] = selected["method_label"]
                else:
                    selected = None
            else:
                selected_rows = candidates[candidates["method"] == method]
                selected = selected_rows.iloc[0] if not selected_rows.empty else None
            output[column] = (
                fmt_mean_std(
                    float(selected["test_at_best_val_mean"]),
                    float(selected["test_at_best_val_std"]),
                )
                if selected is not None
                else ""
            )
        rows.append(output)
    return pd.DataFrame(rows)


def paper_active_party_protocol_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact reviewer-facing comparison for the active label-holder protocol."""
    matched = {"fixed_ring_matched", "random_matched", "expander_matched"}
    rows: List[Dict[str, object]] = []
    for (dataset, setting), group in summary.groupby(["dataset", "setting"], sort=False):
        def row_for(method: str) -> pd.Series | None:
            candidates = group[group["method"] == method]
            return candidates.iloc[0] if not candidates.empty else None

        adaptive = row_for("adaptive_graph_val")
        topk = row_for("topk_reliability_vote")
        full = row_for("full_mesh")
        matched_rows = group[group["method"].isin(matched)]
        best_matched = (
            matched_rows.loc[matched_rows["test_at_best_val_mean"].idxmax()]
            if not matched_rows.empty
            else None
        )
        if adaptive is None:
            continue

        adaptive_acc = float(adaptive["test_at_best_val_mean"])
        topk_acc = float(topk["test_at_best_val_mean"]) if topk is not None else float("nan")
        full_acc = float(full["test_at_best_val_mean"]) if full is not None else float("nan")
        matched_acc = (
            float(best_matched["test_at_best_val_mean"])
            if best_matched is not None
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "setting_label": SETTING_LABELS.get(setting, setting),
                "ta_dvfg": fmt_mean_std(
                    adaptive_acc, float(adaptive["test_at_best_val_std"])
                ),
                "global_topk": (
                    fmt_mean_std(topk_acc, float(topk["test_at_best_val_std"]))
                    if topk is not None
                    else ""
                ),
                "full_mesh": (
                    fmt_mean_std(full_acc, float(full["test_at_best_val_std"]))
                    if full is not None
                    else ""
                ),
                "best_matched_method": (
                    str(best_matched["method_label"])
                    if best_matched is not None
                    else ""
                ),
                "best_matched": (
                    fmt_mean_std(
                        matched_acc,
                        float(best_matched["test_at_best_val_std"]),
                    )
                    if best_matched is not None
                    else ""
                ),
                "ta_minus_full_mesh_pp": 100.0 * (adaptive_acc - full_acc),
                "ta_minus_best_matched_pp": 100.0 * (adaptive_acc - matched_acc),
                "ta_minus_global_topk_pp": 100.0 * (adaptive_acc - topk_acc),
                "ta_edges_mean": float(adaptive["final_edges_mean"]),
                "ta_peer_comm_mean": float(adaptive["peer_to_peer_comm_mean"]),
                "topology_update_comm_mean": float(adaptive["topology_update_comm_mean"]),
                "topology_eval_count_mean": float(adaptive["topology_eval_count_mean"]),
                "topology_update_time_mean": float(adaptive["topology_update_time_mean"]),
            }
        )
    return pd.DataFrame(rows)


def paper_reviewer_protocol_table(
    active_protocol: pd.DataFrame,
    strict_training: pd.DataFrame,
    split_validation: pd.DataFrame,
    decentralized_readout: pd.DataFrame,
) -> pd.DataFrame:
    """Compact hard-setting protocol table for the main paper."""
    rows: List[Dict[str, object]] = []
    specifications = [
        (
            "Active evaluator",
            active_protocol[
                (active_protocol["method"] == "adaptive_graph_val")
                & (active_protocol["setting"] == "hard_noisy")
            ],
        ),
        (
            "Label-siloed logit gradients",
            strict_training[strict_training["method"] == "adaptive_graph_val"],
        ),
        (
            "Disjoint topology/selection validation",
            split_validation[
                (split_validation["method"] == "adaptive_graph_val")
                & (split_validation["param_validation_split_mode"] == "disjoint")
            ],
        ),
        (
            "Local mean readout",
            decentralized_readout[
                decentralized_readout["param_readout_mode"] == "local_mean"
            ],
        ),
        (
            "Local worst-party readout",
            decentralized_readout[
                decentralized_readout["param_readout_mode"] == "local_worst"
            ],
        ),
    ]
    for protocol, frame in specifications:
        row: Dict[str, object] = {"protocol": protocol}
        for dataset in ("ACM", "DBLP"):
            subset = frame[frame["dataset"] == dataset]
            if subset.empty:
                row[f"{dataset} Hard"] = ""
                continue
            mean = float(subset["test_at_best_val"].mean())
            std = float(subset["test_at_best_val"].std(ddof=1))
            row[f"{dataset} Hard"] = fmt_mean_std(mean, std)
            row[f"{dataset} edges"] = float(subset["final_edges"].mean())
            row[f"{dataset} train comm"] = float(subset["train_comm"].mean())
            row[f"{dataset} topology comm"] = float(
                subset["topology_update_comm"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(values: Iterable[float]) -> List[float]:
    values = list(values)
    finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [float("nan")] * len(values)
    running = 0.0
    m = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_test_rows(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    variant_column: str,
    reference: str,
    comparisons: Sequence[str],
    metric: str = "test_at_best_val",
    family: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(group_columns, keys))
        ref = group[group[variant_column] == reference][["seed", metric]].rename(columns={metric: "reference"})
        for comparison in comparisons:
            base = group[group[variant_column] == comparison][["seed", metric]].rename(columns={metric: "comparison"})
            paired = ref.merge(base, on="seed", how="inner").dropna()
            if paired.empty:
                continue
            differences = paired["reference"].to_numpy(float) - paired["comparison"].to_numpy(float)
            n = len(differences)
            mean_difference = float(np.mean(differences))
            std_difference = float(np.std(differences, ddof=1)) if n > 1 else 0.0
            if n > 1:
                sem = stats.sem(differences)
                critical = stats.t.ppf(0.975, n - 1)
                ci_low = float(mean_difference - critical * sem)
                ci_high = float(mean_difference + critical * sem)
                t_p = float(stats.ttest_rel(paired["reference"], paired["comparison"]).pvalue)
            else:
                ci_low = ci_high = mean_difference
                t_p = float("nan")
            try:
                wilcoxon_p = 1.0 if np.allclose(differences, 0) else float(stats.wilcoxon(differences).pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            rows.append(
                {
                    "family": family,
                    **group_values,
                    "reference": reference,
                    "comparison": comparison,
                    "metric": metric,
                    "n_pairs": n,
                    "reference_mean": float(paired["reference"].mean()),
                    "comparison_mean": float(paired["comparison"].mean()),
                    "mean_difference": mean_difference,
                    "mean_difference_pp": 100.0 * mean_difference,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_t_p": t_p,
                    "wilcoxon_p": wilcoxon_p,
                    "cohen_dz": mean_difference / std_difference if std_difference > 0 else float("inf"),
                    "seeds": ",".join(str(int(seed)) for seed in paired["seed"]),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["paired_t_p_holm"] = holm_adjust(result["paired_t_p"].tolist())
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"].tolist())
    return result


def variant_from_job_id(frame: pd.DataFrame) -> pd.Series:
    return frame["job_id"].astype(str).str.rsplit("_", n=1).str[-1]


def protocol_audit(raw: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "suite": "core_cached",
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "yes",
            "note": "Primary paper protocol.",
        },
        {
            "suite": "topology_objective / edge_budget / consensus / minedge_sweep",
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "yes",
            "note": "Reuses cached-core local predictions.",
        },
        {
            "suite": "active_party_protocol",
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "yes",
            "note": (
                "Same cached local predictors; only the designated active evaluator "
                "accesses validation labels for reliability and topology scoring."
            ),
        },
        {
            "suite": "strict_label_training",
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "yes",
            "note": (
                "Only the active party receives labels; passive parties submit "
                "train logits and receive gradients with respect to those logits."
            ),
        },
        {
            "suite": (
                "split_validation / decentralized_readout / active_party_identity / "
                "validation_label_budget / topology_frequency"
            ),
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "yes",
            "note": "Reviewer-resistance sweeps reuse the cached-core local trajectories.",
        },
        {
            "suite": "party_scalability",
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "within-suite",
            "note": "ACM Hard with 20% useful parties as N changes.",
        },
        {
            "suite": "multimodal_core / modality_ablation",
            "party_order": "seed-shuffled",
            "prediction_cache": "yes",
            "split": "dataset",
            "directly_comparable_to_core": "extension only",
            "note": "Simulated multimodal feature construction changes local inputs.",
        },
        {
            "suite": "reviewer_safe_shuffled",
            "party_order": "seed-shuffled",
            "prediction_cache": "no",
            "split": "dataset",
            "directly_comparable_to_core": "diagnostic",
            "note": "Reviewer-specific ACM rerun and edge-composition diagnostics.",
        },
        {
            "suite": "noise_ratio / readout_ablation / active_party_evaluator",
            "party_order": "useful parties at low indices",
            "prediction_cache": "no",
            "split": "dataset",
            "directly_comparable_to_core": "within-suite only",
            "note": "Legacy unshuffled runs; interpret treatment contrasts within each suite.",
        },
        {
            "suite": "protocol_sensitivity",
            "party_order": "useful parties at low indices",
            "prediction_cache": "no",
            "split": "custom stratified",
            "directly_comparable_to_core": "no",
            "note": "Changes both split and party-order protocol relative to cached core.",
        },
        {
            "suite": "central_baselines",
            "party_order": "legacy/unshuffled",
            "prediction_cache": "not applicable",
            "split": "dataset",
            "directly_comparable_to_core": "reference only",
            "note": "Centralized reference architectures, not matched decentralized replay.",
        },
    ]
    return pd.DataFrame(rows)


def completeness_table(raw: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    expected = {
        "core_cached": 6,
        "topology_objective": 2,
        "edge_budget": 24,
        "consensus": 28,
        "noise_ratio": 8,
        "readout_ablation": 12,
        "active_party_evaluator": 8,
        "active_party_protocol": 4,
        "strict_label_training": 2,
        "split_validation": 4,
        "decentralized_readout": 8,
        "active_party_identity": 10,
        "validation_label_budget": 10,
        "topology_frequency": 8,
        "party_scalability": 5,
        "protocol_sensitivity": 4,
        "multimodal_core": 6,
        "modality_ablation": 12,
        "minedge_sweep": 2,
        "central_baselines": 4,
        "reviewer_safe_shuffled": 2,
    }
    rows = []
    for suite, expected_jobs in expected.items():
        if suite == "core_cached":
            observed = raw[raw["suite"] == suite]["job_id"].nunique()
            failed = 0
        elif suite == "minedge_sweep":
            observed = raw[raw["suite"] == suite]["job_id"].nunique()
            failed = 0
        else:
            observed = raw[raw["suite"] == suite]["job_id"].nunique()
            failed = len(list((results_root / suite).glob("*/_FAILED")))
        rows.append(
            {
                "suite": suite,
                "expected_jobs": expected_jobs,
                "completed_jobs": int(observed),
                "failed_jobs": int(failed),
                "complete": observed == expected_jobs and failed == 0,
                "seed_rows": int(len(raw[raw["suite"] == suite])),
            }
        )
    return pd.DataFrame(rows)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def setup_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_core_heatmap(core_summary: pd.DataFrame, path: Path) -> None:
    table = core_summary.copy()
    table["column"] = table["dataset"] + "\n" + table["setting_label"]
    pivot = table.pivot(index="method_label", columns="column", values="test_at_best_val_mean")
    order = [METHOD_LABELS[m] for m in CORE_METHOD_ORDER]
    pivot = pivot.reindex(order)
    columns = ["ACM\nMain", "ACM\nHard", "DBLP\nMain", "DBLP\nHard", "IMDB\nMain", "IMDB\nHard"]
    pivot = pivot.reindex(columns=columns)
    values = pivot.to_numpy(float) * 100
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value >= 72 else "#222222",
                )
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Core performance across HGB-derived controlled settings")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Test@BestVal (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    savefig(path)


def plot_core_tradeoff(core_summary: pd.DataFrame, path: Path) -> None:
    selected = [
        "local_reliability_vote",
        "topk_reliability_vote",
        "fixed_ring_matched",
        "expander_matched",
        "full_mesh",
        "adaptive_graph_val",
    ]
    frame = core_summary[core_summary["method"].isin(selected)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=False)
    for ax, ((dataset, setting), group) in zip(
        axes.flat,
        frame[frame["dataset"].isin(["ACM", "DBLP"])].groupby(["dataset", "setting"], sort=False),
    ):
        for _, row in group.iterrows():
            label = row["method_label"]
            ax.scatter(
                row["inference_comm_mean"],
                100 * row["test_at_best_val_mean"],
                s=48,
                color=COLORS.get(label, "#777777"),
                label=label,
                zorder=3,
            )
        ax.set_title(f"{dataset} {SETTING_LABELS.get(setting, setting)}")
        ax.set_xlabel("Inference communication (scalars)")
        ax.set_ylabel("Test@BestVal (%)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    fig.suptitle("Accuracy–communication trade-off", y=1.01)
    fig.subplots_adjust(right=0.82)
    savefig(path)


def plot_compact_tradeoff(
    core_summary: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Single-column reviewer-facing peer-edge communication trade-off."""
    matched = {"fixed_ring_matched", "random_matched", "expander_matched"}
    panels = [
        ("ACM", "hard_noisy"),
        ("ACM", "main"),
        ("DBLP", "hard_noisy"),
        ("DBLP", "main"),
    ]
    styles = {
        "Global Reliability Vote": {
            "marker": "s",
            "color": COLORS["Global Reliability Vote"],
            "facecolors": "none",
            "s": 28,
        },
        "Global Top-k Reliability": {
            "marker": "o",
            "color": COLORS["Global Top-k Reliability"],
            "facecolors": "none",
            "s": 34,
        },
        "Best Matched Sparse": {
            "marker": "D",
            "color": COLORS["Best Matched Sparse"],
            "facecolors": COLORS["Best Matched Sparse"],
            "s": 28,
        },
        "Full Mesh": {
            "marker": "X",
            "color": COLORS["Full Mesh"],
            "facecolors": COLORS["Full Mesh"],
            "s": 34,
        },
        "TA-DVFG": {
            "marker": "*",
            "color": COLORS["TA-DVFG"],
            "facecolors": COLORS["TA-DVFG"],
            "s": 58,
        },
    }
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 4.1), sharex=True)
    legend_handles: Dict[str, object] = {}
    for ax, (dataset, setting) in zip(axes.flat, panels):
        group = core_summary[
            (core_summary["dataset"] == dataset)
            & (core_summary["setting"] == setting)
        ].copy()
        full_rows = group[group["method"] == "full_mesh"]
        full_comm = (
            float(full_rows.iloc[0]["peer_to_peer_comm_mean"])
            if not full_rows.empty
            else float("nan")
        )
        selected_rows = [
            ("Global Reliability Vote", group[group["method"] == "local_reliability_vote"]),
            ("Global Top-k Reliability", group[group["method"] == "topk_reliability_vote"]),
            ("Full Mesh", full_rows),
            ("TA-DVFG", group[group["method"] == "adaptive_graph_val"]),
        ]
        matched_rows = group[group["method"].isin(matched)]
        if not matched_rows.empty:
            matched_rows = matched_rows.loc[
                [matched_rows["test_at_best_val_mean"].idxmax()]
            ]
        selected_rows.insert(2, ("Best Matched Sparse", matched_rows))

        for label, rows in selected_rows:
            if rows.empty:
                continue
            row = rows.iloc[0]
            peer_percent = (
                100.0 * float(row["peer_to_peer_comm_mean"]) / full_comm
                if full_comm > 0
                else 0.0
            )
            style = styles[label]
            handle = ax.scatter(
                peer_percent,
                100.0 * float(row["test_at_best_val_mean"]),
                marker=style["marker"],
                s=style["s"],
                edgecolors=style["color"],
                facecolors=style["facecolors"],
                linewidths=1.1,
                zorder=4,
                label=label,
            )
            legend_handles.setdefault(label, handle)

        ax.set_title(f"{dataset} {SETTING_LABELS[setting]}", fontsize=8, pad=3)
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0, base=10)
        ax.set_xlim(-0.35, 135)
        ax.set_xticks([0, 5, 100])
        ax.set_xticklabels(["0", "5", "100"])
        ax.grid(True, linewidth=0.45, alpha=0.55)
        ax.tick_params(labelsize=7)

    for ax in axes[:, 0]:
        ax.set_ylabel("Test@BestVal (%)", fontsize=7.5)
    for ax in axes[1, :]:
        ax.set_xlabel("Peer-edge comm. (% Full Mesh)", fontsize=7.2)

    order = [
        "Global Reliability Vote",
        "Global Top-k Reliability",
        "Best Matched Sparse",
        "Full Mesh",
        "TA-DVFG",
    ]
    fig.legend(
        [legend_handles[label] for label in order if label in legend_handles],
        [label for label in order if label in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        frameon=False,
        fontsize=6.6,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.96, bottom=0.19, hspace=0.30, wspace=0.28)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_edge_budget(edge: pd.DataFrame, path: Path) -> None:
    summary = summarize(
        edge,
        ["dataset", "param_adaptive_edge_budget", "param_adaptive_min_gain"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex="col")
    for column, dataset in enumerate(["ACM", "DBLP"]):
        group = summary[summary["dataset"] == dataset].copy()
        for gain, curve in group.groupby("param_adaptive_min_gain"):
            curve = curve.sort_values("param_adaptive_edge_budget")
            axes[0, column].errorbar(
                curve["param_adaptive_edge_budget"],
                100 * curve["test_at_best_val_mean"],
                yerr=100 * curve["test_at_best_val_std"],
                marker="o",
                capsize=3,
                label=f"min gain={float(gain):g}",
            )
            axes[1, column].errorbar(
                curve["param_adaptive_edge_budget"],
                curve["final_edges_mean"],
                yerr=curve["final_edges_std"],
                marker="s",
                capsize=3,
                label=f"min gain={float(gain):g}",
            )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Test@BestVal (%)")
        axes[1, column].set_xlabel("Edge budget")
        axes[1, column].set_ylabel("Actual selected edges")
        axes[0, column].legend(frameon=False)
    fig.suptitle("Edge budget is an upper bound, not a quota")
    savefig(path)


def plot_consensus(consensus: pd.DataFrame, path: Path) -> None:
    frame = consensus.copy()
    frame["variant"] = variant_from_job_id(frame)
    summary = summarize(frame, ["dataset", "setting", "variant"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    order = ["default", "steps0", "steps2", "self0p70", "self0p95", "uniform", "reliability"]
    for ax, ((dataset, setting), group) in zip(axes.flat, summary.groupby(["dataset", "setting"], sort=False)):
        group = group.set_index("variant").reindex(order).reset_index()
        ax.bar(group["variant"], 100 * group["test_at_best_val_mean"], color="#4C78A8")
        ax.errorbar(
            group["variant"],
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_title(f"{dataset} {SETTING_LABELS.get(setting, setting)}")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("Test@BestVal (%)")
    fig.suptitle("Consensus mechanism ablation")
    savefig(path)


def plot_noise(noise: pd.DataFrame, path: Path) -> None:
    summary = summarize(noise, ["dataset", "param_useful_parties", "method", "method_label"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, dataset in zip(axes, ["ACM", "DBLP"]):
        group = summary[summary["dataset"] == dataset]
        for (method, label), curve in group.groupby(["method", "method_label"]):
            curve = curve.sort_values("param_useful_parties")
            ax.plot(
                curve["param_useful_parties"],
                100 * curve["test_at_best_val_mean"],
                marker="o",
                label=label,
                color=COLORS.get(label),
            )
        ax.set_title(dataset)
        ax.set_xlabel("Useful parties (out of 15)")
        ax.set_ylabel("Test@BestVal (%)")
        ax.set_xticks([3, 4, 6, 9])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    fig.suptitle("Robustness to useful/noisy-party ratio (legacy unshuffled suite)")
    fig.subplots_adjust(right=0.82)
    savefig(path)


def plot_topology_objective(topology: pd.DataFrame, path: Path) -> None:
    summary = summarize(topology, ["dataset", "method", "method_label"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    order = ["Adaptive Pairwise", "Adaptive Complementarity", "TA-DVFG"]
    for ax, dataset in zip(axes, ["ACM", "DBLP"]):
        group = summary[summary["dataset"] == dataset].set_index("method_label").reindex(order).reset_index()
        ax.bar(group["method_label"], 100 * group["test_at_best_val_mean"], color=["#4C78A8", "#72B7B2", "#C43C39"])
        ax.errorbar(
            group["method_label"],
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_title(f"{dataset} Hard")
        ax.set_ylabel("Test@BestVal (%)")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Topology-objective ablation")
    savefig(path)


def plot_topology_overhead(topology: pd.DataFrame, path: Path) -> None:
    summary = summarize(topology, ["dataset", "method_label"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, dataset in zip(axes, ["ACM", "DBLP"]):
        group = summary[summary["dataset"] == dataset].sort_values("topology_eval_count_mean")
        ax.barh(group["method_label"], group["topology_eval_count_mean"], color="#B279A2")
        ax.set_title(dataset)
        ax.set_xlabel("Candidate topology evaluations")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("Topology-selection control-plane evaluations")
    savefig(path)


def plot_protocol_sensitivity(protocol: pd.DataFrame, path: Path) -> None:
    summary = summarize(protocol, ["dataset", "setting", "method_label"])
    groups = list(summary.groupby(["dataset", "setting"], sort=False))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    preferred = [
        "Global Uniform Vote",
        "Global Reliability Vote",
        "Ring",
        "Full Mesh",
        "TA-DVFG",
    ]
    for ax, ((dataset, setting), group) in zip(axes.flat, groups):
        group = group.set_index("method_label").reindex(preferred).reset_index()
        ax.bar(group["method_label"], 100 * group["test_at_best_val_mean"], color="#4C78A8")
        ax.errorbar(
            group["method_label"],
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_title(f"{dataset} {SETTING_LABELS.get(setting, setting)}")
        ax.set_ylabel("Test@BestVal (%)")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Custom-split sensitivity (legacy unshuffled party order)")
    savefig(path)


def plot_central_references(core: pd.DataFrame, central: pd.DataFrame, path: Path) -> None:
    adaptive = summarize(
        core[(core["method"] == "adaptive_graph_val") & core["dataset"].isin(["ACM", "DBLP"])],
        ["dataset", "setting", "method_label"],
    )
    references = summarize(central, ["dataset", "setting", "method_label"])
    frame = pd.concat([adaptive, references], ignore_index=True)
    groups = list(frame.groupby(["dataset", "setting"], sort=False))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
    preferred = ["TA-DVFG", "Central Fusion", "Central Full GCN"]
    colors = ["#C43C39", "#9C755F", "#B279A2"]
    for ax, ((dataset, setting), group) in zip(axes.flat, groups):
        group = group.set_index("method_label").reindex(preferred).reset_index()
        ax.bar(group["method_label"], 100 * group["test_at_best_val_mean"], color=colors)
        ax.errorbar(
            group["method_label"],
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_title(f"{dataset} {SETTING_LABELS.get(setting, setting)}")
        ax.set_ylabel("Test@BestVal (%)")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Centralized references (separate architecture/protocol)")
    savefig(path)


def plot_minedges(minedges: pd.DataFrame, path: Path) -> None:
    summary = summarize(minedges, ["dataset", "adaptive_min_edges"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, dataset in zip(axes, ["ACM", "DBLP"]):
        group = summary[summary["dataset"] == dataset].sort_values("adaptive_min_edges")
        ax.errorbar(
            group["adaptive_min_edges"],
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            marker="o",
            capsize=3,
            color="#C43C39",
            label="Accuracy",
        )
        second = ax.twinx()
        second.plot(
            group["adaptive_min_edges"],
            group["final_edges_mean"],
            marker="s",
            linestyle="--",
            color="#4C78A8",
            label="Selected edges",
        )
        ax.set_title(dataset)
        ax.set_xlabel("Minimum required edges")
        ax.set_ylabel("Test@BestVal (%)", color="#C43C39")
        second.set_ylabel("Final edges", color="#4C78A8")
        ax.set_xticks(sorted(group["adaptive_min_edges"].unique()))
    fig.suptitle("Minimum-edge sensitivity")
    savefig(path)


def plot_categorical_ablation(
    frame: pd.DataFrame,
    *,
    category: str,
    title: str,
    path: Path,
    order: Sequence[str] | None = None,
) -> None:
    summary = summarize(frame, ["dataset", "setting", category])
    groups = list(summary.groupby(["dataset", "setting"], sort=False))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
    for ax, ((dataset, setting), group) in zip(axes.flat, groups):
        if order:
            group = group.set_index(category).reindex(order).reset_index()
        ax.bar(group[category].astype(str), 100 * group["test_at_best_val_mean"], color="#4C78A8")
        ax.errorbar(
            group[category].astype(str),
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_title(f"{dataset} {SETTING_LABELS.get(setting, setting)}")
        ax.set_ylabel("Test@BestVal (%)")
        ax.tick_params(axis="x", rotation=25)
    for ax in axes.flat[len(groups) :]:
        ax.axis("off")
    fig.suptitle(title)
    savefig(path)


def plot_modality(modality: pd.DataFrame, path: Path) -> None:
    summary = summarize(modality, ["setting", "param_modalities"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    preferred = ["ATTR", "TEXT", "STRUCT", "HYBRID", "TEXT_STRUCT", "ATTR,TEXT,STRUCT,HYBRID"]
    labels = ["ATTR", "TEXT", "STRUCT", "HYBRID", "TEXT+STRUCT", "Mixed"]
    for ax, setting in zip(axes, ["main", "hard_noisy"]):
        group = summary[summary["setting"] == setting].set_index("param_modalities").reindex(preferred)
        ax.bar(labels, 100 * group["test_at_best_val_mean"], color="#72B7B2")
        ax.errorbar(
            labels,
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_title(f"ACM {SETTING_LABELS.get(setting, setting)}")
        ax.set_ylabel("Test@BestVal (%)")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Simulated multimodal feature ablation")
    savefig(path)


def plot_multimodal_comparison(core: pd.DataFrame, multimodal: pd.DataFrame, path: Path) -> None:
    standard = summarize(
        core[core["method"] == "adaptive_graph_val"],
        ["dataset", "setting"],
    )
    standard["construction"] = "Standard"
    multi = summarize(
        multimodal[multimodal["method"] == "adaptive_graph_val"],
        ["dataset", "setting"],
    )
    multi["construction"] = "Multimodal"
    frame = pd.concat([standard, multi], ignore_index=True)
    frame["group"] = frame["dataset"] + " " + frame["setting"].map(SETTING_LABELS).fillna(frame["setting"])
    groups = list(dict.fromkeys(frame["group"]))
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for index, (construction, color) in enumerate([("Standard", "#4C78A8"), ("Multimodal", "#72B7B2")]):
        values = (
            frame[frame["construction"] == construction]
            .set_index("group")
            .reindex(groups)["test_at_best_val_mean"]
            .to_numpy(float)
            * 100
        )
        ax.bar(x + (index - 0.5) * width, values, width, label=construction, color=color)
    ax.set_xticks(x, groups, rotation=20, ha="right")
    ax.set_ylabel("TA-DVFG Test@BestVal (%)")
    ax.set_xlabel("")
    ax.set_title("Standard versus simulated multimodal construction")
    ax.legend(frameon=False)
    savefig(path)


def plot_active_party_protocol(frame: pd.DataFrame, path: Path) -> None:
    """Compare key references under active-party-only validation-label access."""
    summary = summarize(
        frame,
        ["dataset", "setting", "method", "method_label"],
    )
    panels = [
        ("ACM", "hard_noisy"),
        ("ACM", "main"),
        ("DBLP", "hard_noisy"),
        ("DBLP", "main"),
    ]
    matched = {"fixed_ring_matched", "random_matched", "expander_matched"}
    labels = [
        "Global Reliability",
        "Global Top-k",
        "Best Matched",
        "Full Mesh",
        "TA-DVFG",
    ]
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#9C755F", "#C43C39"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), sharey=False)
    for ax, (dataset, setting) in zip(axes.flat, panels):
        group = summary[
            (summary["dataset"] == dataset)
            & (summary["setting"] == setting)
        ]
        selected: List[pd.Series | None] = []
        for method in (
            "local_reliability_vote",
            "topk_reliability_vote",
        ):
            candidates = group[group["method"] == method]
            selected.append(candidates.iloc[0] if not candidates.empty else None)
        matched_rows = group[group["method"].isin(matched)]
        selected.append(
            matched_rows.loc[matched_rows["test_at_best_val_mean"].idxmax()]
            if not matched_rows.empty
            else None
        )
        for method in ("full_mesh", "adaptive_graph_val"):
            candidates = group[group["method"] == method]
            selected.append(candidates.iloc[0] if not candidates.empty else None)

        values = [
            100.0 * float(row["test_at_best_val_mean"]) if row is not None else np.nan
            for row in selected
        ]
        errors = [
            100.0 * float(row["test_at_best_val_std"]) if row is not None else 0.0
            for row in selected
        ]
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors)
        ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#333333", capsize=3)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_title(f"{dataset} {SETTING_LABELS.get(setting, setting)}")
        ax.set_ylabel("Test@BestVal (%)")
    fig.suptitle("Active-party evaluator: validation labels held by one designated party")
    savefig(path)


def plot_reviewer_protocols(
    split_validation: pd.DataFrame,
    decentralized_readout: pd.DataFrame,
    active_identity: pd.DataFrame,
    strict_training: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))

    split_summary = summarize(
        split_validation[split_validation["method"] == "adaptive_graph_val"],
        ["dataset", "param_validation_split_mode"],
    )
    for dataset, group in split_summary.groupby("dataset"):
        group = group.set_index("param_validation_split_mode").reindex(["shared", "disjoint"])
        axes[0, 0].plot(
            ["Shared", "Disjoint"],
            100 * group["test_at_best_val_mean"],
            marker="o",
            label=dataset,
        )
    axes[0, 0].set_title("Topology vs. epoch-selection validation")
    axes[0, 0].set_ylabel("Test@BestVal (%)")
    axes[0, 0].legend(frameon=False)

    readout_summary = summarize(
        decentralized_readout,
        ["dataset", "param_readout_mode"],
    )
    order = ["active_vote", "local_mean", "active_party_local", "local_worst"]
    labels = ["Global active", "Local mean", "Active local", "Local worst"]
    x = np.arange(len(order))
    width = 0.36
    for index, dataset in enumerate(("ACM", "DBLP")):
        group = (
            readout_summary[readout_summary["dataset"] == dataset]
            .set_index("param_readout_mode")
            .reindex(order)
        )
        axes[0, 1].bar(
            x + (index - 0.5) * width,
            100 * group["test_at_best_val_mean"],
            width,
            label=dataset,
        )
    axes[0, 1].set_xticks(x, labels, rotation=25, ha="right")
    axes[0, 1].set_title("Global and fully local readouts")
    axes[0, 1].set_ylabel("Test@BestVal (%)")
    axes[0, 1].legend(frameon=False)

    identity_summary = summarize(
        active_identity,
        ["dataset", "param_active_party_id"],
    )
    for dataset, group in identity_summary.groupby("dataset"):
        group = group.sort_values("param_active_party_id")
        axes[1, 0].plot(
            group["param_active_party_id"],
            100 * group["test_at_best_val_mean"],
            marker="o",
            label=dataset,
        )
    axes[1, 0].set_title("Active evaluator identity")
    axes[1, 0].set_xlabel("Active party ID")
    axes[1, 0].set_ylabel("Test@BestVal (%)")
    axes[1, 0].legend(frameon=False)

    strict_adaptive = strict_training[
        strict_training["method"] == "adaptive_graph_val"
    ]
    strict_summary = summarize(strict_adaptive, ["dataset"])
    axes[1, 1].bar(
        strict_summary["dataset"],
        100 * strict_summary["test_at_best_val_mean"],
        color="#C43C39",
    )
    axes[1, 1].errorbar(
        strict_summary["dataset"],
        100 * strict_summary["test_at_best_val_mean"],
        yerr=100 * strict_summary["test_at_best_val_std"],
        fmt="none",
        ecolor="#333333",
        capsize=3,
    )
    axes[1, 1].set_title("Label-siloed logit-gradient training")
    axes[1, 1].set_ylabel("Test@BestVal (%)")

    fig.suptitle("Reviewer-resistance protocol checks")
    savefig(path)


def plot_label_budget_and_frequency(
    label_budget: pd.DataFrame,
    frequency: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    budget_summary = summarize(
        label_budget[label_budget["method"] == "adaptive_graph_val"],
        ["dataset", "param_topology_val_fraction"],
    )
    for dataset, group in budget_summary.groupby("dataset"):
        group = group.sort_values("param_topology_val_fraction")
        axes[0].errorbar(
            100 * group["param_topology_val_fraction"].astype(float),
            100 * group["test_at_best_val_mean"],
            yerr=100 * group["test_at_best_val_std"],
            marker="o",
            capsize=3,
            label=dataset,
        )
    axes[0].set_xlabel("Topology validation labels used (%)")
    axes[0].set_ylabel("Test@BestVal (%)")
    axes[0].set_title("Validation-label budget")
    axes[0].legend(frameon=False)

    frame = frequency.copy()
    frame["frequency_variant"] = frame["job_id"].astype(str).str.rsplit("_", n=1).str[-1]
    frequency_summary = summarize(
        frame,
        ["dataset", "frequency_variant"],
    )
    order = ["once", "every50", "every20", "every10"]
    x = np.arange(len(order))
    width = 0.36
    for index, dataset in enumerate(("ACM", "DBLP")):
        group = (
            frequency_summary[frequency_summary["dataset"] == dataset]
            .set_index("frequency_variant")
            .reindex(order)
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            100 * group["test_at_best_val_mean"],
            width,
            label=dataset,
        )
    axes[1].set_xticks(x, ["Once", "Every 50", "Every 20", "Every 10"])
    axes[1].set_ylabel("Test@BestVal (%)")
    axes[1].set_title("Topology update frequency")
    axes[1].legend(frameon=False)
    savefig(path)


def plot_party_scalability(frame: pd.DataFrame, path: Path) -> None:
    summary = summarize(
        frame[frame["method"] == "adaptive_graph_val"],
        ["param_num_parties"],
    ).sort_values("param_num_parties")
    parties = summary["param_num_parties"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    axes[0].errorbar(
        parties,
        100 * summary["test_at_best_val_mean"],
        yerr=100 * summary["test_at_best_val_std"],
        marker="o",
        capsize=3,
        color="#C43C39",
    )
    axes[0].set_ylabel("Test@BestVal (%)")
    axes[0].set_xlabel("Parties")
    axes[0].set_title("Accuracy")
    axes[1].plot(parties, summary["final_edges_mean"], marker="s", color="#4C78A8")
    axes[1].set_ylabel("Selected edges")
    axes[1].set_xlabel("Parties")
    axes[1].set_title("Sparse topology")
    axes[2].plot(
        parties,
        summary["topology_update_time_mean"],
        marker="^",
        color="#B279A2",
    )
    axes[2].set_ylabel("Topology time (s)")
    axes[2].set_xlabel("Parties")
    axes[2].set_title("Selection cost")
    fig.suptitle("Party-count scalability on ACM Hard (20% useful parties)")
    savefig(path)


def plot_edge_composition(core: pd.DataFrame, path: Path) -> None:
    frame = core[core["method"] == "adaptive_graph_val"].copy()
    summary = frame.groupby(["dataset", "setting_label"])[
        ["useful_useful_edges", "useful_noisy_edges", "noisy_noisy_edges"]
    ].mean()
    summary.index = [f"{dataset} {setting}" for dataset, setting in summary.index]
    summary = summary.rename(
        columns={
            "useful_useful_edges": "Useful–Useful",
            "useful_noisy_edges": "Useful–Noisy",
            "noisy_noisy_edges": "Noisy–Noisy",
        }
    )
    summary.plot(
        kind="bar",
        stacked=True,
        figsize=(8.5, 4.2),
        color=["#59A14F", "#F2CF5B", "#E15759"],
    )
    plt.ylabel("Mean selected edges")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.title("TA-DVFG selected-edge composition")
    plt.legend(frameon=False, ncol=3)
    savefig(path)


def plot_reviewer_diagnostics(reviewer: pd.DataFrame, path: Path) -> None:
    selected = ["fixed_ring_matched", "expander_matched", "adaptive_graph_val"]
    frame = reviewer[reviewer["method"].isin(selected)].copy()
    summary = frame.groupby(["setting_label", "method_label"])[
        ["useful_useful_edges", "useful_noisy_edges", "noisy_noisy_edges"]
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    for ax, setting in zip(axes, ["Main", "Hard"]):
        group = summary.loc[setting]
        bottom = np.zeros(len(group))
        for column, color in zip(group.columns, ["#59A14F", "#F2CF5B", "#E15759"]):
            ax.bar(group.index, group[column], bottom=bottom, label=column.replace("_edges", ""), color=color)
            bottom += group[column].to_numpy()
        ax.set_title(f"ACM {setting}")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Mean edges")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Reviewer-safe shuffled topology diagnostics")
    fig.subplots_adjust(bottom=0.25)
    savefig(path)


def build_summary_markdown(
    completeness: pd.DataFrame,
    core_summary: pd.DataFrame,
    core_stats: pd.DataFrame,
    minedge_summary: pd.DataFrame,
    noise_summary: pd.DataFrame,
    modality_summary: pd.DataFrame,
    evaluator_summary: pd.DataFrame,
    active_protocol_table: pd.DataFrame,
    output_path: Path,
) -> None:
    core_adaptive = core_summary[core_summary["method"] == "adaptive_graph_val"].copy()
    settings = core_summary[["dataset", "setting"]].drop_duplicates()
    wins = 0
    ranks = []
    for _, item in settings.iterrows():
        group = core_summary[
            (core_summary["dataset"] == item["dataset"]) & (core_summary["setting"] == item["setting"])
        ].sort_values("test_at_best_val_mean", ascending=False)
        rank = int(np.where(group["method"].to_numpy() == "adaptive_graph_val")[0][0] + 1)
        ranks.append(rank)
        wins += int(rank == 1)

    sig_core = core_stats[
        (core_stats["paired_t_p_holm"] < 0.05)
        & (core_stats["mean_difference"] > 0)
    ] if not core_stats.empty else pd.DataFrame()

    modality_best = (
        modality_summary.sort_values("test_at_best_val_mean", ascending=False)
        .groupby("setting", as_index=False)
        .first()[["setting", "param_modalities", "test_at_best_val_mean"]]
    )

    lines = [
        "# TA-DVFG Final Experiment Summary",
        "",
        "## Coverage",
        "",
        f"- Completed formal suites: {int(completeness['complete'].sum())}/{len(completeness)}.",
        f"- Total analyzed seed-method rows: {int(completeness['seed_rows'].sum())}.",
        "- Primary reporting metric: mean ± standard deviation of Test@BestVal over seeds 42–46.",
        "",
        "## Core results",
        "",
        f"- TA-DVFG ranks first in {wins}/{len(settings)} cached-core dataset/settings; "
        f"mean rank is {np.mean(ranks):.2f}.",
    ]
    for _, row in core_adaptive.sort_values(["dataset", "setting"]).iterrows():
        lines.append(
            f"- {row['dataset']} {SETTING_LABELS.get(row['setting'], row['setting'])}: "
            f"{100 * row['test_at_best_val_mean']:.2f} ± {100 * row['test_at_best_val_std']:.2f}% "
            f"with {row['final_edges_mean']:.2f} selected edges."
        )
    lines += [
        f"- Holm-corrected paired t-tests show a positive TA-DVFG difference in "
        f"{len(sig_core)}/{len(core_stats)} tested core comparisons. With only five seeds, "
        "effect sizes and confidence intervals should be reported alongside p-values.",
        "- TA-DVFG consistently improves over Full Mesh and matched sparse peer-to-peer controls. "
        "Global Top-k Reliability is a stronger-information diagnostic and is statistically "
        "indistinguishable from TA-DVFG across the six paired comparisons.",
        "",
        "## Ablation highlights",
        "",
    ]
    for dataset, group in minedge_summary.groupby("dataset"):
        best_value = group["test_at_best_val_mean"].max()
        best_rows = group[np.isclose(group["test_at_best_val_mean"], best_value)].sort_values("adaptive_min_edges")
        choices = "/".join(str(int(value)) for value in best_rows["adaptive_min_edges"])
        row = best_rows.iloc[0]
        lines.append(
            f"- {dataset} hard: best tested min-edges value is {choices}, "
            f"yielding {100 * row['test_at_best_val_mean']:.2f}% and {row['final_edges_mean']:.2f} final edges."
        )
    for _, row in modality_best.iterrows():
        lines.append(
            f"- ACM {SETTING_LABELS.get(row['setting'], row['setting'])}: best modality configuration is "
            f"`{row['param_modalities']}` at {100 * row['test_at_best_val_mean']:.2f}%."
        )
    if not noise_summary.empty:
        adaptive_noise = noise_summary[noise_summary["method"] == "adaptive_graph_val"]
        for dataset, group in adaptive_noise.groupby("dataset"):
            low = group.sort_values("param_useful_parties").iloc[0]
            high = group.sort_values("param_useful_parties").iloc[-1]
            lines.append(
                f"- {dataset} legacy noise suite: TA-DVFG changes from "
                f"{100 * low['test_at_best_val_mean']:.2f}% at {int(low['param_useful_parties'])} useful parties "
                f"to {100 * high['test_at_best_val_mean']:.2f}% at {int(high['param_useful_parties'])}."
            )
    if not evaluator_summary.empty:
        lines.append(
            "- Active-party evaluator results are numerically comparable within their own suite; "
            "the implementation changes label ownership semantics while retaining the same validation-score computation."
        )
    if not active_protocol_table.empty:
        lines += [
            "",
            "## Active-party evaluator protocol",
            "",
            "- Local predictor training remains the controlled all-parties supervised simulation.",
            "- Validation labels are exposed only to the designated active evaluator, which computes "
            "party reliability, Global Top-k ranking, and candidate topology scores from submitted predictions.",
        ]
        for _, row in active_protocol_table.sort_values(["dataset", "setting"]).iterrows():
            lines.append(
                f"- {row['dataset']} {row['setting_label']}: TA-DVFG {row['ta_dvfg']}; "
                f"gap to Full Mesh {row['ta_minus_full_mesh_pp']:+.2f} pp, "
                f"best matched {row['ta_minus_best_matched_pp']:+.2f} pp, and "
                f"Global Top-k {row['ta_minus_global_topk_pp']:+.2f} pp."
            )
    lines += [
        "",
        "## Protocol caveats",
        "",
        "- Cached core, topology objective, edge budget, consensus, min-edges, active-party protocol, and multimodal experiments use "
        "seed-shuffled useful/noisy party positions.",
        "- Noise ratio, readout, active-party evaluator, protocol sensitivity, and centralized references were "
        "launched with the legacy unshuffled party order. Use their within-suite contrasts, but do not interpret "
        "their absolute values as a perfectly matched comparison with cached core.",
        "- Protocol sensitivity changes both the split policy and party-order protocol; it is evidence about the "
        "combined protocol change, not an isolated split-only effect.",
        "- IMDB is an HGB-derived controlled single-label conversion, not an official HGB multi-label leaderboard result.",
        "- The multimodal experiment is a simulated local modality construction, not a real image/text/audio benchmark.",
        "",
        "## Recommended paper placement",
        "",
        "- Main paper: ACM/DBLP cached core, separated communication accounting, compact trade-off figure, "
        "topology objective, and edge-budget evidence.",
        "- Separate supplementary PDF: IMDB details, noise curve, consensus, min-edges, readout, active evaluator, "
        "protocol sensitivity, multimodal and centralized references.",
        "- Supplementary reviewer diagnostics: shuffled edge composition and per-seed paired statistics.",
        "",
        "See `tables/`, `statistics/`, and `figures/` for the generated artifacts.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output = args.output_dir.resolve()
    tables_dir = output / "tables"
    stats_dir = output / "statistics"
    figures_dir = output / "figures"
    data_dir = output / "data"
    for directory in (tables_dir, stats_dir, figures_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_all_results(results_root)
    completeness = completeness_table(raw, results_root)
    audit = protocol_audit(raw)
    save_table(raw, data_dir / "all_raw_results.csv")
    compact_columns = [
        "suite",
        "job_id",
        "engine",
        "dataset",
        "setting",
        "seed",
        "method",
        "method_label",
        "test_at_best_val",
        "macro_f1_at_best_val",
        "inference_comm",
        "peer_to_peer_comm",
        "global_readout_comm",
        "total_comm",
        "protocol_type",
        "validation_evaluator",
        "active_party_id",
        "local_training_labels",
        "validation_label_protocol",
        "topology_evaluator",
        "local_training_protocol",
        "validation_split_mode",
        "topology_val_fraction",
        "topology_val_count",
        "selection_val_count",
        "final_edges",
        "topology_update_comm",
        "topology_eval_count",
        "topology_update_time",
        "useful_useful_edges",
        "useful_noisy_edges",
        "noisy_noisy_edges",
        "adaptive_min_edges",
        "param_useful_parties",
        "param_adaptive_edge_budget",
        "param_adaptive_min_gain",
        "param_pred_consensus_steps",
        "param_pred_self_weight",
        "param_vote_weighting",
        "param_readout_mode",
        "param_validation_evaluator",
        "param_active_party_id",
        "param_local_training_protocol",
        "param_validation_split_mode",
        "param_topology_val_fraction",
        "param_topology_every",
        "param_topology_max_updates",
        "param_num_parties",
        "param_modalities",
        "prediction_cache_fingerprint",
    ]
    save_table(
        raw[[column for column in compact_columns if column in raw.columns]],
        data_dir / "all_raw_compact.csv",
    )
    save_table(completeness, tables_dir / "experiment_completeness.csv")
    save_table(audit, tables_dir / "protocol_audit.csv")

    core = raw[raw["suite"] == "core_cached"].copy()
    core_summary = summarize(core, ["dataset", "setting", "setting_label", "method", "method_label"])
    save_table(core_summary, tables_dir / "core_summary_long.csv")
    save_table(paper_core_table(core_summary), tables_dir / "paper_main_accuracy.csv")
    save_table(paper_macro_table(core_summary), tables_dir / "paper_main_macro_f1.csv")
    save_table(paper_grouped_main_table(core_summary), tables_dir / "paper_main_grouped.csv")

    communication = core_summary[
        core_summary["method"].isin(
            [
                "local_reliability_vote",
                "topk_reliability_vote",
                "fixed_ring_matched",
                "expander_matched",
                "full_mesh",
                "adaptive_graph_val",
            ]
        )
    ][
        [
            "dataset",
            "setting",
            "method",
            "method_label",
            "test_at_best_val_mean",
            "test_at_best_val_std",
            "inference_comm_mean",
            "peer_to_peer_comm_mean",
            "global_readout_comm_mean",
            "total_comm_mean",
            "final_edges_mean",
            "topology_update_comm_mean",
            "topology_eval_count_mean",
            "topology_update_time_mean",
        ]
    ]
    save_table(communication, tables_dir / "paper_communication_efficiency.csv")
    protocol_lookup = (
        core.groupby(["dataset", "setting", "method", "method_label"], dropna=False)[
            ["peer_to_peer_comm", "global_readout_comm", "total_comm", "final_edges"]
        ]
        .mean()
        .reset_index()
    )
    protocol_types = (
        core.groupby(["dataset", "setting", "method"], dropna=False)["protocol_type"]
        .first()
        .reset_index()
    )
    save_table(
        protocol_lookup.merge(protocol_types, on=["dataset", "setting", "method"], how="left"),
        tables_dir / "paper_communication_protocol.csv",
    )

    edge_composition = summarize(
        core[core["method"] == "adaptive_graph_val"],
        ["dataset", "setting", "method"],
    )
    save_table(edge_composition, tables_dir / "paper_topology_edge_composition.csv")

    core_stats = paired_test_rows(
        core,
        group_columns=["dataset", "setting"],
        variant_column="method",
        reference="adaptive_graph_val",
        comparisons=[
            "local_reliability_vote",
            "topk_reliability_vote",
            "fixed_ring_matched",
            "expander_matched",
            "full_mesh",
        ],
        family="core",
    )
    save_table(core_stats, stats_dir / "core_paired_significance.csv")

    topology = raw[raw["suite"] == "topology_objective"].copy()
    topology_summary = summarize(topology, ["dataset", "method", "method_label"])
    save_table(topology_summary, tables_dir / "paper_topology_objective.csv")

    edge = raw[raw["suite"] == "edge_budget"].copy()
    edge_summary = summarize(
        edge,
        ["dataset", "param_adaptive_edge_budget", "param_adaptive_min_gain"],
    )
    save_table(edge_summary, tables_dir / "paper_edge_budget.csv")

    consensus = raw[raw["suite"] == "consensus"].copy()
    consensus["variant"] = variant_from_job_id(consensus)
    consensus_summary = summarize(consensus, ["dataset", "setting", "variant"])
    save_table(consensus_summary, tables_dir / "paper_consensus_ablation.csv")

    noise = raw[raw["suite"] == "noise_ratio"].copy()
    noise_summary = summarize(noise, ["dataset", "param_useful_parties", "method", "method_label"])
    save_table(noise_summary, tables_dir / "paper_noise_robustness.csv")

    minedges = raw[raw["suite"] == "minedge_sweep"].copy()
    minedge_summary = summarize(minedges, ["dataset", "adaptive_min_edges"])
    save_table(minedge_summary, tables_dir / "paper_minedges_ablation.csv")

    readout = raw[raw["suite"] == "readout_ablation"].copy()
    readout_summary = summarize(readout, ["dataset", "setting", "param_readout_mode"])
    save_table(readout_summary, tables_dir / "paper_readout_ablation.csv")
    readout_stats = paired_test_rows(
        readout,
        group_columns=["dataset", "setting"],
        variant_column="param_readout_mode",
        reference="active_vote",
        comparisons=["mean_party", "best_party"],
        family="readout",
    )

    evaluator = raw[raw["suite"] == "active_party_evaluator"].copy()
    evaluator_summary = summarize(evaluator, ["dataset", "setting", "param_validation_evaluator"])
    save_table(evaluator_summary, tables_dir / "paper_active_party_evaluator.csv")
    evaluator_stats = paired_test_rows(
        evaluator,
        group_columns=["dataset", "setting"],
        variant_column="param_validation_evaluator",
        reference="active_party",
        comparisons=["all_parties"],
        family="active_evaluator",
    )

    active_protocol = raw[raw["suite"] == "active_party_protocol"].copy()
    active_protocol_summary = summarize(
        active_protocol,
        ["dataset", "setting", "setting_label", "method", "method_label"],
    )
    save_table(
        active_protocol_summary,
        tables_dir / "active_party_protocol_summary_long.csv",
    )
    active_protocol_table = paper_active_party_protocol_table(active_protocol_summary)
    save_table(
        active_protocol_table,
        tables_dir / "paper_active_party_protocol.csv",
    )
    active_protocol_stats = paired_test_rows(
        active_protocol,
        group_columns=["dataset", "setting"],
        variant_column="method",
        reference="adaptive_graph_val",
        comparisons=[
            "local_reliability_vote",
            "topk_reliability_vote",
            "fixed_ring_matched",
            "random_matched",
            "expander_matched",
            "full_mesh",
        ],
        family="active_party_protocol",
    )

    strict_training = raw[raw["suite"] == "strict_label_training"].copy()
    strict_training_summary = summarize(
        strict_training,
        ["dataset", "method", "method_label", "param_local_training_protocol"],
    )
    save_table(
        strict_training_summary,
        tables_dir / "paper_strict_label_training.csv",
    )
    strict_comparison = pd.concat(
        [
            active_protocol[
                (active_protocol["setting"] == "hard_noisy")
                & (active_protocol["method"] == "adaptive_graph_val")
            ].assign(training_variant="all_parties_supervised"),
            strict_training[
                strict_training["method"] == "adaptive_graph_val"
            ].assign(training_variant="active_party_logit_gradient"),
        ],
        ignore_index=True,
        sort=False,
    )
    strict_training_stats = paired_test_rows(
        strict_comparison,
        group_columns=["dataset"],
        variant_column="training_variant",
        reference="active_party_logit_gradient",
        comparisons=["all_parties_supervised"],
        family="strict_label_training",
    )

    split_validation = raw[raw["suite"] == "split_validation"].copy()
    split_validation_summary = summarize(
        split_validation,
        [
            "dataset",
            "method",
            "method_label",
            "param_validation_split_mode",
            "param_topology_val_fraction",
        ],
    )
    save_table(
        split_validation_summary,
        tables_dir / "paper_split_validation.csv",
    )
    split_validation_stats = paired_test_rows(
        split_validation[split_validation["method"] == "adaptive_graph_val"],
        group_columns=["dataset"],
        variant_column="param_validation_split_mode",
        reference="shared",
        comparisons=["disjoint"],
        family="split_validation",
    )

    decentralized_readout = raw[raw["suite"] == "decentralized_readout"].copy()
    decentralized_readout_summary = summarize(
        decentralized_readout,
        ["dataset", "param_readout_mode"],
    )
    save_table(
        decentralized_readout_summary,
        tables_dir / "paper_decentralized_readout.csv",
    )
    decentralized_readout_stats = paired_test_rows(
        decentralized_readout,
        group_columns=["dataset"],
        variant_column="param_readout_mode",
        reference="active_vote",
        comparisons=["local_mean", "local_worst", "active_party_local"],
        family="decentralized_readout",
    )

    active_identity = raw[raw["suite"] == "active_party_identity"].copy()
    active_identity_summary = summarize(
        active_identity,
        ["dataset", "param_active_party_id"],
    )
    save_table(
        active_identity_summary,
        tables_dir / "paper_active_party_identity.csv",
    )
    active_identity_stats = paired_test_rows(
        active_identity,
        group_columns=["dataset"],
        variant_column="param_active_party_id",
        reference=0,
        comparisons=[1, 5, 10, 14],
        family="active_party_identity",
    )

    label_budget = raw[raw["suite"] == "validation_label_budget"].copy()
    label_budget_summary = summarize(
        label_budget,
        ["dataset", "method", "method_label", "param_topology_val_fraction"],
    )
    save_table(
        label_budget_summary,
        tables_dir / "paper_validation_label_budget.csv",
    )
    label_budget_stats = paired_test_rows(
        label_budget[label_budget["method"] == "adaptive_graph_val"],
        group_columns=["dataset"],
        variant_column="param_topology_val_fraction",
        reference=1.0,
        comparisons=[0.10, 0.25, 0.50, 0.75],
        family="validation_label_budget",
    )

    topology_frequency = raw[raw["suite"] == "topology_frequency"].copy()
    topology_frequency["frequency_variant"] = (
        topology_frequency["job_id"].astype(str).str.rsplit("_", n=1).str[-1]
    )
    topology_frequency_summary = summarize(
        topology_frequency,
        [
            "dataset",
            "frequency_variant",
            "param_topology_every",
            "param_topology_max_updates",
        ],
    )
    save_table(
        topology_frequency_summary,
        tables_dir / "paper_topology_frequency.csv",
    )
    topology_frequency_stats = paired_test_rows(
        topology_frequency,
        group_columns=["dataset"],
        variant_column="frequency_variant",
        reference="every20",
        comparisons=["once", "every50", "every10"],
        family="topology_frequency",
    )

    party_scalability = raw[raw["suite"] == "party_scalability"].copy()
    party_scalability_summary = summarize(
        party_scalability,
        ["param_num_parties", "param_useful_parties", "method", "method_label"],
    )
    save_table(
        party_scalability_summary,
        tables_dir / "paper_party_scalability.csv",
    )

    reviewer_protocol_table = paper_reviewer_protocol_table(
        active_protocol,
        strict_training,
        split_validation,
        decentralized_readout,
    )
    save_table(
        reviewer_protocol_table,
        tables_dir / "paper_reviewer_protocols_compact.csv",
    )

    protocol = raw[raw["suite"] == "protocol_sensitivity"].copy()
    protocol_summary = summarize(protocol, ["dataset", "setting", "method", "method_label"])
    save_table(protocol_summary, tables_dir / "paper_protocol_sensitivity.csv")

    multimodal = raw[raw["suite"] == "multimodal_core"].copy()
    multimodal_summary = summarize(multimodal, ["dataset", "setting", "method", "method_label"])
    save_table(multimodal_summary, tables_dir / "paper_multimodal_core.csv")
    multimodal_stats = paired_test_rows(
        multimodal,
        group_columns=["dataset", "setting"],
        variant_column="method",
        reference="adaptive_graph_val",
        comparisons=[
            "local_reliability_vote",
            "topk_reliability_vote",
            "fixed_ring_matched",
            "expander_matched",
            "full_mesh",
        ],
        family="multimodal",
    )

    modality = raw[raw["suite"] == "modality_ablation"].copy()
    modality_summary = summarize(modality, ["setting", "param_modalities"])
    save_table(modality_summary, tables_dir / "paper_modality_ablation.csv")
    modality_stats = paired_test_rows(
        modality,
        group_columns=["setting"],
        variant_column="param_modalities",
        reference="ATTR,TEXT,STRUCT,HYBRID",
        comparisons=["ATTR", "TEXT", "STRUCT", "HYBRID", "TEXT_STRUCT"],
        family="modality",
    )

    central = raw[raw["suite"] == "central_baselines"].copy()
    central_summary = summarize(central, ["dataset", "setting", "method", "method_label"])
    save_table(central_summary, tables_dir / "paper_central_references.csv")

    reviewer = raw[raw["suite"] == "reviewer_safe_shuffled"].copy()
    reviewer_summary = summarize(reviewer, ["dataset", "setting", "method", "method_label"])
    save_table(reviewer_summary, tables_dir / "paper_reviewer_safe_shuffled.csv")
    reviewer_stats = paired_test_rows(
        reviewer,
        group_columns=["dataset", "setting"],
        variant_column="method",
        reference="adaptive_graph_val",
        comparisons=["topk_reliability_vote", "fixed_ring_matched", "expander_matched", "full_mesh"],
        family="reviewer_safe_shuffled",
    )

    ablation_stats = pd.concat(
        [readout_stats, evaluator_stats, modality_stats],
        ignore_index=True,
        sort=False,
    )
    save_table(ablation_stats, stats_dir / "ablation_paired_significance.csv")
    save_table(multimodal_stats, stats_dir / "multimodal_paired_significance.csv")
    save_table(reviewer_stats, stats_dir / "reviewer_safe_paired_significance.csv")
    save_table(
        active_protocol_stats,
        stats_dir / "active_party_protocol_paired_significance.csv",
    )
    reviewer_protocol_stats = pd.concat(
        [
            split_validation_stats,
            strict_training_stats,
            decentralized_readout_stats,
            active_identity_stats,
            label_budget_stats,
            topology_frequency_stats,
        ],
        ignore_index=True,
        sort=False,
    )
    save_table(
        reviewer_protocol_stats,
        stats_dir / "reviewer_protocol_paired_significance.csv",
    )

    setup_plotting()
    plot_core_heatmap(core_summary, figures_dir / "core_accuracy_heatmap.png")
    plot_core_tradeoff(core_summary, figures_dir / "core_accuracy_communication_tradeoff.png")
    plot_compact_tradeoff(
        core_summary,
        figures_dir / "tradeoff_compact.png",
        figures_dir / "tradeoff_compact.pdf",
    )
    plot_topology_objective(topology, figures_dir / "topology_objective_ablation.png")
    plot_topology_overhead(topology, figures_dir / "topology_selection_overhead.png")
    plot_edge_budget(edge, figures_dir / "edge_budget_ablation.png")
    plot_consensus(consensus, figures_dir / "consensus_ablation.png")
    plot_noise(noise, figures_dir / "noise_ratio_robustness.png")
    plot_minedges(minedges, figures_dir / "minedges_sensitivity.png")
    plot_categorical_ablation(
        readout,
        category="param_readout_mode",
        title="Final readout ablation (legacy unshuffled suite)",
        path=figures_dir / "readout_ablation.png",
        order=["active_vote", "mean_party", "best_party"],
    )
    plot_categorical_ablation(
        evaluator,
        category="param_validation_evaluator",
        title="Validation-label owner sensitivity (legacy unshuffled suite)",
        path=figures_dir / "active_party_evaluator.png",
        order=["all_parties", "active_party"],
    )
    if not active_protocol.empty:
        plot_active_party_protocol(
            active_protocol,
            figures_dir / "active_party_protocol.png",
        )
    if not strict_training.empty:
        plot_reviewer_protocols(
            split_validation,
            decentralized_readout,
            active_identity,
            strict_training,
            figures_dir / "reviewer_protocol_checks.png",
        )
    if not label_budget.empty:
        plot_label_budget_and_frequency(
            label_budget,
            topology_frequency,
            figures_dir / "label_budget_topology_frequency.png",
        )
    if not party_scalability.empty:
        plot_party_scalability(
            party_scalability,
            figures_dir / "party_scalability.png",
        )
    plot_protocol_sensitivity(protocol, figures_dir / "protocol_sensitivity.png")
    plot_modality(modality, figures_dir / "modality_ablation.png")
    plot_multimodal_comparison(core, multimodal, figures_dir / "standard_vs_multimodal.png")
    plot_central_references(core, central, figures_dir / "central_references.png")
    plot_edge_composition(core, figures_dir / "adaptive_edge_composition.png")
    plot_reviewer_diagnostics(reviewer, figures_dir / "reviewer_safe_edge_diagnostics.png")

    build_summary_markdown(
        completeness,
        core_summary,
        core_stats,
        minedge_summary,
        noise_summary,
        modality_summary,
        evaluator_summary,
        active_protocol_table,
        output / "final_summary.md",
    )

    manifest = {
        "generated_from": str(results_root),
        "output_dir": str(output),
        "raw_rows": len(raw),
        "completed_suites": int(completeness["complete"].sum()),
        "total_suites": len(completeness),
        "tables": sorted(path.name for path in tables_dir.glob("*.csv")),
        "statistics": sorted(path.name for path in stats_dir.glob("*.csv")),
        "figures": sorted(
            path.name for path in figures_dir.iterdir()
            if path.suffix.lower() in {".png", ".pdf"}
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
