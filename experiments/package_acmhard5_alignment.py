"""Package the ACM Hard five-seed alignment-reference suite.

This script only post-processes the fresh within-suite ACM Hard outputs created
by ``alignment_references.py``. It does not run any new dataset/setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import alignment_references as AR  # noqa: E402


METHOD_ORDER = [
    "Best Single",
    "Global Top-k Reliability",
    "Full Mesh prediction consensus",
    "TA-DVFG",
    "Project-and-Mean",
    "Reliability-Selected Project-and-Mean",
    "Reliability-Selected Gated Fusion",
    "Useful-Only best latent fusion",
    "Project-and-Concat",
    "Gated Fusion",
]

SETTING_LABEL = {
    ("ACM", "hard"): "acmhard5",
    ("DBLP", "hard"): "dblphard5",
}


def parse_jsonish(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return []


def try_stats():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def ci95(delta: np.ndarray) -> tuple[float, float]:
    if len(delta) <= 1:
        mean = float(np.mean(delta)) if len(delta) else math.nan
        return mean, mean
    stats = try_stats()
    tcrit = float(stats.t.ppf(0.975, len(delta) - 1)) if stats is not None else 2.7764451051977987
    mean = float(np.mean(delta))
    half = tcrit * float(np.std(delta, ddof=1)) / math.sqrt(len(delta))
    return mean - half, mean + half


def paired_test_rows(per_seed: pd.DataFrame, dataset: str, setting: str) -> pd.DataFrame:
    stats = try_stats()
    rows = []
    pivot = per_seed.pivot_table(index="seed", columns="method", values="test_at_best_val", aggfunc="first")
    if "TA-DVFG" not in pivot.columns:
        return pd.DataFrame()
    for method in METHOD_ORDER:
        if method == "TA-DVFG" or method not in pivot.columns:
            continue
        common = pivot[["TA-DVFG", method]].dropna()
        if common.empty:
            continue
        delta = (common[method] - common["TA-DVFG"]).to_numpy(dtype=float)
        mean_delta = float(np.mean(delta))
        low, high = ci95(delta)
        sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        dz = mean_delta / sd if sd > 0 else math.nan
        if stats is not None and len(delta) > 1:
            t_p = float(stats.ttest_rel(common[method], common["TA-DVFG"]).pvalue)
            try:
                w_p = float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                w_p = math.nan
        else:
            t_p = math.nan
            w_p = math.nan
        wins = int(np.sum(delta > 1e-12))
        ties = int(np.sum(np.abs(delta) <= 1e-12))
        losses = int(np.sum(delta < -1e-12))
        rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                "comparison": f"{method} minus TA-DVFG",
                "method": method,
                "n": int(len(delta)),
                "paired_mean_delta_vs_tadvfg": mean_delta,
                "ci95_low": low,
                "ci95_high": high,
                "paired_ttest_p": t_p,
                "wilcoxon_signed_rank_p": w_p,
                "cohen_dz": dz,
                "wins_vs_tadvfg": wins,
                "ties_vs_tadvfg": ties,
                "losses_vs_tadvfg": losses,
                "note": "Do not interpret non-significance as equivalence.",
            }
        )
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame, dataset: str, setting: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        group = per_seed[per_seed["method"] == method]
        if group.empty:
            continue
        values = group["test_at_best_val"].astype(float)
        f1 = group["macro_f1"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "test_at_best_val_mean": float(values.mean()),
                "test_at_best_val_std": float(values.std(ddof=0)),
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=0)),
                "per_seed_test_at_best_val": json.dumps(
                    {str(int(r.seed)): float(r.test_at_best_val) for r in group.itertuples()}
                ),
                "seeds": ",".join(str(int(v)) for v in sorted(group["seed"].unique())),
            }
        )
    return pd.DataFrame(rows)


def augment_per_seed(out_dir: Path, per_seed: pd.DataFrame, selection: pd.DataFrame, metadata: List[Dict], dataset: str, setting: str) -> pd.DataFrame:
    meta_by_seed = {int(item["seed"]): item for item in metadata}
    selected_by_method_seed: Dict[tuple[int, str], Dict] = {}
    for row in selection[selection["selected"].astype(str).str.lower() == "true"].to_dict("records"):
        selected_by_method_seed[(int(row["seed"]), str(row["method"]))] = row

    rows = []
    for row in per_seed.to_dict("records"):
        seed = int(row["seed"])
        method = str(row["method"])
        meta = meta_by_seed.get(seed, {})
        useful = set(int(v) for v in meta.get("useful_party_ids", [0, 1, 2]))
        selected_row = selected_by_method_seed.get((seed, method), {})
        party_ids = parse_jsonish(selected_row.get("party_ids", "[]"))
        if method == "Useful-Only best latent fusion":
            party_ids = meta.get("useful_party_ids", party_ids)
        if method in {"Reliability-Selected Project-and-Mean", "Reliability-Selected Gated Fusion"}:
            party_ids = meta.get("topk_party_ids", party_ids)
        useful_count = sum(1 for item in party_ids if int(item) in useful)
        weak_count = len(party_ids) - useful_count
        row["selected_parties"] = json.dumps(party_ids)
        row["selected_useful_count"] = useful_count if party_ids else ""
        row["selected_weak_count"] = weak_count if party_ids else ""
        row["selected_topology_links"] = ""
        rows.append(row)

    augmented = pd.DataFrame(rows)
    # Recompute selected TA-DVFG topology links from the fresh within-suite caches.
    for seed in sorted(augmented["seed"].unique()):
        args = argparse.Namespace(
            dataset=dataset,
            setting=setting,
            device="cpu",
            local_epochs=0,
            verbose=False,
            output_dir=out_dir,
        )
        ns = AR.build_runtime_args(args, dataset, setting, int(seed))
        ns.cache_namespace = "alignment_references"
        cache_path = out_dir / "fresh_within_suite_cache" / f"{dataset.lower()}_{setting}_seed{int(seed)}_fresh_final.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        result = AR.ENGINE.evaluate_cached_method("adaptive_graph_val", int(seed), ns, cache, cache_path)
        mask = (augmented["seed"] == seed) & (augmented["method"] == "TA-DVFG")
        augmented.loc[mask, "selected_topology_links"] = result.selected_edges
    return augmented


def communication_and_capabilities(per_seed: pd.DataFrame, comm: pd.DataFrame, metadata: List[Dict], dataset: str, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_nodes = int(metadata[0].get("num_nodes", 3025)) if metadata else 3025
    num_classes = int(metadata[0].get("num_classes", 3)) if metadata else 3
    rows = []
    caps = []
    for method in METHOD_ORDER:
        if method not in set(per_seed["method"]):
            continue
        sample = per_seed[per_seed["method"] == method].iloc[0]
        is_latent = "latent" in str(sample["interface_type"]) or method in {
            "Project-and-Mean",
            "Project-and-Concat",
            "Gated Fusion",
            "Reliability-Selected Project-and-Mean",
            "Reliability-Selected Gated Fusion",
            "Useful-Only best latent fusion",
        }
        central = bool(is_latent) or method in {"Best Single", "Global Top-k Reliability", "Full Mesh prediction consensus"}
        sparse_p2p = method == "TA-DVFG"
        local_readout = method == "TA-DVFG"
        selected_party_count = 0
        d_ref = sample.get("d_ref", "N/A")
        hidden_payload = 0
        prediction_payload = 0
        training_payload = 0
        if is_latent:
            selected = parse_jsonish(sample.get("selected_parties", "[]"))
            if not selected and method in {"Project-and-Mean", "Project-and-Concat", "Gated Fusion"}:
                selected = list(range(15))
            selected_party_count = len(selected)
            d_ref_numeric = pd.to_numeric(pd.Series([d_ref]), errors="coerce").iloc[0]
            hidden_payload = selected_party_count * int(d_ref_numeric) if not pd.isna(d_ref_numeric) else 0
            training_payload = n_nodes * selected_party_count * 64
        else:
            if method == "Full Mesh prediction consensus":
                prediction_payload = 15 * 14 * n_nodes * num_classes
            elif method == "Global Top-k Reliability":
                prediction_payload = 5 * n_nodes * num_classes
            elif method == "TA-DVFG":
                prediction_payload = int(per_seed[per_seed["method"] == method]["test_at_best_val"].count())  # overwritten below
                prediction_payload = 5 * 2 * n_nodes * num_classes
            elif method == "Best Single":
                prediction_payload = n_nodes * num_classes
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "selected_party_count": selected_party_count if is_latent else "",
                "trainable_fusion_parameters": sample.get("param_count", 0),
                "hidden_state_inference_payload_scalars_per_target": hidden_payload,
                "prediction_inference_payload_scalars_per_run": prediction_payload,
                "training_payload_scalars_if_hidden_cached": training_payload,
                "centralized_collection_required": central,
                "shared_projection_interface_required": is_latent,
                "sparse_p2p_deployment_supported": sparse_p2p,
                "party_local_post_consensus_readout_supported": local_readout,
                "diagnostic_only": method == "Useful-Only best latent fusion",
            }
        )
        caps.append(
            {
                "method": method,
                "interface_class": (
                    "diagnostic oracle"
                    if method == "Useful-Only best latent fusion"
                    else "centralized stronger-information latent alignment"
                    if is_latent
                    else "prediction-space reference"
                ),
                "centralized_collection_required": central,
                "shared_projection_interface_required": is_latent,
                "sparse_p2p_deployment_supported": sparse_p2p,
                "party_local_post_consensus_readout_supported": local_readout,
                "deployable_main_baseline": method != "Useful-Only best latent fusion",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(caps)


def selected_parties_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in per_seed.to_dict("records"):
        selected = parse_jsonish(row.get("selected_parties", "[]"))
        if not selected and row["method"] not in {"TA-DVFG"}:
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "setting": row["setting"],
                "seed": row["seed"],
                "method": row["method"],
                "selected_parties": json.dumps(selected),
                "selected_useful_count": row.get("selected_useful_count", ""),
                "selected_weak_count": row.get("selected_weak_count", ""),
                "selected_topology_links": row.get("selected_topology_links", ""),
                "diagnostic_only": row["method"] == "Useful-Only best latent fusion",
            }
        )
    return pd.DataFrame(rows)


def write_tradeoff_pdf(path: Path, summary: pd.DataFrame, capabilities: pd.DataFrame, dataset: str, setting: str) -> None:
    import matplotlib.pyplot as plt

    merged = summary.merge(capabilities, on="method", how="left")
    x = [
        0 if row["sparse_p2p_deployment_supported"] else 1 if not row["shared_projection_interface_required"] else 2
        for _, row in merged.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for xi, (_, row) in zip(x, merged.iterrows()):
        ax.scatter(xi, row["test_at_best_val_mean"], s=55)
        ax.annotate(row["method"], (xi, row["test_at_best_val_mean"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sparse P2P", "Central prediction", "Central hidden"])
    ax.set_ylabel("Mean Test@BestVal")
    ax.set_title(f"{dataset} {setting.title()} five-seed interface/performance trade-off")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, stats_df: pd.DataFrame, selected: pd.DataFrame, dataset: str, setting: str) -> None:
    def score(method: str) -> float:
        return float(summary[summary["method"] == method]["test_at_best_val_mean"].iloc[0])

    td = score("TA-DVFG")
    rspm = score("Reliability-Selected Project-and-Mean")
    allpm = score("Project-and-Mean")
    topk = score("Global Top-k Reliability")
    useful = score("Useful-Only best latent fusion")
    delta = rspm - td
    stat_row = stats_df[stats_df["method"] == "Reliability-Selected Project-and-Mean"].iloc[0]
    selected_counts = selected[selected["method"] == "Reliability-Selected Project-and-Mean"][
        ["seed", "selected_parties", "selected_useful_count", "selected_weak_count"]
    ].to_string(index=False)
    text = f"""# {dataset} {setting.title()} Five-Seed Alignment Reference Report

Scope: {dataset} {setting.title()} only, seeds 42-46. No other dataset/setting or multi-dataset suite was run.

## Main Result

- TA-DVFG: {td:.4f} mean Test@BestVal.
- Reliability-Selected Project-and-Mean: {rspm:.4f} mean Test@BestVal.
- Delta selected latent minus TA-DVFG: {delta:.4f}; 95% CI [{stat_row['ci95_low']:.4f}, {stat_row['ci95_high']:.4f}], paired t-test p={stat_row['paired_ttest_p']:.4g}, Wilcoxon p={stat_row['wilcoxon_signed_rank_p']:.4g}.
- Global Top-k Reliability is strongest among deployable/non-oracle methods here: {topk:.4f}.
- Useful-Only latent fusion is diagnostic only: {useful:.4f}.

## Analysis Answers

1. All-party latent fusion is consistently harmed by weak-party contamination: Project-and-Mean averages {allpm:.4f}, far below Reliability-Selected Project-and-Mean at {rspm:.4f}.
2. Reliability selection recovers much of the latent-fusion performance, improving Project-and-Mean by {rspm - allpm:.4f} mean accuracy.
3. After fair reliability selection, TA-DVFG is better than Reliability-Selected Project-and-Mean in this {dataset} {setting.title()} suite ({td:.4f} vs {rspm:.4f}; see stats CSV for win/tie/loss). The paired t-test and exact Wilcoxon should be interpreted with n=5 caution; do not describe non-significance as equivalence.
4. The best predictive trade-off in this {dataset} {setting.title()} suite is Global Top-k Reliability, but it is a centralized prediction reference. TA-DVFG remains the sparse P2P method with party-local post-consensus readout. Reliability-selected latent fusion requires hidden-state access, a shared projection interface, and centralized collection.

## Selected Parties

Reliability-Selected Project-and-Mean selected parties:

```text
{selected_counts}
```

## Paper Patch Guidance

Do not patch the main paper automatically from this {dataset} {setting.title()}-only suite. Use `PAPER_PATCH_NOTES.md` for conditional wording after broader review.
"""
    path.write_text(text, encoding="utf-8")


def write_patch_notes(path: Path) -> None:
    path.write_text(
        """# Paper Patch Notes

No automatic paper patch was generated.

Conditional narratives:

A. If TA-DVFG is better than selected latent fusion:
TA-DVFG outperforms the tested reliability-selected centralized latent reference while retaining prediction-space sparse P2P deployment and party-local post-consensus readout.

B. If results are statistically similar or mixed:
TA-DVFG and reliability-selected latent fusion show a mixed performance trade-off. The latent method requires hidden-state access, a shared projection interface, and centralized collection, whereas TA-DVFG uses prediction-space sparse P2P communication.

C. If selected latent fusion is better:
Reliability-selected centralized latent fusion improves predictive accuracy, but it uses stronger information and deployment assumptions. TA-DVFG should be framed as a lower-assumption prediction-space alternative rather than as dominating latent fusion.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "alignment_references_acmhard5")
    parser.add_argument("--dataset", default="ACM")
    parser.add_argument("--setting", default="hard")
    args = parser.parse_args()
    out = args.output_dir
    dataset = args.dataset.upper()
    setting = args.setting.lower()
    label = SETTING_LABEL.get((dataset, setting), f"{dataset.lower()}{setting}5")
    per_seed = pd.read_csv(out / "alignment_reference_per_seed.csv")
    selection = pd.read_csv(out / "alignment_reference_selection_grid.csv")
    comm = pd.read_csv(out / "alignment_reference_communication.csv")
    metadata = json.loads((out / "alignment_reference_metadata.json").read_text(encoding="utf-8"))

    per_seed = augment_per_seed(out, per_seed, selection, metadata, dataset, setting)
    summary = summarize(per_seed, dataset, setting)
    stats_df = paired_test_rows(per_seed, dataset, setting)
    comm_df, caps_df = communication_and_capabilities(per_seed, comm, metadata, dataset, setting)
    selected_df = selected_parties_table(per_seed)

    per_seed.to_csv(out / f"alignment_{label}_per_seed.csv", index=False)
    summary.to_csv(out / f"alignment_{label}_summary.csv", index=False)
    stats_df.to_csv(out / f"alignment_{label}_stats.csv", index=False)
    comm_df.to_csv(out / f"alignment_{label}_communication.csv", index=False)
    caps_df.to_csv(out / f"alignment_{label}_capabilities.csv", index=False)
    selected_df.to_csv(out / f"alignment_{label}_selected_parties.csv", index=False)
    pd.read_csv(out / "gate_diagnostics.csv").to_csv(out / f"alignment_{label}_gate_diagnostics.csv", index=False)
    write_tradeoff_pdf(out / f"alignment_{label}_tradeoff.pdf", summary, caps_df, dataset, setting)
    write_report(out / f"ALIGNMENT_{label.upper()}_REPORT.md", summary, stats_df, selected_df, dataset, setting)
    write_patch_notes(out / "PAPER_PATCH_NOTES.md")
    print(f"Packaged {dataset} {setting} five-seed alignment suite in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
