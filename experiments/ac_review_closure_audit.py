"""Targeted AC-review closure audits.

This script inventories existing evidence and audits the two highest-value
concerns that can be closed without launching a new benchmark suite:

1. Best Matched selection provenance.
2. Existing split/disjoint-validation evidence.
"""

from __future__ import annotations

import json
import math
import io
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "ac_review_closure"

SETTINGS = [
    ("ACM", "main", "main experiment/core/acm_main/metrics_summary.csv", "main experiment/core/acm_main/metrics.csv"),
    ("ACM", "hard_noisy", "main experiment/core/acm_hard_noisy/metrics_summary.csv", "main experiment/core/acm_hard_noisy/metrics.csv"),
    ("DBLP", "main", "main experiment/core/dblp_main/metrics_summary.csv", "main experiment/core/dblp_main/metrics.csv"),
    ("DBLP", "hard_noisy", "main experiment/core/dblp_hard_noisy/metrics_summary.csv", "main experiment/core/dblp_hard_noisy/metrics.csv"),
    ("IMDB", "main", "main experiment/core/imdb_main/metrics_summary.csv", "main experiment/core/imdb_main/metrics.csv"),
    ("IMDB", "hard_noisy", "main experiment/core/imdb_hard_noisy/metrics_summary.csv", "main experiment/core/imdb_hard_noisy/metrics.csv"),
]

MATCHED_METHODS = {
    "fixed_ring_matched": "Ring Matched",
    "random_matched": "Random Matched",
    "expander_matched": "Expander Matched",
}


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / path)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    return float(arr.mean()), float(arr.std(ddof=0))


def paired_stats(a: pd.Series, b: pd.Series) -> Dict[str, object]:
    delta = (a.astype(float) - b.astype(float)).dropna().to_numpy(dtype=float)
    if delta.size == 0:
        return {
            "n": 0,
            "mean_delta": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "paired_ttest_p": math.nan,
            "wilcoxon_p": math.nan,
            "cohen_dz": math.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
    tcrit = 2.7764451051977987 if delta.size == 5 else 1.96
    half = tcrit * sd / math.sqrt(delta.size) if delta.size > 1 else 0.0
    try:
        from scipy import stats

        t_p = float(stats.ttest_1samp(delta, 0.0).pvalue) if delta.size > 1 else math.nan
        try:
            w_p = float(stats.wilcoxon(delta, zero_method="wilcox").pvalue)
        except Exception:
            w_p = math.nan
    except Exception:
        t_p = math.nan
        w_p = math.nan
    return {
        "n": int(delta.size),
        "mean_delta": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "paired_ttest_p": t_p,
        "wilcoxon_p": w_p,
        "cohen_dz": mean / sd if sd > 0 else math.nan,
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
    }


def csv_block(frame: pd.DataFrame) -> str:
    handle = io.StringIO()
    frame.to_csv(handle, index=False)
    return "```csv\n" + handle.getvalue().strip() + "\n```"


def write_claim_map() -> pd.DataFrame:
    rows = [
        {
            "concern": "Global Top-k often matches/exceeds TA-DVFG",
            "current_evidence": "Core HGB tables, active-party evaluator summaries, ACM/DBLP Hard alignment suites show Global Top-k is strong centralized prediction-selection reference.",
            "source_file": "main experiment/core/*/metrics_summary.csv; active_party_evaluator_summary.csv; outputs/alignment_references_*/*summary.csv",
            "already_addressed": "partly",
            "new_run_required": "no",
            "planned_paper_patch": "Reframe TA-DVFG as sparse P2P/local-readout method competitive with Global Top-k, not as uniformly higher active-readout accuracy.",
        },
        {
            "concern": "Real heterogeneous evidence is mainly MovieLens-5",
            "current_evidence": "MovieLens-5 and 15-party constructed weak-view MovieLens results exist, but are constructed vertical interfaces rather than natural organizations.",
            "source_file": "outputs/movielens*; outputs/movielens_final_package",
            "already_addressed": "partly",
            "new_run_required": "no",
            "planned_paper_patch": "Use precise wording: real-data heterogeneous predictor benchmark / constructed vertical party interfaces.",
        },
        {
            "concern": "Algorithmic novelty is incremental",
            "current_evidence": "Topology objective, exact config audit, selected latent-fusion comparisons, and matched controls exist.",
            "source_file": "main experiment/ta_dvfg_hgb_reliability.py; outputs/alignment_references_acmhard5; outputs/alignment_references_dblphard5",
            "already_addressed": "partly",
            "new_run_required": "optional only",
            "planned_paper_patch": "Tighten claim around prediction-space sparse topology and deployment capability; optional exact-search sanity can go supplement.",
        },
        {
            "concern": "Validation overfitting risk",
            "current_evidence": "Completed shared/disjoint split-validation summaries exist for ACM/DBLP Hard.",
            "source_file": "results/split_validation/*hard*/*summary.csv",
            "already_addressed": "yes for Phase 2A audit",
            "new_run_required": "no for Phase 2A; candidate replay optional",
            "planned_paper_patch": "Add concise disjoint-validation result if audit passes.",
        },
        {
            "concern": "Scalability bottleneck",
            "current_evidence": "Party scalability outputs and appendix wording already identify candidate scoring growth.",
            "source_file": "results/party_scalability; appendix.tex",
            "already_addressed": "partly",
            "new_run_required": "no",
            "planned_paper_patch": "Clarify sparse deployment scales with selected edges, exhaustive candidate scoring remains bottleneck.",
        },
        {
            "concern": "No formal privacy",
            "current_evidence": "Appendix already notes no cryptographic privacy; predictions/logits can leak.",
            "source_file": "appendix.tex",
            "already_addressed": "partly",
            "new_run_required": "no",
            "planned_paper_patch": "Clarify prediction-space interface is not privacy guarantee.",
        },
        {
            "concern": "Best Matched selection ambiguity",
            "current_evidence": "Engine and final analysis have two distinct paths: validation-selected best_matched_sparse and post-hoc test-mean envelope over fixed matched controls.",
            "source_file": "main experiment/ta_dvfg_hgb_reliability.py; experiments/final_analysis.py",
            "already_addressed": "no",
            "new_run_required": "no",
            "planned_paper_patch": "If table uses post-hoc envelope, relabel Oracle Best Matched or replace main row with predeclared Random Matched.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "current_claim_map.csv", index=False)
    return frame


def best_matched_audit() -> pd.DataFrame:
    rows = []
    for dataset, setting, summary_path, metrics_path in SETTINGS:
        summary = read_csv(summary_path)
        metrics = read_csv(metrics_path)
        fixed = summary[summary["method"].isin(MATCHED_METHODS)]
        if fixed.empty:
            continue
        selected = fixed.loc[fixed["test_at_best_val_mean"].astype(float).idxmax()]
        per_seed = []
        for seed, group in metrics[metrics["method"].isin(MATCHED_METHODS)].groupby("seed"):
            row = group.loc[group["test_at_best_val"].astype(float).idxmax()]
            per_seed.append(f"{int(seed)}:{MATCHED_METHODS[row['method']]}")
        has_engine_best = "best_matched_sparse" in set(metrics["method"])
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "reported_best_matched_family": MATCHED_METHODS[str(selected["method"])],
                "reported_best_matched_method": str(selected["method"]),
                "reported_value_mean": float(selected["test_at_best_val_mean"]),
                "reported_value_std": float(selected["test_at_best_val_std"]),
                "selection_basis_for_paper_grouped_table": "post-hoc test_at_best_val_mean envelope over fixed_ring_matched/random_matched/expander_matched",
                "choice_granularity_for_paper_grouped_table": "per dataset/setting",
                "per_seed_test_oracle_family_if_selected_per_seed": ";".join(per_seed),
                "engine_best_matched_sparse_available": has_engine_best,
                "engine_best_matched_rule": "per epoch validation-only choice among ring/random/expander matched topologies" if has_engine_best else "not present in core headline metrics",
                "source_code_path": "experiments/final_analysis.py:449-475,504-507,1020-1025; main experiment/ta_dvfg_hgb_reliability.py:2086-2105",
                "audit_conclusion": "Case B for paper grouped row: post-hoc test-mean envelope; relabel as Oracle Best Matched or replace with predeclared fixed control.",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "best_matched_selection_audit.csv", index=False)
    return frame


def disjoint_validation_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [
        ("ACM", "hard", "results/split_validation/acm_hard_shared/acm_hard_shared_validation_summary.csv", "results/split_validation/acm_hard_disjoint/acm_hard_disjoint_validation_summary.csv"),
        ("DBLP", "hard", "results/split_validation/dblp_hard_shared/dblp_hard_shared_validation_summary.csv", "results/split_validation/dblp_hard_disjoint/dblp_hard_disjoint_validation_summary.csv"),
    ]
    audit_rows = []
    stats_rows = []
    for dataset, setting, shared_path, disjoint_path in pairs:
        shared = read_csv(shared_path)
        disjoint = read_csv(disjoint_path)
        for protocol, frame, path in [("shared", shared, shared_path), ("disjoint", disjoint, disjoint_path)]:
            for method in ["topk_reliability_vote", "full_mesh", "fixed_ring_matched", "random_matched", "expander_matched", "adaptive_graph_val"]:
                rows = frame[frame["method"] == method]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                audit_rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "protocol": protocol,
                        "method": method,
                        "test_at_best_val_mean": float(row["test_at_best_val_mean"]),
                        "test_at_best_val_std": float(row["test_at_best_val_std"]),
                        "macro_f1_mean": float(row["macro_f1_mean"]),
                        "validation_split_mode": row.get("validation_split_mode", ""),
                        "topology_val_fraction": row.get("topology_val_fraction", ""),
                        "topology_val_count": row.get("topology_val_count", ""),
                        "selection_val_count": row.get("selection_val_count", ""),
                        "validation_evaluator": row.get("validation_evaluator", ""),
                        "topology_evaluator": row.get("topology_evaluator", ""),
                        "seeds": row.get("seeds", ""),
                        "source_file": path,
                    }
                )
        # Use summary-level comparison; paired per-seed CSV was not present in these dirs.
        for method in ["adaptive_graph_val", "topk_reliability_vote", "full_mesh"]:
            s = shared[shared["method"] == method]
            d = disjoint[disjoint["method"] == method]
            if s.empty or d.empty:
                continue
            stats_rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "method": method,
                    "comparison": "disjoint minus shared",
                    "mean_delta": float(d.iloc[0]["test_at_best_val_mean"]) - float(s.iloc[0]["test_at_best_val_mean"]),
                    "shared_mean": float(s.iloc[0]["test_at_best_val_mean"]),
                    "disjoint_mean": float(d.iloc[0]["test_at_best_val_mean"]),
                    "paired_stats_status": "not_available_from_summary_only",
                    "note": "Existing split-validation summaries are complete and valid for protocol audit; paired tests require per-seed split-validation metrics not found in summary files.",
                }
            )
    audit = pd.DataFrame(audit_rows)
    stats = pd.DataFrame(stats_rows)
    audit.to_csv(OUT_DIR / "disjoint_validation_audit.csv", index=False)
    stats.to_csv(OUT_DIR / "disjoint_validation_stats.csv", index=False)
    return audit, stats


def write_reports(claim_map: pd.DataFrame, best: pd.DataFrame, disjoint: pd.DataFrame, disjoint_stats: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "AC_REVIEW_STATUS.md").write_text(
        "\n".join(
            [
                "# AC Review Closure Status",
                "",
                "Scope: targeted audits only. No new full benchmark suite was launched.",
                "",
                "## Phase Status",
                "",
                "- Phase 0 inventory: complete.",
                "- Phase 1 Best Matched selection audit: complete; blocking issue found for grouped paper row.",
                "- Phase 2A disjoint-validation audit: existing ACM/DBLP Hard split-validation summaries found and audited.",
                "- Phase 2B candidate-level validation-to-test replay: not run in this pass.",
                "- Phases 3-6 wording patches: patch files still pending; do not patch main paper until the blocking Best Matched label is resolved.",
                "",
                "## Highest-Priority Decision",
                "",
                "The paper grouped row named `Best Matched Sparse` is generated in `experiments/final_analysis.py` by selecting the fixed matched family with the largest test mean per dataset/setting. This is a post-hoc test envelope, not a deployable validation-selected baseline. It should be relabeled `Oracle Best Matched` or replaced in the main table by a predeclared matched control.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    best_lines = [
        "# Best Matched Selection Report",
        "",
        "Status: **BLOCKING LABEL ISSUE FOUND**",
        "",
        "Two code paths exist:",
        "",
        "1. `best_matched_sparse` in the HGB engine chooses among Ring/Random/Expander matched topologies using validation score at each epoch. This is validation-only.",
        "2. The paper grouped table in `experiments/final_analysis.py` constructs `Best Matched Sparse` by taking `idxmax()` over `test_at_best_val_mean` among fixed matched controls per dataset/setting. This is a post-hoc test-mean envelope.",
        "",
        "Therefore, if the grouped paper row is used, it should be relabeled `Oracle Best Matched` and described as a post-hoc upper envelope over Ring/Random/Expander matched controls.",
        "",
        "## Per-Setting Selection",
        "",
        csv_block(best),
        "",
    ]
    (OUT_DIR / "BEST_MATCHED_SELECTION_REPORT.md").write_text("\n".join(best_lines), encoding="utf-8")
    disjoint_lines = [
        "# Disjoint Validation Report",
        "",
        "Status: existing ACM/DBLP Hard split-validation evidence found.",
        "",
        "The existing split-validation configs use `validation_split_mode=disjoint` and `topology_val_fraction=0.5`, separating topology-label positions from epoch-selection positions within validation. Test labels are not used for topology selection in this protocol.",
        "",
        "Summary-level means are audited below. Per-seed paired tests are not reconstructed here because only summary files were found in the split-validation output directories.",
        "",
        "## Summary",
        "",
        csv_block(disjoint_stats),
        "",
        "## Audited Rows",
        "",
        csv_block(disjoint),
        "",
    ]
    (OUT_DIR / "DISJOINT_VALIDATION_REPORT.md").write_text("\n".join(disjoint_lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    claim_map = write_claim_map()
    best = best_matched_audit()
    disjoint, disjoint_stats = disjoint_validation_audit()
    write_reports(claim_map, best, disjoint, disjoint_stats)
    print(f"Wrote AC-review closure audits to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    def csv_block(frame: pd.DataFrame) -> str:
        handle = io.StringIO()
        frame.to_csv(handle, index=False)
        return "```csv\n" + handle.getvalue().strip() + "\n```"
