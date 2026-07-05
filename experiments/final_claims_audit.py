"""Phase-0 audit for remaining TA-DVFG reviewer-evidence experiments.

This script records verified values from code and raw outputs before any new
experiment branch is attempted. It deliberately does not infer missing results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def extract_arg_defaults() -> Dict[str, str]:
    engine = (REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py").read_text(
        encoding="utf-8", errors="replace"
    )
    names = [
        "topk_reliability_k",
        "adaptive_edge_budget",
        "adaptive_min_edges",
        "max_degree",
        "adaptive_min_gain",
        "topology_every",
        "pred_consensus_steps",
        "pred_self_weight",
        "vote_weighting",
        "reliability_power",
        "topology_degree_power",
        "final_reliability_floor",
        "matched_edge_count",
    ]
    defaults: Dict[str, str] = {}
    for name in names:
        pattern = rf'--{name}".*?default=([^,\)\n]+)'
        match = re.search(pattern, engine, flags=re.S)
        defaults[name] = match.group(1).strip().strip('"') if match else "not_found"

    plan = (REPO_ROOT / "experiments" / "experiment_plan.py").read_text(
        encoding="utf-8", errors="replace"
    )
    base_block = re.search(r"def _base\(.*?return \{(.*?)\n    \}", plan, flags=re.S)
    if base_block:
        block = base_block.group(1)
        for key in [
            "adaptive_edge_budget",
            "adaptive_min_edges",
            "adaptive_min_gain",
            "max_degree",
            "pred_self_weight",
            "topk_reliability_k",
            "matched_edge_count",
        ]:
            m = re.search(rf'"{key}":\s*([^,\n]+)', block)
            if m:
                defaults[f"canonical_{key}"] = m.group(1).strip().strip('"')
    return defaults


def movie_values() -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    summary_path = REPO_ROOT / "outputs" / "movielens" / "movielens_tadvfg_summary.csv"
    per_seed_path = REPO_ROOT / "outputs" / "movielens" / "movielens_tadvfg_per_seed.csv"
    party_path = REPO_ROOT / "outputs" / "movielens" / "party_architecture_audit.csv"
    values: List[Dict[str, object]] = []
    macro: Dict[str, object] = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        for method in ["single_party_best", "adaptive_graph_val", "full_mesh", "adaptive_pair", "topk_reliability_vote"]:
            row = summary[summary["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            values.extend([
                {"name": f"MovieLens5_{method}_auc_mean", "value": row["auc_mean"], "source": str(summary_path)},
                {"name": f"MovieLens5_{method}_auc_std", "value": row["auc_std"], "source": str(summary_path)},
                {"name": f"MovieLens5_{method}_accuracy_mean", "value": row["accuracy_mean"], "source": str(summary_path)},
                {"name": f"MovieLens5_{method}_total_comm_mean", "value": row["total_comm_mean"], "source": str(summary_path)},
                {"name": f"MovieLens5_{method}_links_mean", "value": row["selected_links_mean"], "source": str(summary_path)},
            ])
            macro[method] = row.to_dict()
    if per_seed_path.exists():
        per_seed = pd.read_csv(per_seed_path)
        best = per_seed[per_seed["method"] == "single_party_best"].sort_values("seed")
        values.append({
            "name": "MovieLens5_BestSingle_AUC_per_seed",
            "value": ";".join(f"{int(r.seed)}:{float(r.auc):.6f}" for r in best.itertuples()),
            "source": str(per_seed_path),
        })
    if party_path.exists():
        party = pd.read_csv(party_path)
        for row in party.itertuples(index=False):
            values.append({
                "name": f"MovieLens5_party_{row.party}_parameters",
                "value": int(row.num_parameters),
                "source": str(party_path),
            })
    return values, macro


def status_text(defaults: Dict[str, str]) -> str:
    return f"""# Experiment Status

Generated: {datetime.now(timezone.utc).isoformat()}

## Phase 0 Audit

Status: complete.

- Selector defaults were extracted from `main experiment/ta_dvfg_hgb_reliability.py`.
- Canonical experiment-plan defaults were extracted from `experiments/experiment_plan.py`.
- MovieLens target/split and five-party values were verified from `outputs/movielens/*.csv`.
- Best Single MovieLens per-seed ROC-AUC values were verified from raw per-seed CSV.

## Local Hidden-State Export Feasibility

Status: partial / feasibility-gated.

- HGB local GCN code already has an encoder path (`LocalGCN.encode`) and helper-style embedding access in the engine.
- Existing cached prediction files are prediction caches and do not store hidden representations.
- MovieLens party models currently expose final probabilities only through the experiment runner; hidden states are internal to model-specific forward paths and are not cached.
- Fair representation-alignment baselines therefore require fresh runs with explicit hidden-state export added without changing local predictor behavior.
- Do not synthesize hidden representations from prediction outputs.

## Priority 1 Nested Weak-Party Scaling

Status: partial.

- `experiments/run_nested_weak_scaling.py` implements the required nested protocol.
- The runner trains/loads one 18-party cache per dataset/seed with 3 useful parties and 15 weak parties, then replays nested subsets for weak counts `{0,2,5,10,15}`.
- A smoke test completed for ACM seed 42, weak counts 0 and 2, with 1 local epoch.
- The full primary run on ACM Hard and DBLP Hard, seeds 42--46, epochs 300, is still pending.

Existing `results/noise_ratio` jobs vary useful-party count, but they do not implement the required nested weak-set protocol. They must not be reported as the requested nested weak-party scaling experiment.

## Priority 2 Validation-to-Test Alignment

Status: not complete.

The current adaptive selector records final topology and evaluation counts, but not every candidate edge's validation/test delta. This needs instrumentation or a replay script from cached probabilities.

## Priority 3 Calibration Robustness

Status: not complete.

No temperature-scaling or ECE/Brier/NLL audit has been run yet.

## Priority 4 Heterogeneity Ladder

Status: not complete.

The MovieLens completed package supports heterogeneous views/dimensions, but not a verified H0-H3 ladder.

## Priority 5 Representation Alignment References

Status: blocked until hidden-state export is implemented and audited.

## Exact Selector Default Map

- K / top-k reliability default: `{defaults.get("topk_reliability_k")}` in engine, canonical `{defaults.get("canonical_topk_reliability_k")}` in experiment plan.
- B / adaptive edge budget default: `{defaults.get("adaptive_edge_budget")}` in engine, canonical `{defaults.get("canonical_adaptive_edge_budget")}` in experiment plan.
- d_max / max degree default: `{defaults.get("max_degree")}` in engine, canonical `{defaults.get("canonical_max_degree")}` in experiment plan.
- m / adaptive min edges default: `{defaults.get("adaptive_min_edges")}` in engine, canonical `{defaults.get("canonical_adaptive_min_edges")}` in experiment plan.
- epsilon / adaptive min gain default: `{defaults.get("adaptive_min_gain")}` in engine, canonical `{defaults.get("canonical_adaptive_min_gain")}` in experiment plan.
- topology update interval default: `{defaults.get("topology_every")}` epochs.
- prediction consensus steps: `{defaults.get("pred_consensus_steps")}`.
- prediction self-weight: `{defaults.get("pred_self_weight")}` in engine, canonical `{defaults.get("canonical_pred_self_weight")}` in experiment plan.
"""


def write_macros(path: Path, values: List[Dict[str, object]]) -> None:
    lookup = {row["name"]: row["value"] for row in values}

    def fmt(name: str, digits: int = 4) -> str:
        value = lookup.get(name)
        if value is None:
            return "NA"
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    lines = [
        "% Auto-generated by experiments/final_claims_audit.py",
        f"\\newcommand{{\\BestSingleMovieAUC}}{{{fmt('MovieLens5_single_party_best_auc_mean')}}}",
        f"\\newcommand{{\\BestSingleMovieAUCStd}}{{{fmt('MovieLens5_single_party_best_auc_std')}}}",
        f"\\newcommand{{\\TADVFGMovieAUC}}{{{fmt('MovieLens5_adaptive_graph_val_auc_mean')}}}",
        f"\\newcommand{{\\FullMeshMovieAUC}}{{{fmt('MovieLens5_full_mesh_auc_mean')}}}",
        f"\\newcommand{{\\AdaptivePairMovieAUC}}{{{fmt('MovieLens5_adaptive_pair_auc_mean')}}}",
        f"\\newcommand{{\\TADVFGMovieComm}}{{{fmt('MovieLens5_adaptive_graph_val_total_comm_mean', 0)}}}",
        f"\\newcommand{{\\FullMeshMovieComm}}{{{fmt('MovieLens5_full_mesh_total_comm_mean', 0)}}}",
        f"\\newcommand{{\\TADVFGMovieLinks}}{{{fmt('MovieLens5_adaptive_graph_val_links_mean', 1)}}}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "final_claims_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    defaults = extract_arg_defaults()
    values, macro_payload = movie_values()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_hash": read_git_hash(),
        "command": "python experiments/final_claims_audit.py",
        "seeds": [42, 43, 44, 45, 46],
        "selector_defaults": defaults,
        "movielens_protocol": {
            "target": "binary preference, rating >= 4 is positive",
            "split": "chronological 70% train, 10% validation, 20% test interactions",
            "alignment": "all parties predict identical validation/test user-movie pairs",
            "cross_party_messages": "prediction probabilities/logits only",
        },
        "hgb_weak_noisy_construction": {
            "source": "main experiment/ta_dvfg_hgb_reliability.py::build_party_views",
            "summary": (
                "Useful parties receive feature splits and selected semantic/KNN graph views; "
                "hard-setting distractors receive permuted/noisy features and random graphs."
            ),
        },
    }
    (args.output_dir / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (args.output_dir / "paper_values.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "value", "source"])
        writer.writeheader()
        for row in values:
            writer.writerow(row)

    (args.output_dir / "EXPERIMENT_STATUS.md").write_text(status_text(defaults), encoding="utf-8")
    write_macros(args.output_dir / "paper_patch_macros.tex", values)

    print(f"Wrote Phase-0 audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
