"""Fail clearly when code or evidence claimed by the supplement is absent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Check:
    name: str
    paths: tuple[str, ...] = ()
    text_path: str | None = None
    tokens: tuple[str, ...] = ()
    any_path: bool = False


CHECKS = (
    Check("MovieLens runner", ("experiments/movielens/run_movielens_tadvfg.py",)),
    Check("MovieLens party models", ("models/movielens_parties.py",)),
    Check("K sensitivity", ("experiments/run_k_sensitivity.py",)),
    Check("nested weak scaling", ("experiments/run_nested_weak_scaling.py",)),
    Check(
        "topology selection implementation",
        ("main experiment/ta_dvfg_hgb_reliability.py",),
        "main experiment/ta_dvfg_hgb_reliability.py",
        ("def update_adaptive_topology", "adaptive_min_edges", "adaptive_min_gain"),
    ),
    Check(
        "consensus implementation",
        ("main experiment/ta_dvfg_hgb_reliability.py",),
        "main experiment/ta_dvfg_hgb_reliability.py",
        ("def post_consensus_party_probs", "pred_self_weight"),
    ),
    Check(
        "active/local/joint objectives",
        ("main experiment/ta_dvfg_hgb_reliability.py", "configs/mechanisms/deployment_objectives.json"),
        "main experiment/ta_dvfg_hgb_reliability.py",
        ("def topology_objective_components", '"local_mean"', '"joint"'),
    ),
    Check(
        "label-location implementation",
        ("main experiment/ta_dvfg_hgb_reliability.py",),
        "main experiment/ta_dvfg_hgb_reliability.py",
        ("active_party_logit_gradient", "logit_grad"),
    ),
    Check(
        "latent-reference implementation",
        ("experiments/alignment_references.py", "configs/alignment/acm_hard.json"),
        "experiments/alignment_references.py",
        ("hidden_export_equivalence", "class ProjectMean"),
    ),
    Check(
        "scalability analysis",
        ("experiments/experiment_plan.py", "experiments/final_analysis.py"),
        "experiments/experiment_plan.py",
        ("def party_scalability_jobs",),
    ),
    Check(
        "label-budget analysis",
        ("experiments/experiment_plan.py", "experiments/final_analysis.py"),
        "experiments/experiment_plan.py",
        ("def validation_label_budget_jobs",),
    ),
    Check(
        "topology-frequency analysis",
        ("experiments/experiment_plan.py", "experiments/final_analysis.py"),
        "experiments/experiment_plan.py",
        ("def topology_frequency_jobs",),
    ),
    Check(
        "m=0 audit",
        ("experiments/run_cached_minedges_sweep.py", "configs/robustness/control_sensitivity.json"),
        "configs/robustness/control_sensitivity.json",
        ('"minimum_link_audit"', "0"),
    ),
    Check(
        "edge-budget/tau sweep",
        ("experiments/experiment_plan.py", "experiments/make_readable_paper_figures.py"),
        "experiments/experiment_plan.py",
        ("def edge_budget_jobs",),
    ),
    Check("statistical analysis", ("experiments/statistical_tests.py", "experiments/final_analysis.py")),
    Check(
        "mechanism replay entry points",
        (
            "analysis/mechanisms/run_peer_exchange_audit.py",
            "analysis/provenance/run_testatbestval_replay.py",
            "analysis/provenance/run_topology_provenance.py",
        ),
    ),
    Check(
        "raw outputs",
        (
            "metadata/headline_run_config_audit.csv",
            "metadata/k_sensitivity_raw.csv",
        ),
    ),
    Check(
        "HGB reported-value table",
        (
            "results/final_analysis/tables/paper_main_accuracy.csv",
            "artifacts/raw/hgb/paper_main_accuracy.csv",
        ),
        any_path=True,
    ),
    Check(
        "MovieLens raw seed output",
        (
            "outputs/movielens_final_package_leakage_safe_20260705/raw_results/movielens_tadvfg_per_seed.csv",
            "artifacts/raw/movielens/movielens_tadvfg_per_seed.csv",
        ),
        any_path=True,
    ),
    Check(
        "figure/table scripts",
        (
            "experiments/final_analysis.py",
            "experiments/make_readable_paper_figures.py",
            "experiments/movielens/extended_movielens_suite.py",
        ),
    ),
    Check("environment snapshot", ("environment/system_info.json", "environment/package_versions.txt")),
    Check("reviewer documentation", ("README.md", "REPRODUCE.md", "docs/SUPPLEMENT_TO_CODE_MAP.md", "docs/RESULT_PROVENANCE.md")),
    Check("lightweight tests", ("tests/unit/test_method_invariants.py", "tests/unit/test_movielens_leakage.py", "tests/smoke/test_cached_artifacts.py")),
)


def evaluate(root: Path, check: Check) -> tuple[bool, str]:
    present = [str(path) for path in check.paths if (root / path).is_file()]
    if check.paths:
        path_ok = bool(present) if check.any_path else len(present) == len(check.paths)
        if not path_ok:
            missing = [path for path in check.paths if not (root / path).is_file()]
            return False, "missing path(s): " + ", ".join(missing)
    if check.text_path:
        path = root / check.text_path
        if not path.is_file():
            return False, f"missing text source: {check.text_path}"
        text = path.read_text(encoding="utf-8", errors="replace")
        missing_tokens = [token for token in check.tokens if token not in text]
        if missing_tokens:
            return False, "missing implementation token(s): " + ", ".join(missing_tokens)
    return True, "present"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    rows = []
    for check in CHECKS:
        ok, detail = evaluate(root, check)
        rows.append({"name": check.name, "status": "PASS" if ok else "MISSING", "detail": detail})
        print(f"[{rows[-1]['status']}] {check.name}: {detail}")

    passed = sum(row["status"] == "PASS" for row in rows)
    coverage = 100.0 * passed / len(rows)
    report = {"root": str(root), "passed": passed, "total": len(rows), "coverage_percent": coverage, "checks": rows}
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Coverage: {passed}/{len(rows)} ({coverage:.1f}%)")
    missing = [row["name"] for row in rows if row["status"] != "PASS"]
    if missing:
        print("Missing required items: " + ", ".join(missing))
        return 1
    print("Missing required items: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
