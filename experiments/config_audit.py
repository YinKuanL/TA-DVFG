"""Audit actual headline HGB run configurations.

This script resolves configuration values from saved run artifacts, not source
defaults. It fails loudly if any headline adaptive/TA-DVFG run mismatches the
paper configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = {
    "m": 5,
    "tau": 0.001,
    "B": 15,
    "d_max": 2,
    "K": 1,
    "self_weight": 0.85,
}
HEADLINE = [
    ("ACM", "main", REPO_ROOT / "main experiment" / "core" / "acm_main"),
    ("ACM", "hard_noisy", REPO_ROOT / "main experiment" / "core" / "acm_hard_noisy"),
    ("DBLP", "main", REPO_ROOT / "main experiment" / "core" / "dblp_main"),
    ("DBLP", "hard_noisy", REPO_ROOT / "main experiment" / "core" / "dblp_hard_noisy"),
    ("IMDB", "main", REPO_ROOT / "main experiment" / "core" / "imdb_main"),
    ("IMDB", "hard_noisy", REPO_ROOT / "main experiment" / "core" / "imdb_hard_noisy"),
]
AUDITED_METHODS = {"adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def resolved_values(config: Dict[str, object]) -> Dict[str, object]:
    return {
        "m": config.get("adaptive_min_edges"),
        "tau": config.get("adaptive_min_gain"),
        "B": config.get("adaptive_edge_budget"),
        "d_max": config.get("max_degree"),
        "K": config.get("pred_consensus_steps"),
        "self_weight": config.get("pred_self_weight"),
    }


def matches_paper(values: Dict[str, object]) -> bool:
    for key, expected in PAPER.items():
        actual = values.get(key)
        if actual is None:
            return False
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                return False
        elif int(actual) != expected:
            return False
    return True


def audit_run(dataset: str, setting: str, run_dir: Path) -> List[Dict[str, object]]:
    config_path = run_dir / "metrics_config.json"
    metrics_path = run_dir / "metrics.csv"
    command_path = run_dir / "command.txt"
    job_path = run_dir / "job.json"
    rows: List[Dict[str, object]] = []
    if not config_path.exists() or not metrics_path.exists():
        for seed in [42, 43, 44, 45, 46]:
            rows.append({
                "dataset": dataset,
                "setting": setting,
                "seed": seed,
                "method": "adaptive_graph_val",
                "result_file": str(metrics_path),
                "run_id": str(run_dir),
                "launch_source": "missing",
                "m": "UNRESOLVED",
                "tau": "UNRESOLVED",
                "B": "UNRESOLVED",
                "d_max": "UNRESOLVED",
                "K": "UNRESOLVED",
                "self_weight": "UNRESOLVED",
                "paper_match": False,
                "audit_status": "UNRESOLVED",
                "evidence_path": str(config_path),
                "notes": "Missing metrics_config.json or metrics.csv",
            })
        return rows

    config = read_json(config_path)
    metrics = pd.read_csv(metrics_path)
    values = resolved_values(config)
    ok = matches_paper(values)
    launch_source = str(config_path)
    if command_path.exists():
        launch_source += f";{command_path}"
    if job_path.exists():
        launch_source += f";{job_path}"

    methods = [m for m in sorted(metrics["method"].unique()) if m in AUDITED_METHODS]
    for seed in sorted(metrics["seed"].unique()):
        for method in methods:
            rows.append({
                "dataset": dataset,
                "setting": setting,
                "seed": int(seed),
                "method": method,
                "result_file": str(metrics_path),
                "run_id": str(run_dir.relative_to(REPO_ROOT)),
                "launch_source": launch_source,
                "m": values["m"],
                "tau": values["tau"],
                "B": values["B"],
                "d_max": values["d_max"],
                "K": values["K"],
                "self_weight": values["self_weight"],
                "paper_match": bool(ok),
                "audit_status": "PASS" if ok else "MISMATCH",
                "evidence_path": str(config_path),
                "notes": "Resolved from saved argparse config; command.txt cross-check available."
            })
    return rows


def write_summary(out_dir: Path, rows: List[Dict[str, object]]) -> bool:
    total = len(rows)
    matching = sum(1 for r in rows if r["audit_status"] == "PASS")
    mismatch = sum(1 for r in rows if r["audit_status"] == "MISMATCH")
    unresolved = sum(1 for r in rows if r["audit_status"] == "UNRESOLVED")
    passed = mismatch == 0 and unresolved == 0 and total > 0
    status = "PASS" if passed else "FAIL"
    lines = [
        "# Headline HGB Run Configuration Audit",
        "",
        f"Status: **{status}**",
        "",
        f"- Headline adaptive/TA-DVFG run rows audited: {total}",
        f"- Matching paper values: {matching}",
        f"- Mismatching: {mismatch}",
        f"- Unresolved: {unresolved}",
        "",
        "## Audited Paper Values",
        "",
        f"- `m` / adaptive minimum edges: {PAPER['m']}",
        f"- `tau` / adaptive minimum gain: {PAPER['tau']}",
        f"- `B` / adaptive edge budget: {PAPER['B']}",
        f"- `d_max` / maximum degree: {PAPER['d_max']}",
        f"- `K` / consensus steps: {PAPER['K']}",
        f"- prediction self-weight: {PAPER['self_weight']}",
        "",
        "## Evidence Priority Used",
        "",
        "Resolved values are taken from saved `metrics_config.json` files under `main experiment/core/<setting>/`.",
        "`command.txt` and `job.json` are recorded as cross-check launch evidence when present.",
        "",
    ]
    if not passed:
        lines += [
            "## Blocking Failures",
            "",
        ]
        for row in rows:
            if row["audit_status"] != "PASS":
                lines.append(
                    f"- {row['dataset']} {row['setting']} seed {row['seed']} {row['method']}: "
                    f"status={row['audit_status']} m={row['m']} tau={row['tau']} B={row['B']} "
                    f"d_max={row['d_max']} K={row['K']} self_weight={row['self_weight']}"
                )
    (out_dir / "CONFIG_AUDIT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def write_macros(out_dir: Path) -> None:
    text = "\n".join([
        "% Auto-generated only after headline config audit PASS",
        "\\newcommand{\\AuditedMinEdges}{5}",
        "\\newcommand{\\AuditedMinGain}{0.001}",
        "\\newcommand{\\AuditedEdgeBudget}{15}",
        "\\newcommand{\\AuditedMaxDegree}{2}",
        "\\newcommand{\\AuditedConsensusSteps}{1}",
        "\\newcommand{\\AuditedSelfWeight}{0.85}",
        "",
    ])
    (out_dir / "paper_config_macros.tex").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "config_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for dataset, setting, run_dir in HEADLINE:
        rows.extend(audit_run(dataset, setting, run_dir))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "headline_run_config_audit.csv", index=False)
    (args.output_dir / "headline_run_config_audit.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    passed = write_summary(args.output_dir, rows)
    macro_path = args.output_dir / "paper_config_macros.tex"
    if passed:
        write_macros(args.output_dir)
        print(f"CONFIG AUDIT PASS: wrote {args.output_dir}")
        return 0
    if macro_path.exists():
        macro_path.unlink()
    print(f"CONFIG AUDIT FAIL: see {args.output_dir / 'CONFIG_AUDIT_SUMMARY.md'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

