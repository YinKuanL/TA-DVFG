"""Validate selected paper headline values against existing raw/summary outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


HGB_EXPECTED = {
    "local_reliability_vote": [(86.06, 2.29), (75.98, 4.32), (90.96, 0.52), (88.08, 1.44), (28.51, 0.76), (28.93, 1.57)],
    "topk_reliability_vote": [(89.82, 1.01), (85.50, 2.78), (91.43, 0.65), (90.57, 1.04), (39.27, 0.87), (30.44, 0.92)],
    "full_mesh": [(85.96, 2.30), (75.52, 5.04), (90.86, 0.49), (88.01, 1.45), (28.49, 0.77), (29.03, 1.33)],
    "adaptive_graph_val": [(89.52, 1.71), (85.54, 2.91), (91.50, 1.37), (90.25, 1.14), (38.65, 0.86), (30.08, 0.97)],
}


class Audit:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, actual: float, expected: float, tolerance: float = 5e-5) -> None:
        if math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
            self.passed += 1
            print(f"[PASS] {name}: {actual}")
        else:
            message = f"{name}: expected {expected}, got {actual}"
            self.failed.append(message)
            print(f"[FAIL] {message}")

    def contains(self, name: str, text: str, expected: str) -> None:
        if expected in text:
            self.passed += 1
            print(f"[PASS] {name}: paper claim present")
        else:
            message = f"{name}: expected paper text not found: {expected}"
            self.failed.append(message)
            print(f"[FAIL] {message}")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def locate(root: Path, *alternatives: str) -> Path:
    for relative in alternatives:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError("required evidence missing; tried: " + ", ".join(alternatives))


def pair(cell: str) -> tuple[float, float]:
    values = re.findall(r"\d+(?:\.\d+)?", cell)
    if len(values) < 2:
        raise ValueError(f"cannot parse mean/std cell: {cell!r}")
    return float(values[0]), float(values[1])


def verify_hgb(root: Path, audit: Audit) -> None:
    path = locate(root, "results/final_analysis/tables/paper_main_accuracy.csv", "artifacts/raw/hgb/paper_main_accuracy.csv")
    by_method = {row["method"]: row for row in rows(path)}
    columns = ["ACM Main", "ACM Hard", "DBLP Main", "DBLP Hard", "IMDB Main", "IMDB Hard"]
    for method, expected_cells in HGB_EXPECTED.items():
        for column, expected in zip(columns, expected_cells):
            actual = pair(by_method[method][column])
            audit.check(f"HGB {method} {column} mean", actual[0], expected[0], 0.005)
            audit.check(f"HGB {method} {column} SD", actual[1], expected[1], 0.005)


def verify_k(root: Path, audit: Audit) -> None:
    data = rows(root / "metadata" / "k_sensitivity_summary.csv")
    by_key = {(row["dataset"], row["method"], int(row["K"])): row for row in data}
    audit.check("ACM Hard TA-DVFG K=1", 100 * float(by_key[("ACM", "adaptive_graph_val", 1)]["test_mean"]), 85.54, 0.005)
    audit.check("DBLP Hard TA-DVFG K=1", 100 * float(by_key[("DBLP", "adaptive_graph_val", 1)]["test_mean"]), 90.25, 0.005)


def verify_movielens(root: Path, audit: Audit) -> None:
    five = locate(
        root,
        "outputs/movielens_final_package_leakage_safe_20260705/raw_results/movielens_tadvfg_summary.csv",
        "artifacts/raw/movielens/movielens_tadvfg_summary.csv",
    )
    five_by = {row["method"]: row for row in rows(five)}
    expected = {
        "single_party_best": (0.7402, 0.0),
        "topk_reliability_vote": (0.7487, 800168.0),
        "adaptive_pair": (0.7510, 5201092.0),
        "full_mesh": (0.7497, 10002100.0),
        "adaptive_graph_val": (0.7516, 3200672.0),
    }
    for method, (auc, comm) in expected.items():
        audit.check(f"MovieLens five-party {method} AUC", float(five_by[method]["auc_mean"]), auc, 5e-5)
        audit.check(f"MovieLens five-party {method} communication", float(five_by[method]["total_comm_mean"]), comm, 0.5)

    fifteen = locate(
        root,
        "outputs/movielens_final_package_leakage_safe_20260705/raw_results/movielens_15party_summary.csv",
        "artifacts/raw/movielens/movielens_15party_summary.csv",
    )
    by_key = {(row["setting"], row["method"]): row for row in rows(fifteen)}
    for setting, auc, links in (("main", 0.7516, 5.2), ("hard", 0.7498, 5.0)):
        row = by_key[(setting, "adaptive_graph_val")]
        audit.check(f"MovieLens 15-party {setting} AUC", float(row["auc_mean"]), auc, 5e-5)
        audit.check(f"MovieLens 15-party {setting} links", float(row["selected_links_mean"]), links, 0.05)


def verify_mechanism(root: Path, audit: Audit) -> None:
    path = locate(
        root,
        "outputs/round2_closure/task2/task2_paired_statistics.csv",
        "artifacts/raw/mechanisms/task2_paired_statistics.csv",
    )
    data = rows(path)
    keyed = {(row["dataset"], row["comparison"], row["metric"]): row for row in data}
    for dataset, delta, p_value in (("ACM", 0.0185392642, 0.0022435361), ("DBLP", 0.0267158077, 0.0000667660)):
        row = keyed[(dataset, "A_vs_C_same_state_exchange", "local_mean_test")]
        audit.check(f"{dataset} same-state local-mean delta", float(row["paired_mean_diff_a_minus_b"]), delta, 1e-9)
        audit.check(f"{dataset} same-state paired t p", float(row["paired_t_p"]), p_value, 1e-9)


def verify_communication(root: Path, audit: Audit) -> None:
    path = locate(
        root,
        "results/final_analysis/tables/paper_communication_efficiency.csv",
        "artifacts/raw/hgb/paper_communication_efficiency.csv",
    )
    data = rows(path)
    row = next(r for r in data if r["dataset"] == "ACM" and r["setting"] == "main" and r["method"] == "adaptive_graph_val")
    audit.check("ACM Main TA-DVFG peer communication", float(row["peer_to_peer_comm_mean"]), 90750.0, 0.01)
    audit.check("ACM Main TA-DVFG readout", float(row["global_readout_comm_mean"]), 61710.0, 0.01)
    audit.check("ACM Main TA-DVFG total", float(row["total_comm_mean"]), 152460.0, 0.01)


def group_mean(data: list[dict[str, str]], **filters: object) -> float:
    selected = [
        float(row["test_at_best_val"])
        for row in data
        if all(str(row.get(key)) == str(value) for key, value in filters.items())
    ]
    if not selected:
        raise ValueError(f"no rows for filters: {filters}")
    return statistics.mean(selected)


def verify_nested_weak_scaling(root: Path, audit: Audit) -> None:
    raw = rows(root / "artifacts/raw/nested_weak_scaling/raw_weak_scaling.csv")
    summary = rows(root / "artifacts/summaries/nested_weak_scaling/weak_scaling_summary.csv")
    required_methods = {
        "adaptive_graph_val",
        "full_mesh",
        "adaptive_pair",
        "adaptive_complementarity",
    }
    expected_seeds = {"42", "43", "44", "45", "46"}
    for dataset in ("ACM", "DBLP"):
        for weak_count in (0, 2, 5, 10, 15):
            for method in sorted(required_methods):
                seeds = {
                    row["seed"] for row in raw
                    if row["dataset"] == dataset
                    and int(row["weak_count"]) == weak_count
                    and row["method"] == method
                }
                if seeds == expected_seeds:
                    audit.passed += 1
                    print(f"[PASS] nested coverage {dataset} weak={weak_count} {method}: seeds 42--46")
                else:
                    audit.failed.append(
                        f"nested coverage {dataset} weak={weak_count} {method}: {sorted(seeds)}"
                    )

    summary_by = {
        (row["dataset"], int(row["weak_count"]), row["method"]): row
        for row in summary
    }
    for dataset, expected_gap in (("ACM", 12.59), ("DBLP", 5.16)):
        tadvfg = group_mean(raw, dataset=dataset, weak_count=15, method="adaptive_graph_val")
        full_mesh = group_mean(raw, dataset=dataset, weak_count=15, method="full_mesh")
        audit.check(f"nested {dataset} final gap points", 100 * (tadvfg - full_mesh), expected_gap, 0.005)
        audit.check(
            f"nested {dataset} TA-DVFG summary aggregation",
            float(summary_by[(dataset, 15, "adaptive_graph_val")]["test_at_best_val_mean"]),
            tadvfg,
            1e-12,
        )
        audit.check(
            f"nested {dataset} TA-DVFG links",
            float(summary_by[(dataset, 15, "adaptive_graph_val")]["selected_links_mean"]),
            5.0,
            0.25,
        )
        audit.check(
            f"nested {dataset} Full Mesh links at 18 parties",
            float(summary_by[(dataset, 15, "full_mesh")]["selected_links_mean"]),
            153.0,
            1e-12,
        )


def verify_alignment_acm_hard(root: Path, audit: Audit) -> None:
    raw_root = root / "artifacts/raw/alignment_acm_hard"
    summary_root = root / "artifacts/summaries/alignment_acm_hard"
    per_seed = rows(raw_root / "alignment_acmhard5_per_seed.csv")
    summary = rows(summary_root / "alignment_acmhard5_summary.csv")
    stats_rows = rows(summary_root / "alignment_acmhard5_stats.csv")
    hidden = rows(raw_root / "hidden_export_equivalence.csv")
    communication = rows(summary_root / "alignment_acmhard5_communication.csv")

    required_methods = {
        "Best Single",
        "Global Top-k Reliability",
        "Full Mesh prediction consensus",
        "TA-DVFG",
        "Project-and-Mean",
        "Reliability-Selected Project-and-Mean",
        "Useful-Only best latent fusion",
    }
    expected_seeds = {"42", "43", "44", "45", "46"}
    for method in sorted(required_methods):
        seeds = {row["seed"] for row in per_seed if row["method"] == method}
        if seeds == expected_seeds:
            audit.passed += 1
            print(f"[PASS] ACM Hard alignment {method}: seeds 42--46")
        else:
            audit.failed.append(f"ACM Hard alignment {method}: seeds {sorted(seeds)}")

    by_method = {row["method"]: row for row in summary}
    audit.check("alignment TA-DVFG mean percent", 100 * float(by_method["TA-DVFG"]["test_at_best_val_mean"]), 85.60, 0.005)
    audit.check("alignment TA-DVFG SD percent", 100 * float(by_method["TA-DVFG"]["test_at_best_val_std"]), 1.16, 0.005)
    audit.check(
        "alignment Reliability-Selected P&M mean percent",
        100 * float(by_method["Reliability-Selected Project-and-Mean"]["test_at_best_val_mean"]),
        77.10,
        0.005,
    )
    audit.check(
        "alignment Reliability-Selected P&M SD percent",
        100 * float(by_method["Reliability-Selected Project-and-Mean"]["test_at_best_val_std"]),
        0.93,
        0.005,
    )
    audit.check(
        "alignment Useful-Only oracle mean percent",
        100 * float(by_method["Useful-Only best latent fusion"]["test_at_best_val_mean"]),
        88.93,
        0.005,
    )
    audit.check(
        "alignment Useful-Only oracle SD percent",
        100 * float(by_method["Useful-Only best latent fusion"]["test_at_best_val_std"]),
        0.94,
        0.005,
    )

    rel_stats = next(row for row in stats_rows if row["method"] == "Reliability-Selected Project-and-Mean")
    audit.check("alignment TA-DVFG minus Reliability-Selected P&M points", -100 * float(rel_stats["paired_mean_delta_vs_tadvfg"]), 8.50, 0.005)
    audit.check("alignment paired CI low points", -100 * float(rel_stats["ci95_high"]), 6.54, 0.005)
    audit.check("alignment paired CI high points", -100 * float(rel_stats["ci95_low"]), 10.47, 0.005)
    audit.check("alignment paired t p", float(rel_stats["paired_ttest_p"]), 2.76e-4, 5e-7)
    audit.check("alignment TA-DVFG wins", float(rel_stats["losses_vs_tadvfg"]), 5.0, 0.0)
    audit.check("alignment TA-DVFG ties", float(rel_stats["ties_vs_tadvfg"]), 0.0, 0.0)
    audit.check("alignment TA-DVFG losses", float(rel_stats["wins_vs_tadvfg"]), 0.0, 0.0)

    audit.check("hidden-export check count", float(len(hidden)), 75.0, 0.0)
    audit.check("hidden-export passing count", float(sum(row["status"] == "PASS" for row in hidden)), 75.0, 0.0)
    audit.check(
        "hidden-export maximum difference",
        max(
            float(row[column])
            for row in hidden
            for column in ("max_logit_diff", "max_probability_diff", "val_metric_diff", "test_metric_diff")
        ),
        0.0,
        0.0,
    )
    communication_methods = {row["method"] for row in communication}
    if required_methods <= communication_methods:
        audit.passed += 1
        print("[PASS] alignment communication/parameter rows cover all requested methods")
    else:
        audit.failed.append(
            "alignment communication rows missing: " + ", ".join(sorted(required_methods - communication_methods))
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit()
    try:
        verify_hgb(root, audit)
        verify_k(root, audit)
        verify_movielens(root, audit)
        verify_mechanism(root, audit)
        verify_communication(root, audit)
        closure_root = root / "artifacts" / "raw" / "nested_weak_scaling"
        if closure_root.is_dir():
            verify_nested_weak_scaling(root, audit)
            verify_alignment_acm_hard(root, audit)
        elif (root / "artifacts" / "raw" / "hgb").is_dir():
            raise FileNotFoundError("package closure evidence is missing")
        main_text = (root / "paper/source/main.tex").read_text(encoding="utf-8", errors="replace") if (root / "paper/source/main.tex").is_file() else ""
        supplement = (root / "paper/source/supplementary.tex").read_text(encoding="utf-8", errors="replace") if (root / "paper/source/supplementary.tex").is_file() else ""
        if main_text:
            audit.contains("main HGB headline", main_text, "85.54")
            audit.contains("main MovieLens headline", main_text, "0.7516")
        if supplement:
            audit.contains("supplement default configuration", supplement, "m=5")
            audit.contains("supplement same-state p-value", supplement, "0.002244")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        audit.failed.append(str(exc))
        print(f"[FAIL] {exc}")

    print(f"Reported-value checks: {audit.passed} passed, {len(audit.failed)} failed")
    if audit.failed:
        for failure in audit.failed:
            print(" - " + failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
