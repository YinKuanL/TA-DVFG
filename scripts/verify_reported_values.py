"""Validate selected paper headline values against existing raw/summary outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
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

