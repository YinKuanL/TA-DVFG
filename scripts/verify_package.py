"""Verify structure, anonymity, and checksums of the minimal code package."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path, PurePosixPath


TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".txt", ".toml", ".yaml", ".yml"
}
FORBIDDEN_PARTS = {
    ".git", ".github", ".idea", ".pytest_cache", ".venv", "__pycache__",
    "cache", "caches", "checkpoint", "checkpoints", "data_hgb", "dataset",
    "datasets", "logs", "notebooks", "paper", "source",
}
FORBIDDEN_SUFFIXES = {
    ".bib", ".ckpt", ".ipynb", ".log", ".pdf", ".pt", ".pth", ".tex"
}
REQUIRED = {
    "README.md",
    "REPRODUCE.md",
    "MANIFEST.txt",
    "SHA256SUMS.txt",
    "requirements.txt",
    "configs/experiment_areas.json",
    "docs/PACKAGE_CONTENT_MAP.md",
    "core/ta_dvfg_hgb_reliability.py",
    "scripts/build_reproducibility_package.py",
    "scripts/regenerate_artifacts.py",
    "scripts/verify_package.py",
    "scripts/verify_reported_values.py",
    "experiments/run_cached_core.py",
    "experiments/run_experiments.py",
    "experiments/deployment_objective_analysis.py",
    "experiments/run_cached_minedges_sweep.py",
    "experiments/run_k_sensitivity.py",
    "experiments/run_nested_weak_scaling.py",
    "experiments/alignment_references.py",
    "experiments/movielens/run_movielens_tadvfg.py",
    "analysis/mechanisms/_historical/task1_2_runner.py",
    "analysis/mechanisms/_historical/task3_4_runner.py",
    "analysis/provenance/_historical/step_c_gate.py",
    "analysis/provenance/_historical/step_c5_topology_audit.py",
    "analysis/alignment/make_alignment_acmhard5_figures.py",
    "artifacts/raw/hgb/seed_results.csv",
    "artifacts/raw/hgb/paper_main_accuracy.csv",
    "artifacts/raw/hgb/paper_communication_efficiency.csv",
    "artifacts/raw/movielens/movielens_tadvfg_per_seed.csv",
    "artifacts/raw/movielens/movielens_tadvfg_summary.csv",
    "artifacts/raw/movielens/movielens_15party_per_seed.csv",
    "artifacts/raw/movielens/movielens_15party_summary.csv",
    "artifacts/raw/mechanisms/task2_per_seed.csv",
    "artifacts/raw/mechanisms/task2_paired_statistics.csv",
    "metadata/k_sensitivity_raw.csv",
    "metadata/k_sensitivity_summary.csv",
    "tests/unit/test_method_invariants.py",
    "tests/unit/test_movielens_leakage.py",
    "tests/smoke/test_cached_artifacts.py",
    "tests/smoke/test_package_local_paths.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, tuple[int, str]]:
    entries: dict[str, tuple[int, str]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "sha256\tbytes\tpath":
        raise ValueError("MANIFEST.txt has an invalid header")
    for line in lines[1:]:
        digest, size, relative = line.split("\t", 2)
        entries[relative] = (int(size), digest)
    return entries


def parse_sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    return entries


def allowed_unmanifested(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        "__pycache__" in path.parts
        or ".pytest_cache" in path.parts
        or relative.startswith("artifacts/regenerated/")
        or path.suffix == ".pyc"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    try:
        manifest = parse_manifest(root / "MANIFEST.txt")
        sums = parse_sums(root / "SHA256SUMS.txt")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    listed = set(manifest) | {"MANIFEST.txt", "SHA256SUMS.txt"}
    missing_required = sorted(REQUIRED - listed)
    if missing_required:
        failures.append("required paths missing: " + ", ".join(missing_required))

    for relative, (expected_size, expected_hash) in sorted(manifest.items()):
        path = root / PurePosixPath(relative)
        if not path.is_file():
            failures.append(f"manifest file missing: {relative}")
            continue
        if path.stat().st_size != expected_size:
            failures.append(f"size mismatch: {relative}")
        if sha256(path) != expected_hash:
            failures.append(f"manifest hash mismatch: {relative}")

    for relative, expected_hash in sorted(sums.items()):
        path = root / PurePosixPath(relative)
        if not path.is_file() or sha256(path) != expected_hash:
            failures.append(f"SHA256SUMS mismatch: {relative}")
    if set(sums) != set(manifest) | {"MANIFEST.txt"}:
        failures.append("SHA256SUMS paths do not match manifest plus MANIFEST.txt")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    extras = sorted(path for path in actual - listed if not allowed_unmanifested(path))
    if extras:
        failures.append("unmanifested files: " + ", ".join(extras))

    identity_patterns = {
        "Windows user path": re.compile(r"(?i)(?:[a-z]:[\\/]+users[\\/]|onedrive[\\/])"),
        "email address": re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
        "GitHub repository URL": re.compile(r"(?i)https?://(?:www\.)?github\.com/[^\s/)]+/[^\s/)]+"),
        "local identity": re.compile("(?i)" + "yin" + r"[ _-]?" + "kuan" + "|" + "yin" + "kuanl"),
    }
    for relative in sorted(manifest):
        pure = PurePosixPath(relative)
        lower_parts = {part.lower() for part in pure.parts}
        if lower_parts & FORBIDDEN_PARTS or pure.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden packaged path: {relative}")
        path = root / pure
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 5_000_000:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in identity_patterns.items():
                if pattern.search(text):
                    failures.append(f"{label} found in {relative}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        print(f"Package verification: 0 passed, {len(failures)} failed")
        return 1
    print(f"[PASS] {len(manifest)} manifested payload files verified")
    print("[PASS] required structure, exclusions, anonymity, and checksums verified")
    print("Package verification: 2 passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
