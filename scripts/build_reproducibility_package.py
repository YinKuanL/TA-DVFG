"""Build the anonymous minimal AAAI reproducibility package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PACKAGE_NAME = "TA-DVFG"
ZIP_NAME = "TA-DVFG_AAAI27_All_Experiments_Code.zip"
RECOVERY_COMMIT = "5cc081568eea994f2de3f5d4bd422b328587178f"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def copy_file(root: Path, stage: Path, source: str, destination: str | None = None) -> None:
    source_path = root / source
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    target = stage / (destination or source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)


def copy_glob(root: Path, stage: Path, source_dir: str, pattern: str, destination_dir: str) -> None:
    files = sorted((root / source_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matched {source_dir}/{pattern}")
    for source in files:
        if source.is_file():
            target = stage / destination_dir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def copy_git_blob(root: Path, stage: Path, revision_path: str, destination: str) -> None:
    """Recover a stable historical source file without altering the worktree."""
    content = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{RECOVERY_COMMIT}:{revision_path}"],
    )
    target = stage / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def payload_files(stage: Path) -> list[Path]:
    return sorted(
        path for path in stage.rglob("*")
        if path.is_file() and path.name not in {"MANIFEST.txt", "SHA256SUMS.txt"}
    )


def sanitize_local_metadata(stage: Path, root: Path) -> None:
    """Redact only identity/path strings; never transform numeric evidence."""
    text_suffixes = {".csv", ".json", ".md", ".py", ".txt"}
    root_forms = {
        str(root),
        str(root).replace("\\", "/"),
        str(root).replace("\\", "\\\\"),
    }
    path_pattern = re.compile(r"(?i)[a-z]:(?:[\\/]+)users(?:[\\/]+)[^,\r\n\"']+")
    for path in stage.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        original = path.read_text(encoding="utf-8", errors="replace")
        cleaned = original
        for form in sorted(root_forms, key=len, reverse=True):
            cleaned = cleaned.replace(form, ".")
        cleaned = path_pattern.sub("<REDACTED_LOCAL_PATH>", cleaned)
        cleaned = re.sub(r"(?i)onedrive", "<REDACTED_STORAGE>", cleaned)
        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8", newline="")


def make_package_local(stage: Path) -> None:
    """Rewrite repository-layout paths in copied code to package-local paths."""
    replacements = {
        'REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"':
            'REPO_ROOT / "core" / "ta_dvfg_hgb_reliability.py"',
        'ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"':
            'ROOT / "core" / "ta_dvfg_hgb_reliability.py"',
        'REPO_ROOT / "multi-modalities" / "ta_dvfg_hgb_multimodal.py"':
            'REPO_ROOT / "core" / "ta_dvfg_hgb_multimodal.py"',
        'REPO_ROOT / "main experiment" / "core"':
            'REPO_ROOT / "configs" / "headline"',
        '`main experiment/ta_dvfg_hgb_reliability.py`':
            '`core/ta_dvfg_hgb_reliability.py`',
        'main experiment/ta_dvfg_hgb_reliability.py':
            'core/ta_dvfg_hgb_reliability.py',
        'return REPO_ROOT / "results" / "core_cached_v2" / prefix / f"{prefix}_cached_5seeds_config.json"':
            'return REPO_ROOT / "configs" / "headline" / prefix / "metrics_config.json"',
        'return REPO_ROOT / "results" / "core_cached_v2" / prefix / f"{prefix}_cached_5seeds.csv"':
            'return REPO_ROOT / "artifacts" / "raw" / "hgb" / "seed_results.csv"',
    }
    for path in stage.rglob("*.py"):
        original = path.read_text(encoding="utf-8", errors="replace")
        rewritten = original
        for old, new in replacements.items():
            rewritten = rewritten.replace(old, new)
        if rewritten != original:
            path.write_text(rewritten, encoding="utf-8", newline="")


def write_integrity_files(stage: Path, distribution: Path) -> None:
    manifest = stage / "MANIFEST.txt"
    lines = ["sha256\tbytes\tpath"]
    for path in payload_files(stage):
        relative = path.relative_to(stage).as_posix()
        lines.append(f"{digest(path)}\t{path.stat().st_size}\t{relative}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sums = stage / "SHA256SUMS.txt"
    sum_paths = payload_files(stage) + [manifest]
    sum_lines = [f"{digest(path)}  {path.relative_to(stage).as_posix()}" for path in sorted(sum_paths)]
    sums.write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    shutil.copyfile(manifest, distribution / "MANIFEST.txt")
    shutil.copyfile(sums, distribution / "SHA256SUMS.txt")


def write_zip(stage: Path, destination: Path) -> None:
    fixed_time = (2026, 7, 14, 0, 0, 0)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(path for path in stage.rglob("*") if path.is_file()):
            relative = Path(PACKAGE_NAME) / source.relative_to(stage)
            info = zipfile.ZipInfo(relative.as_posix(), fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def write_headline_configs(root: Path, stage: Path) -> None:
    """Resolve full engine namespaces from the audited job plan without running jobs."""
    experiments_dir = root / "experiments"
    sys.path.insert(0, str(experiments_dir))
    try:
        from experiment_plan import build_jobs
        from run_experiments import parameter_args
    finally:
        sys.path.pop(0)

    engine_path = root / "main experiment" / "ta_dvfg_hgb_reliability.py"
    spec = importlib.util.spec_from_file_location("tadvfg_package_config_engine", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(engine_path)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)

    previous_argv = sys.argv
    try:
        for job in build_jobs(["core"]):
            sys.argv = [str(engine_path), *parameter_args(job.parameters)]
            resolved = vars(engine.parse_args())
            destination = stage / "configs" / "headline" / job.name / "metrics_config.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(resolved, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
    finally:
        sys.argv = previous_argv


def build(root: Path, distribution: Path) -> tuple[Path, Path]:
    stage = distribution / PACKAGE_NAME
    if distribution.exists():
        shutil.rmtree(distribution)
    stage.mkdir(parents=True)

    root_files = ["README.md", "REPRODUCE.md", "requirements.txt"]
    code_files = [
        "main experiment/ta_dvfg_hgb_reliability.py",
        "models/movielens_parties.py",
        "experiments/aggregate_results.py",
        "experiments/alignment_references.py",
        "experiments/config_audit.py",
        "experiments/deployment_objective_analysis.py",
        "experiments/experiment_plan.py",
        "experiments/final_analysis.py",
        "experiments/final_claims_audit.py",
        "experiments/make_ablation_plots.py",
        "experiments/make_readable_paper_figures.py",
        "experiments/merge_cached_baselines.py",
        "experiments/run_cached_ablations.py",
        "experiments/run_cached_core.py",
        "experiments/run_cached_minedges_sweep.py",
        "experiments/run_experiments.py",
        "experiments/run_k_sensitivity.py",
        "experiments/run_nested_weak_scaling.py",
        "experiments/statistical_tests.py",
        "experiments/strong_setting_analysis.py",
        "experiments/movielens/aggregate_15party.py",
        "experiments/movielens/extended_movielens_suite.py",
        "experiments/movielens/package_final_results.py",
        "experiments/movielens/plot_movielens_results.py",
        "experiments/movielens/prepare_movielens.py",
        "experiments/movielens/run_movielens_tadvfg.py",
        "analysis/mechanisms/run_peer_exchange_audit.py",
        "analysis/provenance/run_testatbestval_replay.py",
        "analysis/provenance/run_topology_provenance.py",
        "outputs/round2_closure/task1_2_runner.py",
        "outputs/round2_closure/step_c/step_c_gate.py",
        "outputs/round2_closure/step_c5/step_c5_topology_audit.py",
        "outputs/alignment_references_acmhard5/make_alignment_acmhard5_figures.py",
    ]
    remapped = {
        "outputs/round2_closure/task1_2_runner.py": "analysis/mechanisms/_historical/task1_2_runner.py",
        "outputs/round2_closure/step_c/step_c_gate.py": "analysis/provenance/_historical/step_c_gate.py",
        "outputs/round2_closure/step_c5/step_c5_topology_audit.py": "analysis/provenance/_historical/step_c5_topology_audit.py",
        "outputs/alignment_references_acmhard5/make_alignment_acmhard5_figures.py": "analysis/alignment/make_alignment_acmhard5_figures.py",
        "main experiment/ta_dvfg_hgb_reliability.py": "core/ta_dvfg_hgb_reliability.py",
    }
    support_files = [
        "scripts/build_reproducibility_package.py",
        "scripts/regenerate_artifacts.py",
        "scripts/verify_package.py",
        "scripts/verify_reported_values.py",
        "scripts/verify_supplement_coverage.py",
        "tests/unit/test_method_invariants.py",
        "tests/unit/test_movielens_leakage.py",
        "tests/smoke/test_cached_artifacts.py",
        "tests/smoke/test_package_local_paths.py",
        "tests/test_experiment_plan.py",
        "tests/test_reviewer_defense.py",
        "docs/PACKAGE_CONTENT_MAP.md",
        "environment/README.md",
        "environment/package_versions.txt",
        "environment/system_info.json",
        "metadata/CONFIG_AUDIT_SUMMARY.md",
        "metadata/headline_run_config_audit.csv",
        "metadata/headline_run_config_audit.json",
        "metadata/k_sensitivity_raw.csv",
        "metadata/k_sensitivity_summary.csv",
        "configs/alignment/acm_hard.json",
        "configs/experiment_areas.json",
        "configs/hgb/defaults.json",
        "configs/mechanisms/deployment_objectives.json",
        "configs/movielens/defaults.json",
        "configs/robustness/control_sensitivity.json",
        "configs/robustness/k_sensitivity.json",
        "configs/robustness/nested_weak_scaling.json",
    ]
    for source in root_files + code_files + support_files:
        copy_file(root, stage, source, remapped.get(source))

    copy_git_blob(
        root,
        stage,
        "experiments/round2_closure/task3_4_runner.py",
        "analysis/mechanisms/_historical/task3_4_runner.py",
    )

    write_headline_configs(root, stage)

    copy_file(root, stage, "results/final_analysis/data/all_raw_compact.csv", "artifacts/raw/hgb/seed_results.csv")
    copy_file(root, stage, "results/final_analysis/tables/paper_main_accuracy.csv", "artifacts/raw/hgb/paper_main_accuracy.csv")
    copy_file(root, stage, "results/final_analysis/tables/paper_communication_efficiency.csv", "artifacts/raw/hgb/paper_communication_efficiency.csv")
    copy_glob(root, stage, "results/final_analysis/tables", "*.csv", "artifacts/summaries/hgb/tables")
    copy_glob(root, stage, "results/final_analysis/statistics", "*.csv", "artifacts/summaries/hgb/statistics")

    movie_source = "outputs/movielens_final_package_leakage_safe_20260705/raw_results"
    copy_glob(root, stage, movie_source, "*.csv", "artifacts/raw/movielens")
    copy_glob(root, stage, movie_source, "*.json", "artifacts/raw/movielens")

    for task in ("task1", "task2", "task3", "task4"):
        copy_glob(root, stage, f"outputs/round2_closure/{task}", "*.csv", "artifacts/raw/mechanisms")
        copy_glob(root, stage, f"outputs/round2_closure/{task}", "*.md", "artifacts/summaries/mechanisms")

    for name in (
        "alignment_reference_communication.csv",
        "alignment_reference_metadata.json",
        "alignment_reference_per_seed.csv",
        "alignment_reference_selection_grid.csv",
        "alignment_reference_stats.csv",
        "alignment_reference_summary.csv",
        "cache_comparability_audit.csv",
        "gate_diagnostics.csv",
        "hidden_export_equivalence.csv",
        "hidden_scale_audit.csv",
    ):
        copy_file(root, stage, f"outputs/alignment_references/{name}", f"artifacts/raw/alignment/{name}")
    copy_glob(root, stage, "outputs/alignment_references_dblphard5", "*.csv", "artifacts/raw/alignment/dblp_hard")

    copy_file(root, stage, "metadata/k_sensitivity_raw.csv", "artifacts/raw/k_sensitivity/k_sensitivity_raw.csv")
    copy_file(root, stage, "metadata/k_sensitivity_summary.csv", "artifacts/summaries/k_sensitivity/k_sensitivity_summary.csv")

    make_package_local(stage)
    sanitize_local_metadata(stage, root)
    write_integrity_files(stage, distribution)
    subprocess.run([sys.executable, str(stage / "scripts" / "verify_package.py"), "--root", str(stage)], check=True)
    archive = distribution / ZIP_NAME
    write_zip(stage, archive)
    (distribution / "ZIP_SHA256.txt").write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    return stage, archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    distribution = (args.output or root / "dist" / "aaai27_all_experiments_code").resolve()
    stage, archive = build(root, distribution)
    print(f"Staged package: {stage}")
    print(f"ZIP: {archive}")
    print(f"ZIP SHA256: {digest(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
