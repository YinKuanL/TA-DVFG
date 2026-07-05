"""Build shared local predictions once, then replay standard ablation sweeps."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from experiment_plan import Job, build_jobs
from run_experiments import parameter_args, printable_command


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
RUNNER = REPO_ROOT / "experiments" / "run_experiments.py"
DEFAULT_SUITES = ["topology_objective", "edge_budget", "consensus"]
ALLOWED_SUITES = set(DEFAULT_SUITES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Build the four ACM/DBLP main+hard local-prediction caches, then "
            "run topology objective, edge-budget, and consensus as strict cache replays."
        ),
    )
    parser.add_argument("--stage", choices=["all", "build-cache", "sweep"], default="all")
    parser.add_argument("--suites", nargs="+", choices=DEFAULT_SUITES, default=DEFAULT_SUITES)
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP"], default=["ACM", "DBLP"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Defaults to core_cache_v2 for 300 epochs and an isolated directory otherwise.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Defaults to results/ for the formal 300-epoch five-seed run and an isolated smoke root otherwise.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Rerun completed sweep jobs; valid caches remain reusable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_python(path: Path) -> Path:
    if path.is_absolute():
        return path
    discovered = shutil.which(str(path))
    return Path(discovered) if discovered else (REPO_ROOT / path).resolve()


def selected_jobs(args: argparse.Namespace) -> List[Job]:
    requested = list(dict.fromkeys(args.suites))
    unknown = set(requested) - ALLOWED_SUITES
    if unknown:
        raise ValueError(f"Unsupported cached ablation suites: {sorted(unknown)}")
    allowed_datasets = set(args.datasets)
    return [job for job in build_jobs(requested) if job.dataset in allowed_datasets]


def cache_identity(job: Job, epochs: int) -> Tuple[object, ...]:
    params = job.parameters
    return (
        job.dataset,
        params["graph_views"],
        params["num_parties"],
        params["useful_parties"],
        params["view_setting"],
        params.get("shuffle_party_positions", False),
        params.get("party_shuffle_seed_offset", 0),
        epochs,
    )


def unique_cache_jobs(jobs: Sequence[Job], epochs: int) -> List[Job]:
    unique: Dict[Tuple[object, ...], Job] = {}
    for job in jobs:
        unique.setdefault(cache_identity(job, epochs), job)
    return list(unique.values())


def build_cache_command(args: argparse.Namespace, job: Job) -> List[str]:
    params = dict(job.parameters)
    params.update(
        {
            "device": args.device,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "methods": "adaptive_graph_val",
            "data_dir": str((REPO_ROOT / "data_hgb").resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "cache_predictions": True,
            "reuse_prediction_cache": True,
            "cache_only": True,
        }
    )
    params.pop("skip_local_training_if_cache_exists", None)
    return [str(args.python), str(ENGINE), *parameter_args(params)]


def run_command(command: Sequence[str], dry_run: bool) -> int:
    print("\n" + printable_command(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(list(command), cwd=REPO_ROOT, check=False).returncode


def build_caches(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    cache_jobs = unique_cache_jobs(jobs, args.epochs)
    print(
        f"Preparing {len(cache_jobs)} shared local-training configuration(s) "
        f"under {args.cache_dir}",
        flush=True,
    )
    for job in cache_jobs:
        return_code = run_command(build_cache_command(args, job), args.dry_run)
        if return_code != 0:
            raise SystemExit(
                f"Prediction-cache preparation failed for {job.dataset}/{job.setting} "
                f"with exit code {return_code}."
            )


def failed_sweep_jobs(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    failures: List[Tuple[str, Path, str]] = []
    for suite in args.suites:
        suite_dir = args.results_root / suite
        for marker in sorted(suite_dir.glob("*/_FAILED")):
            log_path = marker.parent / "run.log"
            last_line = ""
            if log_path.exists():
                lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
                last_line = next((line for line in reversed(lines) if line), "")
            failures.append((f"{suite}/{marker.parent.name}", log_path, last_line))
    return failures


def run_sweeps(args: argparse.Namespace) -> int:
    command = [
        str(args.python),
        str(RUNNER),
        *args.suites,
        "--device",
        args.device,
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--datasets",
        *args.datasets,
        "--cache-dir",
        str(args.cache_dir.resolve()),
        "--results-root",
        str(args.results_root.resolve()),
        "--strict-cache",
        "--continue-on-error",
    ]
    if args.no_plots:
        command.append("--no-plots")
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return_code = run_command(command, False)
    if return_code != 0:
        print("\nCached ablation sweep completed with failed jobs:", flush=True)
        for job_id, log_path, last_line in failed_sweep_jobs(args):
            print(f"  - {job_id}", flush=True)
            if last_line:
                print(f"    {last_line}", flush=True)
            print(f"    log: {log_path}", flush=True)
    return return_code


def main() -> int:
    args = parse_args()
    args.python = resolve_python(args.python)
    seed_values = [value.strip() for value in args.seeds.split(",") if value.strip()]
    formal_run = args.epochs == 300 and seed_values == ["42", "43", "44", "45", "46"]
    if args.cache_dir is None:
        args.cache_dir = (
            REPO_ROOT / "results" / "core_cache_v2"
            if args.epochs == 300
            else REPO_ROOT / "results" / "cache_overrides" / f"epochs{args.epochs}"
        )
    if args.results_root is None:
        seed_tag = "-".join(seed_values) if seed_values else "none"
        args.results_root = (
            REPO_ROOT / "results"
            if formal_run
            else REPO_ROOT / "results" / "smoke_cached_ablations" / f"epochs{args.epochs}_seeds{seed_tag}"
        )
    args.cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
    args.results_root = args.results_root if args.results_root.is_absolute() else REPO_ROOT / args.results_root
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    jobs = selected_jobs(args)
    if args.stage in {"all", "build-cache"}:
        build_caches(args, jobs)
    if args.stage in {"all", "sweep"}:
        return run_sweeps(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
