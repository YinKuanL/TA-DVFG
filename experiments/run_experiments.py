"""Run or print the complete TA-DVFG experiment plan.

Examples:
    python experiments/run_experiments.py core --dry-run
    python experiments/run_experiments.py core --device cuda --epochs 300
    python experiments/run_experiments.py edge_budget noise_ratio --resume
    python experiments/run_experiments.py all --datasets ACM DBLP
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from experiment_plan import Job, available_suites, build_jobs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
ENGINES = {
    "standard": REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py",
    "multimodal": REPO_ROOT / "multi-modalities" / "ta_dvfg_hgb_multimodal.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible runner for the experiment matrix in deep-research-report.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "suites",
        nargs="*",
        default=["core"],
        help=f"Suites to run: {', '.join(available_suites())}, or all.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Override cache_dir for cache-enabled jobs.",
    )
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="For cache-enabled jobs, fail if a compatible cache is absent instead of training.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, help="Override epochs for every selected job.")
    parser.add_argument("--seeds", help="Override comma-separated seeds for every selected job.")
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--match", help="Only include job IDs containing this substring.")
    parser.add_argument("--max-jobs", type=int, help="Run only the first N filtered jobs.")
    parser.add_argument("--list", action="store_true", help="List selected jobs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Write/print commands without executing.")
    parser.add_argument("--force", action="store_true", help="Rerun jobs even if _SUCCESS exists.")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip jobs that already have a _SUCCESS marker.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test override: one seed, two epochs, CPU unless --device is explicit.",
    )
    return parser.parse_args()


def normalize_jobs(args: argparse.Namespace) -> List[Job]:
    jobs = build_jobs(args.suites)
    if args.datasets:
        allowed = set(args.datasets)
        jobs = [job for job in jobs if job.dataset in allowed]
    if args.match:
        jobs = [job for job in jobs if args.match.lower() in job.job_id.lower()]

    normalized: List[Job] = []
    for job in jobs:
        params = dict(job.parameters)
        params["device"] = args.device
        cache_enabled = bool(
            params.get("cache_predictions")
            or params.get("reuse_prediction_cache")
            or params.get("skip_local_training_if_cache_exists")
        )
        if args.cache_dir is not None and cache_enabled:
            cache_dir = args.cache_dir if args.cache_dir.is_absolute() else REPO_ROOT / args.cache_dir
            params["cache_dir"] = str(cache_dir.resolve())
        if args.strict_cache and cache_enabled:
            params["reuse_prediction_cache"] = True
            params["skip_local_training_if_cache_exists"] = True
        if args.epochs is not None:
            if (
                cache_enabled
                and args.cache_dir is None
                and args.epochs != int(params.get("epochs", args.epochs))
            ):
                root = args.results_root if args.results_root.is_absolute() else REPO_ROOT / args.results_root
                params["cache_dir"] = str((root / "cache_overrides" / f"epochs{args.epochs}").resolve())
            params["epochs"] = args.epochs
        if args.seeds:
            params["seeds"] = args.seeds
        if args.quick:
            params["epochs"] = 2 if args.epochs is None else args.epochs
            params["seeds"] = args.seeds or "42"
            if job.suite != "deployment_objective":
                params["methods"] = "topk_reliability_vote,fixed_ring_matched,random_matched,expander_matched,adaptive_graph_val"
            params["topology_every"] = 1
            if (
                params.get("cache_predictions")
                or params.get("reuse_prediction_cache")
                or params.get("skip_local_training_if_cache_exists")
            ):
                smoke_cache = (
                    args.results_root / "smoke_cache"
                    if args.results_root.is_absolute()
                    else REPO_ROOT / args.results_root / "smoke_cache"
                )
                params["cache_dir"] = str(smoke_cache.resolve())
                params["cache_predictions"] = True
                params["reuse_prediction_cache"] = True
                params["skip_local_training_if_cache_exists"] = False
            if args.device == "auto":
                params["device"] = "cpu"
            job = replace(job, suite="smoke", name=f"{job.suite}__{job.name}")
        normalized.append(replace(job, parameters=params, make_plots=job.make_plots and not args.no_plots))

    if args.max_jobs is not None:
        normalized = normalized[: max(args.max_jobs, 0)]
    return normalized


def printable_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def parameter_args(parameters: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in parameters.items():
        if value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            args.append(flag)
        elif isinstance(value, (list, tuple)):
            args.extend([flag, ",".join(str(v) for v in value)])
        else:
            args.extend([flag, str(value)])
    return args


def job_paths(job: Job, results_root: Path) -> Dict[str, Path]:
    out_dir = results_root / job.suite / job.name
    return {
        "dir": out_dir,
        "csv": out_dir / job.output_filename,
        "figures": out_dir / "figures",
        "log": out_dir / "run.log",
        "manifest": out_dir / "job.json",
        "command": out_dir / "command.txt",
        "success": out_dir / "_SUCCESS",
        "failed": out_dir / "_FAILED",
    }


def build_command(job: Job, paths: Dict[str, Path], python: Path) -> List[str]:
    engine = ENGINES[job.engine]
    params = dict(job.parameters)
    params["data_dir"] = str(REPO_ROOT / "data_hgb")
    params["save_csv"] = str(paths["csv"])
    if job.make_plots:
        params["plot"] = True
        params["plot_dir"] = str(paths["figures"])
    return [str(python), str(engine), *parameter_args(params)]


def write_job_metadata(job: Job, paths: Dict[str, Path], command: Sequence[str]) -> None:
    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = asdict(job)
    payload["job_id"] = job.job_id
    payload["repo_root"] = str(REPO_ROOT)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["command"] = list(command)
    paths["manifest"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["command"].write_text(printable_command(command) + "\n", encoding="utf-8")


def run_streaming(command: Sequence[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def run_job(job: Job, args: argparse.Namespace, index: int, total: int) -> bool:
    paths = job_paths(job, args.results_root.resolve())
    command = build_command(job, paths, args.python.resolve())

    print(f"\n[{index}/{total}] {job.job_id}")
    print(printable_command(command))

    if args.dry_run:
        return True
    if paths["success"].exists() and args.resume and not args.force:
        print("  skipped: _SUCCESS already exists")
        return True
    write_job_metadata(job, paths, command)

    paths["success"].unlink(missing_ok=True)
    paths["failed"].unlink(missing_ok=True)
    start = time.time()
    return_code = run_streaming(command, paths["log"])
    elapsed = time.time() - start
    stamp = {
        "job_id": job.job_id,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker = paths["success"] if return_code == 0 else paths["failed"]
    marker.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"  {'completed' if return_code == 0 else 'FAILED'} in {elapsed / 60:.1f} min")
    return return_code == 0


def main() -> int:
    args = parse_args()
    args.results_root = args.results_root if args.results_root.is_absolute() else REPO_ROOT / args.results_root
    if not args.python.is_absolute():
        discovered = shutil.which(str(args.python))
        args.python = Path(discovered) if discovered else REPO_ROOT / args.python
    jobs = normalize_jobs(args)

    if not jobs:
        print("No jobs matched the requested filters.")
        return 0
    for engine_name in sorted({job.engine for job in jobs}):
        engine = ENGINES[engine_name]
        if not engine.exists():
            raise FileNotFoundError(f"Missing {engine_name} engine: {engine}")
    if not args.python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    print(f"Selected {len(jobs)} job(s) under {args.results_root}")
    if args.list:
        for index, job in enumerate(jobs, 1):
            print(f"{index:03d}  {job.job_id:<60} dataset={job.dataset:<4} setting={job.setting}")
        return 0

    failures: List[str] = []
    for index, job in enumerate(jobs, 1):
        ok = run_job(job, args, index, len(jobs))
        if not ok:
            failures.append(job.job_id)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailed jobs:")
        for job_id in failures:
            print(f"  - {job_id}")
        return 1
    print("\nAll selected jobs completed or were skipped successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
