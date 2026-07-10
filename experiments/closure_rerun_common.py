"""Common subprocess runner for adapter-backed closure rerun scripts."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from typing import Callable

from project_adapter_template import ProjectAdapter, RerunJob


def parse_adapter_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--adapter", type=Path, default=Path("experiments") / "project_adapter_template.py")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "round2_closure")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_adapter(path: Path) -> ProjectAdapter:
    spec = importlib.util.spec_from_file_location("round2_project_adapter", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter_cls = getattr(module, "ProjectAdapter", None)
    if adapter_cls is None:
        raise SystemExit(f"{path} does not define ProjectAdapter")
    return adapter_cls()


def run_jobs(description: str, builder: Callable[[ProjectAdapter, Path], list[RerunJob]]) -> int:
    args = parse_adapter_args(description)
    adapter = load_adapter(args.adapter)
    try:
        jobs = builder(adapter, args.out)
    except NotImplementedError as exc:
        raise SystemExit(f"{exc} Run task5_audit_baselines.py first, then implement {args.adapter}.") from exc
    if not jobs:
        raise SystemExit("Adapter returned no jobs.")
    for job in jobs:
        print(f"{job.name}: {subprocess.list2cmdline(list(job.command))}")
        if not args.dry_run:
            subprocess.run(job.command, check=True)
    return 0
