"""Task 5 rerun entry point for audited baselines."""

from __future__ import annotations

from closure_rerun_common import run_jobs


if __name__ == "__main__":
    raise SystemExit(
        run_jobs(
            "Task 5 rerun audited Adaptive Pairwise/Complementarity baselines.",
            lambda adapter, out: adapter.build_baseline_jobs(out),
        )
    )
