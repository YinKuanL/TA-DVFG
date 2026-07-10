"""Task 7 validation-reuse protocol rerun entry point."""

from __future__ import annotations

from closure_rerun_common import run_jobs


if __name__ == "__main__":
    raise SystemExit(
        run_jobs(
            "Task 7 rerun validation-reuse/disjoint-validation controls.",
            lambda adapter, out: adapter.build_validation_reuse_jobs(out),
        )
    )
