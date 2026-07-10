"""Task 6 strict matched-topology control rerun entry point."""

from __future__ import annotations

from closure_rerun_common import run_jobs


if __name__ == "__main__":
    raise SystemExit(
        run_jobs(
            "Task 6 rerun strict matched-topology controls.",
            lambda adapter, out: adapter.build_strict_matched_control_jobs(out),
        )
    )
