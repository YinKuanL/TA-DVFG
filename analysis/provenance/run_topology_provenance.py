"""Stable entry point for the historical best-validation topology audit."""

from __future__ import annotations

import runpy
from pathlib import Path


def _implementation() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parent / "_historical" / "step_c5_topology_audit.py",
        here.parents[2]
        / "outputs"
        / "round2_closure"
        / "step_c5"
        / "step_c5_topology_audit.py",
    )
    for path in candidates:
        if path.is_file():
            return path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Topology provenance implementation not found: {expected}")


if __name__ == "__main__":
    runpy.run_path(str(_implementation()), run_name="__main__")

