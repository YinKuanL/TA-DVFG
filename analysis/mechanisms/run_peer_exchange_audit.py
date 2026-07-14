"""Stable entry point for the historical same-state peer-exchange audit."""

from __future__ import annotations

import runpy
from pathlib import Path


def _implementation() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parent / "_historical" / "task1_2_runner.py",
        here.parents[2] / "outputs" / "round2_closure" / "task1_2_runner.py",
    )
    for path in candidates:
        if path.is_file():
            return path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Peer-exchange audit implementation not found: {expected}")


if __name__ == "__main__":
    runpy.run_path(str(_implementation()), run_name="__main__")

