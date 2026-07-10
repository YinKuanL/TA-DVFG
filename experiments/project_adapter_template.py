"""Adapter contract for plugging repo-specific reruns into Round 2 closure tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RerunJob:
    name: str
    command: Sequence[str]
    output: Path


class ProjectAdapter:
    """Minimal integration surface for rerunning audited baselines.

    Fill these methods from the canonical project engine once Task 5 confirms the
    exact source path and CLI arguments. The closure scripts deliberately avoid
    guessing baseline semantics from cached predictions when the reviewer asked
    for a source-level rerun.
    """

    def build_baseline_jobs(self, output_dir: Path) -> list[RerunJob]:
        raise NotImplementedError("Connect audited Adaptive Pairwise/Complementarity baseline commands here.")

    def build_strict_matched_control_jobs(self, output_dir: Path) -> list[RerunJob]:
        raise NotImplementedError("Connect audited matched-topology control commands here.")

    def build_validation_reuse_jobs(self, output_dir: Path) -> list[RerunJob]:
        raise NotImplementedError("Connect audited disjoint/reused validation protocol commands here.")
