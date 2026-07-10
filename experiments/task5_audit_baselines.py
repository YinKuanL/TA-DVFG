"""Task 5: grep-based source audit for reviewer-requested baselines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from closure_core import write_csv


PATTERNS = {
    "Adaptive Pairwise": re.compile(r"adaptive[_ -]?pair", re.IGNORECASE),
    "Adaptive Complementarity": re.compile(r"adaptive[_ -]?complement", re.IGNORECASE),
    "Matched Topology": re.compile(r"matched[_ -]?topology|matched[_ -]?edge|fixed_ring_matched|random_matched|expander_matched", re.IGNORECASE),
    "Validation Split Reuse": re.compile(r"validation_split|topology_val|val_fraction|disjoint", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("outputs") / "round2_closure" / "task5_baseline_audit.md")
    parser.add_argument("--max-matches", type=int, default=12)
    return parser.parse_args()


def iter_source_files(root: Path):
    ignored = {".git", ".venv", ".venv_py314_broken", "__pycache__", "build", "tmp"}
    for path in root.rglob("*"):
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in {".py", ".ps1", ".sh", ".md", ".json", ".csv"}:
            yield path


def main() -> int:
    args = parse_args()
    rows = []
    for path in iter_source_files(args.root):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = path.relative_to(args.root)
        for line_no, line in enumerate(lines, start=1):
            for topic, pattern in PATTERNS.items():
                if pattern.search(line):
                    rows.append(
                        {
                            "topic": topic,
                            "file": str(rel).replace("\\", "/"),
                            "line": line_no,
                            "evidence": line.strip()[:220],
                        }
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("# Task 5 Baseline Audit\n\n")
        handle.write("This audit is grep-based evidence. It identifies likely implementation sites; it does not claim the baselines are fully rerun.\n\n")
        for topic in PATTERNS:
            topic_rows = [row for row in rows if row["topic"] == topic][: args.max_matches]
            handle.write(f"## {topic}\n\n")
            if not topic_rows:
                handle.write("- No direct source evidence found.\n\n")
                continue
            for row in topic_rows:
                handle.write(f"- `{row['file']}:{row['line']}`: {row['evidence']}\n")
            handle.write("\n")
    write_csv(rows, args.out.with_suffix(".csv"))
    print(f"Wrote audit report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
