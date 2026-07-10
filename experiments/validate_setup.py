"""Validate local dependencies, data, engines, and experiment expansion."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

from experiment_plan import available_suites, build_jobs


REPO_ROOT = Path(__file__).resolve().parents[1]


def check_import(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    engines = [
        REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py",
        REPO_ROOT / "multi-modalities" / "ta_dvfg_hgb_multimodal.py",
    ]
    data_root = REPO_ROOT / "data_hgb" / "hgb"
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
        "engines": {str(path.relative_to(REPO_ROOT)): path.exists() for path in engines},
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "datasets": {
            name: (data_root / name.lower()).exists()
            for name in ("ACM", "DBLP", "IMDB")
        },
        "imports": {
            name: check_import(name)
            for name in ("numpy", "scipy", "pandas", "matplotlib", "torch", "torch_geometric")
        },
        "suite_jobs": {
            suite: len(build_jobs([suite]))
            for suite in available_suites()
        },
    }
    try:
        import torch

        report["torch_cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_version"] = torch.version.cuda
        report["torch_device_count"] = torch.cuda.device_count()
    except Exception:
        pass

    print(json.dumps(report, indent=2, ensure_ascii=False))
    required_ok = (
        all(report["engines"].values())
        and report["data_root_exists"]
        and all(item["ok"] for item in report["imports"].values())
    )
    if not required_ok:
        print("\nSetup is incomplete. See the false/error entries above.")
        return 1
    print("\nSetup looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
