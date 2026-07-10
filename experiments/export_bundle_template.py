"""Export TA-DVFG prediction caches into Round 2 NPZ bundles.

The HGB engine stores per-example local predictor probabilities in
``results/core_cache_v2/*_party_preds.pt``. This script converts those Torch
cache files to the shared Round 2 bundle format consumed by ``closure_core.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "core_cache_v2"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "round2_bundles"
DEFAULT_VENV_SITE_PACKAGES = REPO_ROOT / ".venv" / "Lib" / "site-packages"
HGB_SETTINGS = {
    "main": {"useful_parties": 6, "view_setting": "hard"},
    "hard_noisy": {"useful_parties": 3, "view_setting": "hard"},
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=["ACM", "DBLP"], default=["ACM", "DBLP"])
    parser.add_argument("--settings", nargs="+", choices=sorted(HGB_SETTINGS), default=["main", "hard_noisy"])
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--metric", choices=["accuracy", "roc_auc"], default="accuracy")
    parser.add_argument(
        "--prediction-snapshot",
        choices=["final", "epoch"],
        default="final",
        help="Use final probs_val/probs_test or one indexed probs_*_epochs snapshot.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="1-based epoch to export when --prediction-snapshot epoch is used; -1 means last epoch.",
    )
    parser.add_argument("--venv-site-packages", type=Path, default=DEFAULT_VENV_SITE_PACKAGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def import_torch(venv_site_packages: Path):
    if venv_site_packages.exists():
        sys.path.insert(0, str(venv_site_packages))
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required to read TA-DVFG .pt prediction caches. "
            f"Tried venv site-packages: {venv_site_packages}"
        ) from exc
    return torch


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def cache_path(cache_dir: Path, dataset: str, setting: str, seed: int) -> Path:
    spec = HGB_SETTINGS[setting]
    return cache_dir / (
        f"{dataset.lower()}_{spec['view_setting']}_useful{spec['useful_parties']}"
        f"_seed{seed}_party_preds.pt"
    )


def select_predictions(cache: dict[str, Any], snapshot: str, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    if snapshot == "final":
        return tensor_to_numpy(cache["probs_val"]), tensor_to_numpy(cache["probs_test"])
    val_epochs = cache["probs_val_epochs"]
    test_epochs = cache["probs_test_epochs"]
    n_epochs = int(val_epochs.shape[0])
    index = n_epochs - 1 if epoch == -1 else epoch - 1
    if index < 0 or index >= n_epochs:
        raise ValueError(f"Requested epoch {epoch} outside available range 1..{n_epochs}")
    return tensor_to_numpy(val_epochs[index]), tensor_to_numpy(test_epochs[index])


def validate_cache(cache: dict[str, Any], path: Path, dataset: str, setting: str, seed: int) -> None:
    required = ["y", "val_idx", "test_idx", "probs_val", "probs_test", "training_config"]
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"{path} is missing required cache keys: {missing}")
    if str(cache.get("dataset")) != dataset:
        raise ValueError(f"{path} dataset mismatch: cache={cache.get('dataset')!r}, expected={dataset!r}")
    if int(cache.get("seed")) != int(seed):
        raise ValueError(f"{path} seed mismatch: cache={cache.get('seed')!r}, expected={seed!r}")
    config = cache.get("training_config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} has no training_config dictionary")
    spec = HGB_SETTINGS[setting]
    expected = {
        "dataset": dataset,
        "view_setting": spec["view_setting"],
        "useful_parties": spec["useful_parties"],
        "num_parties": 15,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"{path} training_config mismatch: {mismatches}")


def export_one(
    torch: Any,
    source: Path,
    output: Path,
    dataset: str,
    setting: str,
    seed: int,
    metric: str,
    snapshot: str,
    epoch: int,
) -> dict[str, Any]:
    cache = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError(f"{source} is not a dictionary cache")
    validate_cache(cache, source, dataset, setting, seed)
    val_probs, test_probs = select_predictions(cache, snapshot, epoch)
    y = cache["y"]
    y_val = tensor_to_numpy(y[cache["val_idx"]]).astype(int)
    y_test = tensor_to_numpy(y[cache["test_idx"]]).astype(int)
    if val_probs.shape[1] != y_val.shape[0] or test_probs.shape[1] != y_test.shape[0]:
        raise ValueError(
            f"{source} prediction/label shape mismatch: "
            f"val={val_probs.shape}, y_val={y_val.shape}, test={test_probs.shape}, y_test={y_test.shape}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        val_probs=val_probs,
        test_probs=test_probs,
        y_val=y_val,
        y_test=y_test,
        dataset=dataset,
        setting=setting,
        seed=seed,
        metric=metric,
        source_cache=str(source.resolve()),
        cache_fingerprint=str(cache.get("cache_fingerprint", "")),
        prediction_snapshot=snapshot,
        epoch=epoch,
    )
    return {
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "source_cache": str(source),
        "output": str(output),
        "val_probs_shape": list(val_probs.shape),
        "test_probs_shape": list(test_probs.shape),
        "y_val_shape": list(y_val.shape),
        "y_test_shape": list(y_test.shape),
        "snapshot": snapshot,
        "epoch": epoch,
    }


def main() -> int:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    torch = import_torch(args.venv_site_packages)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for setting in args.settings:
            for seed in seeds:
                source = cache_path(args.cache_dir, dataset, setting, seed)
                output = args.out_dir / f"{dataset.lower()}_{setting}_seed{seed}.npz"
                if not source.exists():
                    raise FileNotFoundError(f"Missing prediction cache: {source}")
                if args.dry_run:
                    rows.append(
                        {
                            "dataset": dataset,
                            "setting": setting,
                            "seed": seed,
                            "source_cache": str(source),
                            "output": str(output),
                        }
                    )
                    continue
                row = export_one(
                    torch,
                    source,
                    output,
                    dataset,
                    setting,
                    seed,
                    args.metric,
                    args.prediction_snapshot,
                    args.epoch,
                )
                rows.append(row)
                print(
                    f"Wrote {output} from {source} "
                    f"val={tuple(row['val_probs_shape'])} test={tuple(row['test_probs_shape'])}"
                )
    manifest = args.out_dir / "manifest.json"
    if not args.dry_run:
        manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote manifest to {manifest}")
    else:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
