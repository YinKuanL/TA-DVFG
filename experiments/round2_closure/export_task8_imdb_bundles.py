from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = REPO_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import torch

CACHE_DIR = REPO_ROOT / "results" / "core_cache_v2"
OUT_DIR = REPO_ROOT / "data" / "round2_bundles"
SEEDS = [42, 43, 44, 45, 46]
SETTINGS = {
    "main": 6,
    "hard_noisy": 3,
}


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def export_one(setting: str, useful: int, seed: int) -> dict[str, Any]:
    source = CACHE_DIR / f"imdb_hard_useful{useful}_seed{seed}_party_preds.pt"
    cache = torch.load(source, map_location="cpu", weights_only=False)
    if str(cache.get("dataset")) != "IMDB":
        raise ValueError(f"{source} dataset mismatch: {cache.get('dataset')!r}")
    if int(cache.get("seed")) != seed:
        raise ValueError(f"{source} seed mismatch: {cache.get('seed')!r}")
    config = cache.get("training_config", {})
    if int(config.get("useful_parties")) != useful or config.get("view_setting") != "hard":
        raise ValueError(f"{source} training_config mismatch: useful/view={config.get('useful_parties')}/{config.get('view_setting')}")
    val_probs = tensor_to_numpy(cache["probs_val"])
    test_probs = tensor_to_numpy(cache["probs_test"])
    y = cache["y"]
    y_val = tensor_to_numpy(y[cache["val_idx"]]).astype(int)
    y_test = tensor_to_numpy(y[cache["test_idx"]]).astype(int)
    output = OUT_DIR / f"imdb_{setting}_seed{seed}.npz"
    np.savez(
        output,
        val_probs=val_probs,
        test_probs=test_probs,
        y_val=y_val,
        y_test=y_test,
        dataset="IMDB",
        setting=setting,
        seed=seed,
        metric="accuracy",
        source_cache=str(source.resolve()),
        cache_fingerprint=str(cache.get("cache_fingerprint", "")),
        prediction_snapshot="final",
        epoch=-1,
    )
    return {
        "dataset": "IMDB",
        "setting": setting,
        "seed": seed,
        "source_cache": str(source),
        "output": str(output),
        "val_probs_shape": list(val_probs.shape),
        "test_probs_shape": list(test_probs.shape),
        "y_val_shape": list(y_val.shape),
        "y_test_shape": list(y_test.shape),
        "snapshot": "final",
        "epoch": -1,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting, useful in SETTINGS.items():
        for seed in SEEDS:
            row = export_one(setting, useful, seed)
            rows.append(row)
            print(f"Wrote {row['output']} from {row['source_cache']}")
    manifest_path = OUT_DIR / "manifest_task8_imdb.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
