from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


movie = load_module("movielens_leakage_test", ROOT / "experiments" / "movielens" / "run_movielens_tadvfg.py")
engine = movie.load_engine()


def test_preparation_fits_rating_and_timestamp_statistics_on_train_only() -> None:
    ratings, users, movies = movie.synthetic_smoke_data(42, n_users=20, n_movies=30, n_rows=300)
    train_idx, val_idx, test_idx = movie.chronological_split(ratings)
    original = movie.build_features(ratings, users, movies, train_idx)

    changed = ratings.copy()
    held_out = np.concatenate([val_idx, test_idx])
    changed.loc[held_out, "rating"] = np.where(changed.loc[held_out, "rating"] >= 4, 1, 5)
    changed.loc[held_out, "timestamp"] += 9_000_000_000
    altered = movie.build_features(changed, users, movies, train_idx)

    original_train = original["ratings"].loc[train_idx]
    altered_train = altered["ratings"].loc[train_idx]
    assert np.allclose(original["pair_stats"](original_train), altered["pair_stats"](altered_train))


def test_topology_selection_api_accepts_validation_labels_not_test_labels() -> None:
    signature = inspect.signature(engine.score_topology_on_validation)
    assert "y_val" in signature.parameters
    assert all("test" not in name.lower() for name in signature.parameters)
