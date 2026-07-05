"""MovieLens cached-prediction experiment for TA-DVFG.

This runner adds a real-world recommender experiment without changing the
core TA-DVFG topology or consensus implementation. Local parties train on
private MovieLens views and export only aligned 2-class probabilities.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.movielens_parties import (  # noqa: E402
    GenreKNNParty,
    MFParty,
    MovieMetadataParty,
    PopularityTemporalParty,
    UserProfileParty,
    make_genre_knn_features,
    party_specs,
    predict_proba,
    train_classifier,
)


ENGINE_PATH = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"


def load_engine():
    spec = importlib.util.spec_from_file_location("tadvfg_hgb_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TA-DVFG engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = load_engine()


GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_seeds(text: str) -> List[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def read_movielens_1m(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings_path = data_dir / "ratings.dat"
    users_path = data_dir / "users.dat"
    movies_path = data_dir / "movies.dat"
    if not ratings_path.exists() or not users_path.exists() or not movies_path.exists():
        missing = [str(p) for p in (ratings_path, users_path, movies_path) if not p.exists()]
        raise FileNotFoundError("Missing MovieLens 1M files: " + ", ".join(missing))
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    users = pd.read_csv(
        users_path,
        sep="::",
        names=["user_id", "gender", "age", "occupation", "zip"],
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    return ratings, users, movies


def synthetic_smoke_data(seed: int, n_users: int = 80, n_movies: int = 100, n_rows: int = 2200):
    rng = np.random.default_rng(seed)
    user_ids = np.arange(1, n_users + 1)
    movie_ids = np.arange(1, n_movies + 1)
    users = pd.DataFrame({
        "user_id": user_ids,
        "gender": rng.choice(["M", "F"], size=n_users),
        "age": rng.choice([1, 18, 25, 35, 45, 50, 56], size=n_users),
        "occupation": rng.integers(0, 21, size=n_users),
        "zip": "00000",
    })
    movie_genres = []
    titles = []
    for movie_id in movie_ids:
        chosen = rng.choice(GENRES, size=int(rng.integers(1, 4)), replace=False)
        movie_genres.append("|".join(chosen))
        titles.append(f"Smoke Movie {movie_id} ({int(rng.integers(1980, 2001))})")
    movies = pd.DataFrame({"movie_id": movie_ids, "title": titles, "genres": movie_genres})
    user_bias = rng.normal(0, 0.8, size=n_users)
    movie_bias = rng.normal(0, 0.8, size=n_movies)
    rows = []
    for t in range(n_rows):
        u = int(rng.integers(0, n_users))
        m = int(rng.integers(0, n_movies))
        score = 3.2 + user_bias[u] + movie_bias[m] + rng.normal(0, 0.9)
        rating = int(np.clip(round(score), 1, 5))
        rows.append((u + 1, m + 1, rating, 1_000_000_000 + t))
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return ratings, users, movies


def chronological_split(ratings: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = ratings.sort_values(["timestamp", "user_id", "movie_id"]).index.to_numpy()
    n = len(ordered)
    train_end = int(math.floor(0.70 * n))
    val_end = int(math.floor(0.80 * n))
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def movie_year(title: str) -> float:
    match = re.search(r"\((\d{4})\)", str(title))
    return float(match.group(1)) if match else 1995.0


def build_features(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    train_idx: np.ndarray,
) -> Dict[str, object]:
    user_ids = sorted(ratings["user_id"].unique().tolist())
    movie_ids = sorted(ratings["movie_id"].unique().tolist())
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_map = {mid: i for i, mid in enumerate(movie_ids)}
    ratings = ratings.copy()
    ratings["u"] = ratings["user_id"].map(user_map).astype(int)
    ratings["m"] = ratings["movie_id"].map(movie_map).astype(int)
    labels = (ratings["rating"].to_numpy() >= 4).astype(np.int64)

    users_by_id = users.set_index("user_id").reindex(user_ids)
    age = users_by_id["age"].fillna(users_by_id["age"].median()).astype(float).to_numpy()
    age = (age - age.mean()) / max(age.std(), 1e-6)
    gender = (users_by_id["gender"].fillna("M").to_numpy() == "M").astype(np.float32)[:, None]
    occ = users_by_id["occupation"].fillna(0).astype(int).to_numpy()
    occ_oh = np.zeros((len(user_ids), max(21, int(occ.max()) + 1)), dtype=np.float32)
    occ_oh[np.arange(len(user_ids)), np.clip(occ, 0, occ_oh.shape[1] - 1)] = 1.0
    user_features = np.concatenate([age[:, None].astype(np.float32), gender, occ_oh], axis=1)

    movies_by_id = movies.set_index("movie_id").reindex(movie_ids)
    genre_oh = np.zeros((len(movie_ids), len(GENRES)), dtype=np.float32)
    genre_map = {g: i for i, g in enumerate(GENRES)}
    for row, genres in enumerate(movies_by_id["genres"].fillna("").astype(str)):
        for genre in genres.split("|"):
            if genre in genre_map:
                genre_oh[row, genre_map[genre]] = 1.0
    years = movies_by_id["title"].map(movie_year).to_numpy(dtype=np.float32)
    years = ((years - years.mean()) / max(years.std(), 1e-6))[:, None]
    movie_features = np.concatenate([genre_oh, years], axis=1).astype(np.float32)
    knn_movie_features = make_genre_knn_features(movie_features, k=min(20, max(1, len(movie_ids) - 1)))

    train = ratings.loc[train_idx]
    global_like = float((train["rating"] >= 4).mean())
    movie_count = train.groupby("m").size().reindex(range(len(movie_ids)), fill_value=0).to_numpy(dtype=np.float32)
    user_count = train.groupby("u").size().reindex(range(len(user_ids)), fill_value=0).to_numpy(dtype=np.float32)
    movie_mean = train.groupby("m")["rating"].mean().reindex(range(len(movie_ids)), fill_value=3.0).to_numpy(dtype=np.float32)
    user_mean = train.groupby("u")["rating"].mean().reindex(range(len(user_ids)), fill_value=3.0).to_numpy(dtype=np.float32)
    movie_like = train.assign(like=train["rating"] >= 4).groupby("m")["like"].mean().reindex(
        range(len(movie_ids)), fill_value=global_like
    ).to_numpy(dtype=np.float32)
    user_like = train.assign(like=train["rating"] >= 4).groupby("u")["like"].mean().reindex(
        range(len(user_ids)), fill_value=global_like
    ).to_numpy(dtype=np.float32)
    train_ts_mean = float(train["timestamp"].mean())
    train_ts_std = max(float(train["timestamp"].std()), 1e-6)

    def pair_stats(frame: pd.DataFrame) -> np.ndarray:
        u = frame["u"].to_numpy(dtype=np.int64)
        m = frame["m"].to_numpy(dtype=np.int64)
        timestamp = frame["timestamp"].to_numpy(dtype=np.float32)
        ts = (timestamp - train_ts_mean) / train_ts_std
        return np.stack([
            np.log1p(user_count[u]) / max(np.log1p(user_count).max(), 1.0),
            np.log1p(movie_count[m]) / max(np.log1p(movie_count).max(), 1.0),
            (user_mean[u] - 3.0) / 2.0,
            (movie_mean[m] - 3.0) / 2.0,
            user_like[u],
            movie_like[m],
            ts,
        ], axis=1).astype(np.float32)

    return {
        "ratings": ratings,
        "labels": labels,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_features": user_features.astype(np.float32),
        "movie_features": movie_features,
        "knn_movie_features": knn_movie_features.astype(np.float32),
        "pair_stats": pair_stats,
    }


def tensor_batches(features: Dict[str, object], frame: pd.DataFrame, device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        "users": torch.tensor(frame["u"].to_numpy(), dtype=torch.long, device=device),
        "movies": torch.tensor(frame["m"].to_numpy(), dtype=torch.long, device=device),
        "user_features": torch.tensor(features["user_features"], dtype=torch.float32, device=device),
        "movie_features": torch.tensor(features["movie_features"], dtype=torch.float32, device=device),
        "knn_movie_features": torch.tensor(features["knn_movie_features"], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(features["pair_stats"](frame), dtype=torch.float32, device=device),
    }


def train_five_parties(args: argparse.Namespace, seed: int, features: Dict[str, object], train_idx, val_idx, test_idx):
    ratings = features["ratings"]
    train = ratings.loc[train_idx]
    val = ratings.loc[val_idx]
    test = ratings.loc[test_idx]
    y_train = torch.tensor(features["labels"][train_idx], dtype=torch.long)
    train_batches = tensor_batches(features, train)
    val_batches = tensor_batches(features, val)
    test_batches = tensor_batches(features, test)
    num_users = len(features["user_ids"])
    num_movies = len(features["movie_ids"])
    models = [
        MFParty(num_users, num_movies, 64),
        UserProfileParty(num_movies, features["user_features"].shape[1], 32),
        MovieMetadataParty(num_users, features["movie_features"].shape[1], 128),
        GenreKNNParty(num_users, features["knn_movie_features"].shape[1], 48),
        PopularityTemporalParty(train_batches["pair_features"].shape[1], 16),
    ]
    val_probs, test_probs = [], []
    for party_id, model in enumerate(models):
        train_classifier(
            model,
            train_batches,
            y_train,
            epochs=args.local_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=seed + 100 * party_id,
        )
        val_probs.append(predict_proba(model, val_batches))
        test_probs.append(predict_proba(model, test_batches))
    return torch.stack(val_probs, dim=0), torch.stack(test_probs, dim=0)


def expand_to_15_parties(
    base_val: torch.Tensor,
    base_test: torch.Tensor,
    labels_val: np.ndarray,
    seed: int,
    setting: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    rng = np.random.default_rng(seed + (17 if setting == "main" else 29))
    useful_count = 6 if setting == "main" else 3
    vals = [base_val[i].clone() for i in range(5)]
    tests = [base_test[i].clone() for i in range(5)]
    if useful_count >= 6:
        vals.append(0.75 * base_val[0] + 0.25 * base_val[4])
        tests.append(0.75 * base_test[0] + 0.25 * base_test[4])
    while len(vals) < 15:
        src = int(rng.integers(0, 5))
        strength = float(rng.uniform(0.35, 0.75))
        noise_val = torch.tensor(rng.dirichlet([1.5, 1.5], size=base_val.size(1)), dtype=torch.float32)
        noise_test = torch.tensor(rng.dirichlet([1.5, 1.5], size=base_test.size(1)), dtype=torch.float32)
        degraded_val = strength * base_val[src] + (1.0 - strength) * noise_val
        degraded_test = strength * base_test[src] + (1.0 - strength) * noise_test
        if len(vals) % 4 == 0:
            shuffled = rng.permutation(labels_val)
            leak_free_prior = torch.tensor(np.stack([1 - shuffled.mean(), shuffled.mean()]), dtype=torch.float32)
            degraded_val = 0.8 * degraded_val + 0.2 * leak_free_prior
            degraded_test = 0.8 * degraded_test + 0.2 * torch.tensor([0.5, 0.5])
        vals.append(degraded_val / degraded_val.sum(dim=1, keepdim=True).clamp_min(1e-12))
        tests.append(degraded_test / degraded_test.sum(dim=1, keepdim=True).clamp_min(1e-12))
    order = rng.permutation(15).tolist()
    useful_mask = [1 if i < useful_count else 0 for i in range(15)]
    return (
        torch.stack(vals, dim=0)[order],
        torch.stack(tests, dim=0)[order],
        [useful_mask[i] for i in order],
    )


def make_cache(args, seed: int, data) -> Tuple[Dict[str, object], Dict[str, object]]:
    ratings, users, movies = data
    train_idx, val_idx, test_idx = chronological_split(ratings)
    features = build_features(ratings, users, movies, train_idx)
    base_val, base_test = train_five_parties(args, seed, features, train_idx, val_idx, test_idx)
    y_all = torch.full((len(ratings),), -1, dtype=torch.long)
    y_all[:] = torch.tensor(features["labels"], dtype=torch.long)
    labels_val = features["labels"][val_idx]
    if args.num_parties > 5:
        val_probs, test_probs, useful_mask = expand_to_15_parties(
            base_val, base_test, labels_val, seed, args.setting
        )
    else:
        val_probs, test_probs = base_val, base_test
        useful_mask = [1] * val_probs.size(0)
    specs = party_specs(args.num_parties, args.setting)
    if len(specs) < args.num_parties:
        specs = specs + specs[: args.num_parties - len(specs)]
    split_hash = ENGINE.split_fingerprint(
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )
    training_config = {
        "dataset": "MovieLens1M",
        "seed": seed,
        "split": "chronological_70_10_20",
        "label": "rating>=4",
        "num_parties": int(args.num_parties),
        "setting": args.setting,
        "local_epochs": int(args.local_epochs),
        "architectures": [s.architecture for s in specs[: args.num_parties]],
        "hidden_dims": [s.hidden_dim for s in specs[: args.num_parties]],
        "private_views": [s.private_view for s in specs[: args.num_parties]],
    }
    fingerprint = ENGINE.stable_fingerprint({
        "training_config": training_config,
        "split_fingerprint": split_hash,
    })
    cache = {
        "cache_version": int(ENGINE.CACHE_VERSION),
        "cache_fingerprint": fingerprint,
        "training_config": training_config,
        "split_fingerprint": split_hash,
        "dataset": "MovieLens1M",
        "seed": int(seed),
        "target_node_type": "user_movie_pair",
        "train_idx": torch.tensor(train_idx, dtype=torch.long),
        "val_idx": torch.tensor(val_idx, dtype=torch.long),
        "test_idx": torch.tensor(test_idx, dtype=torch.long),
        "y": y_all,
        "probs_train": torch.empty((args.num_parties, len(train_idx), 2)),
        "probs_val": val_probs,
        "probs_test": test_probs,
        "probs_all": torch.empty((args.num_parties, len(ratings), 2)),
        "probs_val_epochs": val_probs.unsqueeze(0),
        "probs_test_epochs": test_probs.unsqueeze(0),
        "party_reliabilities": (val_probs.argmax(dim=2) == torch.tensor(labels_val)).float().mean(dim=1),
        "party_permutation": list(range(args.num_parties)),
        "useful_party_mask": [bool(x) for x in useful_mask],
        "view_names": [s.name for s in specs[: args.num_parties]],
        "num_classes": 2,
        "num_parties": int(args.num_parties),
        "target_node_count": int(len(test_idx)),
        "local_training_protocol": "all_parties_supervised",
        "local_training_labels": "all_parties_supervised_simulation",
        "training_active_party_id": int(args.active_party_id),
        "training_control_comm": 0,
        "val_user_ids": ratings.loc[val_idx, "user_id"].astype(int).tolist(),
        "test_user_ids": ratings.loc[test_idx, "user_id"].astype(int).tolist(),
        "val_movie_ids": ratings.loc[val_idx, "movie_id"].astype(int).tolist(),
        "test_movie_ids": ratings.loc[test_idx, "movie_id"].astype(int).tolist(),
        "party_specs": [s.__dict__ for s in specs[: args.num_parties]],
    }
    meta = {"features": features, "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    return cache, meta


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def ndcg_at_10(user_ids: Iterable[int], labels: np.ndarray, scores: np.ndarray) -> float:
    frame = pd.DataFrame({"user_id": list(user_ids), "label": labels, "score": scores})
    values = []
    for _, group in frame.groupby("user_id"):
        ranked = group.sort_values("score", ascending=False).head(10)
        gains = ranked["label"].to_numpy(dtype=np.float64)
        if gains.size == 0:
            continue
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
        dcg = float((gains * discounts).sum())
        ideal = np.sort(group["label"].to_numpy(dtype=np.float64))[::-1][:10]
        idcg = float((ideal * discounts[: ideal.size]).sum())
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else float("nan")


def method_probs(method: str, result, cache: Dict[str, object], args: argparse.Namespace) -> torch.Tensor:
    probs_test = [p for p in cache["probs_test"]]
    y_val = cache["y"][cache["val_idx"]]
    probs_val = [p for p in cache["probs_val"]]
    rel = ENGINE.compute_party_reliabilities(probs_val, y_val, args.validation_evaluator, args.active_party_id)
    if method == "single_party_best":
        vals = [(p.argmax(dim=1) == y_val).float().mean().item() for p in probs_val]
        return probs_test[int(np.argmax(vals))]
    if method in {"local_uniform_vote", "local_vote"}:
        return ENGINE.vote_from_probs(probs_test, rel, mode="uniform")
    if method == "local_reliability_vote":
        return ENGINE.vote_from_probs(probs_test, rel, mode="reliability")
    if method in {"topk_reliability_vote", "global_topk_reliability"}:
        k = int(getattr(args, "_resolved_topk_reliability_k", args.topk_reliability_k))
        return ENGINE.topk_reliability_vote(probs_test, rel, k if k > 0 else 5)
    return ENGINE.topology_prediction_consensus(probs_test, result.final_adj, rel, args)


def default_engine_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.dataset = "MovieLens1M"
    ns.useful_parties = 6 if args.setting == "main" else 3
    ns.view_setting = args.setting
    ns.validation_evaluator = "all_parties"
    ns.active_party_id = int(args.active_party_id)
    ns.local_training_labels = "all_parties_supervised_simulation"
    ns.local_training_protocol = "all_parties_supervised"
    ns.validation_split_mode = "shared"
    ns.topology_val_fraction = 1.0
    ns.validation_split_seed_offset = 4096
    ns.label_protocol = "all_supervised"
    ns.training_protocol = "supervised"
    ns.validation_protocol = "shared_val"
    ns.passive_label_access = True
    ns.topology_objective = "active"
    ns.joint_lambda = 0.5
    ns.dropout_protocol = "none"
    ns.dropout_rate = 0.0
    ns.dropout_seed = 0
    ns.pred_consensus_steps = int(args.pred_consensus_steps)
    ns.pred_self_weight = float(args.pred_self_weight)
    ns.consensus_mode = "standard"
    ns.consensus_reliability_margin = 0.0
    ns.vote_weighting = "topology_reliability"
    ns.reliability_power = 1.0
    ns.topology_degree_power = 0.5
    ns.final_reliability_floor = 0.0
    ns.topk_reliability_k = int(args.topk_reliability_k)
    ns.readout_mode = "active_vote"
    ns.max_degree = int(args.max_degree)
    ns.matched_edge_count = int(args.matched_edge_count)
    ns.adaptive_score = "graph_val"
    ns._active_adaptive_score = "graph_val"
    ns.adaptive_edge_budget = int(args.adaptive_edge_budget)
    ns.adaptive_min_edges = int(args.adaptive_min_edges)
    ns.adaptive_min_gain = float(args.adaptive_min_gain)
    ns.adaptive_candidate_edges = int(args.adaptive_candidate_edges)
    ns.pair_acc_weight = 0.45
    ns.pair_gain_weight = 0.20
    ns.reliability_weight = 0.25
    ns.corrective_weight = 0.30
    ns.both_wrong_weight = 0.20
    ns.diversity_weight = 0.0
    ns.disagreement_weight = 0.05
    ns.topology_max_updates = 0
    ns.topology_every = 1
    ns.setting_name = args.setting
    return ns


def write_party_performance(cache: Dict[str, object], out_dir: Path, seed: int) -> None:
    y_val = cache["y"][cache["val_idx"]].numpy()
    y_test = cache["y"][cache["test_idx"]].numpy()
    path = out_dir / "movielens_party_performance.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "party", "architecture", "hidden_dim", "private_view",
            "val_auc", "val_accuracy", "test_auc", "test_accuracy", "test_f1",
        ])
        if not exists:
            writer.writeheader()
        for i, spec in enumerate(cache["party_specs"]):
            val = cache["probs_val"][i].numpy()
            test = cache["probs_test"][i].numpy()
            writer.writerow({
                "seed": seed,
                "party": spec["name"],
                "architecture": spec["architecture"],
                "hidden_dim": spec["hidden_dim"],
                "private_view": spec["private_view"],
                "val_auc": roc_auc_binary(y_val, val[:, 1]),
                "val_accuracy": float((val.argmax(axis=1) == y_val).mean()),
                "test_auc": roc_auc_binary(y_test, test[:, 1]),
                "test_accuracy": float((test.argmax(axis=1) == y_test).mean()),
                "test_f1": binary_f1(y_test, test.argmax(axis=1)),
            })


def run_one_seed(args: argparse.Namespace, seed: int, data, out_dir: Path):
    set_seed(seed)
    engine_args = default_engine_args(args)
    cache, _ = make_cache(args, seed, data)
    cache_path = out_dir / "cache" / f"movielens_{args.setting}_n{args.num_parties}_seed{seed}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    write_party_performance(cache, out_dir, seed)
    results = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if engine_args.topk_reliability_k <= 0 and "adaptive_graph_val" in methods:
        methods = [m for m in methods if m not in {"topk_reliability_vote", "global_topk_reliability"}] + [
            m for m in methods if m in {"topk_reliability_vote", "global_topk_reliability"}
        ]
    for method in methods:
        if method in {"topk_reliability_vote", "global_topk_reliability"}:
            engine_args._resolved_topk_reliability_k = max(1, int(args.matched_edge_count))
        result = ENGINE.evaluate_cached_method(method, seed, engine_args, cache, cache_path)
        probs = method_probs(method, result, cache, engine_args).numpy()
        y_test = cache["y"][cache["test_idx"]].numpy()
        pred = probs.argmax(axis=1)
        result.auc = roc_auc_binary(y_test, probs[:, 1])
        result.binary_f1 = binary_f1(y_test, pred)
        result.ndcg_at_10 = ndcg_at_10(cache["test_user_ids"], y_test, probs[:, 1])
        results.append(result)
        print(
            f"seed={seed} method={method} auc={result.auc:.4f} "
            f"acc={result.test_at_best_val:.4f} f1={result.binary_f1:.4f} edges={result.final_edges}"
        )
    return results


def write_outputs(results, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = out_dir / "movielens_tadvfg_per_seed.csv"
    ENGINE.write_csv(str(per_seed), results)
    frame = pd.read_csv(per_seed)
    frame["auc"] = [getattr(r, "auc", np.nan) for r in results]
    frame["f1"] = [getattr(r, "binary_f1", np.nan) for r in results]
    frame["ndcg_at_10"] = [getattr(r, "ndcg_at_10", np.nan) for r in results]
    frame.to_csv(per_seed, index=False)
    rows = []
    for method, group in frame.groupby("method"):
        rows.append({
            "method": method,
            "auc_mean": group["auc"].mean(),
            "auc_std": group["auc"].std(ddof=0),
            "accuracy_mean": group["test_at_best_val"].mean(),
            "accuracy_std": group["test_at_best_val"].std(ddof=0),
            "f1_mean": group["f1"].mean(),
            "f1_std": group["f1"].std(ddof=0),
            "ndcg_at_10_mean": group["ndcg_at_10"].mean(),
            "ndcg_at_10_std": group["ndcg_at_10"].std(ddof=0),
            "selected_links_mean": group["final_edges"].mean(),
            "peer_comm_mean": group["peer_to_peer_comm"].mean(),
            "readout_comm_mean": group["global_readout_comm"].mean(),
            "total_comm_mean": group["total_comm"].mean(),
            "topology_eval_count_mean": group["topology_eval_count"].mean(),
            "topology_update_time_mean": group["topology_update_time"].mean(),
        })
    pd.DataFrame(rows).to_csv(out_dir / "movielens_tadvfg_summary.csv", index=False)
    topologies = {
        f"{r.seed}:{r.method}": {
            "edges": json.loads(r.selected_edges),
            "party_reliabilities": json.loads(r.party_reliabilities),
            "party_views": json.loads(r.party_view_names),
        }
        for r in results
    }
    (out_dir / "movielens_selected_topologies.json").write_text(
        json.dumps(topologies, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "movielens" / "ml-1m")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "movielens")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--num-parties", type=int, default=5)
    parser.add_argument("--setting", choices=["five", "main", "hard"], default="five")
    parser.add_argument("--local-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--methods", default="single_party_best,local_uniform_vote,local_reliability_vote,adaptive_graph_val,topk_reliability_vote,fixed_ring,random_matched,full_mesh,adaptive_pair,adaptive_complementarity")
    parser.add_argument("--adaptive-edge-budget", type=int, default=6)
    parser.add_argument("--adaptive-min-edges", type=int, default=2)
    parser.add_argument("--adaptive-min-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-candidate-edges", type=int, default=0)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--matched-edge-count", type=int, default=2)
    parser.add_argument("--topk-reliability-k", type=int, default=0)
    parser.add_argument("--pred-consensus-steps", type=int, default=1)
    parser.add_argument("--pred-self-weight", type=float, default=0.85)
    parser.add_argument("--active-party-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.setting in {"main", "hard"} and args.num_parties == 5:
        args.num_parties = 15
        args.matched_edge_count = max(args.matched_edge_count, 5)
        args.adaptive_min_edges = max(args.adaptive_min_edges, 5)
        args.adaptive_edge_budget = max(args.adaptive_edge_budget, 15)
    seeds = parse_seeds(args.seeds)
    if args.synthetic_smoke:
        data = synthetic_smoke_data(seeds[0])
    else:
        data = read_movielens_1m(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    party_file = args.output_dir / "movielens_party_performance.csv"
    if party_file.exists():
        party_file.unlink()
    all_results = []
    for seed in seeds:
        all_results.extend(run_one_seed(args, seed, data, args.output_dir))
    write_outputs(all_results, args.output_dir)
    print(f"Saved MovieLens outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
