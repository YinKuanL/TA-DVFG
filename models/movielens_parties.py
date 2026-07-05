"""Heterogeneous local recommender parties for MovieLens TA-DVFG caches.

The classes in this file are intentionally local: each party owns a different
feature view and hidden dimension, and only exports 2-class probabilities for
aligned user-movie pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PartySpec:
    name: str
    architecture: str
    hidden_dim: int
    private_view: str
    useful: bool = True


class MFParty(nn.Module):
    def __init__(self, num_users: int, num_movies: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(num_users, dim)
        self.movie = nn.Embedding(num_movies, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        u = self.user(users)
        m = self.movie(movies)
        return self.head(torch.cat([u, m, u * m], dim=1))


class UserProfileParty(nn.Module):
    def __init__(self, num_movies: int, user_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.user_encoder = nn.Sequential(nn.Linear(user_dim, hidden_dim), nn.ReLU())
        self.movie = nn.Embedding(num_movies, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, users: torch.Tensor, movies: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        u = self.user_encoder(user_features[users])
        m = self.movie(movies)
        return self.head(torch.cat([u, m, u * m], dim=1))


class MovieMetadataParty(nn.Module):
    def __init__(self, num_users: int, movie_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.user = nn.Embedding(num_users, hidden_dim)
        self.movie_encoder = nn.Sequential(nn.Linear(movie_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, users: torch.Tensor, movies: torch.Tensor, movie_features: torch.Tensor) -> torch.Tensor:
        u = self.user(users)
        m = self.movie_encoder(movie_features[movies])
        return self.head(torch.cat([u, m, u * m], dim=1))


class GenreKNNParty(nn.Module):
    def __init__(self, num_users: int, movie_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.user = nn.Embedding(num_users, hidden_dim)
        self.movie_encoder = nn.Sequential(nn.Linear(movie_dim, hidden_dim), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, users: torch.Tensor, movies: torch.Tensor, knn_movie_features: torch.Tensor) -> torch.Tensor:
        u = self.user(users)
        m = self.movie_encoder(knn_movie_features[movies])
        return self.head(torch.cat([u, m, torch.abs(u - m)], dim=1))


class PopularityTemporalParty(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features)


def party_specs(num_parties: int = 5, setting: str = "five") -> List[PartySpec]:
    base = [
        PartySpec("rating_interaction_mf", "matrix_factorization", 64, "train rating interaction graph"),
        PartySpec("user_profile_mlp", "profile MLP + movie ID embedding", 32, "age, gender, occupation"),
        PartySpec("movie_metadata_mlp", "movie metadata MLP + user ID embedding", 128, "genres and release year"),
        PartySpec("genre_knn_mlp", "genre-KNN smoothed MLP", 48, "genre semantic KNN graph"),
        PartySpec("popularity_temporal_mlp", "small MLP", 16, "train-only popularity and temporal statistics"),
    ]
    if num_parties <= 5:
        return base[:num_parties]
    useful = 6 if setting == "main" else 3
    specs = list(base)
    specs.append(PartySpec("sparse_rating_mf", "sparse matrix_factorization", 40, "sparse train interactions", useful=True))
    weak_templates = [
        ("masked_genres", "metadata MLP with masked genres", 24, "partially masked movie genres"),
        ("corrupt_user_profile", "profile MLP with corrupted features", 20, "corrupted age/gender/occupation"),
        ("outdated_popularity", "small MLP", 12, "early-train popularity statistics"),
        ("low_degree_interaction", "low-rank matrix_factorization", 24, "low-degree interaction subset"),
        ("random_project_movie", "random projection MLP", 18, "projected movie metadata"),
        ("weak_genre_knn", "weak KNN MLP", 18, "small-k noisy genre graph"),
        ("highk_noisy_knn", "noisy KNN MLP", 18, "high-k noisy genre graph"),
        ("partial_profile", "partial profile MLP", 12, "age-only user profile"),
        ("shuffled_feature_diagnostic", "diagnostic MLP", 10, "shuffled train-view features"),
    ]
    for name, arch, dim, view in weak_templates:
        specs.append(PartySpec(name, arch, dim, view, useful=False))
    specs = specs[:num_parties]
    for i, spec in enumerate(specs):
        spec.useful = i < useful
    return specs


def make_genre_knn_features(movie_features: np.ndarray, k: int = 20) -> np.ndarray:
    x = movie_features.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.maximum(norms, 1e-12)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -1.0)
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        nn_idx = np.argpartition(-sim[i], kth=min(k, x.shape[0] - 1))[:k]
        weights = np.maximum(sim[i, nn_idx], 0.0)
        if weights.sum() <= 0:
            out[i] = x[i]
        else:
            out[i] = (weights[:, None] * x[nn_idx]).sum(axis=0) / weights.sum()
    return 0.5 * x + 0.5 * out


def train_classifier(
    model: nn.Module,
    batches: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = labels.numel()
    order = torch.arange(n)
    batch_size = min(4096, max(64, n))
    for epoch in range(max(1, epochs)):
        order = order[torch.randperm(n)]
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            logits = _forward_model(model, batches, idx)
            loss = F.cross_entropy(logits, labels[idx])
            loss.backward()
            opt.step()
    return model


def predict_proba(model: nn.Module, batches: Dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    outs = []
    n = next(iter(batches.values())).shape[0]
    with torch.no_grad():
        for start in range(0, n, 8192):
            idx = torch.arange(start, min(start + 8192, n))
            outs.append(F.softmax(_forward_model(model, batches, idx), dim=1).cpu())
    return torch.cat(outs, dim=0)


def _forward_model(model: nn.Module, batches: Dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    if isinstance(model, MFParty):
        return model(batches["users"][idx], batches["movies"][idx])
    if isinstance(model, UserProfileParty):
        return model(batches["users"][idx], batches["movies"][idx], batches["user_features"])
    if isinstance(model, MovieMetadataParty):
        return model(batches["users"][idx], batches["movies"][idx], batches["movie_features"])
    if isinstance(model, GenreKNNParty):
        return model(batches["users"][idx], batches["movies"][idx], batches["knn_movie_features"])
    if isinstance(model, PopularityTemporalParty):
        return model(batches["pair_features"][idx])
    raise TypeError(f"Unsupported MovieLens party model: {type(model)!r}")

