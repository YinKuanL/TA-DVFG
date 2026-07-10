"""
TA-DVFG HGB multi-dataset experiment
====================================

Generic dataset script for TA-DVFG-style experiments on heterogeneous graph
benchmarks: ACM, DBLP, IMDB, and optional Freebase.

This script is designed to extend the existing ACM-only experiment to multiple
heterogeneous graph datasets. It reuses the same experimental logic:

    - target nodes are converted into target-target graph views via meta-paths;
    - each party owns a private feature block and one private graph view;
    - useful parties receive semantic/KNN views;
    - distractor parties receive noisy features and random graphs;
    - each party trains a local GCN;
    - decentralized methods perform prediction-level consensus;
    - TA-DVFG learns sparse party-party topology by validation utility.
    - reliability-aware consensus is constrained by the learned topology, not a plain global reliability vote.

Main dependencies:
    torch, numpy, scipy, scikit-learn, matplotlib(optional), torch-geometric

Install PyG following your local CUDA/torch version:
    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Examples:
    python ta_dvfg_hgb_final.py --dataset ACM --num_parties 15 --useful_parties 6 --graph_views PAP,PSP,KNN --seeds 42,43,44,45,46 --plot --save_csv acm_hgb.csv
    python ta_dvfg_hgb_final.py --dataset DBLP --num_parties 15 --useful_parties 6 --graph_views APA,APCPA,APTPA,KNN --seeds 42,43,44,45,46 --plot --save_csv dblp_hgb.csv
    python ta_dvfg_hgb_final.py --dataset IMDB --num_parties 15 --useful_parties 6 --graph_views MAM,MDM,MKM,KNN --seeds 42,43,44,45,46 --plot --save_csv imdb_hgb.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_seed_list(seed_text: str) -> List[int]:
    return [int(s.strip()) for s in seed_text.split(",") if s.strip()]


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return torch.device("cpu")
    if not requested.startswith("cuda"):
        raise ValueError("--device must be auto, cpu, cuda, or cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(requested)
    torch.cuda.set_device(device.index if device.index is not None else 0)
    return device


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def accuracy_from_probs(probs: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    pred = probs[idx].argmax(dim=1)
    return (pred == y[idx]).float().mean().item()


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits[idx].argmax(dim=1)
    return (pred == y[idx]).float().mean().item()


def macro_f1_from_probs(probs: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_classes: int) -> float:
    pred = probs[idx].argmax(dim=1).detach().cpu().numpy()
    true = y[idx].detach().cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(f1s))


def row_normalize_dense(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    rowsum = np.abs(x).sum(axis=1, keepdims=True)
    return x / np.maximum(rowsum, 1e-12)


def row_normalize_sparse(mat: sp.spmatrix) -> sp.csr_matrix:
    mat = mat.tocsr().astype(np.float32)
    rowsum = np.asarray(mat.sum(axis=1)).flatten()
    inv = np.zeros_like(rowsum, dtype=np.float32)
    mask = rowsum > 0
    inv[mask] = 1.0 / rowsum[mask]
    return sp.diags(inv).dot(mat).tocsr()


def binarize_no_self(mat: sp.spmatrix) -> sp.csr_matrix:
    mat = mat.tocsr().astype(np.float32)
    if mat.shape[0] == mat.shape[1]:
        mat.setdiag(0)
    mat.eliminate_zeros()
    mat.data[:] = 1.0
    return mat.tocsr()


def scipy_adj_to_edge_list(adj: sp.csr_matrix, max_edges: Optional[int] = None, seed: int = 0) -> List[Tuple[int, int]]:
    adj = sp.triu(adj, k=1).tocsr()
    row, col = adj.nonzero()
    edges = list(zip(row.tolist(), col.tolist()))
    if max_edges is not None and max_edges > 0 and len(edges) > max_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[int(i)] for i in idx]
    return edges


def edge_list_to_sparse_norm(edges: List[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    rows, cols = [], []
    for i, j in edges:
        if i == j:
            continue
        rows += [int(i), int(j)]
        cols += [int(j), int(i)]
    for i in range(num_nodes):
        rows.append(i)
        cols.append(i)
    idx = torch.tensor([rows, cols], dtype=torch.long, device=DEVICE)
    vals = torch.ones(len(rows), dtype=torch.float32, device=DEVICE)
    adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=DEVICE).coalesce()
    row, col = adj.indices()
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=DEVICE)
    deg.scatter_add_(0, row, adj.values())
    deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
    norm_vals = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape, device=DEVICE).coalesce()


def make_knn_edges_from_numpy(x_np: np.ndarray, k: int, block: int = 4096) -> List[Tuple[int, int]]:
    """Construct undirected KNN graph by cosine similarity."""
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    x = F.normalize(x, dim=1)
    n = x.shape[0]
    edges = set()
    with torch.no_grad():
        for start in range(0, n, block):
            end = min(start + block, n)
            sim = x[start:end] @ x.t()
            for local_i, global_i in enumerate(range(start, end)):
                sim[local_i, global_i] = -1.0
            nn_idx = sim.topk(k=min(k, max(n - 1, 1)), dim=1).indices.detach().cpu().numpy()
            for local_i, global_i in enumerate(range(start, end)):
                for j in nn_idx[local_i]:
                    a, b = min(global_i, int(j)), max(global_i, int(j))
                    if a != b:
                        edges.add((a, b))
    return sorted(edges)


def make_random_edges(num_nodes: int, degree: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    target = max(num_nodes * degree // 2, 1)
    edges = set()
    max_possible = num_nodes * (num_nodes - 1) // 2
    target = min(target, max_possible)
    while len(edges) < target:
        i = int(rng.integers(0, num_nodes))
        j = int(rng.integers(0, num_nodes))
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        edges.add((a, b))
    return sorted(edges)


def split_features(x: torch.Tensor, num_parties: int) -> List[torch.Tensor]:
    dims = np.array_split(np.arange(x.size(1)), num_parties)
    out = []
    for idx in dims:
        if len(idx) == 0:
            # Some datasets may have fewer feature dimensions than parties.
            out.append(torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device))
        else:
            out.append(x.index_select(1, torch.as_tensor(idx, dtype=torch.long, device=x.device)).clone())
    return out


def stratified_split(y: torch.Tensor, train_ratio: float, val_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    train, val, test = [], [], []
    for c in sorted(set(y_np.tolist())):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        n_train = max(1, int(train_ratio * len(idx)))
        n_val = max(1, int(val_ratio * len(idx)))
        train.extend(idx[:n_train])
        val.extend(idx[n_train:n_train + n_val])
        test.extend(idx[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return (
        torch.tensor(train, dtype=torch.long, device=DEVICE),
        torch.tensor(val, dtype=torch.long, device=DEVICE),
        torch.tensor(test, dtype=torch.long, device=DEVICE),
    )


# =============================================================================
# Heterogeneous dataset loading and meta-path graph construction
# =============================================================================


def import_pyg_hgb():
    try:
        from torch_geometric.datasets import HGBDataset  # type: ignore
        return HGBDataset
    except Exception as exc:
        raise ImportError(
            "torch-geometric is required for DBLP/IMDB/Freebase HGB experiments. "
            "Install PyG for your Torch/CUDA version. Original error: " + repr(exc)
        )


def _first_attr(obj, names: Iterable[str]):
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return None


def infer_target_node_type(data, requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested.lower()
    # Prefer node types with labels.
    for ntype in data.node_types:
        store = data[ntype]
        if hasattr(store, "y") and store.y is not None:
            return ntype
    # Dataset-specific fallback.
    names = {t.lower(): t for t in data.node_types}
    for cand in ["paper", "author", "movie", "book"]:
        if cand in names:
            return names[cand]
    return data.node_types[0]


def get_node_features(data, ntype: str, max_identity_dim: int = 5000) -> np.ndarray:
    store = data[ntype]
    x = _first_attr(store, ["x", "feat", "features"])
    n = int(store.num_nodes)
    if x is not None:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np[:, None]
        return row_normalize_dense(x_np.astype(np.float32))

    # Fallback for featureless targets: low-dimensional deterministic identity/random projection.
    # This is a fallback only. Prefer datasets with real target-node features.
    print(f"Warning: target node type '{ntype}' has no features. Using deterministic identity/random fallback.")
    if n <= max_identity_dim:
        return np.eye(n, dtype=np.float32)
    rng = np.random.default_rng(12345)
    return rng.standard_normal((n, min(256, max_identity_dim))).astype(np.float32) / math.sqrt(256)


def has_node_features(data, ntype: str) -> bool:
    return _first_attr(data[ntype], ["x", "feat", "features"]) is not None


def graph_structural_features(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    out_dim: int,
    seed: int,
) -> np.ndarray:
    """Build deterministic target-node features from one private graph view.

    This is used only for featureless labelled targets such as HGB Freebase
    `book`. The features are derived from the party's own target-target graph
    view, not from labels or from other parties' views.
    """
    out_dim = max(4, int(out_dim))
    hash_dim = out_dim - 3
    feats = np.zeros((num_nodes, out_dim), dtype=np.float32)
    if not edges:
        return feats

    deg = np.zeros(num_nodes, dtype=np.float32)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            deg[u] += 1.0
            deg[v] += 1.0
            bu = (u * 1315423911 + seed) % hash_dim
            bv = (v * 2654435761 + seed) % hash_dim
            su = 1.0 if ((u + seed) & 1) == 0 else -1.0
            sv = 1.0 if ((v + seed) & 1) == 0 else -1.0
            feats[v, 3 + int(bu)] += su
            feats[u, 3 + int(bv)] += sv

    max_deg = float(max(1.0, deg.max()))
    feats[:, 0] = deg / max_deg
    feats[:, 1] = np.log1p(deg) / math.log1p(max_deg)
    feats[:, 2] = (deg > 0).astype(np.float32)

    # Column-scale the hashed structural channels for stable GCN training.
    scale = np.maximum(1.0, np.std(feats[:, 3:], axis=0, keepdims=True))
    feats[:, 3:] = feats[:, 3:] / scale
    return row_normalize_dense(feats)


def featureless_target_party_features(
    graph_edges: Dict[str, List[Tuple[int, int]]],
    requested_views: List[str],
    num_nodes: int,
    num_parties: int,
    seed: int,
    per_party_dim: int = 16,
) -> np.ndarray:
    blocks: List[np.ndarray] = []
    for p in range(num_parties):
        view = requested_views[p % len(requested_views)]
        edges = graph_edges.get(view, [])
        blocks.append(graph_structural_features(edges, num_nodes, per_party_dim, seed + stable_name_seed(view) + p))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def get_labels_and_split(data, ntype: str, args: argparse.Namespace, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    store = data[ntype]
    if not hasattr(store, "y") or store.y is None:
        raise ValueError(f"Target node type '{ntype}' has no labels y.")
    y = store.y
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    y = y.detach().cpu()
    if y.ndim > 1:
        # For multi-label variants, convert to the dominant class for this node-classification script.
        # If you want true multi-label classification, use BCE loss and micro-F1 separately.
        y = y.argmax(dim=1)
    y_np = y.numpy().astype(np.int64)
    valid = y_np >= 0
    # Relabel classes compactly on valid nodes.
    unique = np.unique(y_np[valid])
    mapping = {int(c): i for i, c in enumerate(unique.tolist())}
    y_compact = np.array([mapping.get(int(v), -1) for v in y_np], dtype=np.int64)

    y_t = torch.tensor(y_compact, dtype=torch.long, device=DEVICE)
    valid_idx = np.where(valid)[0]

    train_mask = _first_attr(store, ["train_mask"])
    val_mask = _first_attr(store, ["val_mask"])
    test_mask = _first_attr(store, ["test_mask"])

    if train_mask is not None and val_mask is not None and test_mask is not None and not args.ignore_dataset_split:
        def mask_to_idx(mask) -> torch.Tensor:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)
            if mask_np.ndim > 1:
                mask_np = mask_np[:, 0]
            if mask_np.dtype == bool:
                idx = np.where(mask_np & valid)[0]
            else:
                idx = mask_np.reshape(-1).astype(np.int64)
                idx = idx[y_compact[idx] >= 0]
            return torch.tensor(idx, dtype=torch.long, device=DEVICE)
        return y_t, mask_to_idx(train_mask), mask_to_idx(val_mask), mask_to_idx(test_mask)

    # If there is a valid label mask, split only labelled nodes.
    y_valid = y_t[torch.tensor(valid_idx, dtype=torch.long, device=DEVICE)]
    train_local, val_local, test_local = stratified_split(y_valid, args.train_ratio, args.val_ratio, seed)
    valid_t = torch.tensor(valid_idx, dtype=torch.long, device=DEVICE)
    return y_t, valid_t[train_local], valid_t[val_local], valid_t[test_local]


def edge_index_to_csr(edge_index: torch.Tensor, src_n: int, dst_n: int) -> sp.csr_matrix:
    ei = edge_index.detach().cpu().numpy()
    row = ei[0].astype(np.int64)
    col = ei[1].astype(np.int64)
    vals = np.ones(len(row), dtype=np.float32)
    return sp.csr_matrix((vals, (row, col)), shape=(src_n, dst_n), dtype=np.float32)


def get_adj_between(data, src: str, dst: str) -> Optional[sp.csr_matrix]:
    """Return adjacency from src nodes to dst nodes, summing all relations. Uses reverse edges if needed."""
    mats = []
    for etype in data.edge_types:
        s, _, d = etype
        edge_index = data[etype].edge_index
        if s == src and d == dst:
            mats.append(edge_index_to_csr(edge_index, int(data[src].num_nodes), int(data[dst].num_nodes)))
        elif s == dst and d == src:
            rev = edge_index_to_csr(edge_index, int(data[dst].num_nodes), int(data[src].num_nodes)).T.tocsr()
            mats.append(rev)
    if not mats:
        return None
    out = mats[0].copy()
    for m in mats[1:]:
        out = out + m
    out.data[:] = 1.0
    return out.tocsr()


def build_metapath_adj(data, node_seq: Sequence[str]) -> Optional[sp.csr_matrix]:
    if len(node_seq) < 3 or node_seq[0] != node_seq[-1]:
        raise ValueError("node_seq must be a target-returning meta-path, e.g. ['paper','author','paper']")
    mat = None
    for a, b in zip(node_seq[:-1], node_seq[1:]):
        adj = get_adj_between(data, a, b)
        if adj is None:
            return None
        mat = adj if mat is None else mat @ adj
    assert mat is not None
    return binarize_no_self(mat)


def dataset_default_views(dataset: str, target: str, data) -> Dict[str, Sequence[str]]:
    ds = dataset.upper()
    nt = {t.lower(): t for t in data.node_types}
    target_real = target

    if ds == "ACM":
        paper = nt.get("paper", target_real)
        author = nt.get("author")
        subject = nt.get("subject") or nt.get("field") or nt.get("term")
        views = {}
        if author:
            views["PAP"] = [paper, author, paper]
        if subject:
            views["PSP"] = [paper, subject, paper]
            views["PLP"] = [paper, subject, paper]
        return views

    if ds == "DBLP":
        author = nt.get("author", target_real)
        paper = nt.get("paper")
        term = nt.get("term")
        conf = nt.get("conference") or nt.get("conf") or nt.get("venue")
        views = {}
        if paper:
            views["APA"] = [author, paper, author]
        if paper and conf:
            views["APCPA"] = [author, paper, conf, paper, author]
        if paper and term:
            views["APTPA"] = [author, paper, term, paper, author]
        return views

    if ds == "IMDB":
        movie = nt.get("movie", target_real)
        actor = nt.get("actor")
        director = nt.get("director")
        keyword = nt.get("keyword") or nt.get("keywords") or nt.get("genre")
        views = {}
        if actor:
            views["MAM"] = [movie, actor, movie]
        if director:
            views["MDM"] = [movie, director, movie]
        if keyword:
            views["MKM"] = [movie, keyword, movie]
            views["MGM"] = [movie, keyword, movie]
        return views

    # Freebase / unknown: auto-generate target-X-target views for all neighbor types.
    views = {}
    for other in data.node_types:
        if other == target_real:
            continue
        if get_adj_between(data, target_real, other) is not None:
            key = f"{target_real[:1].upper()}{other[:1].upper()}{target_real[:1].upper()}"
            views[key] = [target_real, other, target_real]
    return views


def load_hgb_dataset(args: argparse.Namespace, seed: int) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, List[Tuple[int, int]]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    str,
]:
    HGBDataset = import_pyg_hgb()
    dataset_name = args.dataset.upper()
    root = str(Path(args.data_dir) / "hgb")
    dataset = HGBDataset(root=root, name=dataset_name)
    data = dataset[0]

    target = infer_target_node_type(data, args.target_node)
    y, train_idx, val_idx, test_idx = get_labels_and_split(data, target, args, seed)

    num_nodes = int(data[target].num_nodes)
    default_views = dataset_default_views(dataset_name, target, data)
    graph_edges: Dict[str, List[Tuple[int, int]]] = {}

    for name, seq in default_views.items():
        mat = build_metapath_adj(data, seq)
        if mat is not None and mat.shape[0] == num_nodes:
            graph_edges[name.upper()] = scipy_adj_to_edge_list(
                mat,
                max_edges=args.max_metapath_edges,
                seed=seed + stable_name_seed(name),
            )

    if not graph_edges:
        raise RuntimeError(
            f"Could not construct any target-target graph views for dataset={dataset_name}, target={target}. "
            f"Available node_types={data.node_types}, edge_types={data.edge_types}."
            )

    requested_views = [v.strip().upper() for v in args.graph_views.split(",") if v.strip()]
    if not requested_views:
        requested_views = [k for k in graph_edges.keys() if k not in {"RANDOM", "FULL"}]
    if not requested_views:
        requested_views = ["KNN"]

    if has_node_features(data, target):
        x_np = get_node_features(data, target)
    elif args.featureless_target_features == "structural":
        print(
            f"Warning: target node type '{target}' has no features. "
            "Using deterministic private-graph structural features."
        )
        x_np = featureless_target_party_features(
            graph_edges=graph_edges,
            requested_views=requested_views,
            num_nodes=num_nodes,
            num_parties=args.num_parties,
            seed=seed,
            per_party_dim=args.structural_feature_dim,
        )
    else:
        x_np = get_node_features(data, target)

    graph_edges["KNN"] = make_knn_edges_from_numpy(x_np, args.graph_k)
    graph_edges["RANDOM"] = make_random_edges(num_nodes, args.random_graph_degree, seed + 999)
    # FULL = union of non-random semantic views.
    full = set()
    for k, edges in graph_edges.items():
        if k not in {"RANDOM", "KNN"}:
            full.update(edges)
    if full:
        graph_edges["FULL"] = sorted(full)
    else:
        graph_edges["FULL"] = graph_edges["KNN"]

    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    print(
        f"Loaded {dataset_name}: target={target}, nodes={x.size(0)}, features={x.size(1)}, "
        f"classes={int(y.max().item()+1)}, views={sorted(graph_edges.keys())}"
    )
    return x, y, graph_edges, (train_idx, val_idx, test_idx), target


def build_party_views(
    x: torch.Tensor,
    graph_edges: Dict[str, List[Tuple[int, int]]],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[bool], List[int]]:
    xs = split_features(x, args.num_parties)
    num_nodes = x.size(0)
    rng = np.random.default_rng(seed + 123)
    requested_views = [v.strip().upper() for v in args.graph_views.split(",") if v.strip()]
    if not requested_views:
        requested_views = [k for k in graph_edges.keys() if k not in {"RANDOM", "FULL"}]
    if not requested_views:
        requested_views = ["KNN"]

    useful_parties = max(1, min(args.useful_parties, args.num_parties))
    view_names: List[str] = []
    adjs: List[torch.Tensor] = []
    useful_party_mask = [p < useful_parties for p in range(args.num_parties)]

    if args.view_setting == "hard":
        for p in range(useful_parties, args.num_parties):
            perm = torch.as_tensor(rng.permutation(num_nodes), dtype=torch.long, device=x.device)
            xs[p] = xs[p].index_select(0, perm) + torch.randn_like(xs[p]) * args.distractor_feature_noise

    for p in range(args.num_parties):
        if args.view_setting == "hard" and p >= useful_parties:
            view = "RANDOM"
            edges = make_random_edges(num_nodes, args.random_graph_degree, seed + 7000 + p)
        else:
            view = requested_views[p % len(requested_views)]
            if view not in graph_edges:
                print(f"Warning: requested view {view} not found; fallback to KNN.")
                view = "KNN"
            edges = graph_edges[view]
        view_names.append(view)
        adjs.append(edge_list_to_sparse_norm(edges, num_nodes))

    party_permutation = list(range(args.num_parties))
    if args.shuffle_party_positions:
        party_permutation = np.random.default_rng(seed + args.party_shuffle_seed_offset).permutation(args.num_parties).tolist()
        xs = [xs[i] for i in party_permutation]
        adjs = [adjs[i] for i in party_permutation]
        view_names = [view_names[i] for i in party_permutation]
        useful_party_mask = [useful_party_mask[i] for i in party_permutation]

    return xs, adjs, view_names, useful_party_mask, party_permutation


# =============================================================================
# Models
# =============================================================================


class LocalGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(adj, x)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        h = F.relu(self.lin2(h))
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.cls(self.encode(x, adj))


class CentralFusionModel(nn.Module):
    def __init__(self, in_dims: List[int], hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleList([LocalGCN(d, hidden_dim, num_classes, dropout) for d in in_dims])
        self.fusion = nn.Sequential(
            nn.Linear(len(in_dims) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, xs: List[torch.Tensor], adjs: List[torch.Tensor]) -> torch.Tensor:
        hs = [enc.encode(x, adj) for enc, x, adj in zip(self.encoders, xs, adjs)]
        return self.fusion(torch.cat(hs, dim=1))


@dataclass
class Party:
    model: LocalGCN
    optimizer: torch.optim.Optimizer
    reliability: float = 1.0


@dataclass
class RunResult:
    seed: int
    dataset: str
    method: str
    best_val: float
    test_at_best_val: float
    macro_f1_at_best_val: float
    best_observed_test: float
    final_test: float
    train_comm: int
    inference_comm: int
    final_edges: int
    final_adj: Optional[np.ndarray]
    peer_to_peer_comm: int = 0
    global_readout_comm: int = 0
    total_comm: int = 0
    protocol_type: str = ""
    validation_evaluator: str = "all_parties"
    active_party_id: int = 0
    local_training_labels: str = "all_parties_supervised_simulation"
    validation_label_protocol: str = "legacy_shared_evaluator"
    topology_evaluator: str = "all_parties"
    local_training_protocol: str = "all_parties_supervised"
    validation_split_mode: str = "shared"
    topology_val_fraction: float = 1.0
    topology_val_count: int = 0
    selection_val_count: int = 0
    topology_update_comm: int = 0
    topology_eval_count: int = 0
    topology_update_time: float = 0.0
    useful_useful_edges: int = 0
    useful_noisy_edges: int = 0
    noisy_noisy_edges: int = 0
    selected_edges: str = "[]"
    useful_party_mask: str = "[]"
    party_view_names: str = "[]"
    party_reliabilities: str = "[]"
    party_permutation: str = "[]"
    adaptive_min_edges: int = 0
    prediction_cache_path: str = ""
    prediction_cache_fingerprint: str = ""
    setting: str = ""
    label_protocol: str = "all_supervised"
    training_protocol: str = "supervised"
    validation_protocol: str = "shared_val"
    passive_label_access: bool = True
    topology_objective: str = "active"
    joint_lambda: float = 0.5
    active_readout_acc: float = -1.0
    local_post_consensus_mean_acc: float = -1.0
    active_party_post_consensus_acc: float = -1.0
    local_worst_acc: float = -1.0
    local_best_acc: float = -1.0
    dropout_protocol: str = "none"
    dropout_rate: float = 0.0
    dropout_seed: int = 0
    remaining_parties: int = 0
    dropped_party_ids: str = "[]"
    test_acc_after_dropout: float = -1.0
    local_post_consensus_acc_after_dropout: float = -1.0
    control_plane_comm: int = 0


# =============================================================================
# Topologies and consensus
# =============================================================================


def edge_count(adj: np.ndarray) -> int:
    return int(adj.sum() // 2)


def make_ring_topology(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        if i != j:
            adj[i, j] = adj[j, i] = 1.0
    return adj


def make_ring_like_topology(n: int, requested_edges: int, seed: int = 0) -> np.ndarray:
    """Build a deterministic ring-like graph with exactly the requested edges."""
    max_edges = n * (n - 1) // 2
    target = max(0, min(int(requested_edges), max_edges))
    adj = np.zeros((n, n), dtype=np.float32)
    if target == 0:
        return adj
    offset = int(np.random.default_rng(seed + 1401).integers(0, max(n, 1))) if n else 0
    order = [(offset + i) % n for i in range(n)]
    for skip in range(1, n):
        for position in range(n):
            if edge_count(adj) >= target:
                return adj
            i = order[position]
            j = order[(position + skip) % n]
            a, b = min(i, j), max(i, j)
            if a != b and adj[a, b] == 0:
                adj[a, b] = adj[b, a] = 1.0
    return adj


def make_full_topology(n: int) -> np.ndarray:
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def make_random_regular_topology(n: int, max_degree: int, edge_budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 321)
    adj = np.zeros((n, n), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(candidates)
    for i, j in candidates:
        if edge_count(adj) >= edge_budget:
            break
        if deg[i] < max_degree and deg[j] < max_degree:
            adj[i, j] = adj[j, i] = 1.0
            deg[i] += 1
            deg[j] += 1
    return adj


def make_random_matched_topology(n: int, max_degree: int, edge_budget: int, seed: int) -> np.ndarray:
    """Sample exactly edge_budget edges, respecting max_degree whenever feasible."""
    target = max(0, min(int(edge_budget), n * (n - 1) // 2))
    adj = make_random_regular_topology(n, max_degree, target, seed)
    if edge_count(adj) >= target:
        return adj

    rng = np.random.default_rng(seed + 654)
    remaining = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
    rng.shuffle(remaining)
    for i, j in remaining:
        if edge_count(adj) >= target:
            break
        adj[i, j] = adj[j, i] = 1.0
    return adj


def make_expander_like_topology(n: int, edge_budget: int) -> np.ndarray:
    """Simple circulant sparse graph baseline: ring plus skip connections."""
    adj = np.zeros((n, n), dtype=np.float32)
    skips = [1, 2, 3, 5, 7]
    for skip in skips:
        for i in range(n):
            if edge_count(adj) >= edge_budget:
                return adj
            j = (i + skip) % n
            a, b = min(i, j), max(i, j)
            if a != b:
                adj[a, b] = adj[b, a] = 1.0
    return adj


def make_expander_matched_topology(n: int, edge_budget: int, seed: int) -> np.ndarray:
    """Rotated circulant prefix, fair under seed-shuffled party positions."""
    target = max(0, min(int(edge_budget), n * (n - 1) // 2))
    adj = np.zeros((n, n), dtype=np.float32)
    if target == 0:
        return adj
    offset = int(np.random.default_rng(seed + 1701).integers(0, max(n, 1))) if n else 0
    order = [(offset + i) % n for i in range(n)]
    for skip in (1, 2, 3, 5, 7):
        for position in range(n):
            if edge_count(adj) >= target:
                return adj
            i = order[position]
            j = order[(position + skip) % n]
            a, b = min(i, j), max(i, j)
            if a != b and adj[a, b] == 0:
                adj[a, b] = adj[b, a] = 1.0
    return adj


def topology_edge_list(adj: np.ndarray) -> List[List[int]]:
    return [[i, j] for i in range(adj.shape[0]) for j in range(i + 1, adj.shape[1]) if adj[i, j] > 0]


def topology_edge_types(adj: np.ndarray, useful_party_mask: Sequence[bool]) -> Tuple[int, int, int]:
    useful_useful = useful_noisy = noisy_noisy = 0
    for i, j in topology_edge_list(adj):
        if useful_party_mask[i] and useful_party_mask[j]:
            useful_useful += 1
        elif useful_party_mask[i] or useful_party_mask[j]:
            useful_noisy += 1
        else:
            noisy_noisy += 1
    return useful_useful, useful_noisy, noisy_noisy


def get_party_probs(parties: List[Party], xs: List[torch.Tensor], adjs: List[torch.Tensor]) -> List[torch.Tensor]:
    probs = []
    for p, x, adj in zip(parties, xs, adjs):
        p.model.eval()
        with torch.no_grad():
            probs.append(F.softmax(p.model(x, adj), dim=1))
    return probs


def get_party_embeddings(parties: List[Party], xs: List[torch.Tensor], adjs: List[torch.Tensor]) -> List[torch.Tensor]:
    embs = []
    for p, x, adj in zip(parties, xs, adjs):
        p.model.eval()
        with torch.no_grad():
            embs.append(p.model.encode(x, adj))
    return embs


def compute_party_reliabilities(
    val_probs_by_party: List[torch.Tensor],
    y_val: torch.Tensor,
    evaluator_mode: str,
    active_party_id: int,
) -> List[float]:
    """Compute party reliabilities under the declared validation-label owner.

    ``active_party`` simulates passive parties submitting validation
    probabilities to one label-holding evaluator. Only this function receives
    ``y_val``; passive parties never compute their own validation accuracy.
    The numerical result matches centralized evaluation because the submitted
    predictions and labels are unchanged.
    """
    if evaluator_mode not in {"all_parties", "active_party"}:
        raise ValueError(f"Unknown evaluator mode: {evaluator_mode}")
    if evaluator_mode == "active_party" and not 0 <= active_party_id < len(val_probs_by_party):
        raise ValueError("--active_party_id must identify an existing party")
    # Eq. (3): validation accuracy defines each party reliability r_i.
    return [
        float((prob.argmax(dim=1) == y_val).float().mean().item())
        for prob in val_probs_by_party
    ]


def evaluator_reliabilities(
    probs: List[torch.Tensor],
    y: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> List[float]:
    """Backward-compatible wrapper using full-node prediction tensors."""
    return compute_party_reliabilities(
        [prob[val_idx] for prob in probs],
        y[val_idx],
        args.validation_evaluator,
        args.active_party_id,
    )


def update_reliability(
    parties: List[Party],
    xs: List[torch.Tensor],
    adjs: List[torch.Tensor],
    y: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    probs = get_party_probs(parties, xs, adjs)
    for party, reliability in zip(parties, evaluator_reliabilities(probs, y, val_idx, args)):
        party.reliability = reliability


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p / m).log()).sum(dim=1) + 0.5 * (q * (q / m).log()).sum(dim=1)
    return float(js.mean().item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.mean(dim=0), dim=0)
    b = F.normalize(b.mean(dim=0), dim=0)
    return float(1.0 - torch.dot(a, b).item())


def vote_from_probs(probs: List[torch.Tensor], reliabilities: List[float], mode: str = "uniform") -> torch.Tensor:
    """Global voting for local baselines and diagnostic fallbacks."""
    if mode == "uniform":
        return torch.stack(probs, dim=0).mean(dim=0)
    if mode in {"reliability", "topology_reliability"}:
        weights = torch.tensor([max(float(r), 1e-3) for r in reliabilities], dtype=probs[0].dtype, device=probs[0].device)
        weights = weights / weights.sum().clamp_min(1e-12)
        return (weights[:, None, None] * torch.stack(probs, dim=0)).sum(dim=0)
    raise ValueError("vote mode must be uniform, reliability, or topology_reliability")


def topk_reliability_vote(
    probs: List[torch.Tensor],
    reliabilities: List[float],
    k: int,
) -> torch.Tensor:
    k = max(1, min(int(k), len(probs)))
    selected = sorted(range(len(probs)), key=lambda i: (-float(reliabilities[i]), i))[:k]
    selected_probs = [probs[i] for i in selected]
    selected_rel = [reliabilities[i] for i in selected]
    return vote_from_probs(selected_probs, selected_rel, mode="reliability")


def active_party_indices(adj: np.ndarray, n: int) -> List[int]:
    if adj is not None and edge_count(adj) > 0:
        active = np.where(adj.sum(axis=1) > 0)[0].tolist()
        if active:
            return active
    return list(range(n))


def topology_filtered_reliability_vote(
    probs: List[torch.Tensor],
    reliabilities: List[float],
    adj: np.ndarray,
    args: argparse.Namespace,
) -> torch.Tensor:
    """
    Reliability-aware final readout constrained by the learned sparse topology.

    This differs from plain global reliability vote. The learned topology decides
    which parties are active; reliability decides how much to trust each selected
    party. Raw features, raw graphs, and local encoders remain local.
    """
    n = len(probs)
    degree = adj.sum(axis=1) if adj is not None else np.zeros(n, dtype=np.float32)

    active = active_party_indices(adj, n)

    floor = getattr(args, "final_reliability_floor", 0.0)
    if floor > 0:
        filtered = [i for i in active if reliabilities[i] >= floor]
        if filtered:
            active = filtered

    if not active:
        active = list(range(n))

    rel_power = getattr(args, "reliability_power", 1.0)
    deg_power = getattr(args, "topology_degree_power", 0.5)
    raw_weights = []
    # Eq. (6): active readout weights selected parties by reliability and topology degree.
    for i in active:
        rel = max(float(reliabilities[i]), 1e-3) ** rel_power
        topo = max(float(degree[i]) + 1.0, 1.0) ** deg_power
        raw_weights.append(rel * topo)

    weights = torch.tensor(raw_weights, dtype=probs[0].dtype, device=probs[0].device)
    weights = weights / weights.sum().clamp_min(1e-12)
    stacked = torch.stack([probs[i] for i in active], dim=0)
    return (weights[:, None, None] * stacked).sum(dim=0)


def party_accuracy_scores(
    probs: List[torch.Tensor],
    labels: torch.Tensor,
) -> List[float]:
    return [
        float((prob.argmax(dim=1) == labels).float().mean().item())
        for prob in probs
    ]


def post_consensus_party_probs(
    probs: List[torch.Tensor],
    adj: np.ndarray,
    reliabilities: List[float],
    args: argparse.Namespace,
) -> List[torch.Tensor]:
    current = [p.clone() for p in probs]
    if args.pred_consensus_steps <= 0 or edge_count(adj) == 0:
        return current
    n = len(current)
    for _ in range(args.pred_consensus_steps):
        new_probs = []
        for i in range(n):
            neigh = np.where(adj[i] > 0)[0].tolist()
            if args.consensus_mode == "gated":
                neigh = [j for j in neigh if reliabilities[j] >= reliabilities[i] + args.consensus_reliability_margin]
            if not neigh:
                new_probs.append(current[i])
                continue
            # Eqs. (4)-(5): one prediction-consensus step mixes local and neighbor probabilities.
            w = torch.tensor([max(float(reliabilities[j]), 1e-3) for j in neigh], dtype=current[i].dtype, device=current[i].device)
            w = w / w.sum().clamp_min(1e-12)
            neigh_prob = sum(wj * current[j] for wj, j in zip(w, neigh))
            new_probs.append(args.pred_self_weight * current[i] + (1 - args.pred_self_weight) * neigh_prob)
        current = new_probs
    return current


def topology_objective_components(
    edge_set: np.ndarray,
    val_probs_by_party: List[torch.Tensor],
    y_val: torch.Tensor,
    reliabilities: List[float],
    active_party_id: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Return deployment-aware validation scores for a candidate topology.

    The returned values are computed only on the topology-validation split.
    They are used for graph selection and never touch test labels.
    """
    post_probs = post_consensus_party_probs(
        val_probs_by_party,
        edge_set,
        reliabilities,
        args,
    )
    local_scores = party_accuracy_scores(post_probs, y_val)
    if args.vote_weighting == "topology_reliability" and edge_count(edge_set) > 0:
        active_vote = topology_filtered_reliability_vote(
            post_probs,
            reliabilities,
            edge_set,
            args,
        )
    else:
        active_vote = vote_from_probs(
            post_probs,
            reliabilities,
            mode=args.vote_weighting,
        )
    active_score = float((active_vote.argmax(dim=1) == y_val).float().mean().item())
    return {
        "active": active_score,
        "local_mean": float(np.mean(local_scores)),
        "active_party_local": float(local_scores[active_party_id]),
        "local_worst": float(np.min(local_scores)),
        "local_best": float(np.max(local_scores)),
    }


def party_dropout_indices(
    num_parties: int,
    active_party_id: int,
    dropout_rate: float,
    experiment_seed: int,
    dropout_seed: int,
) -> List[int]:
    """Select inference-only party failures while always retaining the label holder."""
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("--dropout_rate must be in [0, 1)")
    candidates = [i for i in range(num_parties) if i != active_party_id]
    count = min(len(candidates), int(round(dropout_rate * num_parties)))
    if count <= 0:
        return []
    rng = np.random.default_rng(
        int(experiment_seed) * 1_000_003 + int(dropout_seed) * 9_973 + 811
    )
    return sorted(int(i) for i in rng.choice(candidates, size=count, replace=False))


def inference_readout_metrics(
    method: str,
    probs: List[torch.Tensor],
    labels: torch.Tensor,
    adj: np.ndarray,
    reliabilities: List[float],
    args: argparse.Namespace,
    experiment_seed: int,
) -> Dict[str, object]:
    """Evaluate global and party-local readouts from the same inference state."""
    n = len(probs)
    dropped = (
        party_dropout_indices(
            n,
            int(args.active_party_id),
            float(args.dropout_rate),
            experiment_seed,
            int(args.dropout_seed),
        )
        if args.dropout_protocol == "random_party_dropout"
        else []
    )
    remaining = [i for i in range(n) if i not in set(dropped)]
    index_map = {old: new for new, old in enumerate(remaining)}
    active_sub_id = index_map[int(args.active_party_id)]
    sub_probs = [probs[i] for i in remaining]
    sub_reliabilities = [float(reliabilities[i]) for i in remaining]
    sub_adj = adj[np.ix_(remaining, remaining)].astype(np.float32, copy=True)

    if method == "active_local_only":
        post_probs = [prob.clone() for prob in sub_probs]
    else:
        post_probs = post_consensus_party_probs(
            sub_probs,
            sub_adj,
            sub_reliabilities,
            args,
        )

    local_scores = party_accuracy_scores(post_probs, labels)
    local_mean = float(np.mean(local_scores))
    active_local = float(local_scores[active_sub_id])
    local_worst = float(np.min(local_scores))
    local_best = float(np.max(local_scores))

    if method == "active_local_only":
        active_vote = post_probs[active_sub_id]
    elif method in {"topk_reliability_vote", "global_topk_reliability"}:
        k = int(getattr(args, "_resolved_topk_reliability_k", args.topk_reliability_k))
        active_vote = topk_reliability_vote(
            sub_probs,
            sub_reliabilities,
            max(1, min(k if k > 0 else 5, len(sub_probs))),
        )
    elif args.vote_weighting == "topology_reliability" and edge_count(sub_adj) > 0:
        active_vote = topology_filtered_reliability_vote(
            post_probs,
            sub_reliabilities,
            sub_adj,
            args,
        )
    else:
        active_vote = vote_from_probs(
            post_probs,
            sub_reliabilities,
            mode=args.vote_weighting,
        )
    active_acc = float((active_vote.argmax(dim=1) == labels).float().mean().item())
    return {
        "active_readout_acc": active_acc,
        "local_post_consensus_mean_acc": local_mean,
        "active_party_post_consensus_acc": active_local,
        "local_worst_acc": local_worst,
        "local_best_acc": local_best,
        "remaining_parties": len(remaining),
        "dropped_party_ids": dropped,
        "adj": sub_adj,
    }


def readout_post_consensus(
    party_probs: List[torch.Tensor],
    adj: np.ndarray,
    reliabilities: List[float],
    args: argparse.Namespace,
    y: Optional[torch.Tensor] = None,
    val_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if args.readout_mode == "mean_party":
        return torch.stack(party_probs, dim=0).mean(dim=0)
    if args.readout_mode == "best_party":
        if y is not None and val_idx is not None:
            post_rel = evaluator_reliabilities(party_probs, y, val_idx, args)
        else:
            post_rel = reliabilities
        best = max(range(len(party_probs)), key=lambda i: (float(post_rel[i]), -i))
        return party_probs[best]
    if args.readout_mode != "active_vote":
        raise ValueError(f"Unknown readout mode: {args.readout_mode}")
    if args.vote_weighting == "topology_reliability" and edge_count(adj) > 0:
        return topology_filtered_reliability_vote(party_probs, reliabilities, adj, args)
    return vote_from_probs(party_probs, reliabilities, mode=args.vote_weighting)


def topology_prediction_consensus(
    probs: List[torch.Tensor],
    adj: np.ndarray,
    reliabilities: List[float],
    args: argparse.Namespace,
    y: Optional[torch.Tensor] = None,
    val_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    current = post_consensus_party_probs(probs, adj, reliabilities, args)
    return readout_post_consensus(current, adj, reliabilities, args, y=y, val_idx=val_idx)


def score_topology_on_validation(
    edge_set: np.ndarray,
    val_probs_by_party: List[torch.Tensor],
    y_val: torch.Tensor,
    reliabilities: List[float],
    evaluator_mode: str,
    active_party_id: int,
    args: argparse.Namespace,
) -> float:
    """Score one candidate graph at the configured validation evaluator.

    In ``active_party`` mode, this function represents the label-holding
    evaluator receiving submitted validation predictions, applying candidate
    consensus, and returning only a scalar topology score.
    """
    if evaluator_mode not in {"all_parties", "active_party"}:
        raise ValueError(f"Unknown evaluator mode: {evaluator_mode}")
    if evaluator_mode == "active_party" and not 0 <= active_party_id < len(val_probs_by_party):
        raise ValueError("--active_party_id must identify an existing party")
    objective = getattr(args, "topology_objective", "active")
    if objective not in {"active", "local_mean", "joint"}:
        raise ValueError(f"Unknown topology objective: {objective}")
    components = topology_objective_components(
        edge_set,
        val_probs_by_party,
        y_val,
        reliabilities,
        active_party_id,
        args,
    )
    if objective == "active":
        # Eq. (7): active deployment objective.
        return float(components["active"])
    if objective == "local_mean":
        # Eq. (8): local deployment objective.
        return float(components["local_mean"])
    lam = float(getattr(args, "joint_lambda", 0.5))
    if not 0.0 <= lam <= 1.0:
        raise ValueError("--joint_lambda must be in [0, 1]")
    # Eq. (9): joint active/local deployment objective.
    return float(lam * components["active"] + (1.0 - lam) * components["local_mean"])


GLOBAL_DIAGNOSTIC_METHODS = {
    "local_vote",
    "local_uniform_vote",
    "local_reliability_vote",
    "topk_reliability_vote",
    "global_topk_reliability",
}


def estimate_communication(
    method: str,
    adj: np.ndarray,
    n_nodes: int,
    n_classes: int,
    args: argparse.Namespace,
) -> Tuple[int, int, int, str]:
    """Separate peer consensus from global prediction-readout communication."""
    prediction_size = int(n_nodes * n_classes)
    n_parties = int(adj.shape[0]) if adj is not None else int(args.num_parties)

    if method in GLOBAL_DIAGNOSTIC_METHODS:
        selected = n_parties
        if method in {"topk_reliability_vote", "global_topk_reliability"}:
            selected = int(getattr(args, "_resolved_topk_reliability_k", args.topk_reliability_k))
            selected = max(1, min(selected if selected > 0 else 5, n_parties))
        global_readout = int(selected * prediction_size)
        return 0, global_readout, global_readout, "global_diagnostic"

    if method in {"single_party_avg", "single_party_best", "active_local_only"}:
        return 0, 0, 0, "local_reference"

    peer_to_peer = 0
    if args.pred_consensus_steps > 0:
        peer_to_peer = int(
            2 * edge_count(adj) * args.pred_consensus_steps * prediction_size
        )

    # The reported active-party evaluation explicitly performs a final
    # prediction-level readout. Count it separately from peer-edge exchange.
    if args.readout_mode == "active_vote":
        degrees = adj.sum(axis=1) if adj is not None else np.zeros(n_parties)
        active_count = int(np.count_nonzero(degrees > 0))
        if active_count == 0:
            active_count = n_parties
        global_readout = int(active_count * prediction_size)
    elif args.readout_mode == "mean_party":
        global_readout = int(n_parties * prediction_size)
    elif args.readout_mode == "best_party":
        global_readout = prediction_size
    elif args.readout_mode in {"local_mean", "local_worst", "active_party_local"}:
        global_readout = 0
    else:
        global_readout = 0

    return (
        peer_to_peer,
        global_readout,
        peer_to_peer + global_readout,
        "peer_to_peer_topology",
    )


def estimate_inference_comm(method: str, adj: np.ndarray, n_nodes: int, n_classes: int, args: argparse.Namespace) -> int:
    """Backward-compatible alias for peer-to-peer inference communication."""
    return estimate_communication(method, adj, n_nodes, n_classes, args)[0]


def evaluator_protocol_metadata(args: argparse.Namespace) -> Dict[str, object]:
    topology_evaluator = (
        "active_party_id"
        if args.validation_evaluator == "active_party"
        else "all_parties"
    )
    return {
        "validation_evaluator": args.validation_evaluator,
        "active_party_id": int(args.active_party_id),
        "local_training_labels": args.local_training_labels,
        "validation_label_protocol": (
            "active_party_only"
            if args.validation_evaluator == "active_party"
            else "all_parties"
        ),
        "topology_evaluator": topology_evaluator,
        "local_training_protocol": args.local_training_protocol,
        "validation_split_mode": args.validation_split_mode,
        "topology_val_fraction": float(args.topology_val_fraction),
        "setting": str(getattr(args, "setting_name", "") or (
            "hard" if int(args.useful_parties) <= 3 else "main"
        )),
        "label_protocol": args.label_protocol,
        "training_protocol": args.training_protocol,
        "validation_protocol": args.validation_protocol,
        "passive_label_access": bool(args.passive_label_access),
        "topology_objective": str(getattr(args, "topology_objective", "active")),
        "joint_lambda": float(getattr(args, "joint_lambda", 0.5)),
        "dropout_protocol": args.dropout_protocol,
        "dropout_rate": float(args.dropout_rate),
        "dropout_seed": int(args.dropout_seed),
    }


def score_topology_edges(
    parties: List[Party],
    xs: List[torch.Tensor],
    adjs: List[torch.Tensor],
    y: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[torch.Tensor], List[torch.Tensor], List[float]]:
    n = len(parties)
    probs = get_party_probs(parties, xs, adjs)
    val_probs = [prob[val_idx] for prob in probs]
    y_val = y[val_idx]
    scored_reliabilities = compute_party_reliabilities(
        val_probs,
        y_val,
        args.validation_evaluator,
        args.active_party_id,
    )
    for party, reliability in zip(parties, scored_reliabilities):
        party.reliability = reliability
    embs = get_party_embeddings(parties, xs, adjs)
    reliabilities = [p.reliability for p in parties]
    utility = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            p_i = val_probs[i]
            p_j = val_probs[j]
            acc_i = scored_reliabilities[i]
            acc_j = scored_reliabilities[j]
            reliability = 0.5 * (acc_i + acc_j)
            diversity = cosine_distance(embs[i][val_idx], embs[j][val_idx])
            disagreement = js_divergence(p_i, p_j)
            pair_probs = 0.5 * (probs[i] + probs[j])
            pair_acc = accuracy_from_probs(pair_probs, y, val_idx)
            pair_gain = pair_acc - max(acc_i, acc_j)
            pred_i = p_i.argmax(dim=1)
            pred_j = p_j.argmax(dim=1)
            correct_i = pred_i.eq(y_val)
            correct_j = pred_j.eq(y_val)
            corrective = (correct_i ^ correct_j).float().mean().item()
            both_wrong = ((~correct_i) & (~correct_j)).float().mean().item()

            # Pair score is similarity/reliability oriented.
            if args._active_adaptive_score == "pair":
                score = (
                    args.pair_acc_weight * pair_acc
                    + args.reliability_weight * reliability
                    + args.diversity_weight * diversity
                    + args.disagreement_weight * disagreement
                )
            else:
                # Complementarity / graph_val candidate score.
                score = (
                    args.pair_acc_weight * pair_acc
                    + args.pair_gain_weight * pair_gain
                    + args.reliability_weight * reliability
                    + args.corrective_weight * corrective
                    + args.diversity_weight * diversity
                    + args.disagreement_weight * disagreement
                    - args.both_wrong_weight * both_wrong
                )
            utility[i, j] = utility[j, i] = float(score)
    return utility, probs, embs, reliabilities


def select_topology_from_utility(utility: np.ndarray, max_degree: int, edge_budget: int) -> np.ndarray:
    n = utility.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    degree = np.zeros(n, dtype=np.int32)
    candidates = [(float(utility[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    candidates.sort(reverse=True)
    for _, i, j in candidates:
        if edge_count(adj) >= edge_budget:
            break
        if degree[i] < max_degree and degree[j] < max_degree:
            adj[i, j] = adj[j, i] = 1.0
            degree[i] += 1
            degree[j] += 1
    return adj


def update_adaptive_topology(
    parties: List[Party],
    xs: List[torch.Tensor],
    adjs: List[torch.Tensor],
    y: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, int]:
    n = len(parties)
    max_edges = int(n * args.max_degree // 2)
    edge_budget = min(args.adaptive_edge_budget if args.adaptive_edge_budget >= 0 else max_edges, max_edges)
    min_edges = min(args.adaptive_min_edges if args.adaptive_min_edges >= 0 else n, edge_budget)

    utility, probs, _, reliabilities = score_topology_edges(parties, xs, adjs, y, val_idx, args)

    if args._active_adaptive_score in {"pair", "complementarity"}:
        return select_topology_from_utility(utility, args.max_degree, edge_budget), 0

    # graph_val: greedy topology selection by actual validation consensus accuracy.
    candidates = [(float(utility[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    candidates.sort(reverse=True)
    if args.adaptive_candidate_edges > 0:
        candidates = candidates[: args.adaptive_candidate_edges]

    adj = np.zeros((n, n), dtype=np.float32)
    degree = np.zeros(n, dtype=np.int32)
    topology_eval_count = 0

    def val_score(candidate_adj: np.ndarray) -> float:
        return score_topology_on_validation(
            candidate_adj,
            [prob[val_idx] for prob in probs],
            y[val_idx],
            reliabilities,
            args.validation_evaluator,
            args.active_party_id,
            args,
        )

    current_val = val_score(adj)
    while edge_count(adj) < edge_budget:
        best_edge = None
        best_val = -1.0
        best_util = -1e30
        for util, i, j in candidates:
            if adj[i, j] > 0:
                continue
            if degree[i] >= args.max_degree or degree[j] >= args.max_degree:
                continue
            trial = adj.copy()
            trial[i, j] = trial[j, i] = 1.0
            topology_eval_count += 1
            trial_val = val_score(trial)
            if (trial_val > best_val + 1e-12) or (abs(trial_val - best_val) <= 1e-12 and util > best_util):
                best_val = trial_val
                best_util = util
                best_edge = (i, j)
        if best_edge is None:
            break
        must_fill = edge_count(adj) < min_edges
        improves = best_val >= current_val + args.adaptive_min_gain
        if not must_fill and not improves:
            break
        i, j = best_edge
        adj[i, j] = adj[j, i] = 1.0
        degree[i] += 1
        degree[j] += 1
        current_val = best_val
    return adj, topology_eval_count


# =============================================================================
# Prediction cache
# =============================================================================


CACHE_VERSION = 3
LEGACY_STANDARD_CACHE_VERSION = 2
V3_STANDARD_CACHE_DEFAULTS = {
    "modality_setting": None,
    "modalities": None,
    "modality_dim": None,
    "cache_namespace": "standard",
}
CACHE_CONFIG_KEYS = (
    "dataset",
    "target_node",
    "data_dir",
    "ignore_dataset_split",
    "graph_views",
    "graph_k",
    "max_metapath_edges",
    "random_graph_degree",
    "num_parties",
    "useful_parties",
    "view_setting",
    "distractor_feature_noise",
    "shuffle_party_positions",
    "party_shuffle_seed_offset",
    "train_ratio",
    "val_ratio",
    "epochs",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "modality_setting",
    "modalities",
    "modality_dim",
)


def prediction_cache_training_config(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    config = {key: getattr(args, key, None) for key in CACHE_CONFIG_KEYS}
    config.update(
        {
            "seed": int(seed),
            "model_type": getattr(args, "cache_model_type", "LocalGCN"),
            "local_training_seed": int(seed + 1000),
            "feature_construction": getattr(
                args,
                "cache_feature_construction",
                "vertical_feature_split+private_graph_view",
            ),
            "cache_namespace": getattr(args, "cache_namespace", "standard"),
        }
    )
    local_training_protocol = getattr(
        args,
        "local_training_protocol",
        "all_parties_supervised",
    )
    if local_training_protocol != "all_parties_supervised":
        config["local_training_protocol"] = local_training_protocol
        config["training_active_party_id"] = int(args.active_party_id)
    return config


def stable_fingerprint(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_name_seed(name: str, modulo: int = 10000) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo


def split_fingerprint(train_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor) -> str:
    payload = {
        "train_idx": train_idx.detach().cpu().tolist(),
        "val_idx": val_idx.detach().cpu().tolist(),
        "test_idx": test_idx.detach().cpu().tolist(),
    }
    return stable_fingerprint(payload)


def prediction_cache_path(args: argparse.Namespace, seed: int) -> Path:
    dataset = args.dataset.lower()
    setting = args.view_setting.lower()
    namespace = str(getattr(args, "cache_namespace", "standard")).strip().lower()
    namespace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in namespace)
    prefix = "" if namespace in {"", "standard"} else f"{namespace}_"
    party_tag = "" if int(args.num_parties) == 15 else f"_n{int(args.num_parties)}"
    filename = (
        f"{prefix}{dataset}_{setting}{party_tag}_useful{args.useful_parties}"
        f"_seed{seed}_party_preds.pt"
    )
    return Path(args.cache_dir) / filename


def validation_protocol_positions(
    y_val: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return topology-label and epoch-selection positions within validation.

    ``shared`` uses a stratified label budget for topology selection while
    retaining the complete validation set for epoch selection. ``disjoint``
    reserves complementary examples for epoch selection, preventing topology
    search and model selection from reusing the same labels.
    """
    mode = str(args.validation_split_mode)
    fraction = float(args.topology_val_fraction)
    if mode not in {"shared", "disjoint"}:
        raise ValueError("--validation_split_mode must be shared or disjoint")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--topology_val_fraction must be in (0, 1]")
    if mode == "disjoint" and fraction >= 1.0:
        raise ValueError("disjoint validation requires --topology_val_fraction < 1")

    labels = y_val.detach().cpu().numpy()
    rng = np.random.default_rng(seed + int(args.validation_split_seed_offset))
    selected: List[int] = []
    for label in sorted(np.unique(labels).tolist()):
        positions = np.where(labels == label)[0]
        rng.shuffle(positions)
        if mode == "disjoint" and len(positions) > 1:
            count = min(len(positions) - 1, max(1, int(round(len(positions) * fraction))))
        else:
            count = max(1, int(round(len(positions) * fraction)))
        selected.extend(int(value) for value in positions[:count])
    topology = torch.tensor(sorted(selected), dtype=torch.long, device=y_val.device)
    if mode == "shared":
        selection = torch.arange(y_val.numel(), dtype=torch.long, device=y_val.device)
    else:
        mask = torch.ones(y_val.numel(), dtype=torch.bool, device=y_val.device)
        mask[topology] = False
        selection = torch.where(mask)[0]
        if selection.numel() == 0:
            raise ValueError("disjoint validation produced an empty selection split")
    return topology, selection


def train_local_parties_one_epoch(
    parties: List[Party],
    xs: List[torch.Tensor],
    adjs: List[torch.Tensor],
    y: torch.Tensor,
    train_idx: torch.Tensor,
    args: argparse.Namespace,
) -> int:
    """Train one local epoch and return label-control communication scalars.

    In ``active_party_logit_gradient`` mode, passive parties submit train-node
    logits to the label-holding active party. The active party computes
    cross-entropy gradients and returns only gradients with respect to those
    logits. Passive-party model code never receives labels.
    """
    protocol = str(args.local_training_protocol)
    if protocol not in {"all_parties_supervised", "active_party_logit_gradient"}:
        raise ValueError(f"Unknown local training protocol: {protocol}")
    communication = 0
    y_train = y[train_idx]
    for i, party in enumerate(parties):
        party.model.train()
        party.optimizer.zero_grad()
        logits = party.model(xs[i], adjs[i])
        submitted_logits = logits[train_idx]
        if protocol == "all_parties_supervised" or i == int(args.active_party_id):
            loss = F.cross_entropy(submitted_logits, y_train)
            loss.backward()
        else:
            evaluator_logits = submitted_logits.detach().requires_grad_(True)
            evaluator_loss = F.cross_entropy(evaluator_logits, y_train)
            returned_gradient = torch.autograd.grad(
                evaluator_loss,
                evaluator_logits,
            )[0]
            submitted_logits.backward(returned_gradient)
            communication += 2 * submitted_logits.numel()
        party.optimizer.step()
    return int(communication)


def validate_prediction_cache(
    cache: Dict[str, object],
    expected_config: Dict[str, object],
    path: Path,
) -> None:
    if int(cache.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"Prediction cache version mismatch for {path}: "
            f"found {cache.get('cache_version')}, expected {CACHE_VERSION}."
        )
    actual = cache.get("training_config")
    if not isinstance(actual, dict):
        raise ValueError(f"Prediction cache {path} has no valid training_config.")
    mismatches = []
    for key, expected in expected_config.items():
        found = actual.get(key)
        if found != expected:
            mismatches.append(f"{key}: cache={found!r}, requested={expected!r}")
    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise ValueError(f"Prediction cache {path} does not match local-training config:\n  - {detail}")
    train_idx = cache.get("train_idx")
    val_idx = cache.get("val_idx")
    test_idx = cache.get("test_idx")
    if not all(isinstance(value, torch.Tensor) for value in (train_idx, val_idx, test_idx)):
        raise ValueError(f"Prediction cache {path} has invalid split indices.")
    actual_split_fingerprint = split_fingerprint(train_idx, val_idx, test_idx)
    if actual_split_fingerprint != cache.get("split_fingerprint"):
        raise ValueError(f"Prediction cache {path} split indices failed their fingerprint check.")
    expected_fingerprint = stable_fingerprint(
        {
            "training_config": actual,
            "split_fingerprint": cache.get("split_fingerprint"),
        }
    )
    if cache.get("cache_fingerprint") != expected_fingerprint:
        raise ValueError(f"Prediction cache {path} failed its fingerprint integrity check.")


def migrate_legacy_standard_prediction_cache(
    cache: Dict[str, object],
    expected_config: Dict[str, object],
    path: Path,
) -> Dict[str, object]:
    """Upgrade metadata-only v2 standard caches after strict compatibility checks."""
    if int(cache.get("cache_version", -1)) != LEGACY_STANDARD_CACHE_VERSION:
        return cache
    actual = cache.get("training_config")
    if not isinstance(actual, dict):
        raise ValueError(f"Legacy prediction cache {path} has no valid training_config.")
    for key, required in V3_STANDARD_CACHE_DEFAULTS.items():
        if expected_config.get(key) != required:
            raise ValueError(
                f"Legacy prediction cache {path} cannot be migrated for {key}="
                f"{expected_config.get(key)!r}; only standard non-multimodal caches are supported."
            )
    for key, expected in expected_config.items():
        if key in V3_STANDARD_CACHE_DEFAULTS:
            continue
        found = actual.get(key)
        if found != expected:
            raise ValueError(
                f"Legacy prediction cache {path} does not match local-training config: "
                f"{key}: cache={found!r}, requested={expected!r}"
            )

    train_idx = cache.get("train_idx")
    val_idx = cache.get("val_idx")
    test_idx = cache.get("test_idx")
    if not all(isinstance(value, torch.Tensor) for value in (train_idx, val_idx, test_idx)):
        raise ValueError(f"Legacy prediction cache {path} has invalid split indices.")
    split_hash = split_fingerprint(train_idx, val_idx, test_idx)
    if split_hash != cache.get("split_fingerprint"):
        raise ValueError(f"Legacy prediction cache {path} split indices failed their fingerprint check.")
    legacy_fingerprint = stable_fingerprint(
        {
            "training_config": actual,
            "split_fingerprint": split_hash,
        }
    )
    if legacy_fingerprint != cache.get("cache_fingerprint"):
        raise ValueError(f"Legacy prediction cache {path} failed its fingerprint integrity check.")

    migrated = dict(cache)
    migrated_config = dict(actual)
    migrated_config.update(V3_STANDARD_CACHE_DEFAULTS)
    migrated["cache_version"] = CACHE_VERSION
    migrated["training_config"] = migrated_config
    migrated["cache_fingerprint"] = stable_fingerprint(
        {
            "training_config": migrated_config,
            "split_fingerprint": split_hash,
        }
    )
    validate_prediction_cache(migrated, expected_config, path)
    return migrated


def load_prediction_cache(path: Path, expected_config: Dict[str, object]) -> Dict[str, object]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError(f"Prediction cache {path} is not a dictionary.")
    if int(cache.get("cache_version", -1)) == LEGACY_STANDARD_CACHE_VERSION:
        cache = migrate_legacy_standard_prediction_cache(cache, expected_config, path)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(cache, temporary_path)
        temporary_path.replace(path)
        print(
            f"Migrated compatible standard prediction cache {path} "
            f"from v{LEGACY_STANDARD_CACHE_VERSION} to v{CACHE_VERSION} "
            f"(fingerprint={str(cache['cache_fingerprint'])[:12]})."
        )
    validate_prediction_cache(cache, expected_config, path)
    return cache


def train_local_prediction_cache(
    seed: int,
    args: argparse.Namespace,
    x: torch.Tensor,
    y: torch.Tensor,
    xs: List[torch.Tensor],
    adjs: List[torch.Tensor],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    target: str,
    useful_party_mask: Sequence[bool],
    view_names: Sequence[str],
    party_permutation: Sequence[int],
) -> Dict[str, object]:
    if args.lambda_kd != 0:
        raise ValueError("Prediction caching requires --lambda_kd 0 because local models must train independently.")

    set_seed(seed + 1000)
    num_classes = int(y[y >= 0].max().item() + 1)
    parties = init_parties(xs, args.hidden_dim, num_classes, args.dropout, args.lr, args.weight_decay)
    val_epochs: List[torch.Tensor] = []
    test_epochs: List[torch.Tensor] = []
    training_control_comm = 0

    for epoch in range(1, args.epochs + 1):
        training_control_comm += train_local_parties_one_epoch(
            parties,
            xs,
            adjs,
            y,
            train_idx,
            args,
        )

        probs = get_party_probs(parties, xs, adjs)
        val_epochs.append(torch.stack([prob[val_idx].detach().cpu() for prob in probs], dim=0))
        test_epochs.append(torch.stack([prob[test_idx].detach().cpu() for prob in probs], dim=0))
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(f"[cache/{args.dataset}] epoch={epoch:03d}/{args.epochs}")

    final_probs = get_party_probs(parties, xs, adjs)
    probs_all = torch.stack([prob.detach().cpu() for prob in final_probs], dim=0)
    probs_train = probs_all[:, train_idx.detach().cpu()]
    probs_val = probs_all[:, val_idx.detach().cpu()]
    probs_test = probs_all[:, test_idx.detach().cpu()]
    reliabilities = [
        accuracy_from_probs(prob, y, val_idx)
        for prob in final_probs
    ]
    training_config = prediction_cache_training_config(args, seed)
    split_hash = split_fingerprint(train_idx, val_idx, test_idx)
    fingerprint = stable_fingerprint({"training_config": training_config, "split_fingerprint": split_hash})
    return {
        "cache_version": CACHE_VERSION,
        "cache_fingerprint": fingerprint,
        "training_config": training_config,
        "split_fingerprint": split_hash,
        "dataset": args.dataset,
        "seed": int(seed),
        "target_node_type": target,
        "train_idx": train_idx.detach().cpu(),
        "val_idx": val_idx.detach().cpu(),
        "test_idx": test_idx.detach().cpu(),
        "y": y.detach().cpu(),
        "probs_train": probs_train,
        "probs_val": probs_val,
        "probs_test": probs_test,
        "probs_all": probs_all,
        "probs_val_epochs": torch.stack(val_epochs, dim=0),
        "probs_test_epochs": torch.stack(test_epochs, dim=0),
        "party_reliabilities": torch.tensor(reliabilities, dtype=torch.float32),
        "party_permutation": list(party_permutation),
        "useful_party_mask": [bool(value) for value in useful_party_mask],
        "view_names": list(view_names),
        "num_classes": num_classes,
        "num_parties": len(xs),
        "target_node_count": int(x.size(0)),
        "local_training_protocol": args.local_training_protocol,
        "local_training_labels": args.local_training_labels,
        "training_active_party_id": int(args.active_party_id),
        "training_control_comm": int(training_control_comm),
    }


def save_prediction_cache(cache: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(
        f"Saved prediction cache to {path} "
        f"(fingerprint={str(cache['cache_fingerprint'])[:12]})"
    )


def cached_edge_utility(
    probs_val: List[torch.Tensor],
    y_val: torch.Tensor,
    reliabilities: List[float],
    args: argparse.Namespace,
) -> np.ndarray:
    if args.diversity_weight != 0:
        raise ValueError(
            "Cached topology evaluation currently requires --diversity_weight 0 "
            "because encoder embeddings are intentionally not cached."
        )
    n = len(probs_val)
    utility = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            p_i = probs_val[i]
            p_j = probs_val[j]
            acc_i = float((p_i.argmax(dim=1) == y_val).float().mean().item())
            acc_j = float((p_j.argmax(dim=1) == y_val).float().mean().item())
            reliability = 0.5 * (acc_i + acc_j)
            disagreement = js_divergence(p_i, p_j)
            pair_probs = 0.5 * (p_i + p_j)
            pair_acc = float((pair_probs.argmax(dim=1) == y_val).float().mean().item())
            pair_gain = pair_acc - max(acc_i, acc_j)
            correct_i = p_i.argmax(dim=1).eq(y_val)
            correct_j = p_j.argmax(dim=1).eq(y_val)
            corrective = (correct_i ^ correct_j).float().mean().item()
            both_wrong = ((~correct_i) & (~correct_j)).float().mean().item()
            if args._active_adaptive_score == "pair":
                score = (
                    args.pair_acc_weight * pair_acc
                    + args.reliability_weight * reliability
                    + args.disagreement_weight * disagreement
                )
            else:
                score = (
                    args.pair_acc_weight * pair_acc
                    + args.pair_gain_weight * pair_gain
                    + args.reliability_weight * reliability
                    + args.corrective_weight * corrective
                    + args.disagreement_weight * disagreement
                    - args.both_wrong_weight * both_wrong
                )
            utility[i, j] = utility[j, i] = float(score)
    return utility


def update_adaptive_topology_from_cache(
    probs_val: List[torch.Tensor],
    y_val: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, int]:
    n = len(probs_val)
    max_edges = int(n * args.max_degree // 2)
    edge_budget = min(args.adaptive_edge_budget if args.adaptive_edge_budget >= 0 else max_edges, max_edges)
    min_edges = min(args.adaptive_min_edges if args.adaptive_min_edges >= 0 else n, edge_budget)
    reliabilities = compute_party_reliabilities(
        probs_val,
        y_val,
        args.validation_evaluator,
        args.active_party_id,
    )
    utility = cached_edge_utility(probs_val, y_val, reliabilities, args)
    if args._active_adaptive_score in {"pair", "complementarity"}:
        return select_topology_from_utility(utility, args.max_degree, edge_budget), 0

    candidates = [(float(utility[i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    candidates.sort(reverse=True)
    if args.adaptive_candidate_edges > 0:
        candidates = candidates[: args.adaptive_candidate_edges]
    adj = np.zeros((n, n), dtype=np.float32)
    degree = np.zeros(n, dtype=np.int32)
    topology_eval_count = 0
    # Algorithm 1 / S1: greedily evaluate feasible one-edge extensions on validation labels.
    def val_score(candidate_adj: np.ndarray) -> float:
        return score_topology_on_validation(
            candidate_adj,
            probs_val,
            y_val,
            reliabilities,
            args.validation_evaluator,
            args.active_party_id,
            args,
        )

    current_val = val_score(adj)
    while edge_count(adj) < edge_budget:
        best_edge = None
        best_val = -1.0
        best_util = -1e30
        for util, i, j in candidates:
            if adj[i, j] > 0:
                continue
            if degree[i] >= args.max_degree or degree[j] >= args.max_degree:
                continue
            trial = adj.copy()
            trial[i, j] = trial[j, i] = 1.0
            topology_eval_count += 1
            trial_val = val_score(trial)
            if (trial_val > best_val + 1e-12) or (abs(trial_val - best_val) <= 1e-12 and util > best_util):
                best_val = trial_val
                best_util = util
                best_edge = (i, j)
        if best_edge is None:
            break
        must_fill = edge_count(adj) < min_edges
        improves = best_val >= current_val + args.adaptive_min_gain
        if not must_fill and not improves:
            break
        i, j = best_edge
        adj[i, j] = adj[j, i] = 1.0
        degree[i] += 1
        degree[j] += 1
        current_val = best_val
    return adj, topology_eval_count


def cached_readout_pair(
    probs_val: List[torch.Tensor],
    probs_test: List[torch.Tensor],
    adj: np.ndarray,
    reliabilities: List[float],
    y_val: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    post_val = post_consensus_party_probs(probs_val, adj, reliabilities, args)
    post_test = post_consensus_party_probs(probs_test, adj, reliabilities, args)
    if args.readout_mode == "mean_party":
        return torch.stack(post_val).mean(dim=0), torch.stack(post_test).mean(dim=0)
    if args.readout_mode == "best_party":
        post_rel = [
            float((prob.argmax(dim=1) == y_val).float().mean().item())
            for prob in post_val
        ]
        best = max(range(len(post_val)), key=lambda i: (post_rel[i], -i))
        return post_val[best], post_test[best]
    if args.vote_weighting == "topology_reliability" and edge_count(adj) > 0:
        return (
            topology_filtered_reliability_vote(post_val, reliabilities, adj, args),
            topology_filtered_reliability_vote(post_test, reliabilities, adj, args),
        )
    return (
        vote_from_probs(post_val, reliabilities, mode=args.vote_weighting),
        vote_from_probs(post_test, reliabilities, mode=args.vote_weighting),
    )


def evaluate_cached_method(
    method: str,
    seed: int,
    args: argparse.Namespace,
    cache: Dict[str, object],
    cache_path: Path,
) -> RunResult:
    if method in {"central_fusion", "central_full_gcn"}:
        raise ValueError(f"{method} cannot be evaluated from local prediction caches.")

    val_epochs = cache["probs_val_epochs"]
    test_epochs = cache["probs_test_epochs"]
    if not isinstance(val_epochs, torch.Tensor) or not isinstance(test_epochs, torch.Tensor):
        raise ValueError(f"Cache {cache_path} is missing per-epoch validation/test predictions.")
    y = cache["y"]
    val_idx = cache["val_idx"]
    test_idx = cache["test_idx"]
    assert isinstance(y, torch.Tensor) and isinstance(val_idx, torch.Tensor) and isinstance(test_idx, torch.Tensor)
    y_val = y[val_idx].to(DEVICE)
    y_test = y[test_idx].to(DEVICE)
    topology_positions, selection_positions = validation_protocol_positions(
        y_val,
        args,
        seed,
    )
    y_topology = y_val[topology_positions]
    y_selection = y_val[selection_positions]
    num_classes = int(cache["num_classes"])
    n = int(cache["num_parties"])
    n_nodes = int(cache["target_node_count"])
    useful_party_mask = [bool(value) for value in cache["useful_party_mask"]]
    view_names = list(cache["view_names"])
    party_permutation = list(cache["party_permutation"])
    local_methods = {
        "single_party_avg",
        "single_party_best",
        "local_vote",
        "local_uniform_vote",
        "local_reliability_vote",
        "topk_reliability_vote",
        "global_topk_reliability",
        "active_local_only",
    }
    collab_adj = method_to_initial_topology(method, n, args, seed)
    best_val = test_at_best = f1_at_best = best_observed = final_test = -1.0
    topology_update_comm = topology_eval_count = 0
    topology_update_time = 0.0
    topology_updates_done = 0
    final_reliabilities: List[float] = [1.0] * n
    best_readout_metrics: Dict[str, object] = {
        "active_readout_acc": -1.0,
        "local_post_consensus_mean_acc": -1.0,
        "active_party_post_consensus_acc": -1.0,
        "local_worst_acc": -1.0,
        "local_best_acc": -1.0,
        "remaining_parties": n,
        "dropped_party_ids": [],
        "adj": collab_adj,
    }

    for epoch_index in range(val_epochs.size(0)):
        epoch = epoch_index + 1
        probs_val = [tensor.to(DEVICE) for tensor in val_epochs[epoch_index]]
        probs_topology = [prob[topology_positions] for prob in probs_val]
        probs_selection = [prob[selection_positions] for prob in probs_val]
        probs_test = [tensor.to(DEVICE) for tensor in test_epochs[epoch_index]]
        reliabilities = compute_party_reliabilities(
            probs_topology,
            y_topology,
            args.validation_evaluator,
            args.active_party_id,
        )
        final_reliabilities = reliabilities

        if method == "best_matched_sparse":
            candidates = [
                make_ring_like_topology(n, args.matched_edge_count, seed),
                make_random_matched_topology(
                    n,
                    args.max_degree,
                    args.matched_edge_count,
                    seed,
                ),
                make_expander_matched_topology(n, args.matched_edge_count, seed),
            ]
            candidate_scores = [
                score_topology_on_validation(
                    candidate,
                    probs_topology,
                    y_topology,
                    reliabilities,
                    args.validation_evaluator,
                    args.active_party_id,
                    args,
                )
                for candidate in candidates
            ]
            collab_adj = candidates[int(np.argmax(candidate_scores))]
            topology_eval_count += len(candidates)
            topology_update_comm += n * y_topology.numel() * num_classes

        update_allowed = (
            int(args.topology_max_updates) <= 0
            or topology_updates_done < int(args.topology_max_updates)
        )
        if (
            method in {"similarity", "adaptive", "adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"}
            and epoch % args.topology_every == 0
            and update_allowed
        ):
            old_score = getattr(args, "_active_adaptive_score", args.adaptive_score)
            args._active_adaptive_score = active_adaptive_score_for_method(method, args.adaptive_score)
            synchronize_device()
            start = time.perf_counter()
            collab_adj, evaluated = update_adaptive_topology_from_cache(
                probs_topology,
                y_topology,
                args,
            )
            synchronize_device()
            topology_update_time += time.perf_counter() - start
            topology_eval_count += evaluated
            topology_update_comm += n * y_topology.numel() * num_classes
            topology_updates_done += 1
            args._active_adaptive_score = old_score

        if method == "single_party_avg":
            val_acc = float(np.mean([
                (prob.argmax(dim=1) == y_selection).float().mean().item()
                for prob in probs_selection
            ]))
            tests = [
                (prob.argmax(dim=1) == y_test).float().mean().item()
                for prob in probs_test
            ]
            f1s = [
                macro_f1_from_probs(prob, y_test, torch.arange(y_test.numel(), device=DEVICE), num_classes)
                for prob in probs_test
            ]
            test_acc, test_f1 = float(np.mean(tests)), float(np.mean(f1s))
        elif method == "single_party_best":
            values = [
                (prob.argmax(dim=1) == y_selection).float().mean().item()
                for prob in probs_selection
            ]
            best_party = int(np.argmax(values))
            val_acc = float(values[best_party])
            test_acc = float((probs_test[best_party].argmax(dim=1) == y_test).float().mean().item())
            test_f1 = macro_f1_from_probs(
                probs_test[best_party],
                y_test,
                torch.arange(y_test.numel(), device=DEVICE),
                num_classes,
            )
        elif method in local_methods:
            if method == "active_local_only":
                active = int(args.active_party_id)
                val_vote = probs_selection[active]
                test_vote = probs_test[active]
            elif method in {"topk_reliability_vote", "global_topk_reliability"}:
                k = int(getattr(args, "_resolved_topk_reliability_k", args.topk_reliability_k))
                k = k if k > 0 else 5
                val_vote = topk_reliability_vote(probs_selection, reliabilities, k)
                test_vote = topk_reliability_vote(probs_test, reliabilities, k)
            else:
                mode = "uniform" if method in {"local_vote", "local_uniform_vote"} else "reliability"
                val_vote = vote_from_probs(probs_selection, reliabilities, mode=mode)
                test_vote = vote_from_probs(probs_test, reliabilities, mode=mode)
            val_acc = float((val_vote.argmax(dim=1) == y_selection).float().mean().item())
            test_acc = float((test_vote.argmax(dim=1) == y_test).float().mean().item())
            test_f1 = macro_f1_from_probs(
                test_vote,
                y_test,
                torch.arange(y_test.numel(), device=DEVICE),
                num_classes,
            )
        else:
            if args.readout_mode in {"local_mean", "local_worst", "active_party_local"}:
                post_val = post_consensus_party_probs(
                    probs_selection,
                    collab_adj,
                    reliabilities,
                    args,
                )
                post_test = post_consensus_party_probs(
                    probs_test,
                    collab_adj,
                    reliabilities,
                    args,
                )
                val_scores = [
                    float((prob.argmax(dim=1) == y_selection).float().mean().item())
                    for prob in post_val
                ]
                test_scores = [
                    float((prob.argmax(dim=1) == y_test).float().mean().item())
                    for prob in post_test
                ]
                test_f1s = [
                    macro_f1_from_probs(
                        prob,
                        y_test,
                        torch.arange(y_test.numel(), device=DEVICE),
                        num_classes,
                    )
                    for prob in post_test
                ]
                if args.readout_mode == "local_mean":
                    val_acc = float(np.mean(val_scores))
                    test_acc = float(np.mean(test_scores))
                    test_f1 = float(np.mean(test_f1s))
                elif args.readout_mode == "local_worst":
                    chosen = int(np.argmin(val_scores))
                    val_acc = float(val_scores[chosen])
                    test_acc = float(test_scores[chosen])
                    test_f1 = float(test_f1s[chosen])
                else:
                    chosen = int(args.active_party_id)
                    val_acc = float(val_scores[chosen])
                    test_acc = float(test_scores[chosen])
                    test_f1 = float(test_f1s[chosen])
            else:
                val_vote, test_vote = cached_readout_pair(
                    probs_selection,
                    probs_test,
                    collab_adj,
                    reliabilities,
                    y_selection,
                    args,
                )
                val_acc = float((val_vote.argmax(dim=1) == y_selection).float().mean().item())
                test_acc = float((test_vote.argmax(dim=1) == y_test).float().mean().item())
                test_f1 = macro_f1_from_probs(
                    test_vote,
                    y_test,
                    torch.arange(y_test.numel(), device=DEVICE),
                    num_classes,
                )

        readout_metrics = inference_readout_metrics(
            method,
            probs_test,
            y_test,
            collab_adj,
            reliabilities,
            args,
            seed,
        )
        if args.dropout_protocol == "random_party_dropout":
            test_acc = float(readout_metrics["active_readout_acc"])
            test_f1 = -1.0

        final_test = test_acc
        best_observed = max(best_observed, test_acc)
        if val_acc > best_val:
            best_val = val_acc
            test_at_best = test_acc
            f1_at_best = test_f1
            best_readout_metrics = readout_metrics

    reported_adj = np.asarray(best_readout_metrics["adj"], dtype=np.float32)
    remaining_mask = [
        useful_party_mask[i]
        for i in range(n)
        if i not in set(best_readout_metrics["dropped_party_ids"])
    ]
    selected_edges = topology_edge_list(reported_adj)
    useful_useful, useful_noisy, noisy_noisy = topology_edge_types(
        reported_adj,
        remaining_mask,
    )
    peer_comm, readout_comm, total_comm, protocol_type = estimate_communication(
        method, reported_adj, n_nodes, num_classes, args
    )
    return RunResult(
        seed=seed,
        dataset=str(cache["dataset"]),
        method=method,
        best_val=best_val,
        test_at_best_val=test_at_best,
        macro_f1_at_best_val=f1_at_best,
        best_observed_test=best_observed,
        final_test=final_test,
        train_comm=int(cache.get("training_control_comm", 0)),
        inference_comm=peer_comm,
        final_edges=edge_count(reported_adj),
        final_adj=reported_adj.astype(int),
        peer_to_peer_comm=peer_comm,
        global_readout_comm=readout_comm,
        total_comm=total_comm,
        protocol_type=protocol_type,
        **evaluator_protocol_metadata(args),
        topology_val_count=int(y_topology.numel()),
        selection_val_count=int(y_selection.numel()),
        topology_update_comm=topology_update_comm,
        topology_eval_count=topology_eval_count,
        topology_update_time=topology_update_time,
        active_readout_acc=float(best_readout_metrics["active_readout_acc"]),
        local_post_consensus_mean_acc=float(
            best_readout_metrics["local_post_consensus_mean_acc"]
        ),
        active_party_post_consensus_acc=float(
            best_readout_metrics["active_party_post_consensus_acc"]
        ),
        local_worst_acc=float(best_readout_metrics["local_worst_acc"]),
        local_best_acc=float(best_readout_metrics["local_best_acc"]),
        remaining_parties=int(best_readout_metrics["remaining_parties"]),
        dropped_party_ids=json.dumps(best_readout_metrics["dropped_party_ids"]),
        test_acc_after_dropout=(
            float(best_readout_metrics["active_readout_acc"])
            if args.dropout_protocol == "random_party_dropout"
            else -1.0
        ),
        local_post_consensus_acc_after_dropout=(
            float(best_readout_metrics["local_post_consensus_mean_acc"])
            if args.dropout_protocol == "random_party_dropout"
            else -1.0
        ),
        control_plane_comm=topology_update_comm,
        useful_useful_edges=useful_useful,
        useful_noisy_edges=useful_noisy,
        noisy_noisy_edges=noisy_noisy,
        selected_edges=json.dumps(selected_edges, separators=(",", ":")),
        useful_party_mask=json.dumps(useful_party_mask),
        party_view_names=json.dumps(view_names),
        party_reliabilities=json.dumps(final_reliabilities),
        party_permutation=json.dumps(party_permutation),
        adaptive_min_edges=int(args.adaptive_min_edges),
        prediction_cache_path=str(cache_path),
        prediction_cache_fingerprint=str(cache["cache_fingerprint"]),
    )


# =============================================================================
# Training
# =============================================================================


def init_parties(xs: List[torch.Tensor], hidden_dim: int, num_classes: int, dropout: float, lr: float, weight_decay: float) -> List[Party]:
    parties = []
    for x in xs:
        model = LocalGCN(x.size(1), hidden_dim, num_classes, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        parties.append(Party(model, opt))
    return parties


def evaluate_local_baseline(method: str, parties: List[Party], xs: List[torch.Tensor], adjs: List[torch.Tensor], y: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor, num_classes: int, args: argparse.Namespace) -> Tuple[float, float, float]:
    probs = get_party_probs(parties, xs, adjs)
    rel = [p.reliability for p in parties]

    if method == "single_party_avg":
        vals, tests, f1s = [], [], []
        for prob in probs:
            vals.append(accuracy_from_probs(prob, y, val_idx))
            tests.append(accuracy_from_probs(prob, y, test_idx))
            f1s.append(macro_f1_from_probs(prob, y, test_idx, num_classes))
        return float(np.mean(vals)), float(np.mean(tests)), float(np.mean(f1s))

    if method == "single_party_best":
        vals = [accuracy_from_probs(prob, y, val_idx) for prob in probs]
        best = int(np.argmax(vals))
        return vals[best], accuracy_from_probs(probs[best], y, test_idx), macro_f1_from_probs(probs[best], y, test_idx, num_classes)

    if method in {"topk_reliability_vote", "global_topk_reliability"}:
        configured_k = int(getattr(args, "_resolved_topk_reliability_k", args.topk_reliability_k))
        k = configured_k if configured_k > 0 else 5
        vote = topk_reliability_vote(probs, rel, k)
        return accuracy_from_probs(vote, y, val_idx), accuracy_from_probs(vote, y, test_idx), macro_f1_from_probs(vote, y, test_idx, num_classes)

    vote_mode = "uniform" if method in {"local_vote", "local_uniform_vote"} else "reliability"
    vote = vote_from_probs(probs, rel, mode=vote_mode)
    return accuracy_from_probs(vote, y, val_idx), accuracy_from_probs(vote, y, test_idx), macro_f1_from_probs(vote, y, test_idx, num_classes)


def method_to_initial_topology(method: str, n: int, args: argparse.Namespace, seed: int) -> np.ndarray:
    max_edges = n * (n - 1) // 2
    budget = args.adaptive_edge_budget if args.adaptive_edge_budget >= 0 else int(n * args.max_degree // 2)
    budget = max(0, min(budget, max_edges))
    if method == "fixed_ring":
        return make_ring_topology(n)
    if method == "fixed_ring_matched":
        return make_ring_like_topology(n, args.matched_edge_count, seed)
    if method == "full_mesh":
        return make_full_topology(n)
    if method == "random_regular":
        return make_random_regular_topology(n, args.max_degree, budget, seed)
    if method == "random_matched":
        return make_random_matched_topology(n, args.max_degree, args.matched_edge_count, seed)
    if method == "expander":
        return make_expander_like_topology(n, budget)
    if method == "expander_matched":
        return make_expander_matched_topology(n, args.matched_edge_count, seed)
    if method == "best_matched_sparse":
        return make_ring_like_topology(n, args.matched_edge_count, seed)
    if method in {"adaptive", "adaptive_pair", "adaptive_complementarity", "adaptive_graph_val", "similarity"}:
        return make_ring_topology(n)
    return np.zeros((n, n), dtype=np.float32)


def active_adaptive_score_for_method(method: str, default: str) -> str:
    if method == "adaptive_pair" or method == "similarity":
        return "pair"
    if method == "adaptive_complementarity":
        return "complementarity"
    if method == "adaptive_graph_val" or method == "adaptive":
        return "graph_val"
    return default


def train_decentralized(
    method: str,
    seed: int,
    args: argparse.Namespace,
    dataset_name: str,
    xs: List[torch.Tensor],
    adjs: List[torch.Tensor],
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    useful_party_mask: Sequence[bool],
    view_names: Sequence[str],
    party_permutation: Sequence[int],
) -> RunResult:
    set_seed(seed + 1000)
    num_classes = int(y[y >= 0].max().item() + 1)
    parties = init_parties(xs, args.hidden_dim, num_classes, args.dropout, args.lr, args.weight_decay)
    n = len(parties)
    local_methods = {
        "single_party_avg",
        "single_party_best",
        "local_vote",
        "local_uniform_vote",
        "local_reliability_vote",
        "topk_reliability_vote",
        "global_topk_reliability",
        "active_local_only",
    }
    collab_adj = method_to_initial_topology(method, n, args, seed)

    best_val, test_at_best, f1_at_best, best_observed, final_test = -1.0, -1.0, -1.0, -1.0, -1.0
    train_comm = 0
    topology_update_comm = 0
    topology_eval_count = 0
    topology_update_time = 0.0

    for epoch in range(1, args.epochs + 1):
        neighbor_probs = get_party_probs(parties, xs, adjs) if method not in local_methods else None

        for i, party in enumerate(parties):
            party.model.train()
            party.optimizer.zero_grad()
            logits = party.model(xs[i], adjs[i])
            loss = F.cross_entropy(logits[train_idx], y[train_idx])

            if method not in local_methods and args.lambda_kd > 0 and neighbor_probs is not None:
                neigh = np.where(collab_adj[i] > 0)[0].tolist()
                if neigh:
                    target = torch.stack([neighbor_probs[j] for j in neigh], dim=0).mean(dim=0).detach()
                    kd = F.kl_div(F.log_softmax(logits[train_idx], dim=1), target[train_idx], reduction="batchmean")
                    loss = loss + args.lambda_kd * kd
            loss.backward()
            party.optimizer.step()

        # Update topology.
        if method in {"similarity", "adaptive", "adaptive_pair", "adaptive_complementarity", "adaptive_graph_val"} and epoch % args.topology_every == 0:
            old_score = getattr(args, "_active_adaptive_score", args.adaptive_score)
            args._active_adaptive_score = active_adaptive_score_for_method(method, args.adaptive_score)
            synchronize_device()
            update_start = time.perf_counter()
            collab_adj, evaluated = update_adaptive_topology(parties, xs, adjs, y, val_idx, args)
            synchronize_device()
            topology_update_time += time.perf_counter() - update_start
            topology_eval_count += evaluated
            topology_update_comm += n * len(val_idx) * num_classes
            args._active_adaptive_score = old_score

        if method not in local_methods and args.lambda_kd > 0:
            train_comm += 2 * edge_count(collab_adj) * y.numel() * num_classes

        update_reliability(parties, xs, adjs, y, val_idx, args)

        if method in local_methods:
            val_acc, test_acc, test_f1 = evaluate_local_baseline(method, parties, xs, adjs, y, val_idx, test_idx, num_classes, args)
        else:
            probs = get_party_probs(parties, xs, adjs)
            rel = [p.reliability for p in parties]
            vote = topology_prediction_consensus(probs, collab_adj, rel, args, y=y, val_idx=val_idx)
            val_acc = accuracy_from_probs(vote, y, val_idx)
            test_acc = accuracy_from_probs(vote, y, test_idx)
            test_f1 = macro_f1_from_probs(vote, y, test_idx, num_classes)

        final_test = test_acc
        best_observed = max(best_observed, test_acc)
        if val_acc > best_val:
            best_val = val_acc
            test_at_best = test_acc
            f1_at_best = test_f1

        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(f"[{dataset_name}/{method}] epoch={epoch:03d} val={val_acc:.4f} test={test_acc:.4f} edges={edge_count(collab_adj)}")

    selected_edges = topology_edge_list(collab_adj)
    useful_useful, useful_noisy, noisy_noisy = topology_edge_types(collab_adj, useful_party_mask)
    final_reliabilities = [float(p.reliability) for p in parties]
    if method in {"fixed_ring_matched", "expander_matched", "adaptive_graph_val"}:
        print(f"Seed {seed}/{method}: selected_edges={selected_edges}")
    print(
        f"Seed {seed}/{method}: reliabilities={[round(value, 6) for value in final_reliabilities]}; "
        f"edge_types=UU:{useful_useful},UN:{useful_noisy},NN:{noisy_noisy}"
    )

    peer_comm, readout_comm, total_comm, protocol_type = estimate_communication(
        method, collab_adj, y.numel(), num_classes, args
    )
    return RunResult(
        seed=seed,
        dataset=dataset_name,
        method=method,
        best_val=best_val,
        test_at_best_val=test_at_best,
        macro_f1_at_best_val=f1_at_best,
        best_observed_test=best_observed,
        final_test=final_test,
        train_comm=train_comm,
        inference_comm=peer_comm,
        final_edges=edge_count(collab_adj),
        final_adj=collab_adj.astype(int),
        peer_to_peer_comm=peer_comm,
        global_readout_comm=readout_comm,
        total_comm=total_comm,
        protocol_type=protocol_type,
        **evaluator_protocol_metadata(args),
        topology_update_comm=topology_update_comm,
        topology_eval_count=topology_eval_count,
        topology_update_time=topology_update_time,
        useful_useful_edges=useful_useful,
        useful_noisy_edges=useful_noisy,
        noisy_noisy_edges=noisy_noisy,
        selected_edges=json.dumps(selected_edges, separators=(",", ":")),
        useful_party_mask=json.dumps([bool(value) for value in useful_party_mask]),
        party_view_names=json.dumps(list(view_names)),
        party_reliabilities=json.dumps(final_reliabilities),
        party_permutation=json.dumps(list(party_permutation)),
        adaptive_min_edges=int(args.adaptive_min_edges),
    )


def train_central_fusion(seed: int, args: argparse.Namespace, dataset_name: str, xs: List[torch.Tensor], adjs: List[torch.Tensor], y: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor) -> RunResult:
    set_seed(seed + 2000)
    num_classes = int(y[y >= 0].max().item() + 1)
    model = CentralFusionModel([x.size(1) for x in xs], args.hidden_dim, num_classes, args.dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, test_at_best, f1_at_best, best_observed, final_test = -1.0, -1.0, -1.0, -1.0, -1.0
    train_comm = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(xs, adjs)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        train_comm += len(xs) * y.numel() * args.hidden_dim

        model.eval()
        with torch.no_grad():
            logits = model(xs, adjs)
            probs = F.softmax(logits, dim=1)
        val_acc = accuracy_from_logits(logits, y, val_idx)
        test_acc = accuracy_from_logits(logits, y, test_idx)
        test_f1 = macro_f1_from_probs(probs, y, test_idx, num_classes)
        final_test = test_acc
        best_observed = max(best_observed, test_acc)
        if val_acc > best_val:
            best_val = val_acc
            test_at_best = test_acc
            f1_at_best = test_f1

    inference = len(xs) * y.numel() * args.hidden_dim
    return RunResult(
        seed, dataset_name, "central_fusion", best_val, test_at_best,
        f1_at_best, best_observed, final_test, train_comm, inference,
        len(xs) * (len(xs)-1)//2, None,
        peer_to_peer_comm=0,
        global_readout_comm=inference,
        total_comm=inference,
        protocol_type="centralized_reference",
    )


def train_central_full_gcn(seed: int, args: argparse.Namespace, dataset_name: str, x: torch.Tensor, full_edges: List[Tuple[int, int]], y: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor) -> RunResult:
    set_seed(seed + 3000)
    num_classes = int(y[y >= 0].max().item() + 1)
    adj = edge_list_to_sparse_norm(full_edges, x.size(0))
    model = LocalGCN(x.size(1), args.hidden_dim, num_classes, args.dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, test_at_best, f1_at_best, best_observed, final_test = -1.0, -1.0, -1.0, -1.0, -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            probs = F.softmax(logits, dim=1)
        val_acc = accuracy_from_logits(logits, y, val_idx)
        test_acc = accuracy_from_logits(logits, y, test_idx)
        test_f1 = macro_f1_from_probs(probs, y, test_idx, num_classes)
        final_test = test_acc
        best_observed = max(best_observed, test_acc)
        if val_acc > best_val:
            best_val = val_acc
            test_at_best = test_acc
            f1_at_best = test_f1

    return RunResult(
        seed, dataset_name, "central_full_gcn", best_val, test_at_best,
        f1_at_best, best_observed, final_test, 0, 0, 0, None,
        peer_to_peer_comm=0,
        global_readout_comm=0,
        total_comm=0,
        protocol_type="centralized_reference",
    )


# =============================================================================
# Driver / reporting
# =============================================================================


def run_seed_with_prediction_cache(seed: int, args: argparse.Namespace) -> List[RunResult]:
    expected_config = prediction_cache_training_config(args, seed)
    cache_path = prediction_cache_path(args, seed)
    cache: Optional[Dict[str, object]] = None

    should_try_load = args.reuse_prediction_cache or args.skip_local_training_if_cache_exists
    if should_try_load and cache_path.exists():
        try:
            cache = load_prediction_cache(cache_path, expected_config)
            print(
                f"Loaded prediction cache {cache_path} "
                f"(fingerprint={str(cache['cache_fingerprint'])[:12]}); local training skipped."
            )
        except ValueError:
            if args.skip_local_training_if_cache_exists:
                raise
            print(f"Cache {cache_path} is invalid for this config; rebuilding it.")
    elif args.skip_local_training_if_cache_exists:
        raise FileNotFoundError(
            f"Prediction cache not found: {cache_path}. "
            "Remove --skip_local_training_if_cache_exists to train and create it."
        )

    if cache is None:
        set_seed(seed)
        x, y, graph_edges, split, target = load_hgb_dataset(args, seed)
        train_idx, val_idx, test_idx = split
        xs, adjs, view_names, useful_party_mask, party_permutation = build_party_views(x, graph_edges, args, seed)
        print(f"Seed {seed}: training local parties once for prediction cache.")
        print(f"Seed {seed}: useful_party_mask={useful_party_mask}")
        print(f"Seed {seed}: party_permutation={party_permutation}")
        print(f"Seed {seed}: view_names={view_names}")
        cache = train_local_prediction_cache(
            seed,
            args,
            x,
            y,
            xs,
            adjs,
            train_idx,
            val_idx,
            test_idx,
            target,
            useful_party_mask,
            view_names,
            party_permutation,
        )
        save_prediction_cache(cache, cache_path)

    if args.cache_only:
        return []

    args.party_permutation_by_seed[str(seed)] = list(cache["party_permutation"])
    args.useful_party_mask_by_seed[str(seed)] = [
        bool(value) for value in cache["useful_party_mask"]
    ]
    print(
        f"Seed {seed}: validation_evaluator={args.validation_evaluator}; "
        f"active_party_id={args.active_party_id}; "
        f"topology_evaluator={args.topology_evaluator}"
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    topk_method_aliases = {"topk_reliability_vote", "global_topk_reliability"}
    if args.topk_reliability_k <= 0 and topk_method_aliases.intersection(methods) and "adaptive_graph_val" in methods:
        deferred = [m for m in methods if m in topk_method_aliases]
        methods = [m for m in methods if m not in topk_method_aliases] + deferred
    results: List[RunResult] = []
    for method in methods:
        if method in topk_method_aliases:
            resolved_k = args.topk_reliability_k
            if resolved_k <= 0:
                adaptive_rows = [row for row in results if row.method == "adaptive_graph_val" and row.final_adj is not None]
                resolved_k = (
                    max(1, int(round(np.mean([
                        len(active_party_indices(row.final_adj, int(cache["num_parties"])))
                        for row in adaptive_rows
                    ]))))
                    if adaptive_rows
                    else 5
                )
            args._resolved_topk_reliability_k = resolved_k
            args.resolved_topk_reliability_k_by_seed[str(seed)] = int(resolved_k)
        result = evaluate_cached_method(method, seed, args, cache, cache_path)
        results.append(result)
        print(
            f"Cache eval seed={seed} method={method}: "
            f"test@best_val={result.test_at_best_val:.4f}, edges={result.final_edges}"
        )
    if hasattr(args, "_resolved_topk_reliability_k"):
        delattr(args, "_resolved_topk_reliability_k")
    return results


def run_seed(seed: int, args: argparse.Namespace) -> List[RunResult]:
    if args.cache_predictions or args.reuse_prediction_cache or args.skip_local_training_if_cache_exists or args.cache_only:
        return run_seed_with_prediction_cache(seed, args)

    set_seed(seed)
    x, y, graph_edges, split, target = load_hgb_dataset(args, seed)
    train_idx, val_idx, test_idx = split
    xs, adjs, view_names, useful_party_mask, party_permutation = build_party_views(x, graph_edges, args, seed)
    print(f"Seed {seed}: useful_party_mask={useful_party_mask}")
    print(f"Seed {seed}: party_permutation={party_permutation}")
    print(f"Seed {seed}: view_names={view_names}")
    args.party_permutation_by_seed[str(seed)] = list(party_permutation)
    args.useful_party_mask_by_seed[str(seed)] = [
        bool(value) for value in useful_party_mask
    ]
    print(
        f"Seed {seed}: validation_evaluator={args.validation_evaluator}; "
        f"active_party_id={args.active_party_id}; "
        f"topology_evaluator={args.topology_evaluator}"
    )
    print(
        f"Seed {seed}: dataset={args.dataset}, target={target}, nodes={x.size(0)}, features={x.size(1)}, "
        f"parties={args.num_parties}, views={view_names}, split={len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
    )

    results: List[RunResult] = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    topk_method_aliases = {"topk_reliability_vote", "global_topk_reliability"}
    if args.topk_reliability_k <= 0 and topk_method_aliases.intersection(methods) and "adaptive_graph_val" in methods:
        deferred = [m for m in methods if m in topk_method_aliases]
        methods = [m for m in methods if m not in topk_method_aliases] + deferred
    for method in methods:
        if method in topk_method_aliases:
            resolved_k = args.topk_reliability_k
            if resolved_k <= 0:
                adaptive_rows = [r for r in results if r.method == "adaptive_graph_val" and r.final_adj is not None]
                if adaptive_rows:
                    active_counts = [len(active_party_indices(r.final_adj, args.num_parties)) for r in adaptive_rows]
                    resolved_k = max(1, int(round(float(np.mean(active_counts)))))
                else:
                    resolved_k = 5
            args._resolved_topk_reliability_k = resolved_k
            args.resolved_topk_reliability_k_by_seed[str(seed)] = int(resolved_k)
            print(f"Seed {seed}: {method} uses k={resolved_k}")
        if method == "central_fusion":
            results.append(train_central_fusion(seed, args, args.dataset.upper(), xs, adjs, y, train_idx, val_idx, test_idx))
        elif method == "central_full_gcn":
            full_edges = graph_edges.get("FULL") or graph_edges.get("KNN") or graph_edges.get("RANDOM")
            results.append(train_central_full_gcn(seed, args, args.dataset.upper(), x, full_edges, y, train_idx, val_idx, test_idx))
        else:
            results.append(
                train_decentralized(
                    method,
                    seed,
                    args,
                    args.dataset.upper(),
                    xs,
                    adjs,
                    y,
                    train_idx,
                    val_idx,
                    test_idx,
                    useful_party_mask,
                    view_names,
                    party_permutation,
                )
            )
    if hasattr(args, "_resolved_topk_reliability_k"):
        delattr(args, "_resolved_topk_reliability_k")
    return results


def summarize(results: List[RunResult]) -> None:
    by_key: Dict[Tuple[str, str], List[RunResult]] = {}
    for r in results:
        by_key.setdefault((r.dataset, r.method), []).append(r)
    print("\n=== Summary ===")
    print(f"{'Dataset':<8} | {'Method':<24} | {'Test@BestVal':>20} | {'Macro-F1':>20} | {'InferComm':>14} | {'Edges':>10}")
    print("-" * 112)
    for (dataset, method), rows in by_key.items():
        acc_m, acc_s = mean_std([r.test_at_best_val for r in rows])
        f1_m, f1_s = mean_std([r.macro_f1_at_best_val for r in rows])
        infer_m, infer_s = mean_std([r.inference_comm for r in rows])
        edge_m, edge_s = mean_std([r.final_edges for r in rows])
        topo_comm_m, _ = mean_std([r.topology_update_comm for r in rows])
        topo_eval_m, _ = mean_std([r.topology_eval_count for r in rows])
        topo_time_m, _ = mean_std([r.topology_update_time for r in rows])
        print(f"{dataset:<8} | {method:<24} | {acc_m:.4f} ± {acc_s:.4f} | {f1_m:.4f} ± {f1_s:.4f} | {infer_m:9.0f} ± {infer_s:4.0f} | {edge_m:5.2f} ± {edge_s:4.2f}")


def summarize_active_party_protocol(results: List[RunResult]) -> None:
    if not results or not any(r.validation_evaluator == "active_party" for r in results):
        return
    by_method: Dict[str, List[RunResult]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result)

    def mean_metric(method: str, attribute: str) -> float:
        rows = by_method.get(method, [])
        return float(np.mean([getattr(row, attribute) for row in rows])) if rows else float("nan")

    matched_methods = ["best_matched_sparse", "fixed_ring_matched", "random_matched", "expander_matched"]
    matched_scores = {
        method: mean_metric(method, "test_at_best_val")
        for method in matched_methods
        if method in by_method
    }
    best_matched_method = max(matched_scores, key=matched_scores.get) if matched_scores else ""
    best_matched = matched_scores.get(best_matched_method, float("nan"))
    ta = mean_metric("adaptive_graph_val", "test_at_best_val")
    topk = mean_metric("global_topk_reliability", "test_at_best_val")
    if math.isnan(topk):
        topk = mean_metric("topk_reliability_vote", "test_at_best_val")
    full = mean_metric("full_mesh", "test_at_best_val")

    print("\n=== Active-party evaluator compact summary ===")
    print(f"Local training labels: {results[0].local_training_labels}")
    print(
        "Validation labels/topology evaluator: "
        f"{results[0].validation_label_protocol}"
    )
    print(f"TA-DVFG Test@BestVal:               {ta:.4f}")
    print(f"Global Top-k Reliability:           {topk:.4f}")
    print(f"Full Mesh:                          {full:.4f}")
    print(f"Best Matched Sparse ({best_matched_method or 'n/a'}): {best_matched:.4f}")
    print(f"TA-DVFG edges:                      {mean_metric('adaptive_graph_val', 'final_edges'):.2f}")
    print(f"TA-DVFG peer communication:         {mean_metric('adaptive_graph_val', 'peer_to_peer_comm'):.0f}")
    print(f"Topology update communication:      {mean_metric('adaptive_graph_val', 'topology_update_comm'):.0f}")
    print(f"Topology evaluation count:          {mean_metric('adaptive_graph_val', 'topology_eval_count'):.1f}")
    print(f"Topology update time (seconds):      {mean_metric('adaptive_graph_val', 'topology_update_time'):.3f}")
    print(f"TA-DVFG - Full Mesh gap:             {ta - full:+.4f}")
    print(f"TA-DVFG - Best Matched Sparse gap:   {ta - best_matched:+.4f}")
    print(f"TA-DVFG - Global Top-k gap:          {ta - topk:+.4f}")


def write_csv(path: str, results: List[RunResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "setting", "seed", "method", "adaptive_min_edges", "best_val", "test_at_best_val", "macro_f1", "macro_f1_at_best_val", "best_observed_test", "final_test", "train_comm", "inference_comm", "peer_to_peer_comm", "global_readout_comm", "total_comm", "protocol_type", "label_protocol", "training_protocol", "validation_protocol", "validation_evaluator", "active_party_id", "passive_label_access", "local_training_labels", "validation_label_protocol", "topology_evaluator", "local_training_protocol", "validation_split_mode", "topology_val_fraction", "topology_val_count", "selection_val_count", "topology_objective", "joint_lambda", "active_readout_acc", "local_post_consensus_mean_acc", "active_party_post_consensus_acc", "local_worst_acc", "local_best_acc", "dropout_protocol", "dropout_rate", "dropout_seed", "remaining_parties", "dropped_party_ids", "test_acc_after_dropout", "local_post_consensus_acc_after_dropout", "num_edges", "peer_comm", "readout_comm", "control_plane_comm", "final_edges", "topology_update_comm", "topology_eval_count", "topology_update_time", "useful_useful_edges", "useful_noisy_edges", "noisy_noisy_edges", "selected_edges", "useful_party_mask", "party_view_names", "party_reliabilities", "party_permutation", "prediction_cache_path", "prediction_cache_fingerprint"])
        for r in results:
            w.writerow([r.dataset, r.setting, r.seed, r.method, r.adaptive_min_edges, r.best_val, r.test_at_best_val, r.macro_f1_at_best_val, r.macro_f1_at_best_val, r.best_observed_test, r.final_test, r.train_comm, r.inference_comm, r.peer_to_peer_comm, r.global_readout_comm, r.total_comm, r.protocol_type, r.label_protocol, r.training_protocol, r.validation_protocol, r.validation_evaluator, r.active_party_id, r.passive_label_access, r.local_training_labels, r.validation_label_protocol, r.topology_evaluator, r.local_training_protocol, r.validation_split_mode, r.topology_val_fraction, r.topology_val_count, r.selection_val_count, r.topology_objective, r.joint_lambda, r.active_readout_acc, r.local_post_consensus_mean_acc, r.active_party_post_consensus_acc, r.local_worst_acc, r.local_best_acc, r.dropout_protocol, r.dropout_rate, r.dropout_seed, r.remaining_parties, r.dropped_party_ids, r.test_acc_after_dropout, r.local_post_consensus_acc_after_dropout, r.final_edges, r.peer_to_peer_comm, r.global_readout_comm, r.control_plane_comm, r.final_edges, r.topology_update_comm, r.topology_eval_count, r.topology_update_time, r.useful_useful_edges, r.useful_noisy_edges, r.noisy_noisy_edges, r.selected_edges, r.useful_party_mask, r.party_view_names, r.party_reliabilities, r.party_permutation, r.prediction_cache_path, r.prediction_cache_fingerprint])
    print(f"Saved CSV to {path}")


def write_summary_csv(path: str, results: List[RunResult]) -> None:
    import pandas as pd
    by_key: Dict[Tuple[str, str, str, float], List[RunResult]] = {}
    for r in results:
        by_key.setdefault((r.dataset, r.method, r.topology_objective, float(r.joint_lambda)), []).append(r)
    rows = []
    for (dataset, method, topology_objective, joint_lambda), vals in by_key.items():
        acc_m, acc_s = mean_std([r.test_at_best_val for r in vals])
        f1_m, f1_s = mean_std([r.macro_f1_at_best_val for r in vals])
        active_m, active_s = mean_std([r.active_readout_acc for r in vals])
        local_m, local_s = mean_std([r.local_post_consensus_mean_acc for r in vals])
        active_local_m, active_local_s = mean_std([r.active_party_post_consensus_acc for r in vals])
        local_worst_m, local_worst_s = mean_std([r.local_worst_acc for r in vals])
        local_best_m, local_best_s = mean_std([r.local_best_acc for r in vals])
        infer_m, infer_s = mean_std([r.inference_comm for r in vals])
        peer_m, peer_s = mean_std([r.peer_to_peer_comm for r in vals])
        readout_m, readout_s = mean_std([r.global_readout_comm for r in vals])
        total_m, total_s = mean_std([r.total_comm for r in vals])
        edge_m, edge_s = mean_std([r.final_edges for r in vals])
        topo_comm_m, topo_comm_s = mean_std([r.topology_update_comm for r in vals])
        topo_eval_m, topo_eval_s = mean_std([r.topology_eval_count for r in vals])
        topo_time_m, topo_time_s = mean_std([r.topology_update_time for r in vals])
        useful_useful_m, useful_useful_s = mean_std([r.useful_useful_edges for r in vals])
        useful_noisy_m, useful_noisy_s = mean_std([r.useful_noisy_edges for r in vals])
        noisy_noisy_m, noisy_noisy_s = mean_std([r.noisy_noisy_edges for r in vals])
        rows.append({
            "dataset": dataset,
            "method": method,
            "topology_objective": topology_objective,
            "joint_lambda": joint_lambda,
            "adaptive_min_edges": vals[0].adaptive_min_edges,
            "test_at_best_val_mean": acc_m,
            "test_at_best_val_std": acc_s,
            "macro_f1_mean": f1_m,
            "macro_f1_std": f1_s,
            "active_readout_acc_mean": active_m,
            "active_readout_acc_std": active_s,
            "local_post_consensus_mean_acc_mean": local_m,
            "local_post_consensus_mean_acc_std": local_s,
            "active_party_post_consensus_acc_mean": active_local_m,
            "active_party_post_consensus_acc_std": active_local_s,
            "local_worst_acc_mean": local_worst_m,
            "local_worst_acc_std": local_worst_s,
            "local_best_acc_mean": local_best_m,
            "local_best_acc_std": local_best_s,
            "inference_comm_mean": infer_m,
            "inference_comm_std": infer_s,
            "peer_to_peer_comm_mean": peer_m,
            "peer_to_peer_comm_std": peer_s,
            "global_readout_comm_mean": readout_m,
            "global_readout_comm_std": readout_s,
            "total_comm_mean": total_m,
            "total_comm_std": total_s,
            "protocol_type": vals[0].protocol_type,
            "validation_evaluator": vals[0].validation_evaluator,
            "active_party_id": vals[0].active_party_id,
            "local_training_labels": vals[0].local_training_labels,
            "validation_label_protocol": vals[0].validation_label_protocol,
            "topology_evaluator": vals[0].topology_evaluator,
            "local_training_protocol": vals[0].local_training_protocol,
            "validation_split_mode": vals[0].validation_split_mode,
            "topology_val_fraction": vals[0].topology_val_fraction,
            "topology_val_count": vals[0].topology_val_count,
            "selection_val_count": vals[0].selection_val_count,
            "final_edges_mean": edge_m,
            "final_edges_std": edge_s,
            "topology_update_comm_mean": topo_comm_m,
            "topology_update_comm_std": topo_comm_s,
            "topology_eval_count_mean": topo_eval_m,
            "topology_eval_count_std": topo_eval_s,
            "topology_update_time_mean": topo_time_m,
            "topology_update_time_std": topo_time_s,
            "useful_useful_edges_mean": useful_useful_m,
            "useful_useful_edges_std": useful_useful_s,
            "useful_noisy_edges_mean": useful_noisy_m,
            "useful_noisy_edges_std": useful_noisy_s,
            "noisy_noisy_edges_mean": noisy_noisy_m,
            "noisy_noisy_edges_std": noisy_noisy_s,
            "seeds": len(vals),
            "prediction_cache_fingerprints": ",".join(sorted({
                value.prediction_cache_fingerprint
                for value in vals
                if value.prediction_cache_fingerprint
            })),
        })
    summary_path = str(Path(path).with_name(Path(path).stem + "_summary.csv"))
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary CSV to {summary_path}")


def write_config_json(path: str, args: argparse.Namespace) -> None:
    config_path = str(Path(path).with_name(Path(path).stem + "_config.json"))
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved config JSON to {config_path}")


def plot_single_topology(adj: Optional[np.ndarray], method: str, seed: int, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.axis("off")
    ax.set_title(f"{method} topology (seed {seed})")
    if adj is None:
        ax.text(0, 0, method, ha="center", va="center", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"topology_{method}_seed{seed}.png"), dpi=200)
        plt.close(fig)
        return
    n = adj.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], linewidth=1.8)
    ax.scatter(pos[:, 0], pos[:, 1], s=700, zorder=3)
    for i in range(n):
        ax.text(pos[i, 0], pos[i, 1], f"P{i}", ha="center", va="center", fontsize=9, zorder=4)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"topology_{method}_seed{seed}.png"), dpi=200)
    plt.close(fig)


def plot_results(results: List[RunResult], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return
    os.makedirs(out_dir, exist_ok=True)
    by_method: Dict[str, List[RunResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    methods = list(by_method.keys())
    acc, acc_std, comm = [], [], []
    for m in methods:
        a, s = mean_std([r.test_at_best_val for r in by_method[m]])
        c, _ = mean_std([r.inference_comm for r in by_method[m]])
        acc.append(a)
        acc_std.append(s)
        comm.append(c)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, acc, yerr=acc_std, capsize=5)
    ax.set_ylabel("Test@BestVal accuracy")
    ax.set_title("TA-DVFG HGB benchmark")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_bar.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(comm, acc, yerr=acc_std, fmt="o", capsize=5, markersize=8)
    for m, x_val, y_val in zip(methods, comm, acc):
        ax.annotate(m, (x_val, y_val), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Inference communication")
    ax.set_ylabel("Test@BestVal accuracy")
    ax.set_title("Accuracy--communication tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_comm_tradeoff.png"), dpi=200)
    plt.close(fig)

    topo_dir = os.path.join(out_dir, "topologies")
    for r in results:
        plot_single_topology(r.final_adj, r.method, r.seed, topo_dir)
    print(f"Saved plots to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DBLP", choices=["ACM", "DBLP", "IMDB", "Freebase", "FREEBASE"])
    parser.add_argument("--target_node", type=str, default="auto")
    parser.add_argument("--data_dir", type=str, default="data_hgb")
    parser.add_argument("--ignore_dataset_split", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--num_parties", type=int, default=15)
    parser.add_argument("--graph_views", type=str, default="")
    parser.add_argument("--graph_k", type=int, default=10)
    parser.add_argument("--max_metapath_edges", type=int, default=300000)
    parser.add_argument("--random_graph_degree", type=int, default=4)
    parser.add_argument(
        "--featureless_target_features",
        type=str,
        default="structural",
        choices=["structural", "random_fallback"],
        help=(
            "How to handle labelled target node types with no original x/features. "
            "structural derives deterministic party-private graph features; "
            "random_fallback preserves the previous identity/random fallback."
        ),
    )
    parser.add_argument(
        "--structural_feature_dim",
        type=int,
        default=16,
        help="Per-party feature dimension for featureless target structural features.",
    )
    parser.add_argument("--view_setting", type=str, default="hard", choices=["standard", "hard"])
    parser.add_argument("--useful_parties", type=int, default=6)
    parser.add_argument("--distractor_feature_noise", type=float, default=1.0)
    parser.add_argument("--shuffle_party_positions", action="store_true")
    parser.add_argument("--party_shuffle_seed_offset", type=int, default=2026)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--lambda_kd", type=float, default=0.0)
    parser.add_argument("--pred_consensus_steps", type=int, default=1)
    parser.add_argument("--pred_self_weight", type=float, default=0.85)
    parser.add_argument("--consensus_mode", type=str, default="standard", choices=["standard", "gated"])
    parser.add_argument("--consensus_reliability_margin", type=float, default=0.0)
    parser.add_argument(
        "--vote_weighting",
        type=str,
        default="topology_reliability",
        choices=["uniform", "reliability", "topology_reliability"],
        help=(
            "uniform = simple average; reliability = global validation-weighted vote; "
            "topology_reliability = reliability-aware vote filtered by learned sparse topology."
        ),
    )
    parser.add_argument("--reliability_power", type=float, default=1.0)
    parser.add_argument("--topology_degree_power", type=float, default=0.5)
    parser.add_argument("--final_reliability_floor", type=float, default=0.0)
    parser.add_argument("--topk_reliability_k", type=int, default=5)
    parser.add_argument(
        "--readout_mode",
        type=str,
        default="active_vote",
        choices=[
            "active_vote",
            "mean_party",
            "best_party",
            "local_mean",
            "local_worst",
            "active_party_local",
        ],
    )
    parser.add_argument("--validation_evaluator", type=str, default="all_parties", choices=["all_parties", "active_party"])
    parser.add_argument("--active_party_id", type=int, default=0)
    parser.add_argument(
        "--label_protocol",
        type=str,
        choices=["all_supervised", "active_party_only"],
        help="Canonical label-ownership protocol; maps to the legacy evaluator/training flags.",
    )
    parser.add_argument(
        "--training_protocol",
        type=str,
        choices=["supervised", "prediction_split"],
        help="Canonical training protocol. prediction_split uses exact returned logit gradients.",
    )
    parser.add_argument(
        "--validation_protocol",
        type=str,
        choices=["shared_val", "split_val"],
        help="Canonical validation protocol; split_val separates topology and model-selection labels.",
    )
    parser.add_argument(
        "--local_training_protocol",
        type=str,
        default="all_parties_supervised",
        choices=["all_parties_supervised", "active_party_logit_gradient"],
    )
    parser.add_argument(
        "--validation_split_mode",
        type=str,
        default="shared",
        choices=["shared", "disjoint"],
    )
    parser.add_argument("--topology_val_fraction", type=float, default=1.0)
    parser.add_argument("--validation_split_seed_offset", type=int, default=4096)
    parser.add_argument(
        "--dropout_protocol",
        type=str,
        default="none",
        choices=["none", "random_party_dropout"],
    )
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_seed", type=int, default=0)
    parser.add_argument(
        "--dropout_rates",
        type=str,
        default="",
        help="Optional comma-separated inference dropout rates.",
    )
    parser.add_argument(
        "--dropout_seeds",
        type=str,
        default="",
        help="Optional comma-separated inference dropout seeds.",
    )
    parser.add_argument("--setting_name", type=str, default="")

    parser.add_argument("--topology_every", type=int, default=20)
    parser.add_argument(
        "--topology_max_updates",
        type=int,
        default=0,
        help="Maximum topology updates; 0 means unlimited.",
    )
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--matched_edge_count", type=int, default=5)
    parser.add_argument("--adaptive_score", type=str, default="graph_val", choices=["pair", "complementarity", "graph_val"])
    parser.add_argument(
        "--topology_objective",
        type=str,
        default="active",
        choices=["active", "local_mean", "joint"],
        help="Validation objective used to select deployment topologies.",
    )
    parser.add_argument(
        "--joint_lambda",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "Weight(s) for the joint topology objective: lambda * active + "
            "(1-lambda) * local_mean. Multiple values run a lambda sweep."
        ),
    )
    parser.add_argument("--adaptive_edge_budget", type=int, default=15)
    parser.add_argument("--adaptive_min_edges", type=int, default=15)
    parser.add_argument("--adaptive_min_gain", type=float, default=0.0)
    parser.add_argument("--adaptive_candidate_edges", type=int, default=0)
    parser.add_argument("--pair_acc_weight", type=float, default=0.45)
    parser.add_argument("--pair_gain_weight", type=float, default=0.20)
    parser.add_argument("--reliability_weight", type=float, default=0.25)
    parser.add_argument("--corrective_weight", type=float, default=0.30)
    parser.add_argument("--both_wrong_weight", type=float, default=0.20)
    parser.add_argument("--diversity_weight", type=float, default=0.0)
    parser.add_argument("--disagreement_weight", type=float, default=0.05)

    parser.add_argument(
        "--methods",
        type=str,
        default="single_party_avg,single_party_best,local_uniform_vote,local_reliability_vote,topk_reliability_vote,fixed_ring,fixed_ring_matched,random_regular,random_matched,expander,expander_matched,full_mesh,adaptive_pair,adaptive_complementarity,adaptive_graph_val",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--save_csv", type=str, default="tadvfg_hgb_results.csv")
    parser.add_argument("--cache_predictions", action="store_true")
    parser.add_argument("--reuse_prediction_cache", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="results/cache")
    parser.add_argument("--cache_namespace", type=str, default="standard")
    parser.add_argument(
        "--skip_local_training_if_cache_exists",
        action="store_true",
        help="Strict cache-only mode: require a valid cache and never fall back to local training.",
    )
    parser.add_argument("--cache_only", action="store_true", help="Build/load caches without topology evaluation.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="hgb_figures")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    global DEVICE
    args = parse_args()
    args.dataset = args.dataset.upper() if args.dataset.upper() == "FREEBASE" else args.dataset.upper()
    args.data_dir = str(Path(args.data_dir).resolve())
    args.cache_dir = str(Path(args.cache_dir).resolve())

    if args.label_protocol == "active_party_only":
        args.validation_evaluator = "active_party"
    elif args.label_protocol == "all_supervised":
        args.local_training_protocol = "all_parties_supervised"
    if args.training_protocol == "prediction_split":
        args.local_training_protocol = "active_party_logit_gradient"
        args.validation_evaluator = "active_party"
    elif args.training_protocol == "supervised":
        args.local_training_protocol = "all_parties_supervised"
    if args.validation_protocol == "split_val":
        args.validation_split_mode = "disjoint"
        if args.topology_val_fraction == 1.0:
            args.topology_val_fraction = 0.5
        args.validation_evaluator = "active_party"
    elif args.validation_protocol == "shared_val":
        args.validation_split_mode = "shared"

    args.label_protocol = args.label_protocol or (
        "active_party_only"
        if args.validation_evaluator == "active_party"
        and args.local_training_protocol == "active_party_logit_gradient"
        else "all_supervised"
    )
    args.training_protocol = args.training_protocol or (
        "prediction_split"
        if args.local_training_protocol == "active_party_logit_gradient"
        else "supervised"
    )
    args.validation_protocol = args.validation_protocol or (
        "split_val" if args.validation_split_mode == "disjoint" else "shared_val"
    )
    args.passive_label_access = not (
        args.label_protocol == "active_party_only"
        and args.training_protocol == "prediction_split"
    )

    if args.matched_edge_count < 0:
        raise ValueError("--matched_edge_count must be non-negative")
    if not 0 <= args.active_party_id < args.num_parties:
        raise ValueError("--active_party_id must be in [0, num_parties)")
    if args.topology_every <= 0:
        raise ValueError("--topology_every must be positive")
    if args.topology_max_updates < 0:
        raise ValueError("--topology_max_updates must be non-negative")
    joint_lambda_values = (
        list(args.joint_lambda)
        if isinstance(args.joint_lambda, list)
        else [float(args.joint_lambda)]
    )
    if any(value < 0.0 or value > 1.0 for value in joint_lambda_values):
        raise ValueError("--joint_lambda values must be in [0, 1]")
    if args.topology_objective != "joint" and len(joint_lambda_values) > 1:
        raise ValueError("Multiple --joint_lambda values are only valid with --topology_objective joint")
    args._joint_lambda_values = joint_lambda_values
    args.joint_lambda = float(joint_lambda_values[0])
    if not 0.0 < args.topology_val_fraction <= 1.0:
        raise ValueError("--topology_val_fraction must be in (0, 1]")
    if not 0.0 <= args.dropout_rate < 1.0:
        raise ValueError("--dropout_rate must be in [0, 1)")
    if (
        args.label_protocol == "active_party_only"
        and args.training_protocol != "prediction_split"
    ):
        raise ValueError(
            "active_party_only requires --training_protocol prediction_split "
            "so passive parties never receive training labels."
        )
    uses_cache = bool(
        args.cache_predictions
        or args.reuse_prediction_cache
        or args.skip_local_training_if_cache_exists
        or args.cache_only
    )
    if (
        args.readout_mode in {"local_mean", "local_worst", "active_party_local"}
        or args.validation_split_mode != "shared"
        or args.topology_val_fraction != 1.0
    ) and not uses_cache:
        raise ValueError(
            "Local decentralized readouts and validation partition experiments "
            "currently require prediction-cache mode."
        )
    DEVICE = resolve_device(args.device)
    args._active_adaptive_score = args.adaptive_score
    args.local_training_labels = (
        "active_party_only_logit_gradient"
        if args.local_training_protocol == "active_party_logit_gradient"
        else "all_parties_supervised_simulation"
    )
    args.validation_label_protocol = (
        "active_party_only"
        if args.validation_evaluator == "active_party"
        else "all_parties"
    )
    args.topology_evaluator = (
        "active_party_id"
        if args.validation_evaluator == "active_party"
        else "all_parties"
    )
    # Backward-compatible config aliases.
    args.local_training_label_access = args.local_training_labels
    args.validation_label_access = args.validation_label_protocol
    args.resolved_topk_reliability_k_by_seed = {}
    args.party_permutation_by_seed = {}
    args.useful_party_mask_by_seed = {}
    print(f"Device: {DEVICE}")
    print(
        f"Config: dataset={args.dataset}, target={args.target_node}, N={args.num_parties}, useful={args.useful_parties}, "
        f"views={args.graph_views or 'default'}, adaptive_budget={args.adaptive_edge_budget}, seeds={args.seeds}\n"
    )
    print(
        f"Protocol: local_training_labels={args.local_training_labels}; "
        f"validation_labels={args.validation_label_protocol}; "
        f"topology_evaluator={args.topology_evaluator}; "
        f"active_party_id={args.active_party_id}; readout={args.readout_mode}; "
        f"topology_objective={args.topology_objective}; "
        f"joint_lambda_values={args._joint_lambda_values}; "
        f"label_protocol={args.label_protocol}; "
        f"training_protocol={args.training_protocol}; "
        f"validation_protocol={args.validation_protocol}; "
        f"passive_label_access={args.passive_label_access}"
    )
    seeds = parse_seed_list(args.seeds) if args.seeds else [args.seed]
    results: List[RunResult] = []
    for seed in seeds:
        objective_lambdas = (
            args._joint_lambda_values
            if args.topology_objective == "joint"
            else [float(args.joint_lambda)]
        )
        for lambda_value in objective_lambdas:
            args.joint_lambda = float(lambda_value)
            if args.dropout_protocol == "random_party_dropout":
                rates = (
                    [float(value) for value in args.dropout_rates.split(",") if value.strip()]
                    if args.dropout_rates
                    else [float(args.dropout_rate)]
                )
                dropout_seeds = (
                    parse_seed_list(args.dropout_seeds)
                    if args.dropout_seeds
                    else [int(args.dropout_seed)]
                )
                for rate in rates:
                    if not 0.0 <= rate < 1.0:
                        raise ValueError("--dropout_rates values must be in [0, 1)")
                    for dropout_seed in dropout_seeds:
                        args.dropout_rate = float(rate)
                        args.dropout_seed = int(dropout_seed)
                        results.extend(run_seed(seed, args))
            else:
                args.dropout_rate = 0.0
                results.extend(run_seed(seed, args))
    if args.cache_only:
        print("Prediction cache stage completed.")
        return
    summarize(results)
    summarize_active_party_protocol(results)
    if args.save_csv:
        write_csv(args.save_csv, results)
        write_summary_csv(args.save_csv, results)
        write_config_json(args.save_csv, args)
    if args.plot:
        plot_results(results, args.plot_dir)


if __name__ == "__main__":
    main()
