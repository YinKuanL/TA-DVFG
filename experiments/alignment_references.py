"""HGB hidden-state export and centralized alignment-reference pilot.

The alignment references here deliberately use stronger information than
TA-DVFG: frozen party hidden states are collected centrally and fused by a new
projection/readout module. They are not communication-matched P2P baselines.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
CORE_DIR = REPO_ROOT / "main experiment" / "core"
OUTPUT_DIR = REPO_ROOT / "outputs" / "alignment_references"

SETTING_DIR = {
    ("ACM", "main"): "acm_main",
    ("ACM", "hard"): "acm_hard_noisy",
    ("DBLP", "main"): "dblp_main",
    ("DBLP", "hard"): "dblp_hard_noisy",
    ("IMDB", "main"): "imdb_main",
    ("IMDB", "hard"): "imdb_hard_noisy",
}


def load_engine():
    spec = importlib.util.spec_from_file_location("tadvfg_hgb_engine_alignment", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TA-DVFG HGB engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = load_engine()


def parse_dims(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def read_setting_config(dataset: str, setting: str) -> Dict[str, object]:
    key = (dataset.upper(), setting.lower())
    if key not in SETTING_DIR:
        raise ValueError(f"Unsupported dataset/setting: {dataset} {setting}")
    path = CORE_DIR / SETTING_DIR[key] / "metrics_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing headline config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_runtime_args(args: argparse.Namespace, dataset: str, setting: str, seed: int) -> argparse.Namespace:
    config = read_setting_config(dataset, setting)
    ns = argparse.Namespace(**config)
    ns.dataset = dataset.upper()
    ns.view_setting = setting.lower()
    ns.seed = seed
    ns.seeds = str(seed)
    ns.device = args.device
    ns.data_dir = str((REPO_ROOT / "data_hgb").resolve())
    ns.plot = False
    ns.plot_dir = ""
    ns.save_csv = ""
    ns.verbose = bool(args.verbose)
    ns.epochs = int(args.local_epochs if args.local_epochs > 0 else getattr(ns, "epochs", 300))
    ns.cache_predictions = False
    ns.reuse_prediction_cache = False
    ns.skip_local_training_if_cache_exists = False
    ns.cache_only = False
    ns.cache_dir = str((OUTPUT_DIR / "local_cache").resolve())
    ns.cache_namespace = "alignment_references"
    ns.log_every = int(getattr(ns, "log_every", 20))

    # Fields introduced after some legacy configs were written.
    defaults = {
        "label_protocol": "all_supervised",
        "training_protocol": "supervised",
        "validation_protocol": "shared_val",
        "local_training_protocol": "all_parties_supervised",
        "validation_split_mode": "shared",
        "topology_val_fraction": 1.0,
        "validation_split_seed_offset": 4096,
        "passive_label_access": True,
        "dropout_protocol": "none",
        "dropout_rate": 0.0,
        "dropout_seed": 0,
        "setting_name": f"{dataset.lower()}_{setting.lower()}_alignment",
        "topology_max_updates": 0,
        "local_training_labels": "all_parties_supervised_simulation",
        "validation_label_protocol": "all_parties",
        "topology_evaluator": "all_parties",
        "topology_objective": "active",
        "joint_lambda": 0.5,
        "party_permutation_by_seed": {},
        "useful_party_mask_by_seed": {},
        "resolved_topk_reliability_k_by_seed": {},
    }
    for key, value in defaults.items():
        if not hasattr(ns, key):
            setattr(ns, key, value)
    return ns


def train_frozen_parties(ns: argparse.Namespace, seed: int):
    ENGINE.set_seed(seed)
    x, y, graph_edges, split, target = ENGINE.load_hgb_dataset(ns, seed)
    train_idx, val_idx, test_idx = split
    xs, adjs, view_names, useful_mask, permutation = ENGINE.build_party_views(x, graph_edges, ns, seed)
    num_classes = int(y.max().item() + 1)
    ENGINE.set_seed(seed + 1000)
    parties = ENGINE.init_parties(xs, int(ns.hidden_dim), num_classes, float(ns.dropout), float(ns.lr), float(ns.weight_decay))
    losses: List[float] = []
    val_epochs: List[torch.Tensor] = []
    test_epochs: List[torch.Tensor] = []
    for epoch in range(int(ns.epochs)):
        ENGINE.train_local_parties_one_epoch(parties, xs, adjs, y, train_idx, ns)
        probs = ENGINE.get_party_probs(parties, xs, adjs)
        val_epochs.append(torch.stack([prob[val_idx].detach().cpu() for prob in probs], dim=0))
        test_epochs.append(torch.stack([prob[test_idx].detach().cpu() for prob in probs], dim=0))
        if epoch == 0 or epoch == int(ns.epochs) - 1:
            with torch.no_grad():
                epoch_loss = 0.0
                for party, x_i, adj_i in zip(parties, xs, adjs):
                    party.model.eval()
                    epoch_loss += float(F.cross_entropy(party.model(x_i, adj_i)[train_idx], y[train_idx]).item())
                losses.append(epoch_loss / max(len(parties), 1))
    ENGINE.update_reliability(parties, xs, adjs, y, val_idx, ns)
    for party in parties:
        party.model.eval()
        for param in party.model.parameters():
            param.requires_grad_(False)
    return (
        x,
        y,
        train_idx,
        val_idx,
        test_idx,
        xs,
        adjs,
        parties,
        view_names,
        useful_mask,
        permutation,
        target,
        losses,
        torch.stack(val_epochs, dim=0),
        torch.stack(test_epochs, dim=0),
    )


def default_cache_path(dataset: str, setting: str, seed: int, useful_parties: int) -> Path:
    return REPO_ROOT / "results" / "core_cache_v2" / (
        f"{dataset.lower()}_{setting.lower()}_useful{useful_parties}_seed{seed}_party_preds.pt"
    )


def load_headline_cache(ns: argparse.Namespace, seed: int) -> Tuple[Optional[Dict[str, object]], Optional[Path], str]:
    expected = ENGINE.prediction_cache_training_config(ns, seed)
    candidates = [
        default_cache_path(ns.dataset, ns.view_setting, seed, int(ns.useful_parties)),
        ENGINE.prediction_cache_path(ns, seed),
    ]
    seen = set()
    for path in candidates:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            return ENGINE.load_prediction_cache(path, expected), path, "loaded_with_strict_config"
        except Exception as exc:
            try:
                raw = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    return raw, path, f"loaded_without_strict_validation: {exc}"
            except Exception:
                pass
            return None, path, f"failed_to_load: {exc}"
    return None, None, "missing"


def fresh_probability_tensors(parties, xs, adjs, train_idx, val_idx, test_idx) -> Dict[str, torch.Tensor]:
    probs = torch.stack([prob.detach().cpu() for prob in ENGINE.get_party_probs(parties, xs, adjs)], dim=0)
    return {
        "probs_all": probs,
        "probs_train": probs[:, train_idx.detach().cpu()],
        "probs_val": probs[:, val_idx.detach().cpu()],
        "probs_test": probs[:, test_idx.detach().cpu()],
    }


def cache_comparability_audit(
    dataset: str,
    setting: str,
    seed: int,
    ns: argparse.Namespace,
    parties,
    xs,
    adjs,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    num_classes: int,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, torch.Tensor]]:
    cache, cache_path, status = load_headline_cache(ns, seed)
    fresh = fresh_probability_tensors(parties, xs, adjs, train_idx, val_idx, test_idx)
    rows = []
    report = {
        "cache_path": str(cache_path) if cache_path else "",
        "cache_load_status": status,
        "cache_has_logits": False,
        "fresh_matches_cache": False,
        "must_use_within_suite_references": True,
    }
    if cache is None:
        for party_id in range(len(parties)):
            rows.append(
                {
                    "dataset": dataset.upper(),
                    "setting": setting.lower(),
                    "seed": seed,
                    "party": party_id,
                    "max_logit_diff": "UNAVAILABLE_CACHE_MISSING",
                    "max_probability_diff": "UNAVAILABLE_CACHE_MISSING",
                    "val_prediction_disagreement_count": "UNAVAILABLE_CACHE_MISSING",
                    "test_prediction_disagreement_count": "UNAVAILABLE_CACHE_MISSING",
                    "val_metric_diff": "UNAVAILABLE_CACHE_MISSING",
                    "test_metric_diff": "UNAVAILABLE_CACHE_MISSING",
                    "status": "UNRESOLVED",
                }
            )
        return pd.DataFrame(rows), report, fresh

    cache_has_logits = all(key in cache for key in ["logits_all", "logits_val", "logits_test"])
    report["cache_has_logits"] = bool(cache_has_logits)
    cache_y = cache.get("y")
    y_cpu = y.detach().cpu()
    if isinstance(cache_y, torch.Tensor) and not torch.equal(cache_y.cpu(), y_cpu):
        report["label_alignment_warning"] = "cache labels differ from fresh labels"
    for party_id in range(len(parties)):
        fresh_val = fresh["probs_val"][party_id]
        fresh_test = fresh["probs_test"][party_id]
        cache_val = cache["probs_val"][party_id].cpu()
        cache_test = cache["probs_test"][party_id].cpu()
        max_prob = max(
            float((fresh_val - cache_val).abs().max().item()),
            float((fresh_test - cache_test).abs().max().item()),
        )
        fresh_val_pred = fresh_val.argmax(dim=1)
        fresh_test_pred = fresh_test.argmax(dim=1)
        cache_val_pred = cache_val.argmax(dim=1)
        cache_test_pred = cache_test.argmax(dim=1)
        val_disagree = int((fresh_val_pred != cache_val_pred).sum().item())
        test_disagree = int((fresh_test_pred != cache_test_pred).sum().item())
        val_metric_fresh = float((fresh_val_pred == y_cpu[val_idx.cpu()]).float().mean().item())
        test_metric_fresh = float((fresh_test_pred == y_cpu[test_idx.cpu()]).float().mean().item())
        val_metric_cache = float((cache_val_pred == y_cpu[val_idx.cpu()]).float().mean().item())
        test_metric_cache = float((cache_test_pred == y_cpu[test_idx.cpu()]).float().mean().item())
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "party": party_id,
                "max_logit_diff": "UNAVAILABLE_CACHE_HAS_NO_LOGITS" if not cache_has_logits else "NOT_IMPLEMENTED",
                "max_probability_diff": max_prob,
                "val_prediction_disagreement_count": val_disagree,
                "test_prediction_disagreement_count": test_disagree,
                "val_metric_diff": abs(val_metric_fresh - val_metric_cache),
                "test_metric_diff": abs(test_metric_fresh - test_metric_cache),
                "status": "PASS" if max_prob <= 1e-6 and val_disagree == 0 and test_disagree == 0 else "MISMATCH",
            }
        )
    frame = pd.DataFrame(rows)
    report["fresh_matches_cache"] = bool((frame["status"] == "PASS").all())
    report["must_use_within_suite_references"] = not report["fresh_matches_cache"]
    return frame, report, fresh


def hidden_export_equivalence(dataset: str, setting: str, seed: int, parties, xs, adjs, y, val_idx, test_idx) -> pd.DataFrame:
    rows = []
    for party_id, (party, x_i, adj_i) in enumerate(zip(parties, xs, adjs)):
        party.model.eval()
        with torch.no_grad():
            logits_forward = party.model(x_i, adj_i)
            hidden = party.model.encode(x_i, adj_i)
            logits_export = party.model.cls(hidden)
            probs_forward = F.softmax(logits_forward, dim=1)
            probs_export = F.softmax(logits_export, dim=1)
        val_forward = ENGINE.accuracy_from_logits(logits_forward, y, val_idx)
        val_export = ENGINE.accuracy_from_logits(logits_export, y, val_idx)
        test_forward = ENGINE.accuracy_from_logits(logits_forward, y, test_idx)
        test_export = ENGINE.accuracy_from_logits(logits_export, y, test_idx)
        max_logit_diff = float((logits_forward - logits_export).abs().max().item())
        max_prob_diff = float((probs_forward - probs_export).abs().max().item())
        val_diff = abs(float(val_forward) - float(val_export))
        test_diff = abs(float(test_forward) - float(test_export))
        status = "PASS" if max(max_logit_diff, max_prob_diff, val_diff, test_diff) <= 1e-6 else "FAIL"
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "party": party_id,
                "max_logit_diff": max_logit_diff,
                "max_probability_diff": max_prob_diff,
                "val_metric_diff": val_diff,
                "test_metric_diff": test_diff,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


class ProjectMean(nn.Module):
    def __init__(self, in_dims: Sequence[int], d_ref: int, num_classes: int):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(dim, d_ref) for dim in in_dims])
        self.norms = nn.ModuleList([nn.LayerNorm(d_ref) for _ in in_dims])
        self.classifier = nn.Linear(d_ref, num_classes)

    def forward(self, hiddens: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = [norm(F.relu(proj(h))) for proj, norm, h in zip(self.projectors, self.norms, hiddens)]
        return self.classifier(torch.stack(projected, dim=0).mean(dim=0))


class ProjectConcat(nn.Module):
    def __init__(self, in_dims: Sequence[int], d_ref: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(dim, d_ref) for dim in in_dims])
        self.classifier = nn.Sequential(
            nn.Linear(len(in_dims) * d_ref, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, hiddens: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = [F.relu(proj(h)) for proj, h in zip(self.projectors, hiddens)]
        return self.classifier(torch.cat(projected, dim=1))


class GatedFusion(nn.Module):
    def __init__(self, in_dims: Sequence[int], d_ref: int, num_classes: int):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(dim, d_ref) for dim in in_dims])
        self.scorers = nn.ModuleList([nn.Linear(d_ref, 1) for _ in in_dims])
        self.classifier = nn.Linear(d_ref, num_classes)

    def forward(self, hiddens: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = [F.relu(proj(h)) for proj, h in zip(self.projectors, hiddens)]
        scores = torch.cat([score(h) for score, h in zip(self.scorers, projected)], dim=1)
        gates = F.softmax(scores, dim=1)
        fused = torch.stack(projected, dim=1)
        fused = (gates.unsqueeze(-1) * fused).sum(dim=1)
        return self.classifier(fused)


@dataclass
class TrainResult:
    method: str
    d_ref: int
    lr: float
    weight_decay: float
    val_acc: float
    test_acc: float
    macro_f1: float
    best_epoch: int
    first_loss: float
    best_loss: float
    param_count: int
    train_loss_curve: List[float]
    val_metric_curve: List[float]
    gate_mean: Optional[List[float]] = None
    gate_entropy: Optional[float] = None
    gate_dominant_fraction: Optional[float] = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None


def make_model(method: str, in_dims: Sequence[int], d_ref: int, num_classes: int) -> nn.Module:
    if method == "Project-and-Mean":
        return ProjectMean(in_dims, d_ref, num_classes)
    if method == "Project-and-Concat":
        return ProjectConcat(in_dims, d_ref, num_classes)
    if method == "Gated Fusion":
        return GatedFusion(in_dims, d_ref, num_classes)
    raise ValueError(f"Unknown alignment method: {method}")


def train_alignment_model(
    method: str,
    d_ref: int,
    hiddens: Sequence[torch.Tensor],
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    in_dims = [int(h.size(1)) for h in hiddens]
    model = make_model(method, in_dims, d_ref, num_classes).to(hiddens[0].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -1.0
    best_epoch = -1
    first_loss = math.nan
    best_loss = math.nan
    train_loss_curve: List[float] = []
    val_metric_curve: List[float] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(hiddens)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        train_loss_curve.append(float(loss.item()))
        if epoch == 0:
            first_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits_eval = model(hiddens)
            val_acc = ENGINE.accuracy_from_logits(logits_eval, y, val_idx)
        val_metric_curve.append(float(val_acc))
        if val_acc > best_val:
            best_val = float(val_acc)
            best_epoch = epoch
            best_loss = float(loss.item())
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(hiddens)
        probs = F.softmax(logits, dim=1)
        gate_mean = None
        gate_entropy = None
        gate_dominant_fraction = None
        if isinstance(model, GatedFusion):
            projected = [F.relu(proj(h)) for proj, h in zip(model.projectors, hiddens)]
            scores = torch.cat([score(h) for score, h in zip(model.scorers, projected)], dim=1)
            gates = F.softmax(scores, dim=1)
            gate_mean = [float(v) for v in gates.mean(dim=0).detach().cpu()]
            entropy = -(gates.clamp_min(1e-12) * gates.clamp_min(1e-12).log()).sum(dim=1)
            gate_entropy = float(entropy.mean().item())
            gate_dominant_fraction = float((gates.max(dim=1).values >= 0.8).float().mean().item())
    return TrainResult(
        method=method,
        d_ref=d_ref,
        lr=float(lr),
        weight_decay=float(weight_decay),
        val_acc=float(ENGINE.accuracy_from_logits(logits, y, val_idx)),
        test_acc=float(ENGINE.accuracy_from_logits(logits, y, test_idx)),
        macro_f1=float(ENGINE.macro_f1_from_probs(probs, y, test_idx, num_classes)),
        best_epoch=int(best_epoch),
        first_loss=float(first_loss),
        best_loss=float(best_loss),
        param_count=int(sum(param.numel() for param in model.parameters())),
        train_loss_curve=train_loss_curve,
        val_metric_curve=val_metric_curve,
        gate_mean=gate_mean,
        gate_entropy=gate_entropy,
        gate_dominant_fraction=gate_dominant_fraction,
        state_dict=best_state,
    )


def select_alignment_reference(
    method: str,
    dims: Sequence[int],
    hiddens: Sequence[torch.Tensor],
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> Tuple[TrainResult, List[TrainResult]]:
    all_runs = []
    for d_ref in dims:
        for lr in parse_floats(args.fusion_lr_grid):
            for weight_decay in parse_floats(args.fusion_weight_decay_grid):
                all_runs.append(
                    train_alignment_model(
                        method,
                        d_ref,
                        hiddens,
                        y,
                        train_idx,
                        val_idx,
                        test_idx,
                        num_classes,
                        int(args.fusion_epochs),
                        lr,
                        weight_decay,
                    )
                )
    best = sorted(all_runs, key=lambda item: (item.val_acc, -item.d_ref, -item.lr, -item.weight_decay), reverse=True)[0]
    return best, all_runs


def prediction_reference_rows(dataset: str, setting: str, seed: int) -> List[Dict[str, object]]:
    key = (dataset.upper(), setting.lower())
    rows: List[Dict[str, object]] = []
    if key not in SETTING_DIR:
        return rows
    path = CORE_DIR / SETTING_DIR[key] / "metrics.csv"
    if not path.exists():
        return rows
    frame = pd.read_csv(path)
    mapping = {
        "single_party_best": "Best Single",
        "topk_reliability_vote": "Global Top-k Reliability",
        "full_mesh": "Full Mesh prediction consensus",
        "adaptive_graph_val": "TA-DVFG",
    }
    for source, name in mapping.items():
        subset = frame[(frame["seed"] == seed) & (frame["method"] == source)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "method": name,
                "interface_type": "prediction-space reference",
                "d_ref": "N/A",
                "selected_by": "existing_headline_csv",
                "best_val": float(row["best_val"]),
                "test_at_best_val": float(row["test_at_best_val"]),
                "macro_f1": float(row["macro_f1_at_best_val"]),
                "best_epoch": "N/A",
                "first_loss": "N/A",
                "best_loss": "N/A",
                "param_count": 0,
                "source": str(path),
            }
        )
    return rows


def fresh_best_single_row(dataset: str, setting: str, seed: int, parties, xs, adjs, y, val_idx, test_idx, num_classes: int) -> Dict[str, object]:
    probs = ENGINE.get_party_probs(parties, xs, adjs)
    vals = [ENGINE.accuracy_from_probs(prob, y, val_idx) for prob in probs]
    best = int(np.argmax(vals))
    return {
        "dataset": dataset.upper(),
        "setting": setting.lower(),
        "seed": seed,
        "method": "Best Single",
        "interface_type": "prediction-space reference",
        "d_ref": "N/A",
        "selected_by": "fresh_local_validation_accuracy",
        "best_val": float(vals[best]),
        "test_at_best_val": float(ENGINE.accuracy_from_probs(probs[best], y, test_idx)),
        "macro_f1": float(ENGINE.macro_f1_from_probs(probs[best], y, test_idx, num_classes)),
        "best_epoch": "N/A",
        "first_loss": "N/A",
        "best_loss": "N/A",
        "param_count": 0,
        "source": "fresh_frozen_local_predictors",
    }


def build_fresh_prediction_cache(
    ns: argparse.Namespace,
    seed: int,
    fresh_probs: Dict[str, torch.Tensor],
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    target: str,
    view_names: Sequence[str],
    useful_mask: Sequence[bool],
    permutation: Sequence[int],
    val_epochs: torch.Tensor,
    test_epochs: torch.Tensor,
) -> Dict[str, object]:
    y_cpu = y.detach().cpu()
    val_y = y_cpu[val_idx.detach().cpu()]
    reliabilities = [
        float((fresh_probs["probs_val"][party].argmax(dim=1) == val_y).float().mean().item())
        for party in range(fresh_probs["probs_val"].size(0))
    ]
    split_hash = ENGINE.split_fingerprint(train_idx, val_idx, test_idx)
    training_config = ENGINE.prediction_cache_training_config(ns, seed)
    fingerprint = ENGINE.stable_fingerprint({"training_config": training_config, "split_fingerprint": split_hash})
    return {
        "cache_version": ENGINE.CACHE_VERSION,
        "cache_fingerprint": fingerprint,
        "training_config": training_config,
        "split_fingerprint": split_hash,
        "dataset": ns.dataset,
        "seed": int(seed),
        "target_node_type": target,
        "train_idx": train_idx.detach().cpu(),
        "val_idx": val_idx.detach().cpu(),
        "test_idx": test_idx.detach().cpu(),
        "y": y_cpu,
        "probs_train": fresh_probs["probs_train"],
        "probs_val": fresh_probs["probs_val"],
        "probs_test": fresh_probs["probs_test"],
        "probs_all": fresh_probs["probs_all"],
        "probs_val_epochs": val_epochs.detach().cpu(),
        "probs_test_epochs": test_epochs.detach().cpu(),
        "party_reliabilities": torch.tensor(reliabilities, dtype=torch.float32),
        "party_permutation": list(permutation),
        "useful_party_mask": [bool(value) for value in useful_mask],
        "view_names": list(view_names),
        "num_classes": int(fresh_probs["probs_all"].size(-1)),
        "num_parties": int(fresh_probs["probs_all"].size(0)),
        "target_node_count": int(fresh_probs["probs_all"].size(1)),
    }


def fresh_prediction_reference_rows(
    dataset: str,
    setting: str,
    seed: int,
    ns: argparse.Namespace,
    fresh_probs: Dict[str, torch.Tensor],
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    target: str,
    view_names: Sequence[str],
    useful_mask: Sequence[bool],
    permutation: Sequence[int],
    num_classes: int,
    val_epochs: torch.Tensor,
    test_epochs: torch.Tensor,
) -> List[Dict[str, object]]:
    rows = []
    cache = build_fresh_prediction_cache(
        ns,
        seed,
        fresh_probs,
        y,
        train_idx,
        val_idx,
        test_idx,
        target,
        view_names,
        useful_mask,
        permutation,
        val_epochs,
        test_epochs,
    )
    cache_path = OUTPUT_DIR / "fresh_within_suite_cache" / f"{dataset.lower()}_{setting.lower()}_seed{seed}_fresh_final.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    method_names = {
        "single_party_best": "Best Single",
        "topk_reliability_vote": "Global Top-k Reliability",
        "full_mesh": "Full Mesh prediction consensus",
        "adaptive_graph_val": "TA-DVFG",
    }
    for engine_method, label in method_names.items():
        result = ENGINE.evaluate_cached_method(engine_method, seed, ns, cache, cache_path)
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "method": label,
                "interface_type": "fresh prediction-space within-suite reference",
                "d_ref": "N/A",
                "selected_by": "fresh_final_predictor_cache",
                "best_val": float(result.best_val),
                "test_at_best_val": float(result.test_at_best_val),
                "macro_f1": float(result.macro_f1_at_best_val),
                "best_epoch": "fresh_final",
                "first_loss": "N/A",
                "best_loss": "N/A",
                "param_count": 0,
                "source": str(cache_path),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = per_seed[pd.to_numeric(per_seed["test_at_best_val"], errors="coerce").notna()].copy()
    numeric["test_at_best_val"] = numeric["test_at_best_val"].astype(float)
    numeric["macro_f1"] = numeric["macro_f1"].astype(float)
    grouped = numeric.groupby(["dataset", "setting", "method"], as_index=False).agg(
        test_at_best_val_mean=("test_at_best_val", "mean"),
        test_at_best_val_std=("test_at_best_val", lambda x: float(np.std(x, ddof=0))),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", lambda x: float(np.std(x, ddof=0))),
        seeds=("seed", lambda x: ",".join(str(int(v)) for v in sorted(x))),
    )
    return grouped


def paired_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = per_seed[pd.to_numeric(per_seed["test_at_best_val"], errors="coerce").notna()].copy()
    numeric["test_at_best_val"] = numeric["test_at_best_val"].astype(float)
    for (dataset, setting), group in numeric.groupby(["dataset", "setting"]):
        pivot = group.pivot_table(index="seed", columns="method", values="test_at_best_val", aggfunc="first")
        if "TA-DVFG" not in pivot:
            continue
        for method in pivot.columns:
            if method == "TA-DVFG":
                continue
            common = pivot[["TA-DVFG", method]].dropna()
            if common.empty:
                continue
            delta = common[method] - common["TA-DVFG"]
            mean_delta = float(delta.mean())
            std_delta = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
            ci_half = 1.96 * std_delta / math.sqrt(len(delta)) if len(delta) > 1 else 0.0
            dz = mean_delta / std_delta if std_delta > 0 else math.nan
            rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "method": method,
                    "n": int(len(delta)),
                    "mean_delta_vs_tadvfg": mean_delta,
                    "ci95_low": mean_delta - ci_half,
                    "ci95_high": mean_delta + ci_half,
                    "cohen_dz": dz,
                    "wilcoxon_note": "not run: scipy not required; with pilot n=1 this is not interpretable",
                }
            )
    return pd.DataFrame(rows)


def hidden_scale_rows(
    dataset: str,
    setting: str,
    seed: int,
    hiddens: Sequence[torch.Tensor],
    party_ids: Sequence[int],
    selected_runs: Sequence[TrainResult],
) -> List[Dict[str, object]]:
    rows = []
    for local_position, (party_id, hidden) in enumerate(zip(party_ids, hiddens)):
        h_cpu = hidden.detach().cpu()
        norms = torch.linalg.vector_norm(h_cpu, dim=1)
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "method": "raw_hidden",
                "party": party_id,
                "d_ref": "N/A",
                "hidden_mean": float(h_cpu.mean().item()),
                "hidden_std": float(h_cpu.std(unbiased=False).item()),
                "hidden_l2_mean": float(norms.mean().item()),
                "hidden_l2_std": float(norms.std(unbiased=False).item()),
                "projected_l2_mean": "N/A",
                "projected_l2_std": "N/A",
            }
        )
    for run in selected_runs:
        if run.state_dict is None:
            continue
        in_dims = [int(hidden.size(1)) for hidden in hiddens]
        base_method = run.method.replace("Reliability-Selected ", "").replace("Useful-Only ", "")
        if base_method == "best latent fusion":
            continue
        # Rebuild with the actual class count inferred from classifier weight.
        class_count = 0
        for key, value in run.state_dict.items():
            if key.endswith("classifier.weight") or key.endswith("classifier.2.weight"):
                class_count = int(value.size(0))
        if class_count <= 0:
            continue
        model = make_model(base_method, in_dims, int(run.d_ref), class_count)
        model.load_state_dict(run.state_dict)
        model.eval()
        with torch.no_grad():
            for local_position, (party_id, hidden) in enumerate(zip(party_ids, hiddens)):
                projector = model.projectors[local_position]
                projected = F.relu(projector(hidden.detach().cpu()))
                pnorm = torch.linalg.vector_norm(projected, dim=1)
                h_cpu = hidden.detach().cpu()
                hnorm = torch.linalg.vector_norm(h_cpu, dim=1)
                rows.append(
                    {
                        "dataset": dataset.upper(),
                        "setting": setting.lower(),
                        "seed": seed,
                        "method": run.method,
                        "party": party_id,
                        "d_ref": run.d_ref,
                        "hidden_mean": float(h_cpu.mean().item()),
                        "hidden_std": float(h_cpu.std(unbiased=False).item()),
                        "hidden_l2_mean": float(hnorm.mean().item()),
                        "hidden_l2_std": float(hnorm.std(unbiased=False).item()),
                        "projected_l2_mean": float(pnorm.mean().item()),
                        "projected_l2_std": float(pnorm.std(unbiased=False).item()),
                    }
                )
    return rows


def write_gate_plot(path: Path, gate_rows: Sequence[Dict[str, object]]) -> None:
    if not gate_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.with_suffix(".txt").write_text("matplotlib unavailable; gate plot not generated\n", encoding="utf-8")
        return
    for row in gate_rows:
        weights = json.loads(row["average_gate_weight_by_party"])
        labels = json.loads(row["party_ids"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar([str(v) for v in labels], weights)
        ax.set_ylim(0, max(1.0, max(weights) * 1.1))
        ax.set_xlabel("Party")
        ax.set_ylabel("Average gate weight")
        ax.set_title(f"{row['method']} gates, {row['dataset']} {row['setting']} seed {row['seed']}")
        fig.tight_layout()
        suffix = row["method"].lower().replace(" ", "_").replace("-", "_")
        fig.savefig(path.parent / f"{path.stem}_{suffix}.png", dpi=180)
        plt.close(fig)


def write_cache_comparability_report(path: Path, frame: pd.DataFrame, metadata: Sequence[Dict[str, object]]) -> None:
    if frame.empty:
        path.write_text("# Cache Comparability Report\n\nNo cache audit rows were produced.\n", encoding="utf-8")
        return
    pass_count = int((frame["status"] == "PASS").sum()) if "status" in frame else 0
    mismatch_count = int((frame["status"] == "MISMATCH").sum()) if "status" in frame else 0
    unresolved_count = int((frame["status"] == "UNRESOLVED").sum()) if "status" in frame else 0
    max_prob = pd.to_numeric(frame["max_probability_diff"], errors="coerce").max()
    val_disagree = pd.to_numeric(frame["val_prediction_disagreement_count"], errors="coerce").sum()
    test_disagree = pd.to_numeric(frame["test_prediction_disagreement_count"], errors="coerce").sum()
    cache_meta = metadata[0].get("cache_comparability", {}) if metadata else {}
    if mismatch_count == 0 and unresolved_count == 0:
        decision = "Fresh local predictors match the headline cache probabilities/predictions; direct cache mixing is allowed for probability-level references."
    else:
        decision = "Fresh local predictors do not fully match or could not be resolved; use only fresh within-suite prediction references."
    text = [
        "# Cache Comparability Report",
        "",
        f"Cache path: `{cache_meta.get('cache_path', '')}`",
        f"Cache load status: `{cache_meta.get('cache_load_status', '')}`",
        f"Cache has logits: `{cache_meta.get('cache_has_logits', False)}`",
        "",
        f"Party rows passing: {pass_count}",
        f"Party rows mismatching: {mismatch_count}",
        f"Party rows unresolved: {unresolved_count}",
        f"Maximum probability difference: {max_prob}",
        f"Validation prediction disagreements: {int(val_disagree) if not pd.isna(val_disagree) else 'N/A'}",
        f"Test prediction disagreements: {int(test_disagree) if not pd.isna(test_disagree) else 'N/A'}",
        "",
        "Logit comparison note: the legacy headline cache stores probabilities but not logits, so logit-level cache comparison is marked unavailable.",
        "",
        f"Decision: {decision}",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def write_tradeoff_plot(path: Path, summary: pd.DataFrame, communication: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.with_suffix(".txt").write_text("matplotlib unavailable; tradeoff plot not generated\n", encoding="utf-8")
        return
    merged = summary.merge(communication, on=["dataset", "setting", "method"], how="left")
    merged["per_inference_projected_payload_scalars"] = pd.to_numeric(
        merged["per_inference_projected_payload_scalars"], errors="coerce"
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in merged.iterrows():
        ax.scatter(row["per_inference_projected_payload_scalars"], row["test_at_best_val_mean"], s=45)
        ax.annotate(row["method"], (row["per_inference_projected_payload_scalars"], row["test_at_best_val_mean"]), fontsize=7)
    ax.set_xlabel("Per-target projected representation payload (scalars)")
    ax.set_ylabel("Test@BestVal accuracy")
    ax.set_title("Alignment-reference pilot trade-off")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    equivalence: pd.DataFrame,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    communication: pd.DataFrame,
    local_losses: Sequence[float],
) -> None:
    eq_pass = bool((equivalence["status"] == "PASS").all()) if not equivalence.empty else False
    align_summary = summary[summary["method"].isin(["Project-and-Mean", "Project-and-Concat", "Gated Fusion"])]
    best_row = align_summary.sort_values("test_at_best_val_mean", ascending=False).head(1)
    best_text = "N/A"
    if not best_row.empty:
        row = best_row.iloc[0]
        best_text = f"{row['method']} ({row['test_at_best_val_mean']:.4f} Test@BestVal)"
    tadvfg = summary[summary["method"] == "TA-DVFG"]
    stronger = align_summary["test_at_best_val_mean"].max() if not align_summary.empty else math.nan
    tadvfg_score = tadvfg["test_at_best_val_mean"].iloc[0] if not tadvfg.empty else math.nan
    if math.isnan(stronger) or math.isnan(tadvfg_score):
        comparison = "Insufficient comparison rows."
    elif stronger > tadvfg_score:
        comparison = "A stronger-information alignment reference is higher in this pilot."
    elif stronger == tadvfg_score:
        comparison = "The best stronger-information alignment reference ties TA-DVFG in this pilot."
    else:
        comparison = "TA-DVFG is higher than the tested stronger-information alignment references in this pilot."

    text = [
        "# Alignment Reference Report",
        "",
        f"Scope: HGB {args.dataset.upper()} {args.setting.lower()}, seeds {args.seeds}.",
        "",
        "## Hidden-State Export",
        "",
        f"Status: {'PASS' if eq_pass else 'FAIL'}",
        "",
        "The export path calls the existing frozen `LocalGCN.encode()` and then the unchanged local `cls` head. "
        "No hidden states are synthesized from logits or probabilities.",
        "",
        "## Pilot Training Check",
        "",
        f"Local predictor average training loss snapshots: {', '.join(f'{v:.4f}' for v in local_losses)}",
        "",
        "## Best Alignment Reference",
        "",
        best_text,
        "",
        "## Comparison to TA-DVFG",
        "",
        comparison,
        "",
        "These rows are centralized stronger-information latent-alignment references, not communication-matched P2P baselines.",
        "",
        "## Communication Boundary",
        "",
        "Representation references require hidden-state collection and centralized fusion. TA-DVFG uses prediction-space "
        "peer communication and supports sparse deployment. The payload columns are therefore reported separately rather "
        "than collapsed into one shared communication number.",
        "",
    ]
    (OUTPUT_DIR / "ALIGNMENT_REFERENCE_REPORT.md").write_text("\n".join(text), encoding="utf-8")


def write_paper_patch_files(summary: pd.DataFrame) -> None:
    note = (
        "No paper patch generated from the seed-42 pilot. "
        "Use only after the ACM Hard five-seed suite or the full alignment-reference suite is complete.\n"
    )
    main = "% AUTO-PATCH SLOT: ALIGNMENT_REFERENCE_RESULTS\n% No patch generated from pilot-only alignment-reference evidence.\n"
    (OUTPUT_DIR / "PAPER_PATCH_NOTES.md").write_text(note, encoding="utf-8")
    (OUTPUT_DIR / "main_text_patch.tex").write_text(main, encoding="utf-8")
    (OUTPUT_DIR / "paper_patch_macros.tex").write_text("% Pilot-only; no cross-seed macros generated.\n", encoding="utf-8")
    table_cols = ["dataset", "setting", "method", "test_at_best_val_mean", "macro_f1_mean", "seeds"]
    existing = [col for col in table_cols if col in summary.columns]
    summary[existing].to_latex(OUTPUT_DIR / "supplement_table.tex", index=False, escape=True)


def run_one(args: argparse.Namespace, dataset: str, setting: str, seed: int):
    ns = build_runtime_args(args, dataset, setting, seed)
    (
        x,
        y,
        train_idx,
        val_idx,
        test_idx,
        xs,
        adjs,
        parties,
        view_names,
        useful_mask,
        permutation,
        target,
        losses,
        val_epochs,
        test_epochs,
    ) = train_frozen_parties(ns, seed)
    num_classes = int(y.max().item() + 1)
    equivalence = hidden_export_equivalence(dataset, setting, seed, parties, xs, adjs, y, val_idx, test_idx)
    if not (equivalence["status"] == "PASS").all():
        raise RuntimeError("Hidden export equivalence failed; aborting alignment references.")
    cache_audit, cache_report, fresh_probs = cache_comparability_audit(
        dataset,
        setting,
        seed,
        ns,
        parties,
        xs,
        adjs,
        y,
        train_idx,
        val_idx,
        test_idx,
        num_classes,
    )
    hiddens = ENGINE.get_party_embeddings(parties, xs, adjs)
    hiddens = [hidden.detach() for hidden in hiddens]
    in_dims = [int(hidden.size(1)) for hidden in hiddens]
    rows = fresh_prediction_reference_rows(
        dataset,
        setting,
        seed,
        ns,
        fresh_probs,
        y,
        train_idx,
        val_idx,
        test_idx,
        target,
        view_names,
        useful_mask,
        permutation,
        num_classes,
        val_epochs,
        test_epochs,
    )
    selection_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    scale_rows: List[Dict[str, object]] = []
    selected_runs: List[Tuple[TrainResult, List[torch.Tensor], List[int]]] = []

    method_specs: List[Tuple[str, str, List[torch.Tensor], List[int], str]] = [
        ("Project-and-Mean", "Project-and-Mean", hiddens, list(range(len(hiddens))), "main"),
        ("Project-and-Concat", "Project-and-Concat", hiddens, list(range(len(hiddens))), "main"),
        ("Gated Fusion", "Gated Fusion", hiddens, list(range(len(hiddens))), "main"),
    ]
    reliability_order = sorted(
        range(len(hiddens)),
        key=lambda party: float(fresh_probs["probs_val"][party].argmax(dim=1).eq(y.detach().cpu()[val_idx.detach().cpu()]).float().mean().item()),
        reverse=True,
    )
    topk_party_ids = reliability_order[: int(args.selected_top_k)]
    method_specs.extend(
        [
            (
                "Reliability-Selected Project-and-Mean",
                "Project-and-Mean",
                [hiddens[i] for i in topk_party_ids],
                topk_party_ids,
                "main",
            ),
            (
                "Reliability-Selected Gated Fusion",
                "Gated Fusion",
                [hiddens[i] for i in topk_party_ids],
                topk_party_ids,
                "main",
            ),
        ]
    )
    useful_party_ids = [idx for idx, is_useful in enumerate(useful_mask) if is_useful]
    diagnostic_candidates: List[TrainResult] = []
    diagnostic_selection: List[TrainResult] = []

    for output_method, model_method, method_hiddens, party_ids, role in method_specs:
        best, all_runs = select_alignment_reference(
            model_method,
            parse_dims(args.d_ref_grid),
            method_hiddens,
            y,
            train_idx,
            val_idx,
            test_idx,
            num_classes,
            args,
        )
        best.method = output_method
        selected_runs.append((best, method_hiddens, party_ids))
        for run in all_runs:
            run_name = output_method
            selection_rows.append(
                {
                    "dataset": dataset.upper(),
                    "setting": setting.lower(),
                    "seed": seed,
                    "method": run_name,
                    "d_ref": run.d_ref,
                    "lr": run.lr,
                    "weight_decay": run.weight_decay,
                    "val_acc": run.val_acc,
                    "test_acc": run.test_acc,
                    "macro_f1": run.macro_f1,
                    "best_epoch": run.best_epoch,
                    "first_loss": run.first_loss,
                    "best_loss": run.best_loss,
                    "param_count": run.param_count,
                    "party_ids": json.dumps(party_ids),
                    "role": role,
                    "selected": run.d_ref == best.d_ref and run.lr == best.lr and run.weight_decay == best.weight_decay,
                }
            )
            for epoch, (loss_value, val_value) in enumerate(zip(run.train_loss_curve, run.val_metric_curve)):
                curve_rows.append(
                    {
                        "dataset": dataset.upper(),
                        "setting": setting.lower(),
                        "seed": seed,
                        "method": run_name,
                        "d_ref": run.d_ref,
                        "lr": run.lr,
                        "weight_decay": run.weight_decay,
                        "epoch": epoch,
                        "train_loss": loss_value,
                        "val_metric": val_value,
                        "selected_run": run.d_ref == best.d_ref and run.lr == best.lr and run.weight_decay == best.weight_decay,
                    }
                )
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "method": output_method,
                "interface_type": "centralized stronger-information latent-alignment reference",
                "d_ref": best.d_ref,
                "selected_by": "validation_accuracy",
                "best_val": best.val_acc,
                "test_at_best_val": best.test_acc,
                "macro_f1": best.macro_f1,
                "best_epoch": best.best_epoch,
                "first_loss": best.first_loss,
                "best_loss": best.best_loss,
                "param_count": best.param_count,
                "source": "fresh_hidden_export",
            }
        )
        if best.gate_mean is not None:
            gate_rows.append(
                {
                    "dataset": dataset.upper(),
                    "setting": setting.lower(),
                    "seed": seed,
                    "method": output_method,
                    "party_ids": json.dumps(party_ids),
                    "average_gate_weight_by_party": json.dumps(best.gate_mean),
                    "gate_entropy": best.gate_entropy,
                    "dominant_gate_fraction_ge_0_8": best.gate_dominant_fraction,
                    "max_average_gate_party": party_ids[int(np.argmax(best.gate_mean))],
                    "selected_party_reliabilities": json.dumps([
                        float(fresh_probs["probs_val"][i].argmax(dim=1).eq(y.detach().cpu()[val_idx.detach().cpu()]).float().mean().item())
                        for i in party_ids
                    ]),
                }
            )

    if useful_party_ids:
        for model_method in ["Project-and-Mean", "Gated Fusion"]:
            best, all_runs = select_alignment_reference(
                model_method,
                parse_dims(args.d_ref_grid),
                [hiddens[i] for i in useful_party_ids],
                y,
                train_idx,
                val_idx,
                test_idx,
                num_classes,
                args,
            )
            diagnostic_candidates.append(best)
            diagnostic_selection.extend(all_runs)
        useful_best = sorted(diagnostic_candidates, key=lambda item: (item.val_acc, item.test_acc), reverse=True)[0]
        selected_runs.append((useful_best, [hiddens[i] for i in useful_party_ids], useful_party_ids))
        rows.append(
            {
                "dataset": dataset.upper(),
                "setting": setting.lower(),
                "seed": seed,
                "method": "Useful-Only best latent fusion",
                "interface_type": "diagnostic oracle: known useful parties only",
                "d_ref": useful_best.d_ref,
                "selected_by": "validation_accuracy_over_oracle_candidates",
                "best_val": useful_best.val_acc,
                "test_at_best_val": useful_best.test_acc,
                "macro_f1": useful_best.macro_f1,
                "best_epoch": useful_best.best_epoch,
                "first_loss": useful_best.first_loss,
                "best_loss": useful_best.best_loss,
                "param_count": useful_best.param_count,
                "source": "fresh_hidden_export_known_useful_only",
            }
        )
        for run in diagnostic_selection:
            selection_rows.append(
                {
                    "dataset": dataset.upper(),
                    "setting": setting.lower(),
                    "seed": seed,
                    "method": f"Useful-Only {run.method}",
                    "d_ref": run.d_ref,
                    "lr": run.lr,
                    "weight_decay": run.weight_decay,
                    "val_acc": run.val_acc,
                    "test_acc": run.test_acc,
                    "macro_f1": run.macro_f1,
                    "best_epoch": run.best_epoch,
                    "first_loss": run.first_loss,
                    "best_loss": run.best_loss,
                    "param_count": run.param_count,
                    "party_ids": json.dumps(useful_party_ids),
                    "role": "diagnostic_oracle",
                    "selected": run.d_ref == useful_best.d_ref and run.lr == useful_best.lr and run.weight_decay == useful_best.weight_decay,
                }
            )
    for selected, method_hiddens, party_ids in selected_runs:
        scale_rows.extend(hidden_scale_rows(dataset, setting, seed, method_hiddens, party_ids, [selected]))
    comm_rows = []
    n_nodes = int(y.numel())
    for row in rows:
        method = row["method"]
        if method in {
            "Project-and-Mean",
            "Project-and-Concat",
            "Gated Fusion",
            "Reliability-Selected Project-and-Mean",
            "Reliability-Selected Gated Fusion",
            "Useful-Only best latent fusion",
        }:
            d_ref = int(row["d_ref"])
            if method.startswith("Reliability-Selected"):
                party_count = len(topk_party_ids)
                selected_dims = [in_dims[i] for i in topk_party_ids]
            elif method == "Useful-Only best latent fusion":
                party_count = len(useful_party_ids)
                selected_dims = [in_dims[i] for i in useful_party_ids]
            else:
                party_count = len(in_dims)
                selected_dims = in_dims
            per_node_projected = party_count * d_ref
            cached_hidden = n_nodes * sum(selected_dims)
            comm_rows.append(
                {
                    "dataset": dataset.upper(),
                    "setting": setting.lower(),
                    "seed": seed,
                    "method": method,
                    "original_hidden_dims": json.dumps(selected_dims),
                    "d_ref": d_ref,
                    "param_count": row["param_count"],
                    "one_time_hidden_cache_payload_scalars": cached_hidden,
                    "per_inference_projected_payload_scalars": per_node_projected,
                    "centralized_fusion_required": True,
                    "party_local_post_consensus_supported": False,
                }
            )
        else:
            comm_rows.append(
                {
                    "dataset": dataset.upper(),
                    "setting": setting.lower(),
                    "seed": seed,
                    "method": method,
                    "original_hidden_dims": "N/A",
                    "d_ref": "N/A",
                    "param_count": row.get("param_count", 0),
                    "one_time_hidden_cache_payload_scalars": "N/A",
                    "per_inference_projected_payload_scalars": 0,
                    "centralized_fusion_required": False,
                    "party_local_post_consensus_supported": method == "TA-DVFG",
                }
            )
    metadata = {
        "dataset": dataset.upper(),
        "setting": setting.lower(),
        "seed": seed,
        "target_node_type": target,
        "num_nodes": n_nodes,
        "num_classes": num_classes,
        "view_names": view_names,
        "useful_party_mask": useful_mask,
        "party_permutation": permutation,
        "local_epochs": int(ns.epochs),
        "fusion_epochs": int(args.fusion_epochs),
        "d_ref_grid": parse_dims(args.d_ref_grid),
        "local_loss_snapshots": losses,
        "cache_comparability": cache_report,
        "topk_party_ids": topk_party_ids,
        "useful_party_ids": useful_party_ids,
    }
    return equivalence, cache_audit, rows, selection_rows, curve_rows, scale_rows, gate_rows, comm_rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ACM")
    parser.add_argument("--setting", choices=["main", "hard"], default="hard")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-epochs", type=int, default=0, help="0 uses the headline setting config.")
    parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--fusion-lr", type=float, default=0.01, help="Deprecated; use --fusion-lr-grid.")
    parser.add_argument("--fusion-weight-decay", type=float, default=5e-4, help="Deprecated; use --fusion-weight-decay-grid.")
    parser.add_argument("--fusion-lr-grid", default="0.001,0.0003")
    parser.add_argument("--fusion-weight-decay-grid", default="0,0.0001")
    parser.add_argument("--d-ref-grid", default="32,64,128")
    parser.add_argument("--selected-top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE.DEVICE = ENGINE.resolve_device(args.device)

    all_equivalence = []
    all_cache_audit = []
    all_rows: List[Dict[str, object]] = []
    all_selection: List[Dict[str, object]] = []
    all_curves: List[Dict[str, object]] = []
    all_scales: List[Dict[str, object]] = []
    all_gates: List[Dict[str, object]] = []
    all_comm: List[Dict[str, object]] = []
    metadata = []
    loss_snapshots: List[float] = []
    for seed in [int(item.strip()) for item in args.seeds.split(",") if item.strip()]:
        equivalence, cache_audit, rows, selection_rows, curve_rows, scale_rows, gate_rows, comm_rows, meta = run_one(args, args.dataset, args.setting, seed)
        all_equivalence.append(equivalence)
        all_cache_audit.append(cache_audit)
        all_rows.extend(rows)
        all_selection.extend(selection_rows)
        all_curves.extend(curve_rows)
        all_scales.extend(scale_rows)
        all_gates.extend(gate_rows)
        all_comm.extend(comm_rows)
        metadata.append(meta)
        loss_snapshots.extend(meta["local_loss_snapshots"])

    equivalence_frame = pd.concat(all_equivalence, ignore_index=True) if all_equivalence else pd.DataFrame()
    cache_audit_frame = pd.concat(all_cache_audit, ignore_index=True) if all_cache_audit else pd.DataFrame()
    per_seed = pd.DataFrame(all_rows)
    selection = pd.DataFrame(all_selection)
    curves = pd.DataFrame(all_curves)
    scales = pd.DataFrame(all_scales)
    gates = pd.DataFrame(all_gates)
    comm = pd.DataFrame(all_comm)
    summary = summarize(per_seed)
    stats = paired_stats(per_seed)
    comm_summary = comm.groupby(["dataset", "setting", "method"], as_index=False).agg(
        original_hidden_dims=("original_hidden_dims", "first"),
        d_ref=("d_ref", "first"),
        param_count=("param_count", "first"),
        one_time_hidden_cache_payload_scalars=("one_time_hidden_cache_payload_scalars", "first"),
        per_inference_projected_payload_scalars=("per_inference_projected_payload_scalars", "first"),
        centralized_fusion_required=("centralized_fusion_required", "first"),
        party_local_post_consensus_supported=("party_local_post_consensus_supported", "first"),
    )

    equivalence_frame.to_csv(OUTPUT_DIR / "hidden_export_equivalence.csv", index=False)
    cache_audit_frame.to_csv(OUTPUT_DIR / "cache_comparability_audit.csv", index=False)
    per_seed.to_csv(OUTPUT_DIR / "alignment_reference_per_seed.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "alignment_reference_selection_grid.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "alignment_train_curves.csv", index=False)
    scales.to_csv(OUTPUT_DIR / "hidden_scale_audit.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "gate_diagnostics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "alignment_reference_summary.csv", index=False)
    stats.to_csv(OUTPUT_DIR / "alignment_reference_stats.csv", index=False)
    comm_summary.to_csv(OUTPUT_DIR / "alignment_reference_communication.csv", index=False)
    (OUTPUT_DIR / "alignment_reference_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_cache_comparability_report(OUTPUT_DIR / "CACHE_COMPARABILITY_REPORT.md", cache_audit_frame, metadata)
    write_gate_plot(OUTPUT_DIR / "gate_weight_diagnostic.png", all_gates)
    write_tradeoff_plot(OUTPUT_DIR / "alignment_reference_tradeoff.png", summary, comm_summary)
    write_report(args, equivalence_frame, per_seed, summary, comm_summary, loss_snapshots)
    write_paper_patch_files(summary)

    if not equivalence_frame.empty and not (equivalence_frame["status"] == "PASS").all():
        print("ALIGNMENT REFERENCE FAIL: hidden export equivalence failed")
        return 2
    print(f"ALIGNMENT REFERENCE PASS: wrote {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
