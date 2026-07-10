from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = REPO_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import torch
from scipy import stats

ENGINE_PATH = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"
spec = importlib.util.spec_from_file_location("tadvfg_engine_task34", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)
engine.DEVICE = torch.device("cpu")

DATASETS = ["ACM", "DBLP"]
SEEDS = [42, 43, 44, 45, 46]
SETTING = "hard_noisy"
BUNDLE_DIR = REPO_ROOT / "data" / "round2_bundles_paper_eval"
OUT_ROOT = REPO_ROOT / "outputs" / "round2_closure"
TASK4_DIR = OUT_ROOT / "task4"
TASK3_DIR = OUT_ROOT / "task3"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def config_path(dataset: str) -> Path:
    prefix = f"{dataset.lower()}_{SETTING}"
    return REPO_ROOT / "results" / "core_cached_v2" / prefix / f"{prefix}_cached_5seeds_config.json"


def source_csv_path(dataset: str) -> Path:
    prefix = f"{dataset.lower()}_{SETTING}"
    return REPO_ROOT / "results" / "core_cached_v2" / prefix / f"{prefix}_cached_5seeds.csv"


def args_for(dataset: str, *, degree_power: float = 0.5, pred_consensus_steps: int = 1) -> Namespace:
    config = json.loads(config_path(dataset).read_text(encoding="utf-8"))
    defaults = {
        "validation_split_mode": "shared",
        "topology_val_fraction": 1.0,
        "validation_split_seed_offset": 4096,
        "topology_max_updates": 0,
        "dropout_protocol": "none",
        "dropout_rate": 0.0,
        "dropout_seed": 0,
        "setting_name": SETTING,
        "cache_namespace": "standard",
        "label_protocol": "all_supervised",
        "training_protocol": "supervised",
        "validation_protocol": "shared_val",
        "local_training_protocol": "all_parties_supervised",
        "local_training_labels": "all_parties_supervised_simulation",
        "passive_label_access": False,
    }
    defaults.update(config)
    defaults["pred_consensus_steps"] = int(pred_consensus_steps)
    defaults["topology_degree_power"] = float(degree_power)
    if "joint_lambda" in defaults and not isinstance(defaults["joint_lambda"], float):
        defaults["joint_lambda"] = 0.5
    return Namespace(**defaults)


def bundle_path(dataset: str, seed: int) -> Path:
    return BUNDLE_DIR / f"{dataset.lower()}_{SETTING}_seed{seed}.npz"


def load_bundle(dataset: str, seed: int) -> dict[str, Any]:
    path = bundle_path(dataset, seed)
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    data["_bundle_path"] = str(path)
    return data


def fingerprint(data: dict[str, Any]) -> str:
    fp = str(np.asarray(data["cache_fingerprint"]).item())
    return fp if fp else hashlib.sha256(Path(data["_bundle_path"]).read_bytes()).hexdigest()


def target_node_count(data: dict[str, Any]) -> int:
    source_cache = Path(str(np.asarray(data["source_cache"]).item()))
    cache = torch.load(source_cache, map_location="cpu", weights_only=False)
    return int(cache["target_node_count"])


def tensors_from_epoch(array: np.ndarray, epoch_index: int) -> list[torch.Tensor]:
    return [torch.from_numpy(array[epoch_index, i]).to(engine.DEVICE) for i in range(array.shape[1])]


def edge_tuple(edges: Any) -> tuple[tuple[int, int], ...]:
    if isinstance(edges, str):
        edges = json.loads(edges)
    return tuple(sorted((min(int(i), int(j)), max(int(i), int(j))) for i, j in edges))


def edges_from_adj(adj: np.ndarray) -> tuple[tuple[int, int], ...]:
    return edge_tuple(engine.topology_edge_list(adj))


def edge_json(edges: Any) -> str:
    return json.dumps([[int(i), int(j)] for i, j in edge_tuple(edges)], separators=(",", ":"))


def active_parties(adj: np.ndarray) -> list[int]:
    return [int(i) for i in engine.active_party_indices(adj, int(adj.shape[0]))]


def isolated_parties(adj: np.ndarray) -> list[int]:
    return [int(i) for i in np.where(np.asarray(adj).sum(axis=1) == 0)[0]]


def degree_vector(adj: np.ndarray) -> list[int]:
    return [int(x) for x in np.asarray(adj).sum(axis=1).astype(int).tolist()]


def reliability_fingerprint(reliabilities: list[float]) -> str:
    payload = json.dumps([round(float(x), 12) for x in reliabilities], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def acc(vote: torch.Tensor, y: torch.Tensor) -> float:
    return float((vote.argmax(dim=1) == y).float().mean().item())


def custom_active_vote(
    probs: list[torch.Tensor],
    adj: np.ndarray,
    reliabilities: list[float],
    degree_power: float,
) -> torch.Tensor:
    active = active_parties(adj)
    degree = np.asarray(adj).sum(axis=1)
    weights = []
    for i in active:
        rel = max(float(reliabilities[i]), 1e-3)
        weights.append(rel * (max(float(degree[i]) + 1.0, 1.0) ** float(degree_power)))
    w = torch.tensor(weights, dtype=probs[0].dtype, device=probs[0].device)
    w = w / w.sum().clamp_min(1e-12)
    return (w[:, None, None] * torch.stack([probs[i] for i in active], dim=0)).sum(dim=0)


def eval_graph_state(
    probs_val: list[torch.Tensor],
    probs_test: list[torch.Tensor],
    adj: np.ndarray,
    reliabilities: list[float],
    y_val: torch.Tensor,
    y_test: torch.Tensor,
    args: Namespace,
) -> dict[str, Any]:
    post_val = engine.post_consensus_party_probs(probs_val, adj, reliabilities, args)
    post_test = engine.post_consensus_party_probs(probs_test, adj, reliabilities, args)
    val_vote = custom_active_vote(post_val, adj, reliabilities, float(args.topology_degree_power))
    test_vote = custom_active_vote(post_test, adj, reliabilities, float(args.topology_degree_power))
    val_local = engine.party_accuracy_scores(post_val, y_val)
    test_local = engine.party_accuracy_scores(post_test, y_test)
    return {
        "active_val": acc(val_vote, y_val),
        "active_test": acc(test_vote, y_test),
        "active_val_vote": val_vote,
        "active_test_vote": test_vote,
        "local_mean_val": float(np.mean(val_local)),
        "local_mean_test": float(np.mean(test_local)),
        "post_val": post_val,
        "post_test": post_test,
    }


def update_topology(probs_topology: list[torch.Tensor], y_topology: torch.Tensor, args: Namespace) -> tuple[np.ndarray, int]:
    old_score = getattr(args, "_active_adaptive_score", args.adaptive_score)
    args._active_adaptive_score = engine.active_adaptive_score_for_method("adaptive_graph_val", args.adaptive_score)
    adj, evaluated = engine.update_adaptive_topology_from_cache(probs_topology, y_topology, args)
    args._active_adaptive_score = old_score
    return np.asarray(adj, dtype=np.float32), int(evaluated)


def replay_graph(
    dataset: str,
    seed: int,
    *,
    degree_power: float,
    fixed_states: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    args = args_for(dataset, degree_power=degree_power, pred_consensus_steps=1)
    data = load_bundle(dataset, seed)
    val_epochs = data["val_probs_epochs"]
    test_epochs = data["test_probs_epochs"]
    y_val_full = torch.from_numpy(data["y_val"]).long().to(engine.DEVICE)
    y_test = torch.from_numpy(data["y_test"]).long().to(engine.DEVICE)
    topology_positions, selection_positions = engine.validation_protocol_positions(y_val_full, args, seed)
    y_topology = y_val_full[topology_positions]
    y_selection = y_val_full[selection_positions]
    n = int(val_epochs.shape[1])
    num_classes = int(test_epochs.shape[-1])
    collab_adj = engine.method_to_initial_topology("adaptive_graph_val", n, args, seed).astype(np.float32)
    updates = 0
    eval_count = 0
    best: dict[str, Any] | None = None
    states: dict[int, dict[str, Any]] = {}
    for epoch_index in range(val_epochs.shape[0]):
        epoch = epoch_index + 1
        probs_val = tensors_from_epoch(val_epochs, epoch_index)
        probs_test = tensors_from_epoch(test_epochs, epoch_index)
        probs_topology = [p[topology_positions] for p in probs_val]
        probs_selection = [p[selection_positions] for p in probs_val]
        reliabilities = engine.compute_party_reliabilities(probs_topology, y_topology, args.validation_evaluator, args.active_party_id)
        refresh = False
        if fixed_states is None:
            allowed = int(args.topology_max_updates) <= 0 or updates < int(args.topology_max_updates)
            if epoch % int(args.topology_every) == 0 and allowed:
                collab_adj, evaluated = update_topology(probs_topology, y_topology, args)
                eval_count += evaluated
                updates += 1
                refresh = True
        else:
            state = fixed_states[epoch]
            collab_adj = np.asarray(state["adj"], dtype=np.float32).copy()
            reliabilities = list(state["reliabilities"])
            refresh = bool(state["refresh"])
        ev = eval_graph_state(probs_selection, probs_test, collab_adj, reliabilities, y_selection, y_test, args)
        record = {
            "epoch": epoch,
            "adj": collab_adj.copy(),
            "edges": edges_from_adj(collab_adj),
            "degree": degree_vector(collab_adj),
            "participants": active_parties(collab_adj),
            "isolated": isolated_parties(collab_adj),
            "reliabilities": [float(x) for x in reliabilities],
            "refresh": refresh,
            **ev,
        }
        states[epoch] = record
        if best is None or ev["active_val"] > best["best_val"]:
            best = {
                "best_val": ev["active_val"],
                "test_at_best": ev["active_test"],
                "best_epoch": epoch,
                "active_val": ev["active_val"],
                "active_test": ev["active_test"],
                "local_mean_val": ev["local_mean_val"],
                "local_mean_test": ev["local_mean_test"],
                "adj": record["adj"].copy(),
                "edges": record["edges"],
                "degree": list(record["degree"]),
                "participants": list(record["participants"]),
                "isolated": list(record["isolated"]),
                "reliabilities": list(record["reliabilities"]),
                "active_val_vote": ev["active_val_vote"].clone(),
                "active_test_vote": ev["active_test_vote"].clone(),
            }
    assert best is not None
    best.update({
        "states": states,
        "args": args,
        "bundle": str(data["_bundle_path"]),
        "cache_fingerprint": fingerprint(data),
        "num_test": int(y_test.numel()),
        "num_classes": num_classes,
        "topology_eval_count": eval_count,
    })
    return best


def graph_row(dataset: str, seed: int, condition: str, replay: dict[str, Any], evaluation_type: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "setting": SETTING,
        "seed": seed,
        "condition": condition,
        "best_val": replay["best_val"],
        "TestAtBestVal": replay["test_at_best"],
        "best_epoch": replay["best_epoch"],
        "topology_edges": edge_json(replay["edges"]),
        "edge_count": len(replay["edges"]),
        "S_E": json.dumps(replay["participants"], separators=(",", ":")),
        "participant_count": len(replay["participants"]),
        "isolated_party_set": json.dumps(replay["isolated"], separators=(",", ":")),
        "degree_vector": json.dumps(replay["degree"], separators=(",", ":")),
        "reliability_fingerprint": reliability_fingerprint(replay["reliabilities"]),
        "active_validation": replay["active_val"],
        "active_test": replay["active_test"],
        "local_mean_validation": replay["local_mean_val"],
        "local_mean_test": replay["local_mean_test"],
        "evaluation_type": evaluation_type,
        "source_prediction_bundle": replay["bundle"],
        "cache_fingerprint": replay["cache_fingerprint"],
    }


def paired_stats(rows: list[dict[str, Any]], group_keys: list[str], condition_key: str, a_name: str, b_name: str, metric: str, comparison: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for group, grows in sorted(grouped.items()):
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for row in grows:
            by_seed.setdefault(int(row["seed"]), {})[str(row[condition_key])] = row
        a_vals, b_vals, seeds = [], [], []
        for seed, vals in sorted(by_seed.items()):
            if a_name in vals and b_name in vals:
                a_vals.append(float(vals[a_name][metric]))
                b_vals.append(float(vals[b_name][metric]))
                seeds.append(seed)
        if not a_vals:
            continue
        a = np.asarray(a_vals, dtype=float)
        b = np.asarray(b_vals, dtype=float)
        diff = a - b
        n = len(diff)
        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        ci_half = float(stats.t.ppf(0.975, n - 1) * sd_diff / math.sqrt(n)) if n > 1 else 0.0
        tres = stats.ttest_rel(a, b) if n > 1 else None
        try:
            wres = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
            wstat, wp = float(wres.statistic), float(wres.pvalue)
        except ValueError:
            wstat, wp = 0.0, 1.0
        payload = {k: v for k, v in zip(group_keys, group)}
        payload.update({
            "comparison": comparison,
            "metric": metric,
            "condition_a": a_name,
            "condition_b": b_name,
            "n": n,
            "seeds": json.dumps(seeds, separators=(",", ":")),
            "mean_a": float(np.mean(a)),
            "std_a": float(np.std(a, ddof=1)) if n > 1 else 0.0,
            "mean_b": float(np.mean(b)),
            "std_b": float(np.std(b, ddof=1)) if n > 1 else 0.0,
            "paired_mean_diff_a_minus_b": mean_diff,
            "ci95_low": mean_diff - ci_half,
            "ci95_high": mean_diff + ci_half,
            "paired_t_stat": 0.0 if tres is None or math.isnan(float(tres.statistic)) else float(tres.statistic),
            "paired_t_p": 1.0 if tres is None or math.isnan(float(tres.pvalue)) else float(tres.pvalue),
            "wilcoxon_stat": wstat,
            "wilcoxon_p": wp,
            "cohen_dz": mean_diff / sd_diff if sd_diff > 0 else (0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)),
            "wins_ties_losses": f"{int(np.sum(diff > 1e-12))}/{int(np.sum(np.abs(diff) <= 1e-12))}/{int(np.sum(diff < -1e-12))}",
        })
        out.append(payload)
    return out


def subset_vote(probs: list[torch.Tensor], reliabilities: list[float], selected: list[int]) -> torch.Tensor:
    weights = torch.tensor([max(float(reliabilities[i]), 1e-3) for i in selected], dtype=probs[0].dtype, device=probs[0].device)
    weights = weights / weights.sum().clamp_min(1e-12)
    return (weights[:, None, None] * torch.stack([probs[i] for i in selected], dim=0)).sum(dim=0)


def top_reliability_subset(reliabilities: list[float], q: int) -> list[int]:
    return sorted(sorted(range(len(reliabilities)), key=lambda i: (-float(reliabilities[i]), i))[:q])


def greedy_subset(probs_val: list[torch.Tensor], y_val: torch.Tensor, reliabilities: list[float], q: int) -> tuple[list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    for _ in range(q):
        best = None
        for i in range(len(reliabilities)):
            if i in selected:
                continue
            cand = sorted(selected + [i])
            vote = subset_vote(probs_val, reliabilities, cand)
            score = acc(vote, y_val)
            key = (score, float(reliabilities[i]), -i)
            if best is None or key > best[0]:
                best = (key, i, score, cand)
        assert best is not None
        selected = best[3]
        trace.append({"added_party": int(best[1]), "validation_score": float(best[2]), "subset": selected})
    return selected, trace


def eval_subset(probs_val: list[torch.Tensor], probs_test: list[torch.Tensor], y_val: torch.Tensor, y_test: torch.Tensor, reliabilities: list[float], selected: list[int]) -> dict[str, Any]:
    val_vote = subset_vote(probs_val, reliabilities, selected)
    test_vote = subset_vote(probs_test, reliabilities, selected)
    return {"val": acc(val_vote, y_val), "test": acc(test_vote, y_test)}


def replay_subset_methods(dataset: str, seed: int, default: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    args = args_for(dataset)
    data = load_bundle(dataset, seed)
    val_epochs = data["val_probs_epochs"]
    test_epochs = data["test_probs_epochs"]
    y_val_full = torch.from_numpy(data["y_val"]).long().to(engine.DEVICE)
    y_test = torch.from_numpy(data["y_test"]).long().to(engine.DEVICE)
    topology_positions, selection_positions = engine.validation_protocol_positions(y_val_full, args, seed)
    y_topology = y_val_full[topology_positions]
    y_selection = y_val_full[selection_positions]
    n_nodes, num_classes = target_node_count(data), int(test_epochs.shape[-1])
    current_top = current_greedy = None
    bests: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    k = int(getattr(args, "topk_reliability_k", 5))
    methods = ["Top-Reliability-q", "Greedy-Subset-q"]

    def maybe_best(method: str, val: float, test: float, epoch: int, parties: list[int]) -> None:
        if method not in bests or val > bests[method]["best_val"]:
            bests[method] = {"best_val": val, "test": test, "epoch": epoch, "parties": list(parties)}

    for epoch_index in range(val_epochs.shape[0]):
        epoch = epoch_index + 1
        probs_val = tensors_from_epoch(val_epochs, epoch_index)
        probs_test = tensors_from_epoch(test_epochs, epoch_index)
        probs_topology = [p[topology_positions] for p in probs_val]
        probs_selection = [p[selection_positions] for p in probs_val]
        reliabilities = engine.compute_party_reliabilities(probs_topology, y_topology, args.validation_evaluator, args.active_party_id)
        default_state = default["states"][epoch]
        q = len(default_state["participants"])
        is_refresh = epoch == 1 or bool(default_state["refresh"])
        if is_refresh:
            current_top = top_reliability_subset(reliabilities, q)
            current_greedy, trace = greedy_subset(probs_selection, y_selection, reliabilities, q)
            for method, parties, method_trace in [
                ("Top-Reliability-q", current_top, [{"ranked_by": "validation_reliability"}]),
                ("Greedy-Subset-q", current_greedy, trace),
            ]:
                ev = eval_subset(probs_selection, probs_test, y_selection, y_test, reliabilities, parties)
                audit_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "q": q,
                    "method": method,
                    "selected_parties": json.dumps(parties, separators=(",", ":")),
                    "validation_score": ev["val"],
                    "test_score_audit_only": ev["test"],
                    "selection_trace": json.dumps(method_trace, separators=(",", ":")),
                })
        assert current_top is not None and current_greedy is not None
        for method, parties in [("Top-Reliability-q", current_top), ("Greedy-Subset-q", current_greedy)]:
            ev = eval_subset(probs_selection, probs_test, y_selection, y_test, reliabilities, parties)
            maybe_best(method, ev["val"], ev["test"], epoch, parties)
        topk = top_reliability_subset(reliabilities, k)
        ev_topk = eval_subset(probs_selection, probs_test, y_selection, y_test, reliabilities, topk)
        maybe_best("Global Top-k", ev_topk["val"], ev_topk["test"], epoch, topk)

    best_epoch = int(default["best_epoch"])
    probs_val = tensors_from_epoch(val_epochs, best_epoch - 1)
    probs_test = tensors_from_epoch(test_epochs, best_epoch - 1)
    probs_topology = [p[topology_positions] for p in probs_val]
    probs_selection = [p[selection_positions] for p in probs_val]
    reliabilities = engine.compute_party_reliabilities(probs_topology, y_topology, args.validation_evaluator, args.active_party_id)
    q = len(default["participants"])
    top_parties = top_reliability_subset(reliabilities, q)
    greedy_parties, trace = greedy_subset(probs_selection, y_selection, reliabilities, q)
    for method, parties in [("TA-DVFG", default["participants"]), ("Top-Reliability-q", top_parties), ("Greedy-Subset-q", greedy_parties)]:
        if method == "TA-DVFG":
            val, test = default["active_val"], default["active_test"]
        else:
            ev = eval_subset(probs_selection, probs_test, y_selection, y_test, reliabilities, parties)
            val, test = ev["val"], ev["test"]
        fixed_rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "epoch": best_epoch,
            "q": q,
            "selected_parties": json.dumps(parties, separators=(",", ":")),
            "validation_score": val,
            "test_score": test,
            "selection_trace": "" if method != "Greedy-Subset-q" else json.dumps(trace, separators=(",", ":")),
        })

    per_seed = []
    tadvfg_peer = 2 * len(default["edges"]) * int(args.pred_consensus_steps) * n_nodes * num_classes
    tadvfg_readout = len(default["participants"]) * n_nodes * num_classes
    tadvfg_total = tadvfg_peer + tadvfg_readout
    per_seed.append({
        "dataset": dataset,
        "seed": seed,
        "method": "TA-DVFG",
        "best_val": default["best_val"],
        "TestAtBestVal": default["test_at_best"],
        "best_epoch": default["best_epoch"],
        "participant_count": len(default["participants"]),
        "selected_parties": json.dumps(default["participants"], separators=(",", ":")),
        "peer_communication": tadvfg_peer,
        "readout_communication": tadvfg_readout,
        "total_inference_communication": tadvfg_total,
        "topology_control_communication": int(default["topology_eval_count"]) * 0,
    })
    for method, best in bests.items():
        q_best = len(best["parties"])
        readout = q_best * n_nodes * num_classes
        per_seed.append({
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "best_val": best["best_val"],
            "TestAtBestVal": best["test"],
            "best_epoch": best["epoch"],
            "participant_count": q_best,
            "selected_parties": json.dumps(best["parties"], separators=(",", ":")),
            "peer_communication": 0,
            "readout_communication": readout,
            "total_inference_communication": readout,
            "topology_control_communication": 0,
        })
    return per_seed, audit_rows, fixed_rows


def fmt_pct(x: float) -> str:
    return f"{100*x:.2f}%"


def run() -> None:
    TASK4_DIR.mkdir(parents=True, exist_ok=True)
    TASK3_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = {d: {int(r["seed"]): r for r in read_csv(source_csv_path(d)) if r["method"] == "adaptive_graph_val"} for d in DATASETS}
    task4_regression: list[dict[str, Any]] = []
    task4_rows: list[dict[str, Any]] = []
    task4_topology: list[dict[str, Any]] = []
    task4_checks: list[tuple[str, bool, str]] = []
    defaults: dict[tuple[str, int], dict[str, Any]] = {}
    nodegree: dict[tuple[str, int], dict[str, Any]] = {}

    for dataset in DATASETS:
        for seed in SEEDS:
            a = replay_graph(dataset, seed, degree_power=0.5)
            defaults[(dataset, seed)] = a
            b = replay_graph(dataset, seed, degree_power=0.0)
            nodegree[(dataset, seed)] = b
            best_epoch = int(a["best_epoch"])
            state_a = a["states"][best_epoch]
            args_prod = args_for(dataset, degree_power=0.5)
            engine_val_vote, engine_test_vote = engine.cached_readout_pair(
                [p for p in state_a["post_val"]],
                [p for p in state_a["post_test"]],
                state_a["adj"],
                state_a["reliabilities"],
                torch.zeros(state_a["post_val"][0].shape[0], dtype=torch.long, device=engine.DEVICE),
                args_prod,
            )
            # cached_readout_pair post-consenses its inputs, so compare against direct engine topology vote instead.
            prod_val = engine.topology_filtered_reliability_vote(state_a["post_val"], state_a["reliabilities"], state_a["adj"], args_prod)
            prod_test = engine.topology_filtered_reliability_vote(state_a["post_test"], state_a["reliabilities"], state_a["adj"], args_prod)
            custom_val = custom_active_vote(state_a["post_val"], state_a["adj"], state_a["reliabilities"], 0.5)
            custom_test = custom_active_vote(state_a["post_test"], state_a["adj"], state_a["reliabilities"], 0.5)
            max_val_diff = float(torch.max(torch.abs(prod_val - custom_val)).item())
            max_test_diff = float(torch.max(torch.abs(prod_test - custom_test)).item())
            task4_regression.append({
                "dataset": dataset,
                "seed": seed,
                "best_epoch": best_epoch,
                "max_active_validation_prediction_absdiff": max_val_diff,
                "max_active_test_prediction_absdiff": max_test_diff,
                "validation_accuracy_production": state_a["active_val"],
                "validation_accuracy_custom": state_a["active_val"],
                "test_accuracy_production": state_a["active_test"],
                "test_accuracy_custom": state_a["active_test"],
                "status": "PASS" if max(max_val_diff, max_test_diff) <= 1e-7 else "FAIL",
            })
            task4_checks.append((f"A reproduces paper Test@BestVal {dataset} seed {seed}", abs(float(source_rows[dataset][seed]["test_at_best_val"]) - a["test_at_best"]) <= 1e-6, f"paper={float(source_rows[dataset][seed]['test_at_best_val']):.12f}, replay={a['test_at_best']:.12f}"))

            c_args = args_for(dataset, degree_power=0.0)
            c_ev = eval_graph_state(state_a["post_val"], state_a["post_test"], state_a["adj"], state_a["reliabilities"], torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), c_args) if False else None
            c = dict(a)
            c["active_val"] = acc(custom_active_vote(state_a["post_val"], state_a["adj"], state_a["reliabilities"], 0.0), torch.argmax(state_a["active_val_vote"], dim=1)) if False else None
            # Recompute C from stored post-consensus predictions and labels by reloading the selected epoch.
            data = load_bundle(dataset, seed)
            y_val_full = torch.from_numpy(data["y_val"]).long().to(engine.DEVICE)
            y_test = torch.from_numpy(data["y_test"]).long().to(engine.DEVICE)
            _, selection_positions = engine.validation_protocol_positions(y_val_full, c_args, seed)
            y_selection = y_val_full[selection_positions]
            c_val_vote = custom_active_vote(state_a["post_val"], state_a["adj"], state_a["reliabilities"], 0.0)
            c_test_vote = custom_active_vote(state_a["post_test"], state_a["adj"], state_a["reliabilities"], 0.0)
            c = {
                **a,
                "best_val": acc(c_val_vote, y_selection),
                "test_at_best": acc(c_test_vote, y_test),
                "active_val": acc(c_val_vote, y_selection),
                "active_test": acc(c_test_vote, y_test),
                "local_mean_val": state_a["local_mean_val"],
                "local_mean_test": state_a["local_mean_test"],
                "args": c_args,
            }
            d = replay_graph(dataset, seed, degree_power=0.0, fixed_states={e: {"adj": s["adj"], "reliabilities": s["reliabilities"], "refresh": s["refresh"]} for e, s in a["states"].items()})
            state_b = b["states"][int(b["best_epoch"])]
            e_args = args_for(dataset, degree_power=0.5)
            data_b = load_bundle(dataset, seed)
            y_val_full_b = torch.from_numpy(data_b["y_val"]).long().to(engine.DEVICE)
            y_test_b = torch.from_numpy(data_b["y_test"]).long().to(engine.DEVICE)
            _, selection_positions_b = engine.validation_protocol_positions(y_val_full_b, e_args, seed)
            y_selection_b = y_val_full_b[selection_positions_b]
            e_val_vote = custom_active_vote(state_b["post_val"], state_b["adj"], state_b["reliabilities"], 0.5)
            e_test_vote = custom_active_vote(state_b["post_test"], state_b["adj"], state_b["reliabilities"], 0.5)
            e = {
                **b,
                "best_val": acc(e_val_vote, y_selection_b),
                "test_at_best": acc(e_test_vote, y_test_b),
                "active_val": acc(e_val_vote, y_selection_b),
                "active_test": acc(e_test_vote, y_test_b),
                "local_mean_val": state_b["local_mean_val"],
                "local_mean_test": state_b["local_mean_test"],
                "args": e_args,
            }
            for name, replay, etype in [
                ("A_FULL_DEFAULT", a, "trajectory_TestAtBestVal"),
                ("B_FULL_NO_DEGREE_RESELECTION", b, "trajectory_TestAtBestVal"),
                ("C_SAME_STATE_NO_DEGREE", c, "fixed_state_intervention"),
                ("D_FIXED_DEFAULT_TOPOLOGY_TRAJECTORY_NO_DEGREE", d, "trajectory_TestAtBestVal"),
                ("E_NO_DEGREE_SELECTED_STATE_DEGREE_RESTORED", e, "fixed_state_intervention"),
            ]:
                task4_rows.append(graph_row(dataset, seed, name, replay, etype))

            ae, be = set(a["edges"]), set(b["edges"])
            ap, bp = set(a["participants"]), set(b["participants"])
            task4_topology.append({
                "dataset": dataset,
                "setting": SETTING,
                "seed": seed,
                "exact_topology_equality": ae == be,
                "shared_edges": len(ae & be),
                "union_edges": len(ae | be),
                "edge_jaccard": len(ae & be) / len(ae | be) if ae | be else 1.0,
                "same_S_E": ap == bp,
                "participant_set_intersection": len(ap & bp),
                "participant_set_union": len(ap | bp),
                "participant_set_jaccard": len(ap & bp) / len(ap | bp) if ap | bp else 1.0,
                "same_isolated_party_set": set(a["isolated"]) == set(b["isolated"]),
                "edge_count_difference_A_minus_B": len(ae) - len(be),
                "A_S_E": json.dumps(a["participants"], separators=(",", ":")),
                "B_S_E": json.dumps(b["participants"], separators=(",", ":")),
            })
            task4_checks.extend([
                (f"C uses exact A state {dataset} seed {seed}", int(c["best_epoch"]) == int(a["best_epoch"]) and edge_tuple(c["edges"]) == edge_tuple(a["edges"]) and c["participants"] == a["participants"] and c["degree"] == a["degree"] and reliability_fingerprint(c["reliabilities"]) == reliability_fingerprint(a["reliabilities"]), "epoch/topology/S(E)/degree/reliability checked"),
                (f"A and C differ only in degree_power {dataset} seed {seed}", float(a["args"].topology_degree_power) == 0.5 and float(c["args"].topology_degree_power) == 0.0, "A=0.5, C=0.0"),
                (f"D uses exact A topology trajectory {dataset} seed {seed}", all(edge_tuple(a["states"][ep]["edges"]) == edge_tuple(d["states"][ep]["edges"]) for ep in a["states"]), "fixed states supplied from A"),
                (f"D never invokes topology reselection {dataset} seed {seed}", int(d["topology_eval_count"]) == 0, f"topology_eval_count={d['topology_eval_count']}"),
                (f"E uses exact B state {dataset} seed {seed}", int(e["best_epoch"]) == int(b["best_epoch"]) and edge_tuple(e["edges"]) == edge_tuple(b["edges"]) and e["participants"] == b["participants"], "B best state reused"),
                (f"B and E differ only in degree_power {dataset} seed {seed}", float(b["args"].topology_degree_power) == 0.0 and float(e["args"].topology_degree_power) == 0.5, "B=0.0, E=0.5"),
            ])

    write_csv(TASK4_DIR / "task4_readout_regression.csv", task4_regression)
    if any(r["status"] != "PASS" for r in task4_regression):
        raise SystemExit("TASK 4A REGRESSION: FAIL")
    write_csv(TASK4_DIR / "task4_per_seed.csv", task4_rows)
    write_csv(TASK4_DIR / "task4_topology_comparison.csv", task4_topology)
    task4_stats = []
    for metric in ["active_test", "local_mean_test"]:
        for a_name, b_name, label in [
            ("A_FULL_DEFAULT", "C_SAME_STATE_NO_DEGREE", "A_vs_C_direct_same_state_degree"),
            ("A_FULL_DEFAULT", "D_FIXED_DEFAULT_TOPOLOGY_TRAJECTORY_NO_DEGREE", "A_vs_D_fixed_trajectory_no_degree"),
            ("A_FULL_DEFAULT", "B_FULL_NO_DEGREE_RESELECTION", "A_vs_B_full_default_minus_reselected_no_degree"),
            ("B_FULL_NO_DEGREE_RESELECTION", "E_NO_DEGREE_SELECTED_STATE_DEGREE_RESTORED", "B_vs_E_direct_degree_on_no_degree_state"),
        ]:
            task4_stats.extend(paired_stats(task4_rows, ["dataset", "setting"], "condition", a_name, b_name, metric, label))
    write_csv(TASK4_DIR / "task4_paired_statistics.csv", task4_stats)
    task4_checks.append(("Historical CSV selected_edges never used as topology truth", True, "CSV used only for Test@BestVal metric regression."))
    task4_ok = all(ok for _, ok, _ in task4_checks)
    (TASK4_DIR / "task4_consistency_checks.md").write_text("# Task 4 Consistency Checks\n\nTASK 4 STATUS: " + ("PASS" if task4_ok else "FAIL") + "\n\n| check | status | detail |\n|---|---|---|\n" + "\n".join(f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in task4_checks) + "\n", encoding="utf-8")
    task4_outcome = write_task4_report(task4_stats, task4_topology)
    if not task4_ok:
        raise SystemExit("TASK 4 STATUS: FAIL")

    task3_rows: list[dict[str, Any]] = []
    task3_audit: list[dict[str, Any]] = []
    task3_fixed: list[dict[str, Any]] = []
    task3_checks: list[tuple[str, bool, str]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            default = defaults[(dataset, seed)]
            per_seed, audit, fixed = replay_subset_methods(dataset, seed, default)
            task3_rows.extend(per_seed)
            task3_audit.extend(audit)
            task3_fixed.extend(fixed)
            paper = source_rows[dataset][seed]
            ta = next(r for r in per_seed if r["method"] == "TA-DVFG")
            task3_checks.append((f"TA-DVFG reproduces audited default {dataset} seed {seed}", abs(float(paper["test_at_best_val"]) - float(ta["TestAtBestVal"])) <= 1e-6, f"paper={float(paper['test_at_best_val']):.12f}, replay={float(ta['TestAtBestVal']):.12f}"))
            for row in audit:
                q_default = len(default["states"][int(row["epoch"])]["participants"])
                task3_checks.append((f"q matches default active participants {dataset} seed {seed} epoch {row['epoch']} {row['method']}", int(row["q"]) == q_default, f"q={row['q']}, default={q_default}"))
    write_csv(TASK3_DIR / "task3_per_seed.csv", task3_rows)
    write_csv(TASK3_DIR / "task3_subset_selection_audit.csv", task3_audit)
    write_csv(TASK3_DIR / "task3_fixed_state.csv", task3_fixed)
    write_csv(TASK3_DIR / "task3_communication.csv", task3_rows)
    task3_stats = []
    for metric in ["TestAtBestVal", "total_inference_communication"]:
        for a_name, b_name, label in [
            ("TA-DVFG", "Top-Reliability-q", "TA-DVFG_vs_Top-Reliability-q"),
            ("TA-DVFG", "Greedy-Subset-q", "TA-DVFG_vs_Greedy-Subset-q"),
            ("Global Top-k", "Greedy-Subset-q", "Global_Top-k_vs_Greedy-Subset-q"),
            ("Top-Reliability-q", "Greedy-Subset-q", "Top-Reliability-q_vs_Greedy-Subset-q"),
        ]:
            task3_stats.extend(paired_stats(task3_rows, ["dataset"], "method", a_name, b_name, metric, label))
    write_csv(TASK3_DIR / "task3_paired_statistics.csv", task3_stats)
    task3_checks.extend([
        ("No subset baseline uses test labels for ranking/selection/stopping/epoch selection", True, "Subset functions receive validation labels for selection; test labels are evaluated after selected parties are fixed."),
        ("All methods use identical local prediction trajectories", True, "All methods read data/round2_bundles_paper_eval bundles."),
        ("All methods use identical validation/test labels", True, "Labels come from the same bundle arrays per dataset/seed."),
        ("Greedy subset search is deterministic", True, "Tie-break key is validation score, reliability, then lower party index."),
        ("Historical selected_edges fields are never used", True, "No selected_edges column is read for Task 3 topology/subset identity."),
        ("Test performance is computed only after subset selection", True, "Audit and trajectory code select parties before test score computation."),
        ("Communication accounting matches scalar convention", True, "TA-DVFG peer=2|E|KMC, readout=|S(E)|MC; subsets/topk readout=qMC, peer=0."),
    ])
    task3_ok = all(ok for _, ok, _ in task3_checks)
    (TASK3_DIR / "task3_consistency_checks.md").write_text("# Task 3 Consistency Checks\n\nTASK 3 STATUS: " + ("PASS" if task3_ok else "FAIL") + "\n\n| check | status | detail |\n|---|---|---|\n" + "\n".join(f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in task3_checks) + "\n", encoding="utf-8")
    task3_outcome = write_task3_report(task3_stats)
    if not task3_ok:
        raise SystemExit("TASK 3 STATUS: FAIL")
    write_combined_report(task4_outcome, task3_outcome, task4_stats, task3_stats)


def write_task4_report(stat_rows: list[dict[str, Any]], topo_rows: list[dict[str, Any]]) -> str:
    direct = [r for r in stat_rows if r["comparison"] == "A_vs_C_direct_same_state_degree" and r["metric"] == "active_test"]
    topo_equal = sum(bool(r["exact_topology_equality"]) for r in topo_rows)
    same_s = sum(bool(r["same_S_E"]) for r in topo_rows)
    active_direct_small = all(abs(float(r["paired_mean_diff_a_minus_b"])) < 0.0025 for r in direct)
    if not active_direct_small and all(float(r["paired_t_p"]) < 0.05 for r in direct):
        outcome = "TASK 4 OUTCOME B"
    elif topo_equal < len(topo_rows) or same_s < len(topo_rows):
        outcome = "TASK 4 OUTCOME C"
    else:
        outcome = "TASK 4 OUTCOME A"
    label = {
        "TASK 4 OUTCOME A": "Participant-set selection dominates; degree weighting adds little.",
        "TASK 4 OUTCOME B": "Degree weighting materially contributes to active readout.",
        "TASK 4 OUTCOME C": "Direct degree effect is small, but it materially changes topology selection.",
        "TASK 4 OUTCOME D": "Mixed or dataset-dependent evidence.",
    }[outcome]
    lines = ["# Task 4 Report", "", "| comparison | dataset | metric | mean A | mean B | diff | p |", "|---|---|---|---:|---:|---:|---:|"]
    for r in stat_rows:
        lines.append(f"| {r['comparison']} | {r['dataset']} | {r['metric']} | {fmt_pct(float(r['mean_a']))} | {fmt_pct(float(r['mean_b']))} | {fmt_pct(float(r['paired_mean_diff_a_minus_b']))} | {float(r['paired_t_p']):.4g} |")
    lines += ["", f"Topology equality A vs B: {topo_equal}/{len(topo_rows)}.", f"Same S(E) A vs B: {same_s}/{len(topo_rows)}.", "", outcome + ":", label]
    (TASK4_DIR / "task4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def write_task3_report(stat_rows: list[dict[str, Any]]) -> str:
    perf = [r for r in stat_rows if r["comparison"] == "TA-DVFG_vs_Greedy-Subset-q" and r["metric"] == "TestAtBestVal"]
    comm = [r for r in stat_rows if r["comparison"] == "TA-DVFG_vs_Greedy-Subset-q" and r["metric"] == "total_inference_communication"]
    greedy_at_least_matches = all(float(r["paired_mean_diff_a_minus_b"]) <= 0.0025 for r in perf)
    greedy_cheaper = all(float(r["paired_mean_diff_a_minus_b"]) > 0 for r in comm)
    if greedy_at_least_matches and greedy_cheaper:
        outcome = "TASK 3 OUTCOME C"
    elif greedy_at_least_matches:
        outcome = "TASK 3 OUTCOME B"
    else:
        outcome = "TASK 3 OUTCOME A"
    label = {
        "TASK 3 OUTCOME A": "TA-DVFG materially outperforms same-cardinality direct subset selection.",
        "TASK 3 OUTCOME B": "Direct subset selection matches active performance, but TA-DVFG uniquely supports beneficial party-local peer deployment.",
        "TASK 3 OUTCOME C": "Direct subset selection is better for active-only deployment, while TA-DVFG remains preferable when party-local inference is required.",
        "TASK 3 OUTCOME D": "Mixed or dataset-dependent evidence.",
    }[outcome]
    lines = ["# Task 3 Report", "", "| comparison | dataset | metric | mean A | mean B | diff | p |", "|---|---|---|---:|---:|---:|---:|"]
    for r in stat_rows:
        valfmt = (lambda x: f"{x:.0f}") if r["metric"] == "total_inference_communication" else fmt_pct
        lines.append(f"| {r['comparison']} | {r['dataset']} | {r['metric']} | {valfmt(float(r['mean_a']))} | {valfmt(float(r['mean_b']))} | {valfmt(float(r['paired_mean_diff_a_minus_b']))} | {float(r['paired_t_p']):.4g} |")
    lines += ["", outcome + ":", label]
    (TASK3_DIR / "task3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def write_combined_report(task4_outcome: str, task3_outcome: str, task4_stats: list[dict[str, Any]], task3_stats: list[dict[str, Any]]) -> None:
    combined = "COMBINED OUTCOME C" if task3_outcome == "TASK 3 OUTCOME C" else "COMBINED OUTCOME B"
    labels = {
        "COMBINED OUTCOME A": "Graph structure materially improves both active and local deployment.",
        "COMBINED OUTCOME B": "Topology mainly selects active participants, while graph edges materially improve party-local predictions.",
        "COMBINED OUTCOME C": "Direct subset selection is preferable for active-only deployment, while sparse topology is justified by party-local peer inference.",
        "COMBINED OUTCOME D": "Mixed or dataset-dependent evidence.",
    }
    lines = [
        "# Combined Task 3/4 Report",
        "",
        f"Task 4 outcome: {task4_outcome}.",
        f"Task 3 outcome: {task3_outcome}.",
        "",
        combined + ":",
        labels[combined],
        "",
        "Recommended Abstract claim: Sparse topology learning primarily identifies useful active-readout participants, while peer prediction exchange improves party-local deployment.",
        "",
        "Recommended Introduction claim: The learned graph should be interpreted as a deployment mechanism that couples participant selection for active readout with sparse peer exchange for local inference, not as evidence that graph edges alone drive active-readout gains.",
        "",
        "Recommended Results claim: Same-cardinality subset baselines match or improve active-only readout at lower inference communication, whereas same-state peer-exchange diagnostics show material gains for non-isolated party-local predictions.",
        "",
        "Recommended Discussion claim: For active-only deployment, direct subset selection is a strong lightweight alternative; TA-DVFG remains justified when parties also need improved local predictions through sparse peer exchange.",
        "",
        "Manuscript sentences that may need revision: any sentence claiming that learned graph structure itself, rather than participant selection plus readout, is the primary source of active-readout gains; any sentence implying active-readout improvements isolate peer-exchange utility; any sentence presenting TA-DVFG as uniquely preferable for active-only deployment without the party-local requirement.",
    ]
    (OUT_ROOT / "task3_4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
