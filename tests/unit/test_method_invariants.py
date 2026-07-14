from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"


def load_engine():
    name = "ta_dvfg_engine_invariants"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()


def consensus_args(*, steps: int = 1, eta: float = 0.85) -> SimpleNamespace:
    return SimpleNamespace(
        pred_consensus_steps=steps,
        pred_self_weight=eta,
        consensus_mode="standard",
        consensus_reliability_margin=0.0,
    )


def selector_args(*, budget: int, degree: int, minimum: int, tau: float) -> SimpleNamespace:
    return SimpleNamespace(
        max_degree=degree,
        adaptive_edge_budget=budget,
        adaptive_min_edges=minimum,
        adaptive_min_gain=tau,
        adaptive_candidate_edges=0,
        _active_adaptive_score="graph_val",
        validation_evaluator="all_parties",
        active_party_id=0,
    )


def patch_selector(monkeypatch, score):
    def fake_edge_scores(parties, xs, adjs, y, val_idx, args):
        n = len(parties)
        utility = np.zeros((n, n), dtype=np.float32)
        probs = [torch.tensor([[0.6, 0.4]], dtype=torch.float32) for _ in parties]
        return utility, probs, [], [1.0] * n

    monkeypatch.setattr(engine, "score_topology_edges", fake_edge_scores)
    monkeypatch.setattr(
        engine,
        "score_topology_on_validation",
        lambda adj, *unused, **kwargs: float(score(adj)),
    )


def run_selector(args: SimpleNamespace):
    parties = [object()] * 6
    return engine.update_adaptive_topology(
        parties,
        [],
        [],
        torch.tensor([0]),
        torch.tensor([0]),
        args,
    )[0]


def test_consensus_preserves_simplex_membership() -> None:
    probs = [
        torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]),
        torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]),
        torch.tensor([[0.1, 0.1, 0.8], [0.2, 0.2, 0.6]]),
    ]
    adj = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)
    result = engine.post_consensus_party_probs(probs, adj, [0.8, 0.4, 0.2], consensus_args(steps=3))
    for party in result:
        assert torch.all(party >= 0)
        assert torch.allclose(party.sum(dim=1), torch.ones(party.shape[0]), atol=1e-6)


def test_isolated_parties_retain_predictions() -> None:
    probs = [torch.tensor([[0.9, 0.1]]), torch.tensor([[0.2, 0.8]]), torch.tensor([[0.6, 0.4]])]
    adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    result = engine.post_consensus_party_probs(probs, adj, [0.9, 0.8, 0.7], consensus_args())
    assert torch.equal(result[2], probs[2])


def test_reliability_weights_normalize_over_neighbors() -> None:
    probs = [torch.tensor([[0.5, 0.5]]), torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
    result = engine.post_consensus_party_probs(probs, adj, [1.0, 0.0, 0.003], consensus_args(eta=0.0))
    assert torch.allclose(result[0], torch.tensor([[0.25, 0.75]]), atol=1e-6)
    assert torch.isclose(result[0].sum(), torch.tensor(1.0))


def test_active_readout_participant_fallback_and_filtering() -> None:
    empty = np.zeros((4, 4), dtype=np.float32)
    assert engine.active_party_indices(empty, 4) == [0, 1, 2, 3]
    nonempty = empty.copy()
    nonempty[1, 3] = nonempty[3, 1] = 1.0
    assert engine.active_party_indices(nonempty, 4) == [1, 3]


def test_degree_edge_and_iteration_constraints(monkeypatch) -> None:
    patch_selector(monkeypatch, lambda adj: engine.edge_count(adj))
    adj = run_selector(selector_args(budget=4, degree=1, minimum=4, tau=0.0))
    assert engine.edge_count(adj) <= 4
    assert int(adj.sum(axis=1).max()) <= 1
    assert engine.edge_count(adj) <= 4  # at most B accepted additions


def test_m_zero_permits_empty_graph(monkeypatch) -> None:
    patch_selector(monkeypatch, lambda adj: 0.5)
    adj = run_selector(selector_args(budget=5, degree=2, minimum=0, tau=0.001))
    assert engine.edge_count(adj) == 0


def test_budget_is_upper_bound_not_quota(monkeypatch) -> None:
    patch_selector(monkeypatch, lambda adj: 0.5)
    adj = run_selector(selector_args(budget=5, degree=2, minimum=1, tau=0.001))
    assert engine.edge_count(adj) == 1


def test_k_zero_performs_no_peer_exchange() -> None:
    probs = [torch.tensor([[0.9, 0.1]]), torch.tensor([[0.1, 0.9]])]
    adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
    result = engine.post_consensus_party_probs(probs, adj, [0.8, 0.7], consensus_args(steps=0))
    assert all(torch.equal(before, after) for before, after in zip(probs, result))


def test_k_hop_influence_is_limited_on_path() -> None:
    probs = [torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    k1 = engine.post_consensus_party_probs(probs, adj, [1, 1, 1], consensus_args(steps=1, eta=0.0))
    k2 = engine.post_consensus_party_probs(probs, adj, [1, 1, 1], consensus_args(steps=2, eta=0.0))
    assert torch.equal(k1[0], probs[1])
    assert torch.allclose(k2[0], torch.tensor([[0.5, 0.5]]))


def test_sparse_and_full_mesh_communication_formulas() -> None:
    args = SimpleNamespace(num_parties=4, pred_consensus_steps=2, readout_mode="local_mean", topk_reliability_k=2)
    sparse = np.zeros((4, 4), dtype=np.float32)
    sparse[0, 1] = sparse[1, 0] = 1
    sparse[2, 3] = sparse[3, 2] = 1
    peer, readout, total, _ = engine.estimate_communication("adaptive_graph_val", sparse, 7, 3, args)
    assert peer == 2 * 2 * 2 * 7 * 3
    assert readout == 0 and total == peer

    full = engine.make_full_topology(4)
    peer, _, _, _ = engine.estimate_communication("full_mesh", full, 7, 3, args)
    assert peer == 4 * 3 * 2 * 7 * 3


def test_deterministic_tie_breaking_repeats_edges(monkeypatch) -> None:
    patch_selector(monkeypatch, lambda adj: engine.edge_count(adj))
    args = selector_args(budget=2, degree=2, minimum=2, tau=0.0)
    first = engine.topology_edge_list(run_selector(args))
    second = engine.topology_edge_list(run_selector(args))
    assert first == second
    assert first == [[3, 5], [4, 5]]
