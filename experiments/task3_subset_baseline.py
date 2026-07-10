"""Task 3: same-cardinality direct subset baselines."""

from __future__ import annotations

import numpy as np

from closure_core import _mean_for_parties, active_readout, party_reliability, readout_probs, score_predictions
from closure_task_common import evaluate_method, run_task, topology


def _greedy_subset(bundle, cardinality: int) -> list[int]:
    selected: list[int] = []
    candidates = list(range(bundle.n_parties))
    while len(selected) < cardinality and candidates:
        best = None
        for party in candidates:
            trial = selected + [party]
            probs = _mean_for_parties(bundle.val_probs, trial)
            score = score_predictions(probs, bundle.y_val, bundle.metric)
            candidate = (score, -party, party)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        selected.append(best[2])
        candidates.remove(best[2])
    return selected


def evaluate(bundle, args):
    edges = topology(bundle, args, K=max(1, args.K))
    active_parties = sorted({party for edge in edges for party in edge}) or list(range(min(1, bundle.n_parties)))
    cardinality = len(active_parties)
    reliability = party_reliability(bundle)
    top = list(np.argsort(-reliability, kind="stable")[:cardinality])
    greedy = _greedy_subset(bundle, cardinality)
    return [
        evaluate_method(bundle, "TA_DVFG_active_readout", active_readout(bundle, edges), edges),
        evaluate_method(bundle, "TopReliabilitySameCardinality", _mean_for_parties(bundle.test_probs, top), []),
        evaluate_method(bundle, "GreedySubsetSameCardinality", _mean_for_parties(bundle.test_probs, greedy), []),
        evaluate_method(bundle, "ActivePartiesNoTopologyDegree", readout_probs(bundle, edges, degree_weighted=False), edges),
    ]


if __name__ == "__main__":
    raise SystemExit(
        run_task(
            "Task 3 same-cardinality subset baseline comparison.",
            "task3_subset_baseline",
            evaluate,
            [
                ("TA_DVFG_active_readout", "TopReliabilitySameCardinality"),
                ("TA_DVFG_active_readout", "GreedySubsetSameCardinality"),
            ],
        )
    )
