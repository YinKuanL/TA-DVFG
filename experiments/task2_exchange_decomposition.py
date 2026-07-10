"""Task 2: decompose topology selection from prediction exchange."""

from __future__ import annotations

from closure_core import active_readout, local_readout
from closure_task_common import evaluate_method, run_task, topology


def evaluate(bundle, args):
    edges1 = topology(bundle, args, K=max(1, args.K))
    fixed_edges = [(0, 1)] if bundle.n_parties >= 2 else []
    return [
        evaluate_method(bundle, "A_full_K1", active_readout(bundle, edges1), edges1),
        evaluate_method(bundle, "B_reselect_K0", local_readout(bundle), []),
        evaluate_method(bundle, "C_fixed_K1_topology_no_exchange", local_readout(bundle), fixed_edges),
    ]


if __name__ == "__main__":
    raise SystemExit(
        run_task(
            "Task 2 exchange decomposition for cached prediction bundles.",
            "task2_exchange_decomposition",
            evaluate,
            [
                ("A_full_K1", "B_reselect_K0"),
                ("A_full_K1", "C_fixed_K1_topology_no_exchange"),
            ],
        )
    )
