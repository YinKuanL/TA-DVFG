"""Task 4: active readout degree-factor ablation."""

from __future__ import annotations

from closure_core import active_readout, readout_probs
from closure_task_common import evaluate_method, run_task, topology


def evaluate(bundle, args):
    edges = topology(bundle, args, K=max(1, args.K))
    fixed_edges = [(0, 1)] if bundle.n_parties >= 2 else []
    return [
        evaluate_method(bundle, "default_degree_weight", active_readout(bundle, edges, degree_weighted=True), edges),
        evaluate_method(bundle, "no_degree_weight", active_readout(bundle, edges, degree_weighted=False), edges),
        evaluate_method(
            bundle,
            "fixed_topology_no_degree",
            readout_probs(bundle, fixed_edges, degree_weighted=False),
            fixed_edges,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(
        run_task(
            "Task 4 degree-weighted readout ablation.",
            "task4_degree_weight_ablation",
            evaluate,
            [
                ("default_degree_weight", "no_degree_weight"),
                ("default_degree_weight", "fixed_topology_no_degree"),
            ],
        )
    )
