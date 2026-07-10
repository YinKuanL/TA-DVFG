"""Task 1: exact K=0 versus K=1 cached-prediction comparison."""

from __future__ import annotations

from closure_core import active_readout, local_readout
from closure_task_common import evaluate_method, run_task, topology


def evaluate(bundle, args):
    edges0 = topology(bundle, args, K=0)
    edges1 = topology(bundle, args, K=1)
    return [
        evaluate_method(bundle, "K0_local_mean", local_readout(bundle), edges0),
        evaluate_method(bundle, "K1_active_readout", active_readout(bundle, edges1), edges1),
    ]


if __name__ == "__main__":
    raise SystemExit(
        run_task(
            "Task 1 exact K=0 versus K=1 cached-prediction comparison.",
            "task1_steps0_exact",
            evaluate,
            [("K1_active_readout", "K0_local_mean")],
        )
    )
