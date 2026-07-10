"""Task 8: no-collaboration baselines from cached prediction bundles."""

from __future__ import annotations

import numpy as np

from closure_core import local_readout, score_predictions
from closure_task_common import evaluate_method, run_task


def evaluate(bundle, args):
    party_scores = [score_predictions(p, bundle.y_val, bundle.metric) for p in bundle.val_probs]
    best = int(np.nanargmax(party_scores))
    return [
        evaluate_method(bundle, "BestSingleValidationParty", bundle.test_probs[best], []),
        evaluate_method(bundle, "LocalOnlyMean", local_readout(bundle), []),
    ]


if __name__ == "__main__":
    raise SystemExit(
        run_task(
            "Task 8 no-collaboration baselines.",
            "task8_no_collaboration",
            evaluate,
            [("BestSingleValidationParty", "LocalOnlyMean")],
        )
    )
