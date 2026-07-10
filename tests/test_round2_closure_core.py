from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPO_ROOT / "experiments" / "closure_core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("closure_core_for_tests", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()


class Round2ClosureCoreTests(unittest.TestCase):
    def bundle(self):
        y_val = np.array([0, 1, 0, 1, 0, 1])
        y_test = np.array([0, 1, 1, 0])
        val = np.array(
            [
                [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]],
                [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]],
                [[0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4]],
            ]
        )
        test = np.array(
            [
                [[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]],
                [[0.8, 0.2], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3]],
                [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8]],
            ]
        )
        return core.Bundle(val, test, y_val, y_test, "T", "unit", 42, "accuracy")

    def test_load_bundle_normalizes_binary_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.npz"
            np.savez(
                path,
                val_probs=np.array([[0.1, 0.9], [0.8, 0.2]]),
                test_probs=np.array([[0.2, 0.8], [0.7, 0.3]]),
                y_val=np.array([0, 1]),
                y_test=np.array([0, 1]),
                dataset="T",
                setting="binary",
                seed=7,
                metric="accuracy",
            )
            bundle = core.load_bundle(path)
            self.assertEqual(bundle.val_probs.shape, (2, 2, 2))
            self.assertEqual(bundle.seed, 7)

    def test_select_topology_handles_zero_and_one_edge(self):
        bundle = self.bundle()
        self.assertEqual(core.select_topology(bundle, K=0), [])
        one = core.select_topology(bundle, K=1, m=1, eta=0.0)
        self.assertEqual(len(one), 1)
        self.assertLess(one[0][0], one[0][1])

    def test_empty_active_readout_falls_back_to_local_mean(self):
        bundle = self.bundle()
        np.testing.assert_allclose(core.active_readout(bundle, []), core.local_readout(bundle))

    def test_paired_stats_counts_wins_ties_losses(self):
        stats = core.paired_stats([1.0, 2.0, 3.0], [0.5, 2.0, 4.0])
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["ties"], 1)
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["n_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
