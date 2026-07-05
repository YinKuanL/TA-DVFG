from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "main experiment" / "ta_dvfg_hgb_reliability.py"


def load_engine():
    spec = importlib.util.spec_from_file_location("ta_dvfg_engine_for_tests", ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()


class ReviewerDefenseTests(unittest.TestCase):
    def test_matched_topologies_have_exact_edge_count(self) -> None:
        for requested in (0, 1, 5, 15):
            ring = engine.make_ring_like_topology(15, requested)
            random = engine.make_random_matched_topology(15, 2, requested, 42)
            expander = engine.make_expander_like_topology(15, requested)
            self.assertEqual(requested, engine.edge_count(ring))
            self.assertEqual(requested, engine.edge_count(random))
            self.assertEqual(requested, engine.edge_count(expander))

    def test_matched_topology_prefix_changes_with_seed(self) -> None:
        ring_42 = engine.topology_edge_list(engine.make_ring_like_topology(15, 5, 42))
        ring_43 = engine.topology_edge_list(engine.make_ring_like_topology(15, 5, 43))
        expander_42 = engine.topology_edge_list(engine.make_expander_matched_topology(15, 5, 42))
        expander_43 = engine.topology_edge_list(engine.make_expander_matched_topology(15, 5, 43))
        self.assertNotEqual(ring_42, ring_43)
        self.assertNotEqual(expander_42, expander_43)

    def test_edge_type_diagnostics(self) -> None:
        adj = np.zeros((4, 4), dtype=np.float32)
        for i, j in ((0, 1), (0, 2), (2, 3)):
            adj[i, j] = adj[j, i] = 1.0
        self.assertEqual((1, 1, 1), engine.topology_edge_types(adj, [True, True, False, False]))

    def test_cache_key_excludes_topology_only_arguments(self) -> None:
        required = {
            "dataset": "ACM",
            "target_node": "auto",
            "data_dir": "data_hgb",
            "ignore_dataset_split": False,
            "graph_views": "PAP,PSP,KNN",
            "graph_k": 10,
            "max_metapath_edges": 300000,
            "random_graph_degree": 4,
            "num_parties": 15,
            "useful_parties": 3,
            "view_setting": "hard",
            "distractor_feature_noise": 1.0,
            "shuffle_party_positions": True,
            "party_shuffle_seed_offset": 2026,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "epochs": 300,
            "hidden_dim": 64,
            "dropout": 0.5,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "adaptive_min_edges": 0,
            "methods": "adaptive_graph_val",
        }
        args_a = SimpleNamespace(**required)
        args_b = SimpleNamespace(**{**required, "adaptive_min_edges": 5, "methods": "full_mesh"})
        self.assertEqual(
            engine.prediction_cache_training_config(args_a, 42),
            engine.prediction_cache_training_config(args_b, 42),
        )

    def test_legacy_standard_cache_metadata_migration(self) -> None:
        required = {
            "dataset": "ACM",
            "target_node": "auto",
            "data_dir": "data_hgb",
            "ignore_dataset_split": False,
            "graph_views": "PAP,PSP,KNN",
            "graph_k": 10,
            "max_metapath_edges": 300000,
            "random_graph_degree": 4,
            "num_parties": 15,
            "useful_parties": 6,
            "view_setting": "hard",
            "distractor_feature_noise": 1.0,
            "shuffle_party_positions": True,
            "party_shuffle_seed_offset": 2026,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "epochs": 300,
            "hidden_dim": 64,
            "dropout": 0.5,
            "lr": 0.01,
            "weight_decay": 5e-4,
        }
        expected = engine.prediction_cache_training_config(SimpleNamespace(**required), 42)
        legacy_config = {
            key: value
            for key, value in expected.items()
            if key not in engine.V3_STANDARD_CACHE_DEFAULTS
        }
        train_idx = torch.tensor([0, 1])
        val_idx = torch.tensor([2])
        test_idx = torch.tensor([3])
        split_hash = engine.split_fingerprint(train_idx, val_idx, test_idx)
        cache = {
            "cache_version": 2,
            "training_config": legacy_config,
            "split_fingerprint": split_hash,
            "cache_fingerprint": engine.stable_fingerprint(
                {"training_config": legacy_config, "split_fingerprint": split_hash}
            ),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }
        migrated = engine.migrate_legacy_standard_prediction_cache(
            cache,
            expected,
            Path("legacy.pt"),
        )
        self.assertEqual(engine.CACHE_VERSION, migrated["cache_version"])
        self.assertEqual(expected, migrated["training_config"])
        engine.validate_prediction_cache(migrated, expected, Path("legacy.pt"))

    def test_stable_metapath_seed(self) -> None:
        self.assertEqual(engine.stable_name_seed("PSP"), engine.stable_name_seed("PSP"))
        self.assertNotEqual(engine.stable_name_seed("PAP"), engine.stable_name_seed("PSP"))

    def test_topk_vote_ignores_lower_reliability_parties(self) -> None:
        probs = [
            torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            torch.tensor([[0.8, 0.2]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        ]
        vote = engine.topk_reliability_vote(probs, [0.9, 0.8, 0.1], 2)
        self.assertEqual(0, int(vote.argmax(dim=1).item()))

    def test_active_party_reliability_matches_centralized_computation(self) -> None:
        probs = [
            torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
            torch.tensor([[0.4, 0.6], [0.7, 0.3]], dtype=torch.float32),
            torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32),
        ]
        labels = torch.tensor([0, 1])
        all_parties = engine.compute_party_reliabilities(
            probs, labels, "all_parties", 0
        )
        active_party = engine.compute_party_reliabilities(
            probs, labels, "active_party", 1
        )
        self.assertEqual(all_parties, active_party)
        self.assertEqual([1.0, 0.0, 0.5], active_party)
        with self.assertRaises(ValueError):
            engine.compute_party_reliabilities(probs, labels, "active_party", 3)

    def test_active_party_topology_score_matches_all_parties(self) -> None:
        probs = [
            torch.tensor([[0.9, 0.1], [0.3, 0.7]], dtype=torch.float32),
            torch.tensor([[0.7, 0.3], [0.8, 0.2]], dtype=torch.float32),
            torch.tensor([[0.2, 0.8], [0.1, 0.9]], dtype=torch.float32),
        ]
        labels = torch.tensor([0, 1])
        reliabilities = engine.compute_party_reliabilities(
            probs, labels, "active_party", 0
        )
        adj = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        common = {
            "pred_consensus_steps": 1,
            "consensus_mode": "standard",
            "consensus_reliability_margin": 0.0,
            "pred_self_weight": 0.85,
            "readout_mode": "active_vote",
            "vote_weighting": "topology_reliability",
            "final_reliability_floor": 0.0,
            "reliability_power": 1.0,
            "topology_degree_power": 0.5,
        }
        all_args = SimpleNamespace(
            **common, validation_evaluator="all_parties", active_party_id=0
        )
        active_args = SimpleNamespace(
            **common, validation_evaluator="active_party", active_party_id=0
        )
        all_score = engine.score_topology_on_validation(
            adj, probs, labels, reliabilities, "all_parties", 0, all_args
        )
        active_score = engine.score_topology_on_validation(
            adj, probs, labels, reliabilities, "active_party", 0, active_args
        )
        self.assertEqual(all_score, active_score)

    def test_logit_gradient_training_matches_direct_cross_entropy(self) -> None:
        torch.manual_seed(7)
        xs = [
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                dtype=torch.float32,
            )
            for _ in range(2)
        ]
        indices = torch.arange(3).repeat(2, 1)
        values = torch.ones(3)
        identity = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
        adjs = [identity, identity]
        labels = torch.tensor([0, 1, 0])
        train_idx = torch.tensor([0, 1, 2])

        direct = engine.init_parties(xs, 4, 2, 0.0, 0.01, 0.0)
        strict = engine.init_parties(xs, 4, 2, 0.0, 0.01, 0.0)
        for direct_party, strict_party in zip(direct, strict):
            strict_party.model.load_state_dict(direct_party.model.state_dict())

        direct_comm = engine.train_local_parties_one_epoch(
            direct,
            xs,
            adjs,
            labels,
            train_idx,
            SimpleNamespace(
                local_training_protocol="all_parties_supervised",
                active_party_id=0,
            ),
        )
        strict_comm = engine.train_local_parties_one_epoch(
            strict,
            xs,
            adjs,
            labels,
            train_idx,
            SimpleNamespace(
                local_training_protocol="active_party_logit_gradient",
                active_party_id=0,
            ),
        )
        self.assertEqual(0, direct_comm)
        self.assertEqual(2 * 3 * 2, strict_comm)
        for direct_party, strict_party in zip(direct, strict):
            for direct_parameter, strict_parameter in zip(
                direct_party.model.parameters(),
                strict_party.model.parameters(),
            ):
                self.assertTrue(
                    torch.allclose(
                        direct_parameter,
                        strict_parameter,
                        atol=1e-7,
                        rtol=1e-6,
                    )
                )

    def test_disjoint_validation_positions_do_not_overlap(self) -> None:
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        args = SimpleNamespace(
            validation_split_mode="disjoint",
            topology_val_fraction=0.5,
            validation_split_seed_offset=4096,
        )
        topology, selection = engine.validation_protocol_positions(labels, args, 42)
        self.assertFalse(set(topology.tolist()) & set(selection.tolist()))
        self.assertEqual(set(range(len(labels))), set(topology.tolist()) | set(selection.tolist()))
        self.assertEqual({0, 1, 2}, set(labels[topology].tolist()))
        self.assertEqual({0, 1, 2}, set(labels[selection].tolist()))

    def test_nonstandard_party_count_has_distinct_cache_path(self) -> None:
        common = {
            "dataset": "ACM",
            "view_setting": "hard",
            "useful_parties": 3,
            "cache_namespace": "standard",
            "cache_dir": "results/cache",
        }
        standard = engine.prediction_cache_path(
            SimpleNamespace(**common, num_parties=15),
            42,
        )
        scaled = engine.prediction_cache_path(
            SimpleNamespace(**common, num_parties=20),
            42,
        )
        self.assertNotEqual(standard, scaled)
        self.assertIn("_n20_", scaled.name)

    def test_communication_accounting_separates_peer_and_global_readout(self) -> None:
        args = SimpleNamespace(
            num_parties=15,
            topk_reliability_k=5,
            pred_consensus_steps=1,
            readout_mode="active_vote",
        )
        empty = np.zeros((15, 15), dtype=np.float32)
        peer, readout, total, protocol = engine.estimate_communication(
            "topk_reliability_vote", empty, 100, 3, args
        )
        self.assertEqual((0, 1500, 1500, "global_diagnostic"), (peer, readout, total, protocol))

        sparse = engine.make_ring_like_topology(15, 5, 42)
        peer, readout, total, protocol = engine.estimate_communication(
            "adaptive_graph_val", sparse, 100, 3, args
        )
        self.assertEqual(3000, peer)
        self.assertGreater(readout, 0)
        self.assertEqual(peer + readout, total)
        self.assertEqual("peer_to_peer_topology", protocol)

    def test_readout_modes_use_post_consensus_party_predictions(self) -> None:
        probs = [
            torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            torch.tensor([[0.2, 0.8]], dtype=torch.float32),
        ]
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        common = {
            "pred_consensus_steps": 0,
            "vote_weighting": "topology_reliability",
            "final_reliability_floor": 0.0,
            "reliability_power": 1.0,
            "topology_degree_power": 0.5,
            "validation_evaluator": "all_parties",
            "active_party_id": 0,
        }
        mean_args = SimpleNamespace(**common, readout_mode="mean_party")
        mean_vote = engine.topology_prediction_consensus(probs, adj, [0.9, 0.8], mean_args)
        self.assertTrue(torch.allclose(mean_vote, torch.tensor([[0.55, 0.45]])))

        best_args = SimpleNamespace(**common, readout_mode="best_party")
        labels = torch.tensor([1])
        val_idx = torch.tensor([0])
        best_vote = engine.topology_prediction_consensus(
            probs, adj, [0.9, 0.8], best_args, y=labels, val_idx=val_idx
        )
        self.assertTrue(torch.equal(best_vote, probs[1]))

    def test_party_dropout_never_drops_active_party(self) -> None:
        for active_party_id in (0, 7, 14):
            dropped = engine.party_dropout_indices(
                15,
                active_party_id,
                0.4,
                42,
                1,
            )
            self.assertNotIn(active_party_id, dropped)
            self.assertEqual(6, len(dropped))

    def test_inference_metrics_report_active_and_local_readouts(self) -> None:
        probs = [
            torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32),
            torch.tensor([[0.2, 0.8], [0.3, 0.7]], dtype=torch.float32),
            torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float32),
        ]
        labels = torch.tensor([0, 1])
        adj = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        args = SimpleNamespace(
            active_party_id=0,
            dropout_protocol="random_party_dropout",
            dropout_rate=0.34,
            dropout_seed=0,
            pred_consensus_steps=1,
            consensus_mode="standard",
            consensus_reliability_margin=0.0,
            pred_self_weight=0.85,
            vote_weighting="topology_reliability",
            final_reliability_floor=0.0,
            reliability_power=1.0,
            topology_degree_power=0.5,
            topk_reliability_k=2,
        )
        metrics = engine.inference_readout_metrics(
            "adaptive_graph_val",
            probs,
            labels,
            adj,
            [1.0, 0.0, 0.5],
            args,
            42,
        )
        self.assertEqual(2, metrics["remaining_parties"])
        self.assertIn("active_readout_acc", metrics)
        self.assertIn("local_post_consensus_mean_acc", metrics)
        self.assertIn("active_party_post_consensus_acc", metrics)
        self.assertIn("local_worst_acc", metrics)
        self.assertIn("local_best_acc", metrics)

    def test_topology_objective_scores_active_local_and_joint(self) -> None:
        labels = torch.tensor([0, 1])
        probs = [
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        ]
        adj = np.zeros((2, 2), dtype=np.float32)
        reliabilities = [1.0, 0.0]
        args = SimpleNamespace(
            topology_objective="active",
            joint_lambda=0.25,
            pred_consensus_steps=0,
            vote_weighting="reliability",
            readout_mode="active_vote",
            final_reliability_floor=0.0,
            reliability_power=1.0,
            topology_degree_power=0.5,
        )
        active = engine.score_topology_on_validation(
            adj, probs, labels, reliabilities, "active_party", 0, args
        )
        args.topology_objective = "local_mean"
        local = engine.score_topology_on_validation(
            adj, probs, labels, reliabilities, "active_party", 0, args
        )
        args.topology_objective = "joint"
        joint = engine.score_topology_on_validation(
            adj, probs, labels, reliabilities, "active_party", 0, args
        )
        self.assertAlmostEqual(1.0, active)
        self.assertAlmostEqual(0.5, local)
        self.assertAlmostEqual(0.625, joint)


if __name__ == "__main__":
    unittest.main()
