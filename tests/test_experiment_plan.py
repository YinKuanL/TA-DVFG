from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_plan import build_jobs  # noqa: E402


class ExperimentPlanTests(unittest.TestCase):
    def test_all_job_ids_are_unique(self) -> None:
        jobs = build_jobs(["all"])
        self.assertEqual(187, len(jobs))
        self.assertEqual(len(jobs), len({job.job_id for job in jobs}))

    def test_core_matrix_is_complete(self) -> None:
        jobs = build_jobs(["core"])
        observed = {(job.dataset, job.setting) for job in jobs}
        expected = {
            (dataset, setting)
            for dataset in ("ACM", "DBLP", "IMDB")
            for setting in ("main", "hard_noisy")
        }
        self.assertEqual(expected, observed)

    def test_edge_budgets_are_reachable(self) -> None:
        for job in build_jobs(["edge_budget"]):
            parties = int(job.parameters["num_parties"])
            max_degree = int(job.parameters["max_degree"])
            budget = int(job.parameters["adaptive_edge_budget"])
            self.assertLessEqual(budget, parties * max_degree // 2)

    def test_consensus_values_cover_the_rp(self) -> None:
        jobs = build_jobs(["consensus"])
        steps = {int(job.parameters["pred_consensus_steps"]) for job in jobs}
        self_weights = {float(job.parameters["pred_self_weight"]) for job in jobs}
        weightings = {str(job.parameters["vote_weighting"]) for job in jobs}
        self.assertEqual({0, 1, 2}, steps)
        self.assertEqual({0.70, 0.85, 0.95}, self_weights)
        self.assertEqual({"uniform", "reliability", "topology_reliability"}, weightings)

    def test_modality_ablation_contains_single_and_mixed_views(self) -> None:
        jobs = build_jobs(["modality_ablation"])
        modalities = {str(job.parameters["modalities"]) for job in jobs}
        self.assertTrue(
            {
                "ATTR",
                "TEXT",
                "STRUCT",
                "HYBRID",
                "TEXT_STRUCT",
                "ATTR,TEXT,STRUCT,HYBRID",
            }.issubset(modalities)
        )

    def test_reviewer_safe_output_names_and_methods(self) -> None:
        jobs = build_jobs(["reviewer_safe"])
        self.assertEqual(4, len(jobs))
        expected_names = {
            "acm_main_reviewer_safe_5seeds.csv",
            "acm_hard_reviewer_safe_5seeds.csv",
            "dblp_main_reviewer_safe_5seeds.csv",
            "dblp_hard_reviewer_safe_5seeds.csv",
        }
        self.assertEqual(expected_names, {job.output_filename for job in jobs})
        required_methods = {
            "topk_reliability_vote",
            "fixed_ring_matched",
            "random_matched",
            "expander_matched",
        }
        for job in jobs:
            self.assertTrue(required_methods.issubset(set(str(job.parameters["methods"]).split(","))))

    def test_shuffled_reviewer_safe_outputs(self) -> None:
        jobs = build_jobs(["reviewer_safe_shuffled"])
        self.assertEqual(2, len(jobs))
        self.assertTrue(all(job.parameters["shuffle_party_positions"] for job in jobs))
        self.assertEqual(
            {
                "acm_main_reviewer_safe_shuffled_5seeds.csv",
                "acm_hard_reviewer_safe_shuffled_5seeds.csv",
            },
            {job.output_filename for job in jobs},
        )

    def test_multimodal_jobs_use_prediction_cache(self) -> None:
        jobs = build_jobs(["multimodal_core", "modality_ablation"])
        self.assertTrue(jobs)
        for job in jobs:
            self.assertTrue(job.parameters["cache_predictions"])
            self.assertTrue(job.parameters["reuse_prediction_cache"])
            self.assertTrue(job.parameters["shuffle_party_positions"])
            self.assertIn("multimodal_cache", str(job.parameters["cache_dir"]))

    def test_multimodal_core_includes_imdb_hard(self) -> None:
        jobs = build_jobs(["multimodal_core"])
        self.assertEqual(6, len(jobs))
        imdb_hard = [job for job in jobs if job.name == "imdb_hard_noisy_mixed"]
        self.assertEqual(1, len(imdb_hard))
        self.assertEqual("IMDB", imdb_hard[0].dataset)
        self.assertEqual("hard_noisy", imdb_hard[0].setting)
        self.assertEqual(3, imdb_hard[0].parameters["useful_parties"])

    def test_standard_ablations_share_cached_core_predictions(self) -> None:
        jobs = build_jobs(["topology_objective", "edge_budget", "consensus"])
        self.assertEqual(54, len(jobs))
        local_configs = set()
        for job in jobs:
            self.assertTrue(job.parameters["cache_predictions"])
            self.assertTrue(job.parameters["reuse_prediction_cache"])
            self.assertTrue(job.parameters["shuffle_party_positions"])
            self.assertEqual(2026, job.parameters["party_shuffle_seed_offset"])
            self.assertIn("core_cache", str(job.parameters["cache_dir"]))
            local_configs.add(
                (
                    job.dataset,
                    job.parameters["useful_parties"],
                    job.parameters["view_setting"],
                    job.parameters["graph_views"],
                )
            )
        self.assertEqual(
            {
                ("ACM", 3, "hard", "PAP,PSP,KNN"),
                ("ACM", 6, "hard", "PAP,PSP,KNN"),
                ("DBLP", 3, "hard", "APA,APCPA,APTPA,KNN"),
                ("DBLP", 6, "hard", "APA,APCPA,APTPA,KNN"),
            },
            local_configs,
        )

    def test_active_party_protocol_is_cached_shuffled_and_reviewer_focused(self) -> None:
        jobs = build_jobs(["active_party_protocol"])
        self.assertEqual(4, len(jobs))
        self.assertEqual(
            {
                "acm_hard_active_evaluator.csv",
                "acm_main_active_evaluator.csv",
                "dblp_hard_active_evaluator.csv",
                "dblp_main_active_evaluator.csv",
            },
            {job.output_filename for job in jobs},
        )
        required = {
            "local_reliability_vote",
            "topk_reliability_vote",
            "full_mesh",
            "fixed_ring_matched",
            "random_matched",
            "expander_matched",
            "adaptive_graph_val",
        }
        for job in jobs:
            self.assertEqual("active_party", job.parameters["validation_evaluator"])
            self.assertEqual(0, job.parameters["active_party_id"])
            self.assertEqual(3, job.parameters["max_degree"])
            self.assertTrue(job.parameters["shuffle_party_positions"])
            self.assertTrue(job.parameters["reuse_prediction_cache"])
            self.assertEqual(required, set(str(job.parameters["methods"]).split(",")))

    def test_reviewer_resistance_suites_cover_requested_protocols(self) -> None:
        strict = build_jobs(["strict_label_training"])
        self.assertEqual(2, len(strict))
        for job in strict:
            self.assertEqual(
                "active_party_logit_gradient",
                job.parameters["local_training_protocol"],
            )
            self.assertEqual("active_party_only", job.parameters["label_protocol"])
            self.assertEqual("prediction_split", job.parameters["training_protocol"])
            self.assertEqual("split_val", job.parameters["validation_protocol"])
            self.assertEqual("disjoint", job.parameters["validation_split_mode"])
            self.assertEqual("active_party", job.parameters["validation_evaluator"])
            self.assertTrue(job.parameters["shuffle_party_positions"])

        split = build_jobs(["split_validation"])
        self.assertEqual(4, len(split))
        self.assertEqual(
            {"shared", "disjoint"},
            {job.parameters["validation_split_mode"] for job in split},
        )

        readout = build_jobs(["decentralized_readout"])
        self.assertEqual(8, len(readout))
        self.assertEqual(
            {"active_vote", "local_mean", "local_worst", "active_party_local"},
            {job.parameters["readout_mode"] for job in readout},
        )

        identity = build_jobs(["active_party_identity"])
        self.assertEqual(10, len(identity))
        self.assertEqual(
            {0, 1, 5, 10, 14},
            {job.parameters["active_party_id"] for job in identity},
        )

        label_budget = build_jobs(["validation_label_budget"])
        self.assertEqual(10, len(label_budget))
        self.assertEqual(
            {0.10, 0.25, 0.50, 0.75, 1.00},
            {job.parameters["topology_val_fraction"] for job in label_budget},
        )

        frequency = build_jobs(["topology_frequency"])
        self.assertEqual(8, len(frequency))
        self.assertTrue(
            any(job.parameters["topology_max_updates"] == 1 for job in frequency)
        )

        scalability = build_jobs(["party_scalability"])
        self.assertEqual(5, len(scalability))
        self.assertEqual(
            {5, 10, 15, 20, 30},
            {job.parameters["num_parties"] for job in scalability},
        )

    def test_strong_setting_suite_matches_reviewer_critical_protocol(self) -> None:
        jobs = build_jobs(["strong_setting"])
        self.assertEqual(4, len(jobs))
        methods = {
            "active_local_only",
            "global_topk_reliability",
            "full_mesh",
            "best_matched_sparse",
            "adaptive_graph_val",
        }
        strict = [job for job in jobs if "strict" in job.name]
        dropout = [job for job in jobs if "party_dropout" in job.name]
        self.assertEqual(2, len(strict))
        self.assertEqual(2, len(dropout))
        for job in jobs:
            self.assertEqual(methods, set(str(job.parameters["methods"]).split(",")))
            self.assertEqual("active_party_only", job.parameters["label_protocol"])
            self.assertEqual("prediction_split", job.parameters["training_protocol"])
            self.assertEqual("split_val", job.parameters["validation_protocol"])
            self.assertEqual("active_party", job.parameters["validation_evaluator"])
            self.assertEqual("active_party_logit_gradient", job.parameters["local_training_protocol"])
            self.assertEqual("disjoint", job.parameters["validation_split_mode"])
            self.assertEqual(0.5, job.parameters["topology_val_fraction"])
        for job in dropout:
            self.assertEqual("random_party_dropout", job.parameters["dropout_protocol"])
            self.assertEqual("0.2,0.4", job.parameters["dropout_rates"])
            self.assertEqual("0,1,2", job.parameters["dropout_seeds"])

    def test_deployment_objective_suite_matches_frontier_protocol(self) -> None:
        jobs = build_jobs(["deployment_objective"])
        self.assertEqual(12, len(jobs))
        by_dataset = {}
        for job in jobs:
            by_dataset.setdefault(job.dataset, []).append(job)
            self.assertEqual("active_party_only", job.parameters["label_protocol"])
            self.assertEqual("prediction_split", job.parameters["training_protocol"])
            self.assertEqual("split_val", job.parameters["validation_protocol"])
            self.assertEqual("active_party", job.parameters["validation_evaluator"])
            self.assertEqual("active_party_logit_gradient", job.parameters["local_training_protocol"])
            self.assertEqual("disjoint", job.parameters["validation_split_mode"])
            self.assertEqual(0.5, job.parameters["topology_val_fraction"])
            self.assertTrue(job.parameters["cache_predictions"])
            self.assertTrue(job.parameters["reuse_prediction_cache"])
        self.assertEqual({"ACM", "DBLP"}, set(by_dataset))
        for dataset_jobs in by_dataset.values():
            self.assertEqual(6, len(dataset_jobs))
            baselines = [job for job in dataset_jobs if "baselines" in job.name]
            self.assertEqual(1, len(baselines))
            self.assertEqual(
                {
                    "active_local_only",
                    "global_topk_reliability",
                    "full_mesh",
                    "best_matched_sparse",
                },
                set(str(baselines[0].parameters["methods"]).split(",")),
            )
            adaptive = [job for job in dataset_jobs if "tadvfg" in job.name]
            observed = {
                (
                    job.parameters["topology_objective"],
                    float(job.parameters["joint_lambda"]),
                )
                for job in adaptive
            }
            self.assertEqual(
                {
                    ("active", 0.5),
                    ("local_mean", 0.5),
                    ("joint", 0.25),
                    ("joint", 0.5),
                    ("joint", 0.75),
                },
                observed,
            )


if __name__ == "__main__":
    unittest.main()
