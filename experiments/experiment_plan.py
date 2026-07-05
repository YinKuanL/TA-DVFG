"""Canonical experiment matrix for the TA-DVFG AAAI submission.

Each generated job is one engine invocation.  The engine itself runs every
requested seed and method sequentially, so a "job" is larger than a single
training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SEEDS = "42,43,44,45,46"

CORE_METHODS = [
    "local_uniform_vote",
    "local_reliability_vote",
    "topk_reliability_vote",
    "fixed_ring",
    "fixed_ring_matched",
    "random_regular",
    "random_matched",
    "expander",
    "expander_matched",
    "full_mesh",
    "adaptive_pair",
    "adaptive_complementarity",
    "adaptive_graph_val",
]

ROBUSTNESS_METHODS = [
    "local_uniform_vote",
    "local_reliability_vote",
    "fixed_ring",
    "full_mesh",
    "adaptive_graph_val",
]

REVIEWER_SAFE_METHODS = list(CORE_METHODS)

ACTIVE_PARTY_PROTOCOL_METHODS = [
    "local_reliability_vote",
    "topk_reliability_vote",
    "full_mesh",
    "fixed_ring_matched",
    "random_matched",
    "expander_matched",
    "adaptive_graph_val",
]

STRONG_SETTING_METHODS = [
    "active_local_only",
    "global_topk_reliability",
    "full_mesh",
    "best_matched_sparse",
    "adaptive_graph_val",
]

DEPLOYMENT_BASELINE_METHODS = [
    "active_local_only",
    "global_topk_reliability",
    "full_mesh",
    "best_matched_sparse",
]

DATASETS: Dict[str, Dict[str, str]] = {
    "ACM": {"views": "PAP,PSP,KNN"},
    "DBLP": {"views": "APA,APCPA,APTPA,KNN"},
    "IMDB": {"views": "MAM,MDM,KNN"},
}

SETTINGS = {
    "main": 6,
    "hard_noisy": 3,
}

STANDARD_SHARED_CACHE_DIR = "results/core_cache_v2"

SUITE_ORDER = [
    "core",
    "reviewer_safe",
    "reviewer_safe_shuffled",
    "topology_objective",
    "edge_budget",
    "consensus",
    "noise_ratio",
    "multimodal_core",
    "modality_ablation",
    "central_baselines",
    "protocol_sensitivity",
    "readout_ablation",
    "active_party_evaluator",
    "active_party_protocol",
    "strict_label_training",
    "split_validation",
    "decentralized_readout",
    "active_party_identity",
    "validation_label_budget",
    "topology_frequency",
    "party_scalability",
    "strong_setting",
    "deployment_objective",
]


@dataclass(frozen=True)
class Job:
    suite: str
    name: str
    engine: str
    dataset: str
    setting: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    make_plots: bool = False
    description: str = ""
    output_filename: str = "metrics.csv"

    @property
    def job_id(self) -> str:
        return f"{self.suite}/{self.name}"


def _base(
    dataset: str,
    useful_parties: int,
    methods: Sequence[str],
    *,
    epochs: int = 300,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "graph_views": DATASETS[dataset]["views"],
        "num_parties": 15,
        "view_setting": "hard",
        "useful_parties": useful_parties,
        "epochs": epochs,
        "seeds": SEEDS,
        "methods": ",".join(methods),
        "adaptive_edge_budget": 15,
        "adaptive_min_edges": 5,
        "adaptive_min_gain": 0.001,
        "max_degree": 2,
        "pred_consensus_steps": 1,
        "pred_self_weight": 0.85,
        "vote_weighting": "topology_reliability",
        "reliability_power": 1.0,
        "topology_degree_power": 0.5,
        "final_reliability_floor": 0.0,
        "topk_reliability_k": 5,
        "matched_edge_count": 5,
        "readout_mode": "active_vote",
        "validation_evaluator": "all_parties",
        "active_party_id": 0,
    }


def _job(
    suite: str,
    name: str,
    dataset: str,
    setting: str,
    parameters: Mapping[str, Any],
    *,
    engine: str = "standard",
    make_plots: bool = False,
    description: str = "",
    output_filename: str = "metrics.csv",
) -> Job:
    return Job(
        suite=suite,
        name=name,
        engine=engine,
        dataset=dataset,
        setting=setting,
        parameters=dict(parameters),
        make_plots=make_plots,
        description=description,
        output_filename=output_filename,
    )


def _use_standard_shared_cache(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Replay topology-only experiments from the fair cached-core trajectories."""
    parameters.update(
        {
            "shuffle_party_positions": True,
            "party_shuffle_seed_offset": 2026,
            "cache_predictions": True,
            "reuse_prediction_cache": True,
            "cache_dir": STANDARD_SHARED_CACHE_DIR,
        }
    )
    return parameters


def core_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in DATASETS:
        for setting, useful in SETTINGS.items():
            jobs.append(
                _job(
                    "core",
                    f"{dataset.lower()}_{setting}",
                    dataset,
                    setting,
                    _base(dataset, useful, CORE_METHODS),
                    make_plots=True,
                    description="Nine-method main comparison over five paired seeds.",
                )
            )
    return jobs


def reviewer_safe_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in (("main", 6), ("hard", 3)):
            params = _base(dataset, useful, REVIEWER_SAFE_METHODS)
            jobs.append(
                _job(
                    "reviewer_safe",
                    f"{dataset.lower()}_{setting}",
                    dataset,
                    setting,
                    params,
                    make_plots=True,
                    description="Reviewer-defense comparison with top-k and matched-edge baselines.",
                    output_filename=f"{dataset.lower()}_{setting}_reviewer_safe_5seeds.csv",
                )
            )
    return jobs


def reviewer_safe_shuffled_jobs() -> List[Job]:
    jobs: List[Job] = []
    for setting, useful in (("main", 6), ("hard", 3)):
        params = _base("ACM", useful, REVIEWER_SAFE_METHODS)
        params.update(
            {
                "shuffle_party_positions": True,
                "party_shuffle_seed_offset": 2026,
            }
        )
        jobs.append(
            _job(
                "reviewer_safe_shuffled",
                f"acm_{setting}",
                "ACM",
                setting,
                params,
                make_plots=True,
                description="Reviewer-safe ACM comparison with seed-shuffled useful/noisy party positions.",
                output_filename=f"acm_{setting}_reviewer_safe_shuffled_5seeds.csv",
            )
        )
    return jobs


def topology_objective_jobs() -> List[Job]:
    methods = [
        "adaptive_pair",
        "adaptive_complementarity",
        "adaptive_graph_val",
    ]
    return [
        _job(
            "topology_objective",
            f"{dataset.lower()}_hard_noisy",
            dataset,
            "hard_noisy",
            _use_standard_shared_cache(_base(dataset, 3, methods)),
            make_plots=True,
            description="Pair, complementarity, and graph-level validation objectives.",
        )
        for dataset in ("ACM", "DBLP")
    ]


def edge_budget_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for budget in (5, 10, 15, 20):
            for min_gain in (0.0, 0.001, 0.005):
                params = _base(dataset, 3, ["adaptive_graph_val"])
                params.update(
                    {
                        "adaptive_edge_budget": budget,
                        "adaptive_min_edges": min(5, budget),
                        "adaptive_min_gain": min_gain,
                        # Degree two caps a 15-party graph at 15 edges, which
                        # would silently make the requested budget=20 point
                        # identical to budget=15. Degree three keeps all four
                        # budget values feasible while remaining sparse.
                        "max_degree": 3,
                    }
                )
                _use_standard_shared_cache(params)
                gain_tag = str(min_gain).replace(".", "p")
                jobs.append(
                    _job(
                        "edge_budget",
                        f"{dataset.lower()}_b{budget}_g{gain_tag}",
                        dataset,
                        "hard_noisy",
                        params,
                        description="Sparse edge-budget and early-stop sweep.",
                    )
                )
    return jobs


def consensus_jobs() -> List[Job]:
    # One-factor-at-a-time around the paper default.  This covers every value
    # requested in the RP while keeping each effect separately interpretable.
    variants = [
        ("default", {}),
        ("steps0", {"pred_consensus_steps": 0}),
        ("steps2", {"pred_consensus_steps": 2}),
        ("self0p70", {"pred_self_weight": 0.70}),
        ("self0p95", {"pred_self_weight": 0.95}),
        ("uniform", {"vote_weighting": "uniform"}),
        ("reliability", {"vote_weighting": "reliability"}),
    ]
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in SETTINGS.items():
            for variant, override in variants:
                params = _base(dataset, useful, ["adaptive_graph_val"])
                params.update(override)
                _use_standard_shared_cache(params)
                jobs.append(
                    _job(
                        "consensus",
                        f"{dataset.lower()}_{setting}_{variant}",
                        dataset,
                        setting,
                        params,
                        description="Prediction-consensus weighting/steps ablation.",
                    )
                )
    return jobs


def noise_ratio_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for useful in (3, 4, 6, 9):
            params = _base(dataset, useful, ROBUSTNESS_METHODS)
            jobs.append(
                _job(
                    "noise_ratio",
                    f"{dataset.lower()}_useful{useful}",
                    dataset,
                    f"useful_{useful}",
                    params,
                    description="Robustness curve as useful/noisy party ratio changes.",
                )
            )
    return jobs


def multimodal_core_jobs() -> List[Job]:
    requested = [
        ("ACM", "main", 6),
        ("ACM", "hard_noisy", 3),
        ("DBLP", "main", 6),
        ("DBLP", "hard_noisy", 3),
        ("IMDB", "main", 6),
        ("IMDB", "hard_noisy", 3),
    ]
    jobs: List[Job] = []
    for dataset, setting, useful in requested:
        params = _base(dataset, useful, CORE_METHODS)
        params.update(
            {
                "modality_setting": "multimodal",
                "modalities": "ATTR,TEXT,STRUCT,HYBRID",
                "modality_dim": 64,
                "shuffle_party_positions": True,
                "party_shuffle_seed_offset": 2026,
                "cache_predictions": True,
                "reuse_prediction_cache": True,
                "cache_dir": "results/multimodal_cache_v3",
            }
        )
        jobs.append(
            _job(
                "multimodal_core",
                f"{dataset.lower()}_{setting}_mixed",
                dataset,
                setting,
                params,
                engine="multimodal",
                make_plots=True,
                description="Mixed private-modality extension with all core methods.",
            )
        )
    return jobs


def modality_ablation_jobs() -> List[Job]:
    variants = {
        "attr": "ATTR",
        "text": "TEXT",
        "struct": "STRUCT",
        "hybrid": "HYBRID",
        "text_struct": "TEXT_STRUCT",
        "mixed": "ATTR,TEXT,STRUCT,HYBRID",
    }
    jobs: List[Job] = []
    for setting, useful in SETTINGS.items():
        for variant, modalities in variants.items():
            params = _base("ACM", useful, ["adaptive_graph_val"])
            params.update(
                {
                    "modality_setting": "multimodal",
                    "modalities": modalities,
                    "modality_dim": 64,
                    "shuffle_party_positions": True,
                    "party_shuffle_seed_offset": 2026,
                    "cache_predictions": True,
                    "reuse_prediction_cache": True,
                    "cache_dir": "results/multimodal_cache_v3",
                }
            )
            jobs.append(
                _job(
                    "modality_ablation",
                    f"acm_{setting}_{variant}",
                    "ACM",
                    setting,
                    params,
                    engine="multimodal",
                    description="Controlled private-modality ablation.",
                )
            )
    return jobs


def central_baseline_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in SETTINGS.items():
            jobs.append(
                _job(
                    "central_baselines",
                    f"{dataset.lower()}_{setting}",
                    dataset,
                    setting,
                    _base(dataset, useful, ["central_fusion", "central_full_gcn"]),
                    description="Optional centralized upper/reference baselines.",
                )
            )
    return jobs


def protocol_sensitivity_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in SETTINGS.items():
            params = _base(dataset, useful, ROBUSTNESS_METHODS)
            params["ignore_dataset_split"] = True
            jobs.append(
                _job(
                    "protocol_sensitivity",
                    f"{dataset.lower()}_{setting}_custom_split",
                    dataset,
                    setting,
                    params,
                    description="Self-defined stratified split sensitivity check.",
                )
            )
    return jobs


def readout_ablation_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in SETTINGS.items():
            for readout_mode in ("active_vote", "mean_party", "best_party"):
                params = _base(dataset, useful, ["adaptive_graph_val"])
                params["readout_mode"] = readout_mode
                jobs.append(
                    _job(
                        "readout_ablation",
                        f"{dataset.lower()}_{setting}_{readout_mode}",
                        dataset,
                        setting,
                        params,
                        description="Final post-consensus readout ablation.",
                    )
                )
    return jobs


def active_party_evaluator_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in SETTINGS.items():
            for evaluator in ("all_parties", "active_party"):
                params = _base(
                    dataset,
                    useful,
                    [
                        "global_topk_reliability",
                        "full_mesh",
                        "best_matched_sparse",
                        "adaptive_graph_val",
                    ],
                )
                params.update(
                    {
                        "validation_evaluator": evaluator,
                        "active_party_id": 0,
                        "max_degree": 3,
                    }
                )
                _use_standard_shared_cache(params)
                jobs.append(
                    _job(
                        "active_party_evaluator",
                        f"{dataset.lower()}_{setting}_{evaluator}",
                        dataset,
                        setting,
                        params,
                        description="Validation-label ownership sensitivity check.",
                    )
                )
    return jobs


def active_party_protocol_jobs() -> List[Job]:
    """VFGL-like validation-label ownership using shared cached predictors."""
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for setting, useful in SETTINGS.items():
            params = _base(dataset, useful, ACTIVE_PARTY_PROTOCOL_METHODS)
            params.update(
                {
                    "validation_evaluator": "active_party",
                    "active_party_id": 0,
                    "max_degree": 3,
                    "shuffle_party_positions": True,
                    "party_shuffle_seed_offset": 2026,
                    "cache_predictions": True,
                    "reuse_prediction_cache": True,
                    "cache_dir": STANDARD_SHARED_CACHE_DIR,
                }
            )
            short_setting = "hard" if setting == "hard_noisy" else "main"
            jobs.append(
                _job(
                    "active_party_protocol",
                    f"{dataset.lower()}_{short_setting}_active_evaluator",
                    dataset,
                    setting,
                    params,
                    output_filename=f"{dataset.lower()}_{short_setting}_active_evaluator.csv",
                    description=(
                        "Active label-holder computes reliability, top-k ranking, "
                        "and validation topology scores from submitted predictions."
                    ),
                )
            )
    return jobs


def strict_label_training_jobs() -> List[Job]:
    """Label-siloed local training via active-party logit-gradient returns."""
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        params = _base(dataset, 3, ACTIVE_PARTY_PROTOCOL_METHODS)
        params.update(
            {
                "validation_evaluator": "active_party",
                "active_party_id": 0,
                "local_training_protocol": "active_party_logit_gradient",
                "label_protocol": "active_party_only",
                "training_protocol": "prediction_split",
                "validation_protocol": "split_val",
                "validation_split_mode": "disjoint",
                "topology_val_fraction": 0.5,
                "cache_namespace": "strict_logit_gradient",
                "cache_predictions": True,
                "reuse_prediction_cache": True,
                "cache_dir": "results/strict_label_cache_v1",
                "shuffle_party_positions": True,
                "party_shuffle_seed_offset": 2026,
                "max_degree": 3,
            }
        )
        jobs.append(
            _job(
                "strict_label_training",
                f"{dataset.lower()}_hard_logit_gradient",
                dataset,
                "hard_noisy",
                params,
                output_filename=f"{dataset.lower()}_hard_strict_label_training.csv",
                description=(
                    "Only the active party owns train/validation labels; passive "
                    "parties receive gradients with respect to submitted logits."
                ),
            )
        )
    return jobs


def strong_setting_jobs() -> List[Job]:
    """Reviewer-critical strict label-holder, local readout, and dropout runs."""
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        common = _base(dataset, 3, STRONG_SETTING_METHODS)
        common.update(
            {
                "label_protocol": "active_party_only",
                "training_protocol": "prediction_split",
                "validation_protocol": "split_val",
                "validation_evaluator": "active_party",
                "active_party_id": 0,
                "local_training_protocol": "active_party_logit_gradient",
                "validation_split_mode": "disjoint",
                "topology_val_fraction": 0.5,
                "cache_namespace": "strict_logit_gradient",
                "cache_predictions": True,
                "reuse_prediction_cache": True,
                "cache_dir": "results/strict_label_cache_v1",
                "shuffle_party_positions": True,
                "party_shuffle_seed_offset": 2026,
                "max_degree": 3,
                "setting_name": "hard",
            }
        )
        jobs.append(
            _job(
                "strong_setting",
                f"{dataset.lower()}_hard_strict",
                dataset,
                "hard_noisy",
                common,
                output_filename=f"{dataset.lower()}_hard_strict.csv",
                description=(
                    "Strict active label-holder training with disjoint validation "
                    "and simultaneous active/local post-consensus reporting."
                ),
            )
        )

        dropout = dict(common)
        dropout.update(
            {
                "dropout_protocol": "random_party_dropout",
                "dropout_rates": "0.2,0.4",
                "dropout_seeds": "0,1,2",
            }
        )
        jobs.append(
            _job(
                "strong_setting",
                f"{dataset.lower()}_hard_party_dropout",
                dataset,
                "hard_noisy",
                dropout,
                output_filename=f"{dataset.lower()}_hard_party_dropout.csv",
                description=(
                    "Inference-only random non-active party dropout at 20% and "
                    "40%, with three deterministic dropout seeds."
                ),
            )
        )
    return jobs


def deployment_objective_jobs() -> List[Job]:
    """Deployment-aware topology objectives under the strict label-holder setting."""
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        common = _base(dataset, 3, DEPLOYMENT_BASELINE_METHODS)
        common.update(
            {
                "label_protocol": "active_party_only",
                "training_protocol": "prediction_split",
                "validation_protocol": "split_val",
                "validation_evaluator": "active_party",
                "active_party_id": 0,
                "local_training_protocol": "active_party_logit_gradient",
                "validation_split_mode": "disjoint",
                "topology_val_fraction": 0.5,
                "cache_namespace": "strict_logit_gradient",
                "cache_predictions": True,
                "reuse_prediction_cache": True,
                "cache_dir": "results/strict_label_cache_v1",
                "shuffle_party_positions": True,
                "party_shuffle_seed_offset": 2026,
                "max_degree": 3,
                "setting_name": "hard",
                "topology_objective": "active",
                "joint_lambda": 0.5,
            }
        )
        jobs.append(
            _job(
                "deployment_objective",
                f"{dataset.lower()}_hard_baselines",
                dataset,
                "hard_noisy",
                common,
                output_filename=f"{dataset.lower()}_hard_deployment_baselines.csv",
                description=(
                    "Strict deployment references for active/local topology-frontier plots."
                ),
            )
        )

        adaptive_common = dict(common)
        adaptive_common["methods"] = "adaptive_graph_val"
        for objective in ("active", "local_mean"):
            params = dict(adaptive_common)
            params.update({"topology_objective": objective, "joint_lambda": 0.5})
            jobs.append(
                _job(
                    "deployment_objective",
                    f"{dataset.lower()}_hard_tadvfg_{objective}",
                    dataset,
                    "hard_noisy",
                    params,
                    output_filename=f"{dataset.lower()}_hard_tadvfg_{objective}.csv",
                    description=(
                        "TA-DVFG with an explicit deployment-aware topology objective."
                    ),
                )
            )

        for lam in (0.25, 0.5, 0.75):
            tag = str(lam).replace(".", "p")
            params = dict(adaptive_common)
            params.update({"topology_objective": "joint", "joint_lambda": lam})
            jobs.append(
                _job(
                    "deployment_objective",
                    f"{dataset.lower()}_hard_tadvfg_joint_{tag}",
                    dataset,
                    "hard_noisy",
                    params,
                    output_filename=f"{dataset.lower()}_hard_tadvfg_joint_{tag}.csv",
                    description=(
                        "TA-DVFG joint active/local topology objective lambda sweep."
                    ),
                )
            )
    return jobs


def split_validation_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for mode in ("shared", "disjoint"):
            params = _base(dataset, 3, ACTIVE_PARTY_PROTOCOL_METHODS)
            params.update(
                {
                    "validation_evaluator": "active_party",
                    "active_party_id": 0,
                    "validation_split_mode": mode,
                    "topology_val_fraction": 1.0 if mode == "shared" else 0.5,
                    "max_degree": 3,
                }
            )
            _use_standard_shared_cache(params)
            jobs.append(
                _job(
                    "split_validation",
                    f"{dataset.lower()}_hard_{mode}",
                    dataset,
                    "hard_noisy",
                    params,
                    output_filename=f"{dataset.lower()}_hard_{mode}_validation.csv",
                    description="Shared versus disjoint topology/epoch-selection validation labels.",
                )
            )
    return jobs


def decentralized_readout_jobs() -> List[Job]:
    jobs: List[Job] = []
    modes = ("active_vote", "local_mean", "local_worst", "active_party_local")
    for dataset in ("ACM", "DBLP"):
        for mode in modes:
            params = _base(dataset, 3, ["adaptive_graph_val"])
            params.update(
                {
                    "validation_evaluator": "active_party",
                    "active_party_id": 0,
                    "readout_mode": mode,
                    "max_degree": 3,
                }
            )
            _use_standard_shared_cache(params)
            jobs.append(
                _job(
                    "decentralized_readout",
                    f"{dataset.lower()}_hard_{mode}",
                    dataset,
                    "hard_noisy",
                    params,
                    description="Global active vote versus fully local post-consensus readouts.",
                )
            )
    return jobs


def active_party_identity_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for active_party_id in (0, 1, 5, 10, 14):
            params = _base(dataset, 3, ["adaptive_graph_val"])
            params.update(
                {
                    "validation_evaluator": "active_party",
                    "active_party_id": active_party_id,
                    "max_degree": 3,
                }
            )
            _use_standard_shared_cache(params)
            jobs.append(
                _job(
                    "active_party_identity",
                    f"{dataset.lower()}_hard_party{active_party_id}",
                    dataset,
                    "hard_noisy",
                    params,
                    description="Checks that evaluator identity does not privilege a party index.",
                )
            )
    return jobs


def validation_label_budget_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
            params = _base(
                dataset,
                3,
                ["topk_reliability_vote", "full_mesh", "adaptive_graph_val"],
            )
            params.update(
                {
                    "validation_evaluator": "active_party",
                    "active_party_id": 0,
                    "validation_split_mode": "shared",
                    "topology_val_fraction": fraction,
                    "max_degree": 3,
                }
            )
            _use_standard_shared_cache(params)
            tag = str(fraction).replace(".", "p")
            jobs.append(
                _job(
                    "validation_label_budget",
                    f"{dataset.lower()}_hard_f{tag}",
                    dataset,
                    "hard_noisy",
                    params,
                    description="Sensitivity to the active evaluator's topology-label budget.",
                )
            )
    return jobs


def topology_frequency_jobs() -> List[Job]:
    variants = (
        ("once", 20, 1),
        ("every50", 50, 0),
        ("every20", 20, 0),
        ("every10", 10, 0),
    )
    jobs: List[Job] = []
    for dataset in ("ACM", "DBLP"):
        for name, every, maximum in variants:
            params = _base(dataset, 3, ["adaptive_graph_val"])
            params.update(
                {
                    "validation_evaluator": "active_party",
                    "topology_every": every,
                    "topology_max_updates": maximum,
                    "max_degree": 3,
                }
            )
            _use_standard_shared_cache(params)
            jobs.append(
                _job(
                    "topology_frequency",
                    f"{dataset.lower()}_hard_{name}",
                    dataset,
                    "hard_noisy",
                    params,
                    description="Accuracy/control-plane trade-off from topology update frequency.",
                )
            )
    return jobs


def party_scalability_jobs() -> List[Job]:
    jobs: List[Job] = []
    for parties in (5, 10, 15, 20, 30):
        useful = max(1, int(round(0.2 * parties)))
        max_sparse_edges = parties * 3 // 2
        params = _base(
            "ACM",
            useful,
            ["topk_reliability_vote", "full_mesh", "adaptive_graph_val"],
        )
        params.update(
            {
                "num_parties": parties,
                "topk_reliability_k": min(5, parties),
                "matched_edge_count": min(5, max_sparse_edges),
                "adaptive_edge_budget": min(15, max_sparse_edges),
                "adaptive_min_edges": min(5, max_sparse_edges),
                "max_degree": 3,
                "validation_evaluator": "active_party",
                "cache_namespace": "scalability",
                "cache_predictions": True,
                "reuse_prediction_cache": True,
                "cache_dir": "results/scalability_cache_v1",
                "shuffle_party_positions": True,
                "party_shuffle_seed_offset": 2026,
            }
        )
        jobs.append(
            _job(
                "party_scalability",
                f"acm_hard_n{parties}",
                "ACM",
                "hard_noisy",
                params,
                output_filename=f"acm_hard_scalability_n{parties}.csv",
                description="Party-count scalability at a fixed 20% useful-party ratio.",
            )
        )
    return jobs


SUITE_BUILDERS = {
    "core": core_jobs,
    "reviewer_safe": reviewer_safe_jobs,
    "reviewer_safe_shuffled": reviewer_safe_shuffled_jobs,
    "topology_objective": topology_objective_jobs,
    "edge_budget": edge_budget_jobs,
    "consensus": consensus_jobs,
    "noise_ratio": noise_ratio_jobs,
    "multimodal_core": multimodal_core_jobs,
    "modality_ablation": modality_ablation_jobs,
    "central_baselines": central_baseline_jobs,
    "protocol_sensitivity": protocol_sensitivity_jobs,
    "readout_ablation": readout_ablation_jobs,
    "active_party_evaluator": active_party_evaluator_jobs,
    "active_party_protocol": active_party_protocol_jobs,
    "strict_label_training": strict_label_training_jobs,
    "split_validation": split_validation_jobs,
    "decentralized_readout": decentralized_readout_jobs,
    "active_party_identity": active_party_identity_jobs,
    "validation_label_budget": validation_label_budget_jobs,
    "topology_frequency": topology_frequency_jobs,
    "party_scalability": party_scalability_jobs,
    "strong_setting": strong_setting_jobs,
    "deployment_objective": deployment_objective_jobs,
}


def available_suites() -> List[str]:
    return list(SUITE_ORDER)


def build_jobs(suites: Iterable[str]) -> List[Job]:
    requested = list(suites)
    if "all" in requested:
        requested = available_suites()
    unknown = sorted(set(requested) - set(SUITE_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown suite(s): {', '.join(unknown)}")
    jobs: List[Job] = []
    for suite in SUITE_ORDER:
        if suite in requested:
            jobs.extend(SUITE_BUILDERS[suite]())
    return jobs
