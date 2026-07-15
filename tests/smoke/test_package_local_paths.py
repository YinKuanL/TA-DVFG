from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_packaged_experiment_paths_are_self_contained() -> None:
    """The staged package must not retain research-workspace code paths."""
    if not (ROOT / "core" / "ta_dvfg_hgb_reliability.py").is_file():
        return  # Source-workspace layout; the builder tests the transformed copy.
    required = [
        "experiments/run_cached_core.py",
        "experiments/run_experiments.py",
        "experiments/run_k_sensitivity.py",
        "experiments/run_nested_weak_scaling.py",
        "experiments/movielens/run_movielens_tadvfg.py",
        "experiments/alignment_references.py",
        "analysis/mechanisms/_historical/task1_2_runner.py",
        "analysis/mechanisms/_historical/task3_4_runner.py",
        "analysis/provenance/_historical/step_c_gate.py",
        "analysis/provenance/_historical/step_c5_topology_audit.py",
    ]
    assert all((ROOT / relative).is_file() for relative in required)
    old_engine_layout = ' / "' + "main" + " experiment" + '" / '
    old_multimodal_layout = ' / "' + "multi" + "-modalities" + '" / '
    for path in ROOT.rglob("*.py"):
        if path.name == "build_reproducibility_package.py":
            continue  # Distribution tooling reads the source-workspace layout.
        text = path.read_text(encoding="utf-8", errors="replace")
        assert old_engine_layout not in text, path
        assert old_multimodal_layout not in text, path


def test_package_has_audited_headline_configs() -> None:
    if not (ROOT / "core").is_dir():
        return
    expected = {
        "acm_main", "acm_hard_noisy", "dblp_main", "dblp_hard_noisy",
        "imdb_main", "imdb_hard_noisy",
    }
    actual = {
        path.parent.name
        for path in (ROOT / "configs" / "headline").glob("*/metrics_config.json")
    }
    assert actual == expected
