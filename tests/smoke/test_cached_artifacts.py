from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_k_module():
    name = "k_sensitivity_cached_smoke"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "experiments" / "run_k_sensitivity.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cached_k_table_and_figures_reproduce_headlines(tmp_path: Path) -> None:
    module = load_k_module()
    summary = pd.read_csv(ROOT / "metadata" / "k_sensitivity_summary.csv")
    headline = summary[(summary["method"] == "adaptive_graph_val") & (summary["K"] == 1)]
    means = dict(zip(headline["dataset"], headline["test_mean"]))
    assert round(100 * means["ACM"], 2) == 85.54
    assert round(100 * means["DBLP"], 2) == 90.25

    module.write_tables(summary, tmp_path)
    module.write_figures(summary, tmp_path)
    table = (tmp_path / "k_sensitivity_table.csv").read_text(encoding="utf-8")
    assert "85.54" in table and "90.25" in table
    assert (tmp_path / "figures" / "acm_k_accuracy.png").is_file()
    assert (tmp_path / "figures" / "dblp_k_accuracy.png").is_file()
    assert (tmp_path / "figures" / "k_reachability.png").is_file()

