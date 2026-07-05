# TA-DVFG Supplement Experiment Code

This folder contains the cleaned experiment code used for the paper results.
It excludes raw datasets, caches, Python bytecode, temporary files, and LaTeX.

## Contents

- `main experiment/ta_dvfg_hgb_reliability.py`: core TA-DVFG HGB engine.
- `experiments/`: experiment runners, audits, aggregation, and plotting scripts.
- `experiments/movielens/`: MovieLens 1M experiment suite.
- `models/movielens_parties.py`: heterogeneous MovieLens local predictors.
- `tests/`: lightweight protocol tests.
- `metadata/`: small config/result provenance files.

## Setup

Use Python 3.9+.

```bash
pip install -r requirements.txt
```

The HGB experiments require PyTorch Geometric dataset access. MovieLens requires
MovieLens 1M under `data/movielens/ml-1m/`.

## Main Commands

HGB cached core:

```bash
python experiments/run_cached_core.py --datasets ACM DBLP IMDB --seeds 42,43,44,45,46
```

HGB K sensitivity:

```bash
python experiments/run_k_sensitivity.py
```

Nested weak-party scaling:

```bash
python experiments/run_nested_weak_scaling.py
```

ACM/DBLP alignment references:

```bash
python experiments/alignment_references.py --dataset ACM --setting hard --seeds 42,43,44,45,46
python experiments/alignment_references.py --dataset DBLP --setting hard --seeds 42,43,44,45,46
```

MovieLens 1M:

```bash
python experiments/movielens/prepare_movielens.py --data-dir data/movielens
python experiments/movielens/run_movielens_tadvfg.py --data-dir data/movielens/ml-1m --output-dir outputs/movielens --seeds 42,43,44,45,46 --num-parties 5 --setting five --local-epochs 8
python experiments/movielens/run_movielens_tadvfg.py --data-dir data/movielens/ml-1m --output-dir outputs/movielens_main15 --seeds 42,43,44,45,46 --setting main --local-epochs 8
python experiments/movielens/run_movielens_tadvfg.py --data-dir data/movielens/ml-1m --output-dir outputs/movielens_hard15 --seeds 42,43,44,45,46 --setting hard --local-epochs 8
```

Final aggregation/audit:

```bash
python experiments/final_analysis.py
python experiments/config_audit.py
python experiments/final_claims_audit.py
```

## Notes

- Seeds used in the paper are `42,43,44,45,46`.
- MovieLens target construction uses MovieLens 1M observed ratings only:
  `rating >= 4` is positive; ratings `1--3` are negative; no negative sampling.
- MovieLens split is chronological 70/10/20 over observed interactions.
- Test labels are used only for final reporting.
