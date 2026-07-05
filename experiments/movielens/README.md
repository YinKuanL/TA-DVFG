# MovieLens TA-DVFG Experiment

This adds MovieLens 1M as a real-world cached-prediction experiment. It does
not modify the ACM/DBLP/IMDB experiments or the core TA-DVFG algorithm.

## Protocol

- Dataset: MovieLens 1M `ratings.dat`, `users.dat`, `movies.dat`.
- Split: chronological 70% train, 10% validation, 20% test over interactions.
- Label: `1` when `rating >= 4`, otherwise `0`.
- Alignment: all parties predict the same validation and test user-movie pairs.
- Cross-party data: only `[num_pairs, 2]` prediction probabilities enter
  topology selection and consensus. Hidden embeddings are never exchanged.
- Seeds: `42,43,44,45,46`.
- NDCG@10: ranks the observed test candidate interactions per user using the
  positive-class score. This is reported separately from binary classification.

## Parties

| Party | Model | Hidden dim | Private view |
|---|---|---:|---|
| P1 | matrix factorization | 64 | rating interactions |
| P2 | profile MLP + movie ID embedding | 32 | age, gender, occupation |
| P3 | movie metadata MLP + user ID embedding | 128 | genres, release year |
| P4 | genre-KNN smoothed MLP | 48 | semantic/genre KNN graph |
| P5 | popularity/temporal MLP | 16 | train-only popularity, activity, rating history, timestamp |

The 15-party `main` and `hard` settings add degraded but nontrivial variants of
these real views. `main` uses 6 useful and 9 weak/noisy parties; `hard` uses 3
useful and 12 weak/noisy parties. Party order is shuffled per seed for those
settings.

## Commands

From the repository root, use the base Python with the venv packages if the
`.venv` launcher has Unicode-path issues:

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$PY='C:\Users\Yin Kuan\AppData\Local\Programs\Python\Python314\python.exe'
```

Download and prepare MovieLens 1M:

```powershell
& $PY experiments\movielens\prepare_movielens.py --data-dir data\movielens
```

Small smoke test without downloading data:

```powershell
& $PY experiments\movielens\run_movielens_tadvfg.py `
  --synthetic-smoke --seeds 42 --local-epochs 2 `
  --output-dir outputs\movielens_smoke
```

Five-party MovieLens run:

```powershell
& $PY experiments\movielens\run_movielens_tadvfg.py `
  --data-dir data\movielens\ml-1m `
  --output-dir outputs\movielens `
  --seeds 42,43,44,45,46 `
  --num-parties 5 --setting five --local-epochs 8
```

15-party Main:

```powershell
& $PY experiments\movielens\run_movielens_tadvfg.py `
  --data-dir data\movielens\ml-1m `
  --output-dir outputs\movielens_main15 `
  --seeds 42,43,44,45,46 `
  --setting main --local-epochs 8
```

15-party Hard:

```powershell
& $PY experiments\movielens\run_movielens_tadvfg.py `
  --data-dir data\movielens\ml-1m `
  --output-dir outputs\movielens_hard15 `
  --seeds 42,43,44,45,46 `
  --setting hard --local-epochs 8
```

Generate figures:

```powershell
& $PY experiments\movielens\plot_movielens_results.py --output-dir outputs\movielens
```

## Outputs

- `movielens_party_performance.csv`
- `movielens_tadvfg_per_seed.csv`
- `movielens_tadvfg_summary.csv`
- `movielens_selected_topologies.json`
- `movielens_accuracy_auc_bar.png`
- `movielens_comm_tradeoff.png`
- `movielens_topology_example.png`
- `movielens_party_reliability.png`

## Validation Notes

- The split is chronological, so future interactions are not included in train.
- Popularity and user/movie historical features are computed from train rows
  only.
- Every party receives the same aligned validation and test pair IDs.
- TA-DVFG selection uses validation labels only; test labels are used only for
  final reporting.
- Sparse matched baselines use `--matched-edge-count`; for 15 parties the
  default is raised to 5. Full mesh has `N*(N-1)/2` edges, so 15 parties have
  105 edges.

