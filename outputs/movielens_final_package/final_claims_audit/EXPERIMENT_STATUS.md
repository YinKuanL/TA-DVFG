# Experiment Status

Generated: 2026-07-04T09:06:06.545459+00:00

## Phase 0 Audit

Status: complete.

- Selector defaults were extracted from `main experiment/ta_dvfg_hgb_reliability.py`.
- Canonical experiment-plan defaults were extracted from `experiments/experiment_plan.py`.
- MovieLens target/split and five-party values were verified from `outputs/movielens/*.csv`.
- Best Single MovieLens per-seed ROC-AUC values were verified from raw per-seed CSV.

## Local Hidden-State Export Feasibility

Status: partial / feasibility-gated.

- HGB local GCN code already has an encoder path (`LocalGCN.encode`) and helper-style embedding access in the engine.
- Existing cached prediction files are prediction caches and do not store hidden representations.
- MovieLens party models currently expose final probabilities only through the experiment runner; hidden states are internal to model-specific forward paths and are not cached.
- Fair representation-alignment baselines therefore require fresh runs with explicit hidden-state export added without changing local predictor behavior.
- Do not synthesize hidden representations from prediction outputs.

## Priority 1 Nested Weak-Party Scaling

Status: partial.

- `experiments/run_nested_weak_scaling.py` implements the required nested protocol.
- The runner trains/loads one 18-party cache per dataset/seed with 3 useful parties and 15 weak parties, then replays nested subsets for weak counts `(0, 2, 5, 10, 15)`.
- A smoke test completed for ACM seed 42, weak counts 0 and 2, with 1 local epoch.
- The full primary run on ACM Hard and DBLP Hard, seeds 42--46, epochs 300, is still pending.

Existing `results/noise_ratio` jobs vary useful-party count, but they do not implement the required nested weak-set protocol. They must not be reported as the requested nested weak-party scaling experiment.

## Priority 2 Validation-to-Test Alignment

Status: not complete.

The current adaptive selector records final topology and evaluation counts, but not every candidate edge's validation/test delta. This needs instrumentation or a replay script from cached probabilities.

## Priority 3 Calibration Robustness

Status: not complete.

No temperature-scaling or ECE/Brier/NLL audit has been run yet.

## Priority 4 Heterogeneity Ladder

Status: not complete.

The MovieLens completed package supports heterogeneous views/dimensions, but not a verified H0-H3 ladder.

## Priority 5 Representation Alignment References

Status: blocked until hidden-state export is implemented and audited.

## Exact Selector Default Map

- K / top-k reliability default: `5` in engine, canonical `5` in experiment plan.
- B / adaptive edge budget default: `15` in engine, canonical `15` in experiment plan.
- d_max / max degree default: `2` in engine, canonical `2` in experiment plan.
- m / adaptive min edges default: `15` in engine, canonical `5` in experiment plan.
- epsilon / adaptive min gain default: `0.0` in engine, canonical `0.001` in experiment plan.
- topology update interval default: `20` epochs.
- prediction consensus steps: `1`.
- prediction self-weight: `0.85` in engine, canonical `0.85` in experiment plan.
