# AAAI all-experiments code package report

Date: 2026-07-15

## Deliverable

- Code folder: `C:\Users\Yin Kuan\OneDrive\桌面\Durham_CS\Research\TA-DVFG\dist\aaai27_all_experiments_code\TA-DVFG`
- ZIP: `C:\Users\Yin Kuan\OneDrive\桌面\Durham_CS\Research\TA-DVFG\dist\aaai27_all_experiments_code\TA-DVFG_AAAI27_All_Experiments_Code.zip`
- ZIP SHA-256: `e918e885545846ebad02625fe38d54ea5ce6f928c0522d0285f4b98c1d990b51`
- Manifested payload files: 173

The package was built as one reviewer-facing folder. Experiment paths in the
copied code resolve to that folder's `core/`, `configs/`, `artifacts/`, and local
output directories. The ZIP contains no paper source, PDFs, datasets,
checkpoints, prediction caches, credentials, workstation paths, or identity
metadata.

## Included experiment areas

- HGB cached core for ACM, DBLP, and IMDB Main/Hard settings.
- Reliability, fixed/matched topology, Full Mesh, adaptive pair,
  complementarity, and validation-selected graph comparisons.
- Deployment objectives: active, local, and joint variants.
- Peer-exchange same-state interventions and Test@BestVal/topology provenance.
- Same-cardinality subset baselines and degree/readout audits, including the
  recovered exact historical combined generator.
- Active-party evaluator/identity and strict label-location protocols.
- Multi-hop K sensitivity.
- Nested weak-party scaling runner and audited configuration.
- Validation-label budget and topology-update frequency.
- Party-count scalability.
- Minimum-link `m=0`, edge-budget, and gain-threshold sweeps.
- Consensus/readout, topology-objective, noise, and protocol controls.
- MovieLens five-party and fifteen-party Main/Hard experiments, models,
  aggregation, and plotting.
- Latent-alignment reference models, runner, audit evidence, and figure script.
- Communication accounting, statistics, provenance, and table/figure
  regeneration.
- Multimodal/modality-ablation experiment plans and compact seed-level results.

## Missing or partial areas

- **Multimodal HGB source code (partial):** `experiment_plan.py` references a
  separate historical multimodal engine, but that file is absent from every
  available local and remote branch history checked. The two multimodal suites
  therefore retain plans/configuration and bundled seed evidence but cannot be
  rerun from this package. No substitute implementation was invented.
- **Nested weak-party evidence (partial):** the full runner/configuration is
  included, but no small raw/summary CSV was available. The excluded ten
  prediction caches would be needed for the cached replay.
- **ACM alignment evidence (partial):** the available ACM CSV has one seed,
  although the five-seed runner/configuration is included. Available five-seed
  DBLP supporting evidence is bundled separately.
- **Mechanism replay inputs (partial rerun):** all available generators and
  seed-level/statistical outputs are included; large NPZ replay bundles are
  excluded by the issue constraints.
- Public HGB and MovieLens datasets are intentionally excluded. Full fresh
  reruns require the user to obtain them separately and place them at the
  documented package-local paths.

## Clean-ZIP validation

Validation used a fresh extraction of the final ZIP and did not run any full
experiment.

| Check | Result |
|---|---|
| `python -m compileall -q .` | PASS |
| `python -m pytest tests -q` | PASS: 47 passed, 0 failed, 1 PyTorch warning |
| `python scripts/verify_reported_values.py` | PASS: 71 passed, 0 failed |
| `python scripts/verify_package.py` | PASS: 2 groups, 0 failed; 173 payload files |
| `python scripts/regenerate_artifacts.py` | PASS: 13 files regenerated from bundled evidence |
| Available non-multimodal suite expansion | PASS: 169 jobs across 21 suites |
| Runner `--help` import checks | PASS: 8 passed, 0 failed |

The reported-value verifier performs 71 checks in the anonymous package; four
additional manuscript-text presence checks are intentionally skipped because
paper sources are excluded.

## Protected-source confirmation

The package workflow did not edit `paper/source/` or any historical reported
value file. Pre-existing worktree modifications in the protected paper files
were left untouched and were not staged.
