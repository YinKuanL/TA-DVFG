# AAAI all-experiments code package report

Date: 2026-07-15

## Deliverable

- Code folder: `C:\Users\Yin Kuan\OneDrive\桌面\Durham_CS\Research\TA-DVFG\dist\aaai27_all_experiments_code\TA-DVFG`
- ZIP: `C:\Users\Yin Kuan\OneDrive\桌面\Durham_CS\Research\TA-DVFG\dist\aaai27_all_experiments_code\TA-DVFG_AAAI27_All_Experiments_Code.zip`
- ZIP SHA-256: `851c32f75118d59445507b9da2e668fd32e4ec45427de48b26af797b37b0b99b`
- Manifested payload files: 164

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
- Nested weak-party scaling runner, audited configuration, and complete
  ACM/DBLP seed evidence for weak counts 0/2/5/10/15.
- Validation-label budget and topology-update frequency.
- Party-count scalability.
- Minimum-link `m=0`, edge-budget, and gain-threshold sweeps.
- Consensus/readout, topology-objective, noise, and protocol controls.
- MovieLens five-party and fifteen-party Main/Hard experiments, models,
  aggregation, and plotting.
- ACM Hard latent-alignment reference models, runner, five-seed evidence,
  paired statistics, communication/parameter counts, and 75 hidden-export checks.
- Communication accounting, statistics, provenance, and table/figure
  regeneration.
## Missing or partial areas

- **Mechanism replay inputs (partial rerun):** all available generators and
  seed-level/statistical outputs are included; large NPZ replay bundles are
  excluded by the issue constraints. These commands are explicitly marked as
  non-runnable derived-evidence analyses, not clean-ZIP Level-2 commands.
- Public HGB and MovieLens datasets are intentionally excluded. Full fresh
  reruns require the user to obtain them separately and place them at the
  documented package-local paths.

Multimodal plans, evidence, table inputs, and reviewer-facing references were
removed because the current main paper and supplement do not reference them.

## Artifact-closure provenance

The complete closure files were recovered unchanged from reachable commit
`5cc081568eea994f2de3f5d4bd422b328587178f`:

- `results/nested_weak_scaling/{raw_weak_scaling.csv,weak_scaling_summary.csv,weak_scaling_paired_deltas.csv,manifest.json}`
- `outputs/alignment_references_acmhard5/` seed, summary, statistics,
  communication, capability, selection, diagnostic, metadata, and
  hidden-export CSV/JSON files.

Only workstation path fields are redacted in the package copy. Numerical fields
are byte-for-byte values from the recovered evidence.

## Clean-ZIP validation

Validation used a fresh extraction of the final ZIP and did not run any full
experiment.

| Check | Result |
|---|---|
| `python -m compileall -q .` | PASS |
| `python -m pytest tests/unit tests/smoke -q` | PASS: 16 passed, 0 failed |
| `python scripts/verify_reported_values.py` | PASS: 143 passed, 0 failed |
| `python scripts/verify_package.py` | PASS: 2 groups, 0 failed; 164 payload files |
| Additional plan/defense tests | PASS: 28 passed, 0 failed, 1 PyTorch warning |
| `python scripts/regenerate_artifacts.py` | PASS: 18 files regenerated from bundled evidence |

The 143 anonymous-package checks include complete nested-scaling seed coverage,
the 12.59/5.16-point final gaps, TA-DVFG/Full-Mesh link counts, all requested ACM
alignment means/statistics, and 75/75 hidden-export equivalence checks. Four
additional manuscript-text presence checks are intentionally skipped because
paper sources are excluded.

## Protected-source confirmation

The package workflow did not edit `paper/source/` or any historical reported
value file. Pre-existing worktree modifications in the protected paper files
were left untouched and were not staged.
