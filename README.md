# TA-DVFG: Sparse Prediction Topology Learning

TA-DVFG studies whether heterogeneous graph predictors can collaborate through sparse prediction exchange without representation alignment.

## Research question

When parties train local graph predictors over different feature or view partitions, can they improve downstream prediction by exchanging only task-level predictions over a learned sparse topology?

## Core method

TA-DVFG learns a sparse party-to-party prediction topology and a topology-aware readout over independently trained local predictors. The repository contains the HGB and MovieLens experiment implementations, cached-output analyses, mechanism audits, statistical reporting, and protected paper sources.

TA-DVFG is an **inference-stage topology/readout mechanism**. It does not align
or fuse private hidden representations and the selector does not back-propagate
into independently trained local predictors.

## Information and privacy boundaries

Parties exchange task-level prediction vectors. Hidden states are not required.
Active topology selection is centralized at a low-frequency evaluator using
held-out validation predictions. Deployment consensus is sparse peer-to-peer
over the selected links. An active readout may collect the post-consensus
predictions that it consumes; a party-local readout needs no global collection.

The code provides no formal privacy guarantee. The strict label-location variant
is an API-separation protocol: passive parties submit logits and receive logit
gradients without directly receiving labels. Those gradients may reveal label
information, so the boundary is neither cryptographic nor differential-private.

## Repository map

| Path | Contents |
|---|---|
| `core/ta_dvfg_hgb_reliability.py` (package) | Preserved HGB engine: parties, caches, consensus, topology, readouts, communication |
| `models/` | MovieLens party predictor implementations |
| `experiments/` | HGB, MovieLens, nested scaling, K sensitivity, alignment and analysis runners |
| `analysis/` | Stable entry points for provenance and mechanism audits |
| `configs/` | Machine-readable audited experiment manifests |
| `metadata/` | Tracked paper values and headline configuration audits |
| `results/`, `outputs/` | Local ignored raw results, summaries and large caches |
| `tests/` | Deterministic unit, regression and smoke tests |
| `scripts/` | Coverage, value verification, cached regeneration and package building |
| `environment/` | Recorded experiment-system and package-version evidence |
| `docs/` | Inventory, supplement map, crosswalk, provenance and cleanup report |
| `paper/source/` | Protected main paper, supplement and checklist sources |

## Install

Python 3.14 was used for the reported experiments. A clean CPU environment can
be prepared with:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The complete recorded package snapshot is in `environment/package_versions.txt`.
HGB full reruns additionally require the PyTorch Geometric HGB datasets;
MovieLens full reruns require the GroupLens MovieLens 1M files. Datasets are not
redistributed here.

## Quick validation

No dataset or GPU is needed for the lightweight suite:

```bash
python -m pytest tests/unit tests/smoke -q
```

Run all lightweight checks and coverage/value audits with:

```bash
python scripts/verify_supplement_coverage.py
python scripts/verify_reported_values.py
```

## Reproduction entry points

| Supplement area | Entry point |
|---|---|
| HGB cached core and comparisons | `experiments/run_cached_core.py`, `experiments/experiment_plan.py` |
| Same-state peer exchange | `analysis/mechanisms/run_peer_exchange_audit.py` |
| Topology/Test@BestVal provenance | `analysis/provenance/run_topology_provenance.py`, `run_testatbestval_replay.py` |
| Deployment objectives | `experiments/deployment_objective_analysis.py` |
| Nested weak-party scaling | `experiments/run_nested_weak_scaling.py` |
| MovieLens | `experiments/movielens/run_movielens_tadvfg.py` |
| Latent-alignment reference | `experiments/alignment_references.py` |
| K-hop sensitivity | `experiments/run_k_sensitivity.py` |
| Statistics, HGB tables and figures | `experiments/statistical_tests.py`, `experiments/final_analysis.py` |

Exact code/output mappings and coverage statuses are in
`docs/SUPPLEMENT_TO_CODE_MAP.md` and `docs/RESULT_PROVENANCE.md`.

## Regenerate from existing outputs

These commands do not train local predictors:

```bash
python experiments/final_analysis.py --results-root results --output-dir artifacts/regenerated/hgb
python experiments/deployment_objective_analysis.py --input-root results/deployment_objective_runs --output-dir artifacts/regenerated/deployment
python experiments/movielens/extended_movielens_suite.py --output-dir outputs/movielens_leakage_safe_20260705
```

Generated HGB tables/figures go to the selected `--output-dir`. MovieLens derived
files are written beside the existing leakage-safe raw output. Protected paper
figures are never overwritten by the cleanup workflow.

## Build the reproducibility package

```bash
python scripts/build_reproducibility_package.py
```

The build writes a local reproducibility archive under `dist/` together with
`MANIFEST.txt` and `SHA256SUMS.txt`. The builder excludes datasets, large caches,
credentials, local paths, paper drafts, histories, logs and generated Python
caches, then runs the supplement coverage audit against the staged package.

See `REPRODUCE.md` for no-rerun, lightweight and full-experiment levels,
prerequisites, expected locations and runtime boundaries.

## Hardware and software

The reported runs used Windows 11, an Intel i7-12700H with 31.64 GiB RAM, and a
CPU-only PyTorch 2.12.1 build. Although the host contained an RTX 4060 Laptop GPU
and CUDA 12.6 toolkit, CUDA was unavailable to that PyTorch build. Exact recorded
details are in `environment/`.

## Citation and license

Citation metadata is provided in `CITATION.cff` without author-identifying
details. No repository license was present in the source history; users must
treat the code as all-rights-reserved until an explicit license is added.
