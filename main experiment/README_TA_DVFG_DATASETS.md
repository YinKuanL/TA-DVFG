# TA-DVFG Dataset Scripts

This package contains a generic HGB dataset experiment script for TA-DVFG:

- `ta_dvfg_hgb_final.py`: generic script for ACM, DBLP, IMDB, and optional Freebase via PyTorch Geometric `HGBDataset`.
- `run_tadvfg_all.sh`: example commands for main experiments, edge-budget sweep, and noisy-party sweep.

## Recommended main datasets

Use ACM + DBLP + IMDB as the main complete experiment set.

| Dataset | Target node | Recommended views |
|---|---|---|
| ACM | paper | PAP, PSP/PLP, KNN |
| DBLP | author | APA, APCPA, APTPA, KNN |
| IMDB | movie | MAM, MDM, MKM/KNN |

## Install dependencies

```bash
pip install numpy scipy scikit-learn matplotlib torch
# install PyG according to your torch/CUDA version:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Run main experiments

```bash
bash run_tadvfg_all.sh
```

Or run separately:

```bash
python ta_dvfg_hgb_final.py --dataset DBLP --graph_views APA,APCPA,APTPA,KNN --num_parties 15 --useful_parties 6 --view_setting hard --seeds 42,43,44,45,46 --plot --save_csv dblp_hgb_tadvfg.csv
```

## Notes

1. The script builds target-target graph views from meta-paths using available HGB edge types.
2. If a requested view is not found, the script falls back to KNN and prints a warning.
3. For IMDB, if `MKM` is not available in the downloaded dataset version, use `--graph_views MAM,MDM,KNN`.
4. Freebase is included as optional but may require extra checking of target node type and relations.
