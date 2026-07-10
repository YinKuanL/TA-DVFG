# Paper Source Sync Manifest

Imported date: 2026-07-10

This manifest records the canonical `paper/source/` mirror created during the
initial GitHub and Overleaf synchronization bootstrap.

Note: the configured Overleaf project exposed the three live paper entry files
(`main.tex`, `supplementary.tex`, and `ReproducibilityChecklist.tex`) plus the
AAAI template file `CameraReady2027.tex` through read-only MCP inspection.
Referenced bibliography, style, and figure dependencies were resolved from the
existing local paper mirror because they were required by the inspected source
but were not exposed as separate Overleaf files in the MCP listing.

| File | Role | Source | Imported date | SHA-256 |
| --- | --- | --- | --- | --- |
| `main.tex` | Main paper compile root | Overleaf | 2026-07-10 | `048C14D0CF4001FBF97FE8C23066685812CC4DCAA652CED5EB177155B2A35410` |
| `supplementary.tex` | Supplement compile root | Overleaf | 2026-07-10 | `0042263690101FBB2F3A062075C328DA370E3956CC22971315389ABB053CD388` |
| `ReproducibilityChecklist.tex` | Reproducibility checklist compile root | Overleaf | 2026-07-10 | `B21CC8CA4F6806D827A0C1DA69DF79D26261CFCBF42593CCF59530EFCF2752E8` |
| `references.bib` | Main paper bibliography | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `B281D8BB84C858D2AF9D4860DB8C6BC9AE817F336C8EB6F5FB57A122E6B87A87` |
| `aaai2027.sty` | Required AAAI style file | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `391BCE82815BF698B8E382DD3AE7E30C75D7AB46DF140CB295B1266016BC8623` |
| `aaai2027.bst` | Required AAAI bibliography style file | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `5DB7765BA99DE5C1E4686F9B3940A0ADD9C5E702F2164514462BEC130CCB6E3C` |
| `figures/movielens_pareto_auc_communication.png` | Main paper figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `D0FCA0844748EA48ADA8ACE1E63E4921FA3CEB318CB46A627A4FA0DFDAC80D4D` |
| `figures/movielens_party_val_test_auc.png` | Main paper figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `2E71F60DBD1ED21B7622B587569D9B504BAB306C5FFBB40E2DF2A8A854A2B6F9` |
| `figures/topology_objective_ablation.pdf` | Main paper figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `F9325937C7F45BC2A4BA897E19B10D1EB4EFFB7B3CAF4405C26765952AF43CA9` |
| `figures/deployment_tradeoff.png` | Main paper figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `66BE2A59DF64BE2F9DDD4622C16E52011D57344A39BCD8371CCEF736EBB24D11` |
| `figures/alignment_acmhard5_five_seed.pdf` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `6349C5433AF3F491EF9A13235C53BEDAD99413A0682B695D500B1DB0014C9EA9` |
| `figures/alignment_acmhard5_paired_lines.pdf` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `CE2CF3BAF13E1F901452D5D05FBBA5B98D219B85F06C0807149030560D889A40` |
| `figures/label_budget_topology_frequency_tight.png` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `42A572888EEE92420096977C6C0767EF409AFA0D39DF103C3586E0231CB45D08` |
| `figures/minedges_sensitivity_tight.png` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `2B528BBFCE4D695F95BED7387A105F6895E71DDB8D14E670D28F042A06A24A47` |
| `figures/consensus_ablation_p11.png` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `3605B7F63528A5A1144F2F9C63C14AA031137D04E57DC59F673CB07D9502E8CF` |
| `figures/consensus_ablation_p12.png` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `EFE67001E21874A279C8B17663081E9B708E4C9A2171647C42D3AF86FFF6E0F0` |
| `figures/consensus_ablation_p21.png` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `A3A4381B262E24D9EBC915F5BB8BE2C6FF345070D36E5041CD828A2104413F78` |
| `figures/consensus_ablation_p22.png` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `3A7E544E151079144364CBE85A99FF2FC3E1997E98CFB419277FE22832C9931F` |
| `figures/edge_budget_acm_edges.pdf` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `E85857E659463831E80B8FF5E4B6EF9FB62C3D9183C0316FE5FDAD30A9F8234E` |
| `figures/edge_budget_dblp_edges.pdf` | Supplement figure dependency | Local dependency mirror for inspected Overleaf source | 2026-07-10 | `6D7D5DCCDF5CAE2FD579D29E5819E3DA8855F55CDAB74E90B187794BC0C2A315` |
