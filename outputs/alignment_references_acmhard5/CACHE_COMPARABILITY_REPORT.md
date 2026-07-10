# Cache Comparability Report

Cache path: `C:\Users\Yin Kuan\OneDrive\桌面\Durham_CS\Research\TA-DVFG\results\core_cache_v2\acm_hard_useful3_seed42_party_preds.pt`
Cache load status: `loaded_without_strict_validation: Prediction cache C:\Users\Yin Kuan\OneDrive\桌面\Durham_CS\Research\TA-DVFG\results\core_cache_v2\acm_hard_useful3_seed42_party_preds.pt does not match local-training config:
  - shuffle_party_positions: cache=True, requested=False
  - cache_namespace: cache='standard', requested='alignment_references'`
Cache has logits: `False`

Party rows passing: 0
Party rows mismatching: 75
Party rows unresolved: 0
Maximum probability difference: 1.0
Validation prediction disagreements: 29152
Test prediction disagreements: 29193

Logit comparison note: the legacy headline cache stores probabilities but not logits, so logit-level cache comparison is marked unavailable.

Decision: Fresh local predictors do not fully match or could not be resolved; use only fresh within-suite prediction references.
