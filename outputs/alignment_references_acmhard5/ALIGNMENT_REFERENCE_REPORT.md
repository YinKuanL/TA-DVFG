# Alignment Reference Report

Scope: HGB ACM hard, seeds 42,43,44,45,46.

## Hidden-State Export

Status: PASS

The export path calls the existing frozen `LocalGCN.encode()` and then the unchanged local `cls` head. No hidden states are synthesized from logits or probabilities.

## Pilot Training Check

Local predictor average training loss snapshots: 1.0948, 0.1881, 1.0951, 0.1921, 1.0956, 0.1894, 1.0948, 0.1897, 1.0952, 0.1938

## Best Alignment Reference

Project-and-Mean (0.5941 Test@BestVal)

## Comparison to TA-DVFG

TA-DVFG is higher than the tested stronger-information alignment references in this pilot.

These rows are centralized stronger-information latent-alignment references, not communication-matched P2P baselines.

## Communication Boundary

Representation references require hidden-state collection and centralized fusion. TA-DVFG uses prediction-space peer communication and supports sparse deployment. The payload columns are therefore reported separately rather than collapsed into one shared communication number.
