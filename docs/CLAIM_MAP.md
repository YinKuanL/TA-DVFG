# Claim Map

Use one row per important scientific claim.

| ID | Claim | Location | Evidence | Protocol caveat | Strength | Status |
|---|---|---|---|---|---|---|
| C1 |  |  |  |  | descriptive / comparative / causal | TODO |

## Strength rules

### Descriptive
Example: "The learned topology uses fewer links."

Needs:
- directly measured quantity;
- clearly stated setting.

### Comparative
Example: "Method A outperforms Method B."

Needs:
- matched protocol;
- fair information access;
- clear selection rule;
- uncertainty/statistical context where appropriate.

### Causal or mechanistic
Example: "Component X causes the improvement."

Needs stronger evidence than an ablation alone unless the design supports the causal interpretation.

## Red flags

- [ ] Test-set selection.
- [ ] Oracle/post-hoc method described as deployable.
- [ ] Stronger-information reference described as a baseline.
- [ ] Missing seeds.
- [ ] Incompatible protocols merged into one conclusion.
- [ ] "Consistently" used with exceptions.
- [ ] "Equivalent" used for non-significant difference.
