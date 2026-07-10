# ACM Hard Five-Seed Alignment Reference Report

Scope: ACM Hard only, seeds 42-46. No ACM Main, DBLP, IMDB, MovieLens, or multi-dataset suite was run.

## Main Result

- TA-DVFG: 0.8560 mean Test@BestVal.
- Reliability-Selected Project-and-Mean: 0.7710 mean Test@BestVal.
- Delta selected latent minus TA-DVFG: -0.0850; 95% CI [-0.1047, -0.0654], paired t-test p=0.0002756, Wilcoxon p=0.0625.
- Global Top-k Reliability is strongest among deployable/non-oracle methods here: 0.8619.
- Useful-Only latent fusion is diagnostic only: 0.8893.

## Analysis Answers

1. All-party latent fusion is consistently harmed by weak-party contamination: Project-and-Mean averages 0.5941, far below Reliability-Selected Project-and-Mean at 0.7710.
2. Reliability selection recovers much of the latent-fusion performance, improving Project-and-Mean by 0.1769 mean accuracy.
3. After fair reliability selection, TA-DVFG is better than Reliability-Selected Project-and-Mean in this ACM Hard suite (0.8560 vs 0.7710; 5/5 seed wins). The paired t-test is significant, while the exact Wilcoxon p-value is limited by n=5 resolution; do not describe this as equivalence.
4. The best predictive trade-off in this ACM Hard suite is Global Top-k Reliability, but it is a centralized prediction reference. TA-DVFG remains the sparse P2P method with party-local post-consensus readout. Reliability-selected latent fusion requires hidden-state access, a shared projection interface, and centralized collection.

## Selected Parties

Reliability-Selected Project-and-Mean selected parties:

```text
 seed selected_parties selected_useful_count selected_weak_count
   42  [0, 2, 1, 9, 5]                     3                   2
   43  [0, 2, 1, 3, 5]                     3                   2
   44 [0, 2, 1, 6, 13]                     3                   2
   45 [0, 2, 1, 3, 12]                     3                   2
   46  [0, 2, 1, 9, 5]                     3                   2
```

## Paper Patch Guidance

Do not patch the main paper automatically from this ACM Hard-only suite. Use `PAPER_PATCH_NOTES.md` for conditional wording after broader review.
