# Phase 8 — Within-subject vs Cross-subject Physiological Separability

Run date (UTC): 2026-10-18T09:50:14.438892+00:00

## Executive answer

Adjacent pain levels were difficult even within subjects (subject-averaged AUC 0.593), while cross-subject performance was 0.621. Wide controls were more separable within subjects (0.733) and cross-subject (0.779). All five mean within-minus-cross gaps were negative, so this experiment does not show population-level transfer failure as the dominant limitation. The strongest evidence is instead for within-subject overlap plus the statistical disadvantage of fitting a personalized model from only 32 training trials per fold; individual positive-gap exceptions still show heterogeneous transfer behavior.

These are stimulus-intensity labels, not trial-level subjective pain ratings. Performance therefore does not establish a physiological upper bound on pain recognition.

## Protocol and leakage controls

- Input: Phase 2 filtered physiological feature table, 87 subjects, 20 trials per subject/class, pairs PA1_vs_PA2, PA2_vs_PA3, PA3_vs_PA4, PA1_vs_PA3, BL1_vs_PA4.
- Fixed low-dimensional input: 5 Phase 6 preselected features (`phase6_preselected`), chosen before Phase 8 modeling; no Phase 8 feature search was performed.
- Model: L2 Logistic Regression (`C=1`), median imputation and standardization inside each training fold.
- Within-subject: repeated stratified 5-fold CV × 10 repeats; every trial receives 10 OOF scores which are averaged before scoring.
- Cross-subject: exact Phase 3 subject-independent fold assignments; no held-out subject contributes to fitted preprocessing or model parameters.
- Uncertainty: 2000 stratified trial bootstraps for individual AUC/BA and gap CIs; subject bootstrap for pair summaries.
- The within and cross conditions have very different training sizes (32 trials per within-subject fold versus thousands of trials across training subjects). The paired gap is descriptive and cannot be interpreted as a pure causal subject effect.

## Pair-level results

| pair | within AUC mean | cross AUC mean | paired gap mean (95% CI) | positive gaps | median signal overlap | median abs(g) |
|---|---:|---:|---:|---:|---:|---:|
| PA1_vs_PA2 | 0.521 | 0.566 | -0.044 [-0.072, -0.015] | 33.33% | 0.750 | 0.278 |
| PA2_vs_PA3 | 0.606 | 0.619 | -0.014 [-0.044, 0.017] | 45.98% | 0.700 | 0.370 |
| PA3_vs_PA4 | 0.652 | 0.679 | -0.027 [-0.050, -0.001] | 32.18% | 0.700 | 0.398 |
| PA1_vs_PA3 | 0.636 | 0.687 | -0.050 [-0.072, -0.029] | 29.89% | 0.650 | 0.374 |
| BL1_vs_PA4 | 0.829 | 0.871 | -0.041 [-0.053, -0.030] | 13.79% | 0.450 | 1.073 |

Balanced accuracy and individual bootstrap intervals are retained in the machine-readable tables; the headline comparison uses ROC-AUC because it is threshold-independent.

## Mechanistic interpretation

### 1. Are adjacent levels difficult within the same subject?

Yes. The subject-averaged adjacent within-subject AUC was 0.593. The wide-control minus adjacent within-subject difference was 0.140 (95% subject-bootstrap CI 0.116 to 0.163). The signal-level overlap table independently shows substantial overlap in the fixed Phase 6 features. This supports within-subject overlap as a real part of the difficulty.

### 2. Which subjects show the largest transfer gaps?

A positive gap means within-subject AUC exceeded cross-subject AUC. Group means were negative for every pair, but individual exceptions occurred:

| pair | positive point gaps | bootstrap CI entirely > 0 | largest gap (subject) |
|---|---:|---:|---:|
| PA1_vs_PA2 | 29/87 | 1/87 | 0.423 (subject 42, CI 0.090 to 0.725) |
| PA2_vs_PA3 | 40/87 | 5/87 | 0.497 (subject 48, CI 0.170 to 0.780) |
| PA3_vs_PA4 | 28/87 | 4/87 | 0.325 (subject 59, CI 0.072 to 0.570) |
| PA1_vs_PA3 | 26/87 | 1/87 | 0.215 (subject 55, CI -0.070 to 0.508) |
| BL1_vs_PA4 | 12/87 | 0/87 | 0.052 (subject 80, CI -0.052 to 0.175) |

The clearest individual positive-gap cases were concentrated in adjacent boundaries, but most individual intervals were wide. These are heterogeneity flags, not evidence that subject-specific training is generally superior.

### 3. Adjacent versus wide-gap separability

Wide controls exceeded adjacent pairs by 0.140 AUC within subjects and 0.158 AUC cross-subject. This indicates that larger stimulus separation improves physiological separability in both evaluation regimes; it is not solely a transfer phenomenon.

### 4. What drives the PA2 difficulty?

Across PA1/PA2 and PA2/PA3, mean within-subject AUC was 0.563, mean cross-subject AUC was 0.592, and mean paired gap was -0.029. PA1/PA2 was especially close to chance within subjects (0.521). Because cross-subject AUC was higher rather than lower on average, the present comparison points more strongly to within-subject overlap and small personalized training sets than to a population-level PA2 transfer failure. This is not a variance decomposition because training sizes differ.

### 5. Does Phase 6 physiological consistency explain individual separability?

Phase 6 median absolute monotonicity was positively associated with separability, especially for wider pairs:

| pair | ρ with within AUC (BH p) | ρ with cross AUC (BH p) |
|---|---:|---:|
| PA1_vs_PA2 | 0.083 (0.493) | 0.476 (1.08e-05) |
| PA2_vs_PA3 | 0.326 (0.00368) | 0.511 (1.65e-06) |
| PA3_vs_PA4 | 0.466 (1.63e-05) | 0.370 (0.000883) |
| PA1_vs_PA3 | 0.620 (1.15e-09) | 0.699 (5.16e-13) |
| BL1_vs_PA4 | 0.739 (9.55e-15) | 0.730 (1.61e-14) |

Direction reversals showed corresponding negative associations for several pairs (full corrected table in `phase6_consistency_associations.csv`). This supports consistency as a descriptive correlate of individual separability. However, Phase 6 consistency and Phase 8 AUC use the same subjects and overlapping features/levels, so the association is not an independent predictive validation and should not be called causal explanation.

## Trial-order and drift diagnostics

For pain–pain pairs, the median label separability of the filename trial index ranged from 0.562 to 0.595. The median absolute within-class feature–trial Spearman correlation ranged from 0.225 to 0.301 across pairs.

The number of subjects with trial-index AUC ≥ 0.70 was 4, 13, 5, and 12 for PA1/PA2, PA2/PA3, PA3/PA4, and PA1/PA3 respectively. This makes trial-order imbalance a plausible subject-specific confound even though pair medians were modest.

The local documentation does not explicitly establish that the numeric filename suffix is acquisition chronology, so this is recorded as a trial-index proxy rather than proven elapsed time. BL1 uses indices 081–100 while pain trials use 001–080; consequently BL1/PA4 is structurally confounded with this proxy and cannot diagnose temporal drift from these identifiers alone.

## Limitations

- Labels encode calibrated stimulus intensity, not subjective pain experienced on each trial.
- Forty trials per subject/pair constrain personalized estimation; repeated CV reduces split dependence but does not create independent observations.
- Averaging repeated OOF scores makes efficient use of all trials, but trial-level bootstrap intervals condition on the fitted CV procedure.
- Empirical overlap is a descriptive common-bin histogram intersection with 20 observations per class; it is not a population distribution estimate.
- The five features were fixed from Phase 6 to limit overfitting and connect analyses, but they do not span every retained ECG/EDA/EMG feature.
- The within/cross gap combines subject shift with unequal training-set size and training-population composition.
- Phase 6 association tests are exploratory correlations, not causal explanations.

## Reproducibility and outputs

Run from repository root:

```bash
/opt/anaconda3/bin/python -m src.models.run_within_cross_separability --root . --output-dir experiments/within_cross_separability
```

Key tables: `signal_level_separability.csv`, `trial_order_diagnostics.csv`, `feature_trial_drift.csv`, `within_subject_metrics.csv`, `cross_subject_metrics.csv`, `subject_paired_gaps.csv`, `pair_summary.csv`, `adjacent_vs_wide.csv`, and `phase6_consistency_associations.csv`.
