# Phase 6 Cross-Subject Stability and Personalization Analysis

Run date: 2026-10-09T03:36:27.409168+00:00

## Reused fixed inputs

- Phase 2 feature table: `artifacts/feature_table_filtered.csv`.
- Folds: `experiments/baselines/fold_assignments.csv`; all Phase 6 predictive analyses reuse these subject-independent folds.
- Phase 4 modality context: `experiments/modality_contribution/summary_metrics.csv` and `modality_pain_level_delta.csv`.
- Phase 5 feature context: `experiments/feature_contribution/*` summaries.
- Phase 5.5 adjacent separability context: `experiments/adjacent_class_analysis/binary_pair_summary_metrics.csv`.

## Selected features

The analysis uses five representative features: `eda_scl_delta`, `eda_scl_slope_per_s`, `eda_scr_amplitude_sum`, `eda_scr_count`, and `emg_trapezius__tsfresh_linear_trend_slope`. The requested SCR amplitude-sum concept is named `eda_scr_amplitude_sum` in the local feature table.

## Validation checks

- Subjects in feature table: 87.
- Samples in feature table: 8700.
- Expected subjects accounted for in trajectory summaries: 87.
- Reused folds: 5; subject overlap checks passed for normalization and binary calibration runs.
- Baseline normalization used only each subject's BL1 trials to compute subtraction offsets. PA1-PA4 labels from the held-out subject were not used for normalization.
- Calibration and evaluation trials were disjoint within each held-out subject for `k > 0`.

## Connection to previous phases

Phase 4 showed that the full XGBoost model had macro F1 0.305, while removing EDA reduced macro F1 to 0.234. EDA therefore remained the clearest modality-level contributor in the fixed Phase 4 artifacts.
At the pain-level ablation scale, EDA removal most strongly reduced BL1 and PA4 recall, but PA2 remained weak: the full XGBoost PA2 recall was 0.092.
Phase 5 identified EDA SCL features as globally important, especially `eda_scl_delta`. Phase 6 asks whether those globally important features move in consistent directions across subjects and pain boundaries.

## A. Subject-Level Physiological Trajectories

| feature | median rho | IQR rho | positive rho | negative rho | clearly non-monotonic |
|---|---:|---:|---:|---:|---:|
| eda_scl_delta | 0.900 | 0.300 | 0.989 | 0.011 | 0.667 |
| eda_scl_slope_per_s | 0.900 | 0.300 | 0.989 | 0.011 | 0.667 |
| eda_scr_amplitude_sum | 0.900 | 0.300 | 0.931 | 0.011 | 0.598 |
| eda_scr_count | 0.872 | 0.308 | 0.920 | 0.023 | 0.563 |
| emg_trapezius__tsfresh_linear_trend_slope | -0.100 | 0.850 | 0.448 | 0.506 | 0.977 |

Global predictive importance did not imply universal physiological monotonicity. Several EDA features had positive median subject-level associations with pain, but a substantial minority of subjects showed negative or sign-reversing trajectories.

## B. Boundary-Level Direction Consistency

The median directional consistency across adjacent boundaries was 0.695, compared with 0.862 for wider gaps. This supports the expected pattern that wider pain-level gaps have more stable physiological direction than adjacent boundaries, although the effect varies by feature.

| feature | boundary | median diff | IQR | positive | negative | direction consistency |
|---|---|---:|---:|---:|---:|---:|
| eda_scl_delta | BL1_to_PA1 | 0.01846 | 0.06708 | 0.793 | 0.207 | 0.793 |
| eda_scl_delta | PA1_to_PA2 | 0.009964 | 0.06226 | 0.632 | 0.368 | 0.632 |
| eda_scl_delta | PA2_to_PA3 | 0.03045 | 0.1238 | 0.839 | 0.161 | 0.839 |
| eda_scl_delta | PA3_to_PA4 | 0.08521 | 0.2034 | 0.908 | 0.092 | 0.908 |
| eda_scl_delta | PA1_to_PA3 | 0.04962 | 0.1737 | 0.874 | 0.126 | 0.874 |
| eda_scl_delta | BL1_to_PA4 | 0.2199 | 0.4452 | 0.989 | 0.011 | 0.989 |
| eda_scl_slope_per_s | BL1_to_PA1 | 0.003834 | 0.01283 | 0.805 | 0.195 | 0.805 |
| eda_scl_slope_per_s | PA1_to_PA2 | 0.001924 | 0.01229 | 0.644 | 0.356 | 0.644 |
| eda_scl_slope_per_s | PA2_to_PA3 | 0.005272 | 0.02164 | 0.828 | 0.172 | 0.828 |
| eda_scl_slope_per_s | PA3_to_PA4 | 0.01459 | 0.03943 | 0.874 | 0.126 | 0.874 |
| eda_scl_slope_per_s | PA1_to_PA3 | 0.006627 | 0.03233 | 0.851 | 0.149 | 0.851 |
| eda_scl_slope_per_s | BL1_to_PA4 | 0.03488 | 0.08039 | 0.977 | 0.023 | 0.977 |
| eda_scr_amplitude_sum | BL1_to_PA1 | 0.00575 | 0.02659 | 0.713 | 0.184 | 0.713 |
| eda_scr_amplitude_sum | PA1_to_PA2 | 0.006479 | 0.02886 | 0.609 | 0.299 | 0.609 |
| eda_scr_amplitude_sum | PA2_to_PA3 | 0.008101 | 0.04194 | 0.770 | 0.149 | 0.770 |
| eda_scr_amplitude_sum | PA3_to_PA4 | 0.0328 | 0.07065 | 0.839 | 0.115 | 0.839 |
| eda_scr_amplitude_sum | PA1_to_PA3 | 0.01377 | 0.06304 | 0.793 | 0.149 | 0.793 |
| eda_scr_amplitude_sum | BL1_to_PA4 | 0.06296 | 0.1479 | 0.931 | 0.023 | 0.931 |
| eda_scr_count | BL1_to_PA1 | 0.1 | 0.25 | 0.690 | 0.149 | 0.690 |
| eda_scr_count | PA1_to_PA2 | 0.1 | 0.2 | 0.586 | 0.230 | 0.586 |
| eda_scr_count | PA2_to_PA3 | 0.1 | 0.275 | 0.701 | 0.149 | 0.701 |
| eda_scr_count | PA3_to_PA4 | 0.1 | 0.35 | 0.644 | 0.184 | 0.644 |
| eda_scr_count | PA1_to_PA3 | 0.2 | 0.4 | 0.828 | 0.080 | 0.828 |
| eda_scr_count | BL1_to_PA4 | 0.65 | 0.55 | 0.908 | 0.023 | 0.908 |
| emg_trapezius__tsfresh_linear_trend_slope | BL1_to_PA1 | -8.248e-07 | 9.736e-06 | 0.448 | 0.552 | 0.552 |
| emg_trapezius__tsfresh_linear_trend_slope | PA1_to_PA2 | 1.781e-06 | 1.375e-05 | 0.655 | 0.345 | 0.655 |
| emg_trapezius__tsfresh_linear_trend_slope | PA2_to_PA3 | -6.397e-07 | 1.488e-05 | 0.494 | 0.506 | 0.506 |
| emg_trapezius__tsfresh_linear_trend_slope | PA3_to_PA4 | 1.09e-07 | 1.147e-05 | 0.506 | 0.494 | 0.506 |
| emg_trapezius__tsfresh_linear_trend_slope | PA1_to_PA3 | 1.542e-06 | 1.115e-05 | 0.598 | 0.402 | 0.598 |
| emg_trapezius__tsfresh_linear_trend_slope | BL1_to_PA4 | -6.517e-07 | 1.664e-05 | 0.448 | 0.552 | 0.552 |

Phase 5.5 all-modality logistic ROC-AUC values for the same major pairs were:
- `BL1_vs_PA4`: ROC-AUC 0.881.
- `PA1_vs_PA2`: ROC-AUC 0.561.
- `PA1_vs_PA3`: ROC-AUC 0.677.
- `PA2_vs_PA3`: ROC-AUC 0.594.

The physiological direction-consistency results line up with Phase 5.5: adjacent pairs remain close to chance, while BL1 vs PA4 is much more separable.

## C. PA2 Heterogeneity

For `eda_scl_delta`, PA2 between-subject variance was 0.04168, with a PA2 variance ratio of 0.810 relative to the mean of PA1 and PA3 variance.
PA2 lay physiologically between PA1 and PA3 for 0.494 of subjects, while `PA2 < PA1` occurred for 0.368 and `PA2 > PA3` occurred for 0.161.
This points to heterogeneous subject-specific PA2 responses rather than a single universally harmful feature. It does not establish a physiological cause.

## D. Subject Baseline Normalization

Five-class macro F1 changed from 0.304 with raw features to 0.296 after BL1 subtraction. Balanced accuracy changed from 0.334 to 0.327.

| pair | raw ROC-AUC | baseline-subtracted ROC-AUC | delta |
|---|---:|---:|---:|
| BL1_vs_PA4 | 0.881 | 0.863 | -0.018 |
| PA1_vs_PA2 | 0.561 | 0.557 | -0.004 |
| PA1_vs_PA3 | 0.677 | 0.672 | -0.005 |
| PA2_vs_PA3 | 0.594 | 0.592 | -0.002 |

The normalization result is the direct test of baseline heterogeneity. Here, BL1 subtraction did not materially rescue the adjacent PA1/PA2 or PA2/PA3 boundaries, so absolute baseline offsets alone are not sufficient to explain the problem.

## E. Lightweight Calibration Pilot

| pair | k per class | ROC-AUC mean | balanced accuracy mean | F1 mean | eval rows mean |
|---|---:|---:|---:|---:|---:|
| BL1_vs_PA4 | 0 | 0.881 | 0.808 | 0.797 | 3480.0 |
| BL1_vs_PA4 | 1 | 0.862 | 0.771 | 0.767 | 3306.0 |
| BL1_vs_PA4 | 2 | 0.886 | 0.795 | 0.790 | 3132.0 |
| PA1_vs_PA2 | 0 | 0.561 | 0.547 | 0.505 | 3480.0 |
| PA1_vs_PA2 | 1 | 0.527 | 0.521 | 0.519 | 3306.0 |
| PA1_vs_PA2 | 2 | 0.538 | 0.527 | 0.526 | 3132.0 |
| PA1_vs_PA3 | 0 | 0.677 | 0.634 | 0.600 | 3480.0 |
| PA1_vs_PA3 | 1 | 0.632 | 0.591 | 0.587 | 3306.0 |
| PA1_vs_PA3 | 2 | 0.658 | 0.609 | 0.603 | 3132.0 |
| PA2_vs_PA3 | 0 | 0.594 | 0.567 | 0.546 | 3480.0 |
| PA2_vs_PA3 | 1 | 0.563 | 0.543 | 0.533 | 3306.0 |
| PA2_vs_PA3 | 2 | 0.580 | 0.553 | 0.549 | 3132.0 |

Calibration used only a 1D logistic mapping from the global model's held-out subject scores to labels, with calibration trials excluded from evaluation. This is a score-level pilot, not a subject-specific model retrain. In this run, score calibration did not materially improve the adjacent PA1/PA2 or PA2/PA3 ROC-AUCs; the small calibration sets mostly reduced usable evaluation data and sometimes degraded discrimination.

## F. Global Importance vs Stability

| feature | Phase 5 class | XGB abs SHAP | XGB rank | rho median | adjacent consistency | non-monotonic |
|---|---|---:|---:|---:|---:|---:|
| eda_scl_delta | BL1 | 0.6669 | 1 | 0.900 | 0.793 | 0.667 |
| eda_scl_slope_per_s | BL1 | 0.0805 | 3 | 0.900 | 0.787 | 0.667 |
| eda_scr_amplitude_sum | nan | nan | nan | 0.900 | 0.733 | 0.598 |
| eda_scr_count | nan | nan | nan | 0.872 | 0.655 | 0.563 |
| emg_trapezius__tsfresh_linear_trend_slope | PA1 | 0.0418 | 6 | -0.100 | 0.555 | 0.977 |

This table is the central Phase 6 distinction: a feature can be globally useful to the model and still be directionally inconsistent across subjects or across adjacent boundaries.

## Answers to the Phase 6 Questions

1. **Are globally important features also directionally consistent across subjects?** Partly. `eda_scl_delta` is globally important and has a positive median trajectory, but consistency is not universal; subject-level sign reversals remain common enough to limit a universal monotonic interpretation.
2. **Does physiological consistency decrease at adjacent pain boundaries?** Yes in the aggregate. Adjacent boundaries show lower median direction consistency than wider gaps, matching the weak adjacent-pair ROC-AUC pattern from Phase 5.5.
3. **Is PA2 unusually heterogeneous?** PA2 shows substantial subject-specific non-ordinal behavior for the selected EDA features. The evidence supports PA2 heterogeneity as a plausible contributor to weak PA2 classification, not as a proven cause.
4. **How much of adjacent-class difficulty can be explained by baseline heterogeneity?** Only a limited amount in this pilot. BL1 subtraction did not materially improve the most difficult adjacent ROC-AUCs.
5. **Does minimal subject calibration recover information that a fully global model misses?** Not materially in this pilot. `k=1` and `k=2` score calibration did not rescue PA1/PA2 or PA2/PA3 ROC-AUC, so the current evidence does not show that a tiny labelled calibration set can recover a hidden adjacent-boundary signal.
6. **Main bottleneck?** Baseline shift is not the dominant explanation. The evidence points to visible response-pattern heterogeneity plus intrinsic adjacent-class overlap in the available physiological features; because lightweight calibration also failed to help, the adjacent-boundary bottleneck appears stronger than a simple subject-specific score-threshold problem.

## Restrained Research Conclusion

Phase 6 supports the hypothesis that fine-grained physiological pain-intensity recognition is constrained by cross-subject heterogeneity, especially at adjacent pain boundaries. The evidence does not justify claiming that fully personalized models are required, but it does show that population-level feature importance should not be read as universal physiological consistency.

## Output Files

- `experiments/cross_subject_stability/subject_level_trajectories.csv`
- `experiments/cross_subject_stability/subject_feature_monotonicity.csv`
- `experiments/cross_subject_stability/feature_monotonicity_summary.csv`
- `experiments/cross_subject_stability/boundary_consistency.csv`
- `experiments/cross_subject_stability/pa2_heterogeneity_summary.csv`
- `experiments/cross_subject_stability/normalization_comparison_metrics.csv`
- `experiments/cross_subject_stability/calibration_comparison_metrics.csv`
- `experiments/cross_subject_stability/figures/`
