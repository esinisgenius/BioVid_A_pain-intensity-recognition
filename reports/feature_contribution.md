# Phase 5 Pain-Level-Specific Feature Attribution

Run date: 2026-09-10T10:11:47Z

## Protocol

- Main attribution model: `XGBoost + combined_unique`.
- Folds: exact Phase 3 subject-independent folds from `experiments/baselines/fold_assignments.csv`.
- Preprocessing: `SimpleImputer(strategy="median")` remains inside each fold-local sklearn Pipeline and is fit only on training subjects.
- SHAP source: XGBoost Booster native `pred_contribs=True` TreeSHAP values on held-out fold rows after fold-local imputation.
- Cross-model sensitivity: Random Forest is used for top-feature overlap; Logistic Regression is used for top-feature overlap and targeted group-removal direction.
- Interpretation: class-level contribution means predictive contribution in this 5-class setting, not physiological causality.

## Focused XGBoost SHAP Features

| cell | rank_in_cell | feature_name | provenance | mean_abs_shap | mean_signed_shap | positive_shap_prop | feature_value_shap_corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eda/BL1 | 1 | eda_scl_delta | shared | 0.6669 | 0.2637 | 0.7858 | -0.7025 |
| eda/BL1 | 2 | eda__tsfresh_skewness | generic_automated | 0.2050 | 0.0269 | 0.4864 | -0.7369 |
| eda/BL1 | 3 | eda_scl_slope_per_s | shared | 0.0805 | 0.0462 | 0.8328 | -0.7451 |
| eda/PA2 | 1 | eda_scl_delta | shared | 0.0661 | 0.0017 | 0.7699 | -0.5879 |
| eda/PA2 | 2 | eda__tsfresh_kurtosis | generic_automated | 0.0314 | -0.0011 | 0.5625 | 0.0265 |
| eda/PA2 | 3 | eda_scl_min | shared | 0.0306 | -0.0006 | 0.3541 | 0.8210 |
| eda/PA3 | 1 | eda_scl_delta | shared | 0.2804 | 0.0536 | 0.7028 | 0.4141 |
| eda/PA3 | 2 | eda__tsfresh_skewness | generic_automated | 0.1124 | 0.0090 | 0.6908 | 0.4780 |
| eda/PA3 | 3 | eda__tsfresh_kurtosis | generic_automated | 0.0411 | 0.0010 | 0.3478 | 0.0116 |
| eda/PA4 | 1 | eda_scl_delta | shared | 0.6106 | 0.3429 | 0.7945 | 0.6832 |
| eda/PA4 | 2 | eda__tsfresh_skewness | generic_automated | 0.1889 | 0.0491 | 0.6041 | 0.8088 |
| eda/PA4 | 3 | eda_scl_slope_per_s | shared | 0.1150 | 0.0639 | 0.7362 | 0.7947 |
| emg/PA1 | 1 | emg_trapezius__tsfresh_linear_trend_slope | generic_automated | 0.0418 | -0.0007 | 0.3866 | -0.2130 |
| emg/PA1 | 2 | emg_trapezius__tsfresh_kurtosis | generic_automated | 0.0338 | 0.0020 | 0.4087 | 0.1634 |
| emg/PA1 | 3 | emg_trapezius__tsfresh_maximum | generic_automated | 0.0311 | -0.0031 | 0.7046 | -0.3324 |
| ecg/BL1 | 1 | ecg_rate_slope_bpm_per_s | domain_informed | 0.0550 | 0.0043 | 0.5442 | -0.7807 |
| ecg/BL1 | 2 | ecg__tsfresh_minimum | generic_automated | 0.0252 | 0.0012 | 0.6916 | -0.2101 |
| ecg/BL1 | 3 | ecg_quality_mean | domain_informed | 0.0232 | 0.0004 | 0.6907 | 0.5777 |

## Targeted Feature-Removal Validation

| modality | pain_class | removed_features | mean_delta_f1 | sd_delta_f1 | mean_delta_recall | positive_f1_fold_prop |
| --- | --- | --- | --- | --- | --- | --- |
| ecg | BL1 | ecg_rate_slope_bpm_per_s\|ecg__tsfresh_minimum\|ecg_quality_mean | -0.0001 | 0.0120 | 0.0086 | 0.4000 |
| eda | BL1 | eda_scl_delta\|eda__tsfresh_skewness\|eda_scl_slope_per_s | 0.0418 | 0.0126 | 0.0594 | 1.0000 |
| eda | PA2 | eda_scl_delta\|eda__tsfresh_kurtosis\|eda_scl_min | -0.0036 | 0.0256 | -0.0032 | 0.2000 |
| eda | PA3 | eda_scl_delta\|eda__tsfresh_skewness\|eda__tsfresh_kurtosis | 0.0334 | 0.0114 | 0.0357 | 1.0000 |
| eda | PA4 | eda_scl_delta\|eda__tsfresh_skewness\|eda_scl_slope_per_s | 0.0126 | 0.0145 | 0.0270 | 0.8000 |
| emg | PA1 | emg_trapezius__tsfresh_linear_trend_slope\|emg_trapezius__tsfresh_kurtosis\|emg_trapezius__tsfresh_maximum | 0.0118 | 0.0047 | 0.0125 | 1.0000 |

## Cross-Model Sensitivity

| modality | pain_class | feature_name | xgboost_fold_top10_frequency | random_forest_top10 | logistic_regression_top10 | xgboost_top_feature_group_delta_f1 | logistic_regression_top_feature_group_delta_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eda | BL1 | eda_scl_delta | 1.0000 | True | True | 0.0418 | 0.0464 |
| eda | BL1 | eda__tsfresh_skewness | 1.0000 | True | True | 0.0418 | 0.0464 |
| eda | BL1 | eda_scl_slope_per_s | 1.0000 | True | True | 0.0418 | 0.0464 |
| eda | PA2 | eda_scl_delta | 0.8000 | True | False | -0.0036 | 0.0036 |
| eda | PA2 | eda__tsfresh_kurtosis | 0.8000 | True | True | -0.0036 | 0.0036 |
| eda | PA2 | eda_scl_min | 0.4000 | False | False | -0.0036 | 0.0036 |
| eda | PA3 | eda_scl_delta | 1.0000 | True | True | 0.0334 | -0.0079 |
| eda | PA3 | eda__tsfresh_skewness | 1.0000 | True | False | 0.0334 | -0.0079 |
| eda | PA3 | eda__tsfresh_kurtosis | 0.8000 | True | True | 0.0334 | -0.0079 |
| eda | PA4 | eda_scl_delta | 1.0000 | True | True | 0.0126 | 0.0327 |
| eda | PA4 | eda__tsfresh_skewness | 1.0000 | True | True | 0.0126 | 0.0327 |
| eda | PA4 | eda_scl_slope_per_s | 1.0000 | True | False | 0.0126 | 0.0327 |
| emg | PA1 | emg_trapezius__tsfresh_linear_trend_slope | 0.6000 | False | False | 0.0118 | -0.0010 |
| emg | PA1 | emg_trapezius__tsfresh_kurtosis | 0.6000 | False | False | 0.0118 | -0.0010 |
| emg | PA1 | emg_trapezius__tsfresh_maximum | 0.6000 | False | False | 0.0118 | -0.0010 |
| ecg | BL1 | ecg_rate_slope_bpm_per_s | 1.0000 | True | False | -0.0001 | 0.0024 |
| ecg | BL1 | ecg__tsfresh_minimum | 0.2000 | False | False | -0.0001 | 0.0024 |
| ecg | BL1 | ecg_quality_mean | 0.2000 | False | False | -0.0001 | 0.0024 |

## Interpretation Summary

### EDA cells

- `BL1`: driven most by eda_scl_delta, eda__tsfresh_skewness, eda_scl_slope_per_s. Top-feature-group removal ΔF1=0.0418.
- `PA2`: driven most by eda_scl_delta, eda__tsfresh_kurtosis, eda_scl_min. Top-feature-group removal ΔF1=-0.0036.
- `PA3`: driven most by eda_scl_delta, eda__tsfresh_skewness, eda__tsfresh_kurtosis. Top-feature-group removal ΔF1=0.0334.
- `PA4`: driven most by eda_scl_delta, eda__tsfresh_skewness, eda_scl_slope_per_s. Top-feature-group removal ΔF1=0.0126.

### PA2 EDA pattern

- PA2 shares top EDA features with endpoint or higher-pain cells: eda__tsfresh_kurtosis, eda_scl_delta.
- The negative Phase 4 EDA/PA2 performance contribution is therefore best treated as a multiclass decision-boundary trade-off or class-overlap pattern until Phase 6 tests subject-level stability.
- PA2 signed SHAP directions for the top EDA features:
| feature_name | mean_signed_shap | positive_shap_prop | feature_value_shap_corr |
| --- | --- | --- | --- |
| eda_scl_delta | 0.0017 | 0.7699 | -0.5879 |
| eda__tsfresh_kurtosis | -0.0011 | 0.5625 | 0.0265 |
| eda_scl_min | -0.0006 | 0.3541 | 0.8210 |

### EMG and ECG

- `EMG/PA1`: driven most by emg_trapezius__tsfresh_linear_trend_slope, emg_trapezius__tsfresh_kurtosis, emg_trapezius__tsfresh_maximum; top-feature-group removal ΔF1=0.0118.
- `ECG/BL1`: secondary, smaller effect; top features are ecg_rate_slope_bpm_per_s, ecg__tsfresh_minimum, ecg_quality_mean; top-feature-group removal ΔF1=-0.0001.

### Evidence Strength

- Supported by both XGBoost SHAP rank stability and XGBoost group-removal direction: eda/BL1/eda_scl_delta; eda/BL1/eda__tsfresh_skewness; eda/BL1/eda_scl_slope_per_s; eda/PA3/eda_scl_delta; eda/PA3/eda__tsfresh_skewness; eda/PA3/eda__tsfresh_kurtosis; eda/PA4/eda_scl_delta; eda/PA4/eda__tsfresh_skewness; eda/PA4/eda_scl_slope_per_s; emg/PA1/emg_trapezius__tsfresh_linear_trend_slope; emg/PA1/emg_trapezius__tsfresh_kurtosis; emg/PA1/emg_trapezius__tsfresh_maximum.
- Findings with low fold top-10 frequency or weak cross-model agreement should be carried into Phase 6 as explicit stability questions, not as final feature conclusions.

## Outputs

- `experiments/feature_contribution/class_specific_shap.csv`
- `experiments/feature_contribution/fold_level_feature_attribution.csv`
- `experiments/feature_contribution/xgboost_global_feature_importance.csv`
- `experiments/feature_contribution/targeted_feature_removal_summary.csv`
- `experiments/feature_contribution/targeted_feature_removal_fold_level.csv`
- `experiments/feature_contribution/top_feature_cross_model_consistency.csv`
- `experiments/feature_contribution/figures/feature_pain_level_shap_heatmap.png`
- `experiments/feature_contribution/figures/eda_signed_shap_by_pain_level.png`
- `experiments/feature_contribution/figures/targeted_feature_removal_delta_f1.png`

## Phase 6 Readiness

The Phase 5 outputs are sufficient to enter Phase 6 cross-subject stability. The next phase should compute subject-level attribution profiles and test whether the apparent class-specific EDA/EMG/ECG patterns are broadly shared or driven by subsets of subjects.
