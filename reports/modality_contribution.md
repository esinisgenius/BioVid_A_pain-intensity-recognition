# Phase 4 Cross-Model Modality Contribution Analysis

Run date: 2026-09-10T06:01:15Z

Task: subject-independent 5-class classification (`BL1`, `PA1`, `PA2`, `PA3`, `PA4`) using BioVid Part A physiological features only.

## Protocol

- Main analysis model: `XGBoost + combined_unique`.
- Robustness models: Random Forest, RBF-SVM, Logistic Regression.
- Folds: exact Phase 3 subject-independent folds from `experiments/baselines/fold_assignments.csv`.
- Learned preprocessing remains inside each sklearn Pipeline and is fit only on training subjects in each fold.
- Contribution is defined as `performance(full ECG+EDA+EMG) - performance(without modality)`.
- No SHAP, feature ranking, subject clustering, demographic analysis, CNN attribution, tuning, or binary attribution is run here.

## XGBoost Overall Contribution

| modality | delta_macro_f1 | delta_balanced_accuracy | full_macro_f1 | ablated_macro_f1 |
| --- | ---: | ---: | ---: | ---: |
| ecg | -0.0012 | 0.0001 | 0.3052 | 0.3064 |
| eda | 0.0715 | 0.0937 | 0.3052 | 0.2337 |
| emg | -0.0057 | -0.0064 | 0.3052 | 0.3109 |

## Cross-Model Overall Contribution

| modality | mean_delta_macro_f1 | mean_delta_balanced_accuracy | positive_model_proportion |
| --- | ---: | ---: | ---: |
| ecg | 0.0041 | 0.0024 | 0.5000 |
| eda | 0.0614 | 0.0839 | 1.0000 |
| emg | -0.0008 | -0.0021 | 0.5000 |

## XGBoost Pain-Level ΔF1

| modality | BL1 | PA1 | PA2 | PA3 | PA4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ecg | 0.0012 | 0.0020 | -0.0128 | 0.0101 | -0.0065 |
| eda | 0.1964 | -0.0165 | -0.0470 | 0.0745 | 0.1502 |
| emg | -0.0088 | 0.0070 | -0.0131 | -0.0087 | -0.0049 |

## XGBoost Pain-Level ΔRecall

| modality | BL1 | PA1 | PA2 | PA3 | PA4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ecg | 0.0063 | 0.0052 | -0.0132 | 0.0098 | -0.0075 |
| eda | 0.3006 | -0.0557 | -0.0632 | 0.0753 | 0.2115 |
| emg | -0.0138 | 0.0075 | -0.0109 | -0.0017 | -0.0132 |

## Cross-Model Consistency

| modality | class_name | mean_delta_f1 | xgboost_delta_f1 | random_forest_delta_f1 | rbf_svc_delta_f1 | logistic_regression_delta_f1 | positive_model_proportion | positive_fold_proportion | consistency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ecg | BL1 | 0.0053 | 0.0012 | 0.0156 | 0.0072 | -0.0028 | 0.75 | 0.65 | cross-model robust positive |
| ecg | PA1 | 0.0124 | 0.0020 | 0.0155 | 0.0157 | 0.0165 | 1.00 | 0.60 | moderately consistent |
| ecg | PA2 | 0.0104 | -0.0128 | -0.0276 | 0.0449 | 0.0370 | 0.50 | 0.55 | moderately consistent |
| ecg | PA3 | -0.0022 | 0.0101 | -0.0217 | -0.0083 | 0.0110 | 0.50 | 0.60 | moderately consistent |
| ecg | PA4 | -0.0053 | -0.0065 | 0.0068 | -0.0176 | -0.0038 | 0.25 | 0.40 | moderately consistent |
| eda | BL1 | 0.1709 | 0.1964 | 0.1774 | 0.1384 | 0.1714 | 1.00 | 1.00 | cross-model robust positive |
| eda | PA1 | -0.0199 | -0.0165 | 0.0088 | -0.0812 | 0.0094 | 0.50 | 0.35 | moderately consistent |
| eda | PA2 | -0.0311 | -0.0470 | -0.0432 | -0.0215 | -0.0128 | 0.00 | 0.25 | cross-model robust negative |
| eda | PA3 | 0.0440 | 0.0745 | 0.0381 | 0.0334 | 0.0299 | 1.00 | 0.95 | cross-model robust positive |
| eda | PA4 | 0.1429 | 0.1502 | 0.1351 | 0.1455 | 0.1408 | 1.00 | 1.00 | cross-model robust positive |
| emg | BL1 | -0.0019 | -0.0088 | 0.0014 | 0.0002 | -0.0006 | 0.50 | 0.35 | moderately consistent |
| emg | PA1 | 0.0156 | 0.0070 | -0.0074 | 0.0349 | 0.0279 | 0.75 | 0.70 | cross-model robust positive |
| emg | PA2 | -0.0051 | -0.0131 | -0.0128 | -0.0168 | 0.0224 | 0.25 | 0.50 | moderately consistent |
| emg | PA3 | -0.0056 | -0.0087 | -0.0188 | 0.0007 | 0.0043 | 0.50 | 0.45 | moderately consistent |
| emg | PA4 | -0.0069 | -0.0049 | 0.0002 | -0.0154 | -0.0073 | 0.25 | 0.35 | cross-model robust negative |

## Figures

- `experiments/modality_contribution/figures/xgboost_modality_pain_delta_f1_heatmap.png`
- `experiments/modality_contribution/figures/xgboost_modality_pain_delta_recall_heatmap.png`
- `experiments/modality_contribution/figures/cross_model_contribution_consistency.png`

## Interpretation

- EDA is not the largest XGBoost ΔF1 contributor for every pain level; by level, the highest XGBoost delta is BL1=eda (largest positive, ΔF1=+0.1964), PA1=emg (largest positive, ΔF1=+0.0070), PA2=ecg (least negative, ΔF1=-0.0128), PA3=eda (largest positive, ΔF1=+0.0745), PA4=eda (largest positive, ΔF1=+0.1502).
- PA2 shows a distinct pattern: ecg ΔF1=-0.0128, emg ΔF1=-0.0131, eda ΔF1=-0.0470.
- Some XGBoost pain-level scores improve after removing a modality, most notably eda/PA2 ΔF1=-0.0470; eda/PA1 ΔF1=-0.0165; emg/PA2 ΔF1=-0.0131; ecg/PA2 ΔF1=-0.0128; emg/BL1 ΔF1=-0.0088; emg/PA3 ΔF1=-0.0087; ecg/PA4 ΔF1=-0.0065; emg/PA4 ΔF1=-0.0049.
- Cross-model robust cells: ecg/BL1: cross-model robust positive; eda/BL1: cross-model robust positive; eda/PA2: cross-model robust negative; eda/PA3: cross-model robust positive; eda/PA4: cross-model robust positive; emg/PA1: cross-model robust positive; emg/PA4: cross-model robust negative.
- Population-level modality contribution does depend on model choice: direction is usually aligned for some endpoint classes, but intermediate pain levels have small deltas and frequent sign changes.
- These outputs are sufficient to enter Phase 5 feature-level attribution, but Phase 5 should treat stable modality/pain cells as priorities and explicitly separate robust signals from model-dependent effects.

## Machine-Readable Outputs

- `experiments/modality_contribution/metrics.json`
- `experiments/modality_contribution/summary_metrics.csv`
- `experiments/modality_contribution/per_class_metrics.csv`
- `experiments/modality_contribution/modality_overall_delta.csv`
- `experiments/modality_contribution/modality_pain_level_delta.csv`
- `experiments/modality_contribution/cross_model_consistency.csv`
- `experiments/modality_contribution/fold_level_delta.csv`
- `experiments/modality_contribution/oof_predictions.csv`
