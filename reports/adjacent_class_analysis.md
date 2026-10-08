# Phase 5.5 Adjacent-Class Separability and Ordinal Error Analysis

Run date: 2026-10-07T05:19:59Z

## Protocol

- Reused the existing Phase 3 subject-independent fold assignments from `experiments/baselines/fold_assignments.csv`.
- Reused the current `combined_unique` physiological feature set and did not modify the Phase 4/5 pipeline.
- Binary benchmarks compare PA1 vs PA2, PA2 vs PA3, PA1 vs PA3, and BL1 vs PA4.
- Models: fold-local Logistic Regression and XGBoost; preprocessing remains inside sklearn pipelines fit only on training subjects.
- Targeted interaction analysis is limited to Phase 5-supported EDA and PA1 EMG features; no unrestricted feature mining or permutation importance is run.

## Existing Multiclass OOF Ordinal Error

| model | accuracy | mean_absolute_class_error | within_one_class_proportion | severe_error_rate_abs_ge_2 | exact_error_rate |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.3344 | 1.1757 | 0.6659 | 0.3341 | 0.6656 |
| xgboost | 0.3324 | 1.1476 | 0.6778 | 0.3222 | 0.6676 |

### XGBoost PA2 Error Destinations

| predicted_as | count | error_proportion |
| --- | --- | --- |
| BL1 | 585 | 0.3703 |
| PA1 | 341 | 0.2158 |
| PA4 | 333 | 0.2108 |
| PA3 | 321 | 0.2032 |

## All-Modality XGBoost Pair Separability

| task | overall_roc_auc | fold_roc_auc_mean | fold_roc_auc_std | overall_balanced_accuracy | overall_f1 | overall_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| BL1_vs_PA4 | 0.9021 | 0.9019 | 0.0207 | 0.8256 | 0.8189 | 0.8256 |
| PA1_vs_PA2 | 0.5561 | 0.5561 | 0.0200 | 0.5399 | 0.5234 | 0.5399 |
| PA1_vs_PA3 | 0.6815 | 0.6838 | 0.0319 | 0.6368 | 0.6181 | 0.6368 |
| PA2_vs_PA3 | 0.5931 | 0.5959 | 0.0339 | 0.5693 | 0.5656 | 0.5693 |

## Full Binary Benchmark Table

| task | model | modality_set | overall_roc_auc | fold_roc_auc_mean | overall_balanced_accuracy | overall_f1 | overall_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BL1_vs_PA4 | logistic_regression | all_modalities | 0.8809 | 0.8810 | 0.8080 | 0.7973 | 0.8080 |
| BL1_vs_PA4 | logistic_regression | ecg_eda | 0.8880 | 0.8880 | 0.8121 | 0.8005 | 0.8121 |
| BL1_vs_PA4 | logistic_regression | ecg_emg | 0.6659 | 0.6660 | 0.6270 | 0.5900 | 0.6270 |
| BL1_vs_PA4 | logistic_regression | ecg_only | 0.6713 | 0.6726 | 0.6374 | 0.6011 | 0.6374 |
| BL1_vs_PA4 | logistic_regression | eda_emg | 0.8841 | 0.8851 | 0.8106 | 0.7967 | 0.8106 |
| BL1_vs_PA4 | logistic_regression | eda_only | 0.8910 | 0.8916 | 0.8184 | 0.8034 | 0.8184 |
| BL1_vs_PA4 | logistic_regression | emg_only | 0.5761 | 0.5765 | 0.5563 | 0.4628 | 0.5563 |
| BL1_vs_PA4 | xgboost | all_modalities | 0.9021 | 0.9019 | 0.8256 | 0.8189 | 0.8256 |
| BL1_vs_PA4 | xgboost | ecg_eda | 0.9002 | 0.8993 | 0.8253 | 0.8186 | 0.8253 |
| BL1_vs_PA4 | xgboost | ecg_emg | 0.6625 | 0.6643 | 0.6230 | 0.5845 | 0.6230 |
| BL1_vs_PA4 | xgboost | ecg_only | 0.6607 | 0.6628 | 0.6236 | 0.5854 | 0.6236 |
| BL1_vs_PA4 | xgboost | eda_emg | 0.8966 | 0.8984 | 0.8158 | 0.8100 | 0.8158 |
| BL1_vs_PA4 | xgboost | eda_only | 0.8965 | 0.8983 | 0.8241 | 0.8187 | 0.8241 |
| BL1_vs_PA4 | xgboost | emg_only | 0.5687 | 0.5693 | 0.5443 | 0.5150 | 0.5443 |
| PA1_vs_PA2 | logistic_regression | all_modalities | 0.5612 | 0.5618 | 0.5468 | 0.5046 | 0.5468 |
| PA1_vs_PA2 | logistic_regression | ecg_eda | 0.5618 | 0.5634 | 0.5532 | 0.4963 | 0.5532 |
| PA1_vs_PA2 | logistic_regression | ecg_emg | 0.5074 | 0.5076 | 0.5121 | 0.4876 | 0.5121 |
| PA1_vs_PA2 | logistic_regression | ecg_only | 0.5045 | 0.5045 | 0.5083 | 0.4708 | 0.5083 |
| PA1_vs_PA2 | logistic_regression | eda_emg | 0.5693 | 0.5700 | 0.5489 | 0.4990 | 0.5489 |
| PA1_vs_PA2 | logistic_regression | eda_only | 0.5729 | 0.5739 | 0.5526 | 0.4770 | 0.5526 |
| PA1_vs_PA2 | logistic_regression | emg_only | 0.5026 | 0.5032 | 0.4997 | 0.4661 | 0.4997 |
| PA1_vs_PA2 | xgboost | all_modalities | 0.5561 | 0.5561 | 0.5399 | 0.5234 | 0.5399 |
| PA1_vs_PA2 | xgboost | ecg_eda | 0.5487 | 0.5485 | 0.5282 | 0.5039 | 0.5282 |
| PA1_vs_PA2 | xgboost | ecg_emg | 0.5115 | 0.5118 | 0.5089 | 0.4939 | 0.5089 |
| PA1_vs_PA2 | xgboost | ecg_only | 0.5174 | 0.5173 | 0.5118 | 0.4948 | 0.5118 |
| PA1_vs_PA2 | xgboost | eda_emg | 0.5532 | 0.5533 | 0.5356 | 0.5308 | 0.5356 |
| PA1_vs_PA2 | xgboost | eda_only | 0.5474 | 0.5490 | 0.5299 | 0.5143 | 0.5299 |
| PA1_vs_PA2 | xgboost | emg_only | 0.5207 | 0.5210 | 0.5101 | 0.5124 | 0.5101 |
| PA1_vs_PA3 | logistic_regression | all_modalities | 0.6774 | 0.6798 | 0.6336 | 0.6002 | 0.6336 |
| PA1_vs_PA3 | logistic_regression | ecg_eda | 0.6815 | 0.6852 | 0.6405 | 0.6030 | 0.6405 |
| PA1_vs_PA3 | logistic_regression | ecg_emg | 0.5722 | 0.5754 | 0.5586 | 0.5127 | 0.5586 |
| PA1_vs_PA3 | logistic_regression | ecg_only | 0.5831 | 0.5867 | 0.5704 | 0.5267 | 0.5704 |
| PA1_vs_PA3 | logistic_regression | eda_emg | 0.6790 | 0.6812 | 0.6336 | 0.5954 | 0.6336 |
| PA1_vs_PA3 | logistic_regression | eda_only | 0.6848 | 0.6869 | 0.6348 | 0.5925 | 0.6348 |
| PA1_vs_PA3 | logistic_regression | emg_only | 0.5335 | 0.5382 | 0.5178 | 0.4238 | 0.5178 |
| PA1_vs_PA3 | xgboost | all_modalities | 0.6815 | 0.6838 | 0.6368 | 0.6181 | 0.6368 |
| PA1_vs_PA3 | xgboost | ecg_eda | 0.6796 | 0.6822 | 0.6356 | 0.6210 | 0.6356 |
| PA1_vs_PA3 | xgboost | ecg_emg | 0.5598 | 0.5618 | 0.5391 | 0.5052 | 0.5391 |
| PA1_vs_PA3 | xgboost | ecg_only | 0.5621 | 0.5638 | 0.5480 | 0.5095 | 0.5480 |
| PA1_vs_PA3 | xgboost | eda_emg | 0.6882 | 0.6895 | 0.6402 | 0.6280 | 0.6402 |
| PA1_vs_PA3 | xgboost | eda_only | 0.6909 | 0.6930 | 0.6382 | 0.6193 | 0.6382 |
| PA1_vs_PA3 | xgboost | emg_only | 0.5374 | 0.5385 | 0.5207 | 0.5114 | 0.5207 |
| PA2_vs_PA3 | logistic_regression | all_modalities | 0.5940 | 0.5977 | 0.5672 | 0.5458 | 0.5672 |
| PA2_vs_PA3 | logistic_regression | ecg_eda | 0.5956 | 0.6006 | 0.5707 | 0.5420 | 0.5707 |
| PA2_vs_PA3 | logistic_regression | ecg_emg | 0.5498 | 0.5522 | 0.5388 | 0.4995 | 0.5388 |
| PA2_vs_PA3 | logistic_regression | ecg_only | 0.5534 | 0.5578 | 0.5477 | 0.5016 | 0.5477 |
| PA2_vs_PA3 | logistic_regression | eda_emg | 0.5972 | 0.6002 | 0.5724 | 0.5450 | 0.5724 |
| PA2_vs_PA3 | logistic_regression | eda_only | 0.6001 | 0.6042 | 0.5724 | 0.5353 | 0.5724 |
| PA2_vs_PA3 | logistic_regression | emg_only | 0.5256 | 0.5258 | 0.5103 | 0.4471 | 0.5103 |
| PA2_vs_PA3 | xgboost | all_modalities | 0.5931 | 0.5959 | 0.5693 | 0.5656 | 0.5693 |
| PA2_vs_PA3 | xgboost | ecg_eda | 0.6024 | 0.6060 | 0.5724 | 0.5657 | 0.5724 |
| PA2_vs_PA3 | xgboost | ecg_emg | 0.5249 | 0.5281 | 0.5187 | 0.5107 | 0.5187 |
| PA2_vs_PA3 | xgboost | ecg_only | 0.5417 | 0.5454 | 0.5339 | 0.5147 | 0.5339 |
| PA2_vs_PA3 | xgboost | eda_emg | 0.5856 | 0.5869 | 0.5655 | 0.5592 | 0.5655 |
| PA2_vs_PA3 | xgboost | eda_only | 0.5918 | 0.5940 | 0.5759 | 0.5694 | 0.5759 |
| PA2_vs_PA3 | xgboost | emg_only | 0.5096 | 0.5104 | 0.5069 | 0.5060 | 0.5069 |

## Targeted EDA x EMG Interaction Check

Targeted products: eda_scl_delta__x__emg_trapezius__tsfresh_linear_trend_slope, eda_scl_delta__x__emg_trapezius__tsfresh_kurtosis, eda_scl_slope_per_s__x__emg_trapezius__tsfresh_linear_trend_slope, eda_scl_slope_per_s__x__emg_trapezius__tsfresh_kurtosis, eda__tsfresh_kurtosis__x__emg_trapezius__tsfresh_linear_trend_slope, eda__tsfresh_kurtosis__x__emg_trapezius__tsfresh_kurtosis, eda_scl_min__x__emg_trapezius__tsfresh_linear_trend_slope, eda_scl_min__x__emg_trapezius__tsfresh_kurtosis

| task | model | mean_delta_roc_auc | sd_delta_roc_auc | mean_delta_balanced_accuracy | mean_delta_f1 | positive_auc_fold_prop |
| --- | --- | --- | --- | --- | --- | --- |
| PA1_vs_PA2 | logistic_regression | -0.0002 | 0.0063 | -0.0009 | -0.0012 | 0.6000 |
| PA1_vs_PA2 | xgboost | -0.0028 | 0.0136 | 0.0066 | 0.0114 | 0.2000 |
| PA1_vs_PA3 | logistic_regression | -0.0028 | 0.0016 | 0.0035 | 0.0045 | 0.0000 |
| PA1_vs_PA3 | xgboost | 0.0025 | 0.0051 | 0.0028 | 0.0049 | 0.6000 |
| PA2_vs_PA3 | logistic_regression | -0.0003 | 0.0050 | 0.0005 | 0.0020 | 0.2000 |
| PA2_vs_PA3 | xgboost | 0.0040 | 0.0080 | 0.0004 | -0.0010 | 0.8000 |

## Focused PDP/ICE Figures

- `experiments/adjacent_class_analysis/figures/pdp_ice_PA1_vs_PA2_eda_scl_delta.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA1_vs_PA2_eda_scl_slope_per_s.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA1_vs_PA2_emg_trapezius__tsfresh_linear_trend_slope.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA2_vs_PA3_eda_scl_delta.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA2_vs_PA3_eda_scl_slope_per_s.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA2_vs_PA3_emg_trapezius__tsfresh_linear_trend_slope.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA1_vs_PA3_eda_scl_delta.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA1_vs_PA3_eda_scl_slope_per_s.png`
- `experiments/adjacent_class_analysis/figures/pdp_ice_PA1_vs_PA3_emg_trapezius__tsfresh_linear_trend_slope.png`

## Direct Answers

- Are PA1 vs PA2 and PA2 vs PA3 genuinely less separable than wider pain-level pairs? Yes in the primary all-modality XGBoost benchmark: adjacent mean ROC-AUC is 0.575 versus 0.792 for the wider controls, and adjacent mean balanced accuracy is 0.555 versus 0.731. Logistic Regression shows the same direction with all modalities: adjacent mean ROC-AUC 0.578 versus controls 0.779.
- Does EDA preserve ordinal pain information even when exact PA2 classification is weak? Yes, with qualification: EDA-only XGBoost retains above-chance pairwise ROC-AUC, averaging 0.570 for adjacent pairs and 0.794 for wider controls, while the multiclass OOF errors are mostly within one pain level.
- Do EDA+EMG combinations improve adjacent-class discrimination? The effect is modest and model-dependent in this targeted benchmark; for primary XGBoost, EDA+EMG changes adjacent ROC-AUC by -0.000 on average versus EDA alone.
- Is PA2 difficulty more consistent with class overlap than with a few harmful individual features? Yes. The OOF multiclass pattern has within-±1 proportion 0.678 and severe-error rate 0.322; binary adjacent separability is lower than non-adjacent controls, while targeted EDA x EMG interactions do not produce a large rescue. That points more toward adjacent decision-boundary overlap than a small set of harmful features.

## Artifact Index

- `experiments/adjacent_class_analysis/multiclass_confusion_matrix.csv`
- `experiments/adjacent_class_analysis/multiclass_error_destinations.csv`
- `experiments/adjacent_class_analysis/multiclass_ordinal_error_summary.csv`
- `experiments/adjacent_class_analysis/binary_pair_fold_metrics.csv`
- `experiments/adjacent_class_analysis/binary_pair_summary_metrics.csv`
- `experiments/adjacent_class_analysis/binary_pair_oof_predictions.csv`
- `experiments/adjacent_class_analysis/targeted_interaction_fold_metrics.csv`
- `experiments/adjacent_class_analysis/targeted_interaction_summary.csv`
