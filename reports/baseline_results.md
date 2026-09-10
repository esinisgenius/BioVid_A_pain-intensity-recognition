# Phase 3 Subject-Independent Baseline Results

Run date: 2026-09-09T05:06:50Z

Task: 5-class classification (`BL1`, `PA1`, `PA2`, `PA3`, `PA4`) using BioVid Part A physiological features only.

## Protocol

- Split protocol: `StratifiedGroupKFold`.
- Split reason: Used because every subject has all five classes with identical class counts.
- Folds: 5; random seed: 20260909.
- Grouping: `subject_id`; no subject appears in both train and test in any fold.
- Fold-local preprocessing: `SimpleImputer(strategy="median")` for both models; `StandardScaler` for Logistic Regression only. These steps are inside sklearn pipelines and are fit separately within each training fold.
- Second non-linear baseline: Random Forest, selected because `xgboost` is not installed and Random Forest is available through scikit-learn with fewer dependencies.
- Input feature table: CSV copy of the Phase 2 feature table. The local parquet file raised a pyarrow read error during Phase 3 setup, so CSV was used without changing feature values.

## Overall 5-Class Baselines

| model | feature set | features | macro F1 | fold F1 mean ± std | balanced accuracy | fold bal. acc. mean ± std |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | domain_informed | 36 | 0.2815 | 0.2804 ± 0.0107 | 0.3240 | 0.3240 ± 0.0126 |
| logistic_regression | generic_automated | 36 | 0.2978 | 0.2953 ± 0.0162 | 0.3351 | 0.3352 ± 0.0124 |
| logistic_regression | combined_unique | 59 | 0.3035 | 0.3023 ± 0.0098 | 0.3344 | 0.3345 ± 0.0103 |
| random_forest | domain_informed | 36 | 0.3014 | 0.3009 ± 0.0124 | 0.3186 | 0.3189 ± 0.0178 |
| random_forest | generic_automated | 36 | 0.3037 | 0.3030 ± 0.0064 | 0.3230 | 0.3230 ± 0.0083 |
| random_forest | combined_unique | 59 | 0.2984 | 0.2980 ± 0.0032 | 0.3225 | 0.3226 ± 0.0104 |

## Unimodal Reference Baselines

| model | feature set | features | macro F1 | fold F1 mean ± std | balanced accuracy |
|---|---|---:|---:|---:|---:|
| logistic_regression | ecg_unimodal | 25 | 0.2399 | 0.2374 ± 0.0108 | 0.2533 |
| logistic_regression | eda_gsr_unimodal | 24 | 0.2852 | 0.2847 ± 0.0045 | 0.3347 |
| logistic_regression | emg_trapezius_unimodal | 23 | 0.2136 | 0.2089 ± 0.0141 | 0.2267 |
| random_forest | ecg_unimodal | 25 | 0.2290 | 0.2281 ± 0.0105 | 0.2349 |
| random_forest | eda_gsr_unimodal | 24 | 0.2922 | 0.2916 ± 0.0087 | 0.2990 |
| random_forest | emg_trapezius_unimodal | 23 | 0.2197 | 0.2184 ± 0.0087 | 0.2209 |

## Pain-Level Performance

### logistic_regression / domain_informed

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3197 | 0.6948 | 0.4379 | 1740 |
| PA1 | 0.2446 | 0.1420 | 0.1796 | 1740 |
| PA2 | 0.2201 | 0.0805 | 0.1178 | 1740 |
| PA3 | 0.2644 | 0.1534 | 0.1942 | 1740 |
| PA4 | 0.4226 | 0.5494 | 0.4778 | 1740 |

### logistic_regression / generic_automated

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3382 | 0.6764 | 0.4510 | 1740 |
| PA1 | 0.2348 | 0.1937 | 0.2123 | 1740 |
| PA2 | 0.2393 | 0.0897 | 0.1304 | 1740 |
| PA3 | 0.2819 | 0.1511 | 0.1968 | 1740 |
| PA4 | 0.4464 | 0.5644 | 0.4985 | 1740 |

### logistic_regression / combined_unique

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3466 | 0.6253 | 0.4460 | 1740 |
| PA1 | 0.2613 | 0.2224 | 0.2403 | 1740 |
| PA2 | 0.2314 | 0.1092 | 0.1484 | 1740 |
| PA3 | 0.2529 | 0.1489 | 0.1874 | 1740 |
| PA4 | 0.4407 | 0.5661 | 0.4956 | 1740 |

### random_forest / domain_informed

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3520 | 0.5138 | 0.4178 | 1740 |
| PA1 | 0.2661 | 0.2276 | 0.2454 | 1740 |
| PA2 | 0.2091 | 0.1351 | 0.1641 | 1740 |
| PA3 | 0.2539 | 0.1943 | 0.2201 | 1740 |
| PA4 | 0.4100 | 0.5224 | 0.4594 | 1740 |

### random_forest / generic_automated

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3487 | 0.5483 | 0.4263 | 1740 |
| PA1 | 0.2363 | 0.2000 | 0.2166 | 1740 |
| PA2 | 0.2294 | 0.1408 | 0.1745 | 1740 |
| PA3 | 0.2602 | 0.1943 | 0.2224 | 1740 |
| PA4 | 0.4355 | 0.5316 | 0.4788 | 1740 |

### random_forest / combined_unique

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3500 | 0.5713 | 0.4341 | 1740 |
| PA1 | 0.2569 | 0.2178 | 0.2358 | 1740 |
| PA2 | 0.1943 | 0.1057 | 0.1370 | 1740 |
| PA3 | 0.2484 | 0.1782 | 0.2075 | 1740 |
| PA4 | 0.4288 | 0.5397 | 0.4779 | 1740 |

## Confusion Pattern Summary

| model | feature set | off-diagonal errors | adjacent error share |
|---|---|---:|---:|
| logistic_regression | domain_informed | 5881 | 0.469 |
| logistic_regression | generic_automated | 5785 | 0.493 |
| logistic_regression | combined_unique | 5791 | 0.498 |
| logistic_regression | ecg_unimodal | 6496 | 0.417 |
| logistic_regression | eda_gsr_unimodal | 5788 | 0.475 |
| logistic_regression | emg_trapezius_unimodal | 6728 | 0.397 |
| random_forest | domain_informed | 5928 | 0.504 |
| random_forest | generic_automated | 5890 | 0.519 |
| random_forest | combined_unique | 5894 | 0.524 |
| random_forest | ecg_unimodal | 6656 | 0.422 |
| random_forest | eda_gsr_unimodal | 6099 | 0.534 |
| random_forest | emg_trapezius_unimodal | 6778 | 0.412 |

## Fold Instability

| model | feature set | macro F1 min-max | macro F1 std | balanced accuracy min-max | balanced accuracy std |
|---|---|---:|---:|---:|---:|
| logistic_regression | domain_informed | 0.2660-0.2902 | 0.0107 | 0.3141-0.3389 | 0.0126 |
| logistic_regression | generic_automated | 0.2769-0.3203 | 0.0162 | 0.3178-0.3500 | 0.0124 |
| logistic_regression | combined_unique | 0.2929-0.3167 | 0.0098 | 0.3194-0.3441 | 0.0103 |
| logistic_regression | ecg_unimodal | 0.2275-0.2531 | 0.0108 | 0.2444-0.2618 | 0.0083 |
| logistic_regression | eda_gsr_unimodal | 0.2780-0.2907 | 0.0045 | 0.3239-0.3456 | 0.0090 |
| logistic_regression | emg_trapezius_unimodal | 0.1867-0.2216 | 0.0141 | 0.2089-0.2382 | 0.0144 |
| random_forest | domain_informed | 0.2820-0.3164 | 0.0124 | 0.2894-0.3365 | 0.0178 |
| random_forest | generic_automated | 0.2965-0.3124 | 0.0064 | 0.3135-0.3322 | 0.0083 |
| random_forest | combined_unique | 0.2932-0.3020 | 0.0032 | 0.3078-0.3318 | 0.0104 |
| random_forest | ecg_unimodal | 0.2171-0.2400 | 0.0105 | 0.2217-0.2494 | 0.0118 |
| random_forest | eda_gsr_unimodal | 0.2852-0.3068 | 0.0087 | 0.2883-0.3147 | 0.0101 |
| random_forest | emg_trapezius_unimodal | 0.2098-0.2312 | 0.0087 | 0.2128-0.2365 | 0.0095 |

## Best Observed Baseline Detail

Best macro F1 in this Phase 3 grid: `random_forest` with `generic_automated`.

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3487 | 0.5483 | 0.4263 | 1740 |
| PA1 | 0.2363 | 0.2000 | 0.2166 | 1740 |
| PA2 | 0.2294 | 0.1408 | 0.1745 | 1740 |
| PA3 | 0.2602 | 0.1943 | 0.2224 | 1740 |
| PA4 | 0.4355 | 0.5316 | 0.4788 | 1740 |

Confusion matrix rows=true, columns=predicted, class order `BL1 PA1 PA2 PA3 PA4`:

```text
 954  360  194  139   93
 733  348  233  213  213
 570  324  245  278  323
 339  268  225  338  570
 140  173  171  331  925
```

Adjacent-pain-level share among off-diagonal errors for the best observed baseline: 0.519.

## Leakage Review

- No subject overlap was detected in any fold.
- No metadata, label, sample id, subject id/name, trial number, source path, temperature, corrugator EMG, or zygomaticus EMG feature was included.
- Learned preprocessing was nested inside sklearn pipelines and fit only on training rows for each fold.
- New operational risk: `artifacts/feature_table_filtered.parquet` could not be read by the current pyarrow environment; Phase 3 used `artifacts/feature_table_filtered.csv`, whose Phase 2 SHA-256 is tracked.

Machine-readable outputs are in `experiments/baselines/`: fold assignments, out-of-fold predictions, metrics JSON, feature-set manifest, per-class metrics, fold metrics, confusion matrices, and subject-level out-of-fold accuracy.
