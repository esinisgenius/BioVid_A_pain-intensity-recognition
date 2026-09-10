# Phase 3B Model Robustness and Binary Benchmark

Run date: 2026-09-09T05:55:26Z

Phase 3 results are reused unchanged. This add-on uses the existing subject-independent folds and the `combined_unique` feature set only.

## 5-Class Robustness

| model | macro F1 | fold F1 mean ± std | balanced accuracy | fold bal. acc. mean ± std |
|---|---:|---:|---:|---:|
| logistic_regression | 0.3035 | 0.3023 ± 0.0098 | 0.3344 | 0.3345 ± 0.0103 |
| random_forest | 0.2984 | 0.2980 ± 0.0032 | 0.3225 | 0.3226 ± 0.0104 |
| rbf_svc | 0.2759 | 0.2758 ± 0.0137 | 0.3166 | 0.3167 ± 0.0148 |

### 5-Class Fold-Level Results

| model | fold | macro F1 | balanced accuracy | test subjects |
|---|---:|---:|---:|---:|
| logistic_regression | 1 | 0.3167 | 0.3441 | 17 |
| logistic_regression | 2 | 0.2929 | 0.3300 | 17 |
| logistic_regression | 3 | 0.3073 | 0.3435 | 17 |
| logistic_regression | 4 | 0.2945 | 0.3194 | 18 |
| logistic_regression | 5 | 0.3000 | 0.3356 | 18 |
| random_forest | 1 | 0.2970 | 0.3265 | 17 |
| random_forest | 2 | 0.2987 | 0.3318 | 17 |
| random_forest | 3 | 0.2990 | 0.3159 | 17 |
| random_forest | 4 | 0.2932 | 0.3078 | 18 |
| random_forest | 5 | 0.3020 | 0.3311 | 18 |
| rbf_svc | 1 | 0.2730 | 0.3141 | 17 |
| rbf_svc | 2 | 0.2767 | 0.3235 | 17 |
| rbf_svc | 3 | 0.2910 | 0.3235 | 17 |
| rbf_svc | 4 | 0.2546 | 0.2922 | 18 |
| rbf_svc | 5 | 0.2839 | 0.3300 | 18 |

### 5-Class Per-Class Metrics

#### logistic_regression

| class | precision | recall | F1 |
|---|---:|---:|---:|
| BL1 | 0.3466 | 0.6253 | 0.4460 |
| PA1 | 0.2613 | 0.2224 | 0.2403 |
| PA2 | 0.2314 | 0.1092 | 0.1484 |
| PA3 | 0.2529 | 0.1489 | 0.1874 |
| PA4 | 0.4407 | 0.5661 | 0.4956 |

#### random_forest

| class | precision | recall | F1 |
|---|---:|---:|---:|
| BL1 | 0.3500 | 0.5713 | 0.4341 |
| PA1 | 0.2569 | 0.2178 | 0.2358 |
| PA2 | 0.1943 | 0.1057 | 0.1370 |
| PA3 | 0.2484 | 0.1782 | 0.2075 |
| PA4 | 0.4288 | 0.5397 | 0.4779 |

#### rbf_svc

| class | precision | recall | F1 |
|---|---:|---:|---:|
| BL1 | 0.3058 | 0.7034 | 0.4263 |
| PA1 | 0.2457 | 0.1310 | 0.1709 |
| PA2 | 0.2087 | 0.0989 | 0.1342 |
| PA3 | 0.2557 | 0.1299 | 0.1723 |
| PA4 | 0.4386 | 0.5195 | 0.4757 |

## 5-Class Confusion Matrices

Rows=true, columns=predicted, class order `BL1 PA1 PA2 PA3 PA4`.

### logistic_regression

```text
1088  302  139  112   99
 788  387  172  172  221
 636  353  190  241  320
 433  260  178  259  610
 194  179  142  240  985
```

### random_forest

```text
 994  365  171  127   83
 727  379  214  201  219
 599  341  184  279  337
 374  222  222  310  612
 146  168  156  331  939
```

### rbf_svc

```text
1224  193  137   76  110
 994  228  194  139  185
 845  223  172  190  310
 609  169  184  226  552
 331  115  137  253  904
```

## Binary BL1 vs PA4 Benchmark

| model | accuracy | balanced accuracy | precision | recall | F1 | AUROC | fold F1 mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8080 | 0.8080 | 0.8445 | 0.7552 | 0.7973 | 0.8809 | 0.7971 ± 0.0267 |
| random_forest | 0.8184 | 0.8184 | 0.8428 | 0.7828 | 0.8117 | 0.8959 | 0.8113 ± 0.0250 |
| rbf_svc | 0.7951 | 0.7951 | 0.8496 | 0.7172 | 0.7778 | 0.8663 | 0.7773 ± 0.0317 |

### Binary Fold-Level Results

| model | fold | accuracy | balanced accuracy | F1 | AUROC | test subjects |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 1 | 0.8132 | 0.8132 | 0.7955 | 0.8938 | 17 |
| logistic_regression | 2 | 0.8074 | 0.8074 | 0.8048 | 0.8761 | 17 |
| logistic_regression | 3 | 0.8000 | 0.8000 | 0.7939 | 0.8751 | 17 |
| logistic_regression | 4 | 0.7778 | 0.7778 | 0.7583 | 0.8529 | 18 |
| logistic_regression | 5 | 0.8417 | 0.8417 | 0.8328 | 0.9070 | 18 |
| random_forest | 1 | 0.8162 | 0.8162 | 0.8013 | 0.8934 | 17 |
| random_forest | 2 | 0.8368 | 0.8368 | 0.8360 | 0.9136 | 17 |
| random_forest | 3 | 0.8029 | 0.8029 | 0.8052 | 0.8761 | 17 |
| random_forest | 4 | 0.7944 | 0.7944 | 0.7778 | 0.8860 | 18 |
| random_forest | 5 | 0.8417 | 0.8417 | 0.8362 | 0.9141 | 18 |
| rbf_svc | 1 | 0.8000 | 0.8000 | 0.7834 | 0.8701 | 17 |
| rbf_svc | 2 | 0.8162 | 0.8162 | 0.8109 | 0.8821 | 17 |
| rbf_svc | 3 | 0.7897 | 0.7897 | 0.7741 | 0.8524 | 17 |
| rbf_svc | 4 | 0.7611 | 0.7611 | 0.7261 | 0.8356 | 18 |
| rbf_svc | 5 | 0.8097 | 0.8097 | 0.7921 | 0.8896 | 18 |

Binary confusion matrices use rows=true, columns=predicted, class order `BL1 PA4`.

### logistic_regression

```text
1498  242
 426 1314
```

### random_forest

```text
1486  254
 378 1362
```

### rbf_svc

```text
1519  221
 492 1248
```

## Interpretation

- The three 5-class models are at a similar performance level; RBF-SVC does not remove the central grading difficulty.
- PA1, PA2, and PA3 remain weaker than BL1 and PA4 across linear, tree, and kernel models.
- Subject-level performance remains heterogeneous across models, so the cross-subject stability question is still central.
- BL1 vs PA4 is substantially easier than the 5-class grading task, as expected for a secondary literature-style benchmark.
- These results support moving into Phase 4 contribution analysis, with Logistic Regression as the primary analysis model and RF/SVC as robustness checks rather than optimization targets.
