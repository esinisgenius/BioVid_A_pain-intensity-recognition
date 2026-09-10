# Model Strengthening Benchmark

Run date: 2026-09-10T05:19:44Z

This benchmark strengthens the methods layer only. It reuses the existing subject-independent folds and does not run feature attribution, feature ranking, modality ablation, clustering, or tuning.

## 5-Class Results

| input | model | macro F1 | fold F1 mean ± std | balanced accuracy | fold bal. acc. mean ± std |
|---|---|---:|---:|---:|---:|
| engineered_features | logistic_regression | 0.3035 | 0.3023 ± 0.0098 | 0.3344 | 0.3345 ± 0.0103 |
| engineered_features | random_forest | 0.2984 | 0.2980 ± 0.0032 | 0.3225 | 0.3226 ± 0.0104 |
| engineered_features | rbf_svc | 0.2759 | 0.2758 ± 0.0137 | 0.3166 | 0.3167 ± 0.0148 |
| engineered_features | xgboost | 0.3052 | 0.3047 ± 0.0093 | 0.3324 | 0.3323 ± 0.0108 |
| filtered_raw_signal | tiny_1d_cnn | 0.2199 | 0.2076 ± 0.0156 | 0.2386 | 0.2388 ± 0.0146 |

## 5-Class Per-Class F1

| input | model | BL1 | PA1 | PA2 | PA3 | PA4 |
|---|---|---:|---:|---:|---:|---:|
| engineered_features | logistic_regression | 0.4460 | 0.2403 | 0.1484 | 0.1874 | 0.4956 |
| engineered_features | random_forest | 0.4341 | 0.2358 | 0.1370 | 0.2075 | 0.4779 |
| engineered_features | rbf_svc | 0.4263 | 0.1709 | 0.1342 | 0.1723 | 0.4757 |
| engineered_features | xgboost | 0.4453 | 0.2283 | 0.1277 | 0.2310 | 0.4939 |
| filtered_raw_signal | tiny_1d_cnn | 0.2879 | 0.2112 | 0.1347 | 0.1263 | 0.3394 |

## 5-Class Per-Class Precision / Recall / F1

### engineered_features / logistic_regression

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3466 | 0.6253 | 0.4460 | 1740 |
| PA1 | 0.2613 | 0.2224 | 0.2403 | 1740 |
| PA2 | 0.2314 | 0.1092 | 0.1484 | 1740 |
| PA3 | 0.2529 | 0.1489 | 0.1874 | 1740 |
| PA4 | 0.4407 | 0.5661 | 0.4956 | 1740 |

### engineered_features / random_forest

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3500 | 0.5713 | 0.4341 | 1740 |
| PA1 | 0.2569 | 0.2178 | 0.2358 | 1740 |
| PA2 | 0.1943 | 0.1057 | 0.1370 | 1740 |
| PA3 | 0.2484 | 0.1782 | 0.2075 | 1740 |
| PA4 | 0.4288 | 0.5397 | 0.4779 | 1740 |

### engineered_features / rbf_svc

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3058 | 0.7034 | 0.4263 | 1740 |
| PA1 | 0.2457 | 0.1310 | 0.1709 | 1740 |
| PA2 | 0.2087 | 0.0989 | 0.1342 | 1740 |
| PA3 | 0.2557 | 0.1299 | 0.1723 | 1740 |
| PA4 | 0.4386 | 0.5195 | 0.4757 | 1740 |

### engineered_features / xgboost

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.3583 | 0.5879 | 0.4453 | 1740 |
| PA1 | 0.2512 | 0.2092 | 0.2283 | 1740 |
| PA2 | 0.2089 | 0.0920 | 0.1277 | 1740 |
| PA3 | 0.2606 | 0.2075 | 0.2310 | 1740 |
| PA4 | 0.4383 | 0.5655 | 0.4939 | 1740 |

### filtered_raw_signal / tiny_1d_cnn

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BL1 | 0.2249 | 0.4000 | 0.2879 | 1740 |
| PA1 | 0.2121 | 0.2103 | 0.2112 | 1740 |
| PA2 | 0.1952 | 0.1029 | 0.1347 | 1740 |
| PA3 | 0.2240 | 0.0879 | 0.1263 | 1740 |
| PA4 | 0.2993 | 0.3920 | 0.3394 | 1740 |


## Binary BL1 vs PA4

| input | model | accuracy | balanced accuracy | precision | recall | F1 | AUROC | fold F1 mean ± std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engineered_features | logistic_regression | 0.8080 | 0.8080 | 0.8445 | 0.7552 | 0.7973 | 0.8809 | 0.7971 ± 0.0267 |
| engineered_features | random_forest | 0.8184 | 0.8184 | 0.8428 | 0.7828 | 0.8117 | 0.8959 | 0.8113 ± 0.0250 |
| engineered_features | rbf_svc | 0.7951 | 0.7951 | 0.8496 | 0.7172 | 0.7778 | 0.8663 | 0.7773 ± 0.0317 |
| engineered_features | xgboost | 0.8256 | 0.8256 | 0.8516 | 0.7885 | 0.8189 | 0.9021 | 0.8187 ± 0.0298 |
| filtered_raw_signal | tiny_1d_cnn | 0.6230 | 0.6230 | 0.6524 | 0.5264 | 0.5827 | 0.6469 | 0.5758 ± 0.0930 |

## Subject-Level Heterogeneity

| task | input | model | min acc. | mean acc. | max acc. | subject acc. std |
|---|---|---|---:|---:|---:|---:|
| 5class | engineered_features | logistic_regression | 0.1600 | 0.3344 | 0.5100 | 0.0833 |
| 5class | engineered_features | random_forest | 0.1700 | 0.3225 | 0.5100 | 0.0778 |
| 5class | engineered_features | rbf_svc | 0.1600 | 0.3166 | 0.5000 | 0.0783 |
| 5class | engineered_features | xgboost | 0.1300 | 0.3324 | 0.5200 | 0.0813 |
| 5class | filtered_raw_signal | tiny_1d_cnn | 0.1500 | 0.2386 | 0.4300 | 0.0587 |
| binary_bl1_vs_pa4 | engineered_features | logistic_regression | 0.4500 | 0.8080 | 1.0000 | 0.1543 |
| binary_bl1_vs_pa4 | engineered_features | random_forest | 0.4500 | 0.8184 | 1.0000 | 0.1445 |
| binary_bl1_vs_pa4 | engineered_features | rbf_svc | 0.4750 | 0.7951 | 1.0000 | 0.1518 |
| binary_bl1_vs_pa4 | engineered_features | xgboost | 0.5000 | 0.8256 | 1.0000 | 0.1443 |
| binary_bl1_vs_pa4 | filtered_raw_signal | tiny_1d_cnn | 0.3500 | 0.6230 | 0.9750 | 0.1479 |

## Confusion Matrices

### 5-class / engineered_features / logistic_regression

```text
1088  302  139  112   99
 788  387  172  172  221
 636  353  190  241  320
 433  260  178  259  610
 194  179  142  240  985
```

### 5-class / engineered_features / random_forest

```text
 994  365  171  127   83
 727  379  214  201  219
 599  341  184  279  337
 374  222  222  310  612
 146  168  156  331  939
```

### 5-class / engineered_features / rbf_svc

```text
1224  193  137   76  110
 994  228  194  139  185
 845  223  172  190  310
 609  169  184  226  552
 331  115  137  253  904
```

### 5-class / engineered_features / xgboost

```text
1023  325  158  140   94
 737  364  186  234  219
 585  341  160  321  333
 358  255  151  361  615
 152  164  111  329  984
```

### 5-class / filtered_raw_signal / tiny_1d_cnn

```text
 696  384  201  133  326
 682  366  202  119  371
 682  351  179  120  408
 580  334  181  153  492
 455  291  154  158  682
```

### BL1 vs PA4 / engineered_features / logistic_regression

```text
1498  242
 426 1314
```

### BL1 vs PA4 / engineered_features / random_forest

```text
1486  254
 378 1362
```

### BL1 vs PA4 / engineered_features / rbf_svc

```text
1519  221
 492 1248
```

### BL1 vs PA4 / engineered_features / xgboost

```text
1501  239
 368 1372
```

### BL1 vs PA4 / filtered_raw_signal / tiny_1d_cnn

```text
1252  488
 824  916
```

## Interpretation

- XGBoost does not materially exceed the LR/RF/SVM feature-based baselines in 5-class subject-independent grading.
- The tiny 1D-CNN does not clearly outperform engineered-feature models under this fixed, small architecture.
- PA1, PA2, and PA3 remain difficult across feature-based and raw-signal models, so the difficulty is not only a handcrafted-feature limitation.
- BL1 vs PA4 remains much easier than 5-class grading, confirming that endpoint discrimination is not equivalent to pain-intensity grading.
- Subject-level heterogeneity persists across model families and input representations.

## Recommended Model Roles

- Main explanation model: Logistic Regression with `combined_unique`, because it is competitive and easiest to interpret without turning the paper into model optimization.
- Boosting robustness: XGBoost, as a fixed stronger tree-boosting check rather than a tuned leaderboard model.
- Kernel robustness: RBF-SVC.
- Representation sensitivity baseline: tiny 1D-CNN on filtered ECG/GSR/trapezius waveforms.
- Phase 4/5 should use LR as primary and compare contribution conclusions against RF, XGBoost, and SVC for cross-model consistency where computationally feasible.
