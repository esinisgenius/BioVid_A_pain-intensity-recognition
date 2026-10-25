# Phase 10 — Prediction Reliability, Uncertainty, and Ordinal Intervals

Run date: 2026-10-25T10:45:31Z

## Executive finding

This phase is a reliability feasibility study. The conclusions below distinguish probability reliability, ranking of error risk, and prediction-set coverage; none is treated as evidence of a causal physiological mechanism or clinical validity.

## Leakage-safe protocol

- Reused the exact Phase 3 five-fold outer subject assignments.
- Within every outer fold, 25% of outer-training subjects were assigned to calibration; the remaining subjects trained the fixed Population-only XGBoost model.
- Temperature, conformal thresholds, and ordinal radii were fitted only on fold-local calibration subjects. Outer-test labels were used only for final evaluation.
- Selective curves rank outer-test samples using probability-derived uncertainty only. Coverage cutoffs are label-free empirical quantiles within each outer test fold; no outer-test label selected a threshold.
- Confidence intervals resample subjects as clusters, preserving their repeated trials.

## Probability calibration

| probabilities | accuracy | ECE | multiclass Brier | NLL | mean confidence |
|---|---:|---:|---:|---:|---:|
| raw | 0.3333 | 0.0467 | 0.7474 | 1.4705 | 0.3800 |
| temperature_scaled | 0.3333 | 0.0148 | 0.7440 | 1.4642 | 0.3386 |

Subject-clustered 95% confidence intervals:

| probabilities | metric | estimate | 95% CI |
|---|---|---:|---:|
| raw | ece | 0.0467 | [0.0338, 0.0603] |
| raw | brier | 0.7474 | [0.7355, 0.7594] |
| raw | nll | 1.4705 | [1.4386, 1.5038] |
| temperature_scaled | ece | 0.0148 | [0.0092, 0.0274] |
| temperature_scaled | brier | 0.7440 | [0.7336, 0.7544] |
| temperature_scaled | nll | 1.4642 | [1.4379, 1.4908] |

Paired improvement (`raw - temperature-scaled`; positive favors scaling):

| metric | improvement | subject-clustered 95% CI |
|---|---:|---:|
| ece_improvement | 0.0319 | [0.0126, 0.0397] |
| brier_improvement | 0.0034 | [0.0009, 0.0060] |
| nll_improvement | 0.0063 | [-0.0004, 0.0135] |

Fold temperatures ranged from 1.234 to 1.421 (median 1.326).

Figures: `figures/reliability_diagram_overall.png` and `figures/reliability_diagrams_per_class.png`.

## Uncertainty and selective prediction

| uncertainty | AURC | error-detection AUROC |
|---|---:|---:|
| max_probability | 0.5645 | 0.6207 |
| predictive_entropy | 0.5648 | 0.6227 |

Subject-clustered uncertainty-ranking intervals:

| uncertainty | metric | estimate | 95% CI |
|---|---|---:|---:|
| max_probability | aurc | 0.5645 | [0.5425, 0.5874] |
| max_probability | error_detection_auroc | 0.6207 | [0.6067, 0.6341] |
| predictive_entropy | aurc | 0.5648 | [0.5423, 0.5880] |
| predictive_entropy | error_detection_auroc | 0.6227 | [0.6088, 0.6366] |

Mean fold performance at selected coverage levels:

| method | coverage | accepted accuracy | MAE | severe error rate |
|---|---:|---:|---:|---:|
| max_probability | 50% | 0.4083 | 1.0085 | 0.2809 |
| max_probability | 70% | 0.3710 | 1.0611 | 0.2941 |
| max_probability | 90% | 0.3445 | 1.1139 | 0.3103 |
| max_probability | 100% | 0.3333 | 1.1435 | 0.3215 |
| predictive_entropy | 50% | 0.4063 | 0.9908 | 0.2718 |
| predictive_entropy | 70% | 0.3750 | 1.0538 | 0.2922 |
| predictive_entropy | 90% | 0.3455 | 1.1108 | 0.3102 |
| predictive_entropy | 100% | 0.3333 | 1.1435 | 0.3215 |
| random_rejection | 50% | 0.3332 | 1.1446 | 0.3219 |
| random_rejection | 70% | 0.3335 | 1.1431 | 0.3214 |
| random_rejection | 90% | 0.3329 | 1.1438 | 0.3216 |
| random_rejection | 100% | 0.3333 | 1.1435 | 0.3215 |

Rejected-sample class counts and proportions are recorded in `selective_rejected_class_distribution.csv`; accepted-only improvements must be read together with those rejection profiles.

Pain-level uncertainty:

| pain level | accuracy | mean max probability | mean entropy |
|---|---:|---:|---:|
| BL1 | 0.5874 | 0.3283 | 0.9250 |
| PA1 | 0.2034 | 0.3204 | 0.9314 |
| PA2 | 0.1213 | 0.3222 | 0.9301 |
| PA3 | 0.2063 | 0.3362 | 0.9184 |
| PA4 | 0.5483 | 0.3856 | 0.8781 |

## Conformal prediction

| method | target coverage | empirical coverage | mean width | median width | singleton rate | singleton error rate | contiguity |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinal_contiguous | 95% | 0.9525 | 4.134 | 4.0 | 0.0000 | — | 1.0000 |
| ordinal_contiguous | 90% | 0.8969 | 3.719 | 4.0 | 0.0000 | — | 1.0000 |
| ordinal_contiguous | 80% | 0.8041 | 3.158 | 3.0 | 0.0000 | — | 1.0000 |
| standard_classification | 95% | 0.9524 | 4.094 | 4.0 | 0.0011 | 0.6000 | 0.9869 |
| standard_classification | 90% | 0.9023 | 3.681 | 4.0 | 0.0047 | 0.2927 | 0.9646 |
| standard_classification | 80% | 0.8056 | 3.078 | 3.0 | 0.0133 | 0.3448 | 0.9156 |

The standard comparator is split conformal classification using nonconformity `1 - p(y)`. The ordinal comparator is a split-conformal residual interval around the probability-weighted expected level, using `|y - E[level]|`; it is a baseline construction, not a novel algorithm.

### Coverage heterogeneity at 90% target

| method | subject coverage min–median–max | subject mean-width range |
|---|---:|---:|
| ordinal_contiguous | 0.740–0.920–1.000 | 3.29–3.95 |
| standard_classification | 0.750–0.910–1.000 | 2.72–4.72 |

| method | pain level | coverage | mean width |
|---|---|---:|---:|
| ordinal_contiguous | BL1 | 0.8017 | 3.771 |
| ordinal_contiguous | PA1 | 0.9701 | 3.771 |
| ordinal_contiguous | PA2 | 1.0000 | 3.762 |
| ordinal_contiguous | PA3 | 0.9810 | 3.717 |
| ordinal_contiguous | PA4 | 0.7316 | 3.574 |
| standard_classification | BL1 | 0.9126 | 3.744 |
| standard_classification | PA1 | 0.9109 | 3.788 |
| standard_classification | PA2 | 0.9494 | 3.805 |
| standard_classification | PA3 | 0.8891 | 3.701 |
| standard_classification | PA4 | 0.8494 | 3.368 |

### Interval containment versus point-error severity (90% target)

| method | point prediction | n | interval/set contains truth | mean width |
|---|---|---:|---:|---:|
| ordinal_contiguous | correct | 2900 | 0.9945 | 3.667 |
| ordinal_contiguous | adjacent_error | 3001 | 0.9627 | 3.737 |
| ordinal_contiguous | severe_error | 2799 | 0.7253 | 3.753 |
| standard_classification | correct | 2900 | 1.0000 | 3.472 |
| standard_classification | adjacent_error | 3001 | 0.9680 | 3.737 |
| standard_classification | severe_error | 2799 | 0.7306 | 3.839 |

### Guarantee scope

Classical split-conformal marginal coverage requires exchangeability of calibration and future examples. Trials are clustered within subjects here, so trial-level exchangeability is doubtful. The subject-disjoint construction prevents identity leakage, but it does not establish finite-sample per-subject or per-class coverage. Per-subject and per-class results are diagnostics only; subject-clustered bootstrap intervals quantify sampling uncertainty but do not restore a conformal guarantee.

## Failure cases and limitations

- The fold-local calibration split reduces the data available to train XGBoost, so these predictions are not numerically identical to the earlier model trained on every outer-training subject.
- Temperature scaling changes probability sharpness but not the argmax class, so it cannot improve five-class accuracy.
- A useful selective system needs both low accepted risk and acceptable rejection burden; class-dependent rejection can reduce apparent risk by disproportionately abstaining on intermediate levels.
- Marginal conformal coverage can coexist with serious subject-level undercoverage and wide, low-resolution sets.
- BioVid experimental heat pain does not establish clinical diagnostic or deployment validity.

## Decision on boundary-aware selective resolution

The feasibility gate is only partially met. Max-probability and entropy ranking clearly outperform random rejection, so the model has useful error-risk information. However, the gain is not sufficient for precise grading: at 50% coverage accepted accuracy is only about 41%, and the 90% conformal outputs require roughly 3.7 of five labels on average. The ordinal interval is contiguous, but it is not narrower than the standard set at matched coverage and has zero singleton predictions; it therefore adds interpretability of shape, not resolution.

A narrowly scoped boundary-aware selective-resolution follow-up is scientifically justified because risk ranking exists while standard baselines leave a severe coverage-resolution and subject-heterogeneity gap. This is not yet justification for a complex new model or an innovation claim. Any follow-up should stop unless it improves resolution at matched subject-independent coverage without concentrating rejection or undercoverage in PA2, endpoint classes, or particular subjects.

Machine-readable per-subject, per-class, fold-level, selective, and conformal diagnostics are stored beside this report in `experiments/prediction_reliability/`.
