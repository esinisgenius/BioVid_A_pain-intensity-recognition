# Thiam et al. (2021) supervised DDCAE smoke test

Run date: 2026-10-25T11:51:44.267761+00:00

## Status and interpretation

This is a one-outer-fold engineering smoke test with a small number of epochs. It is not a paper reproduction result and must not be compared directly with the paper's 87-fold LOSO mean. The implementation preserves the reported three modalities, 1,152-sample inputs, six-stage DDCAE encoders/decoders, attention, feature-wise multimodal gate, joint reconstruction/classification objective, learning rate, and batch size. Protocol and paper ambiguities are documented in `reports/thiam2021_feasibility_review.md`.

## Execution

- Device: `mps` (MPS available: `True`).
- Outer fold: 1; fit/validation/calibration/test subjects: 42/10/18/17.
- Epochs run: 2 of 100 paper epochs; batch size: 100; learning rate: 1e-05.
- Epoch wall time: 7.85 s mean (3.60-12.11 s).
- Warm-up-aware steady-state projection: 0.10 h per fold, 0.51 h for five folds x one seed, and 1.53 h for five folds x three seeds. A conservative all-epoch-mean projection is 1.09/3.27 h. Projections exclude cache construction and final reliability evaluation.
- Peak process RSS: 646.6 MB; MPS current/driver allocation: 7.0/66.8 MB.

## Loss sanity check

| epoch | train total | validation total | train CE | validation CE | seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.3541 | 1.4789 | 1.6109 | 1.6106 | 12.11 |
| 2 | 1.3513 | 1.4763 | 1.6106 | 1.6106 | 3.60 |

Best validation checkpoint: epoch 2. With only 2 epochs at the paper's small learning rate, loss direction is a pipeline diagnostic, not evidence of convergence.

## Outer-fold smoke performance and same-trial XGBoost reference

| model | accuracy | macro F1 | balanced accuracy | MAE | QWK | PA2 recall |
|---|---:|---:|---:|---:|---:|---:|
| DDCAE smoke | 0.2000 | 0.0667 | 0.2000 | 1.2000 | 0.0000 | 1.0000 |
| Phase 3-strengthening XGBoost | 0.3329 | 0.3001 | 0.3329 | 1.1500 | 0.4775 | 0.0647 |

The rows contain the exact same outer-fold test trials. The DDCAE row is intentionally under-trained and is not a fair model comparison.

### DDCAE per-class recall

| class | recall |
|---|---:|
| BL1 | 0.0000 |
| PA1 | 0.0000 |
| PA2 | 1.0000 |
| PA3 | 0.0000 |
| PA4 | 0.0000 |

### DDCAE adjacent-level AUC

| pair | AUROC |
|---|---:|
| BL1 vs PA1 | 0.5175 |
| PA1 vs PA2 | 0.5106 |
| PA2 vs PA3 | 0.4812 |
| PA3 vs PA4 | 0.5021 |

DDCAE confusion matrix, rows=true and columns=predicted, class order `BL1 PA1 PA2 PA3 PA4`:

```text
   0    0  340    0    0
   0    0  340    0    0
   0    0  340    0    0
   0    0  340    0    0
   0    0  340    0    0
```

## Reliability smoke diagnostics

- Temperature-scaled ECE: 0.0016 (temperature 10.0000).
- Error-detection AUROC: 0.4925.
- Accuracy at 50% selective coverage: 0.1859.
- 90% split-conformal target: empirical coverage 0.9135, mean set width 4.594 of five labels.

For context, the completed five-fold Phase 10 XGBoost run had temperature-scaled ECE 0.0148, error-detection AUROC 0.6207, 50%-coverage accuracy 0.4083, and 90% conformal coverage/width 0.9023/3.681. The two-epoch DDCAE is worse in risk ranking and resolution; its very low ECE is caused by nearly uniform probabilities and temperature hitting the upper bound, not useful calibration.

These values verify that the Phase 10-style, subject-disjoint calibration path works with DDCAE probabilities. They cannot yet answer whether the reliability-resolution trade-off persists after a published deep model is trained: a two-epoch checkpoint collapsed to PA2 and has not learned the intended classifier. The formal answer is therefore **undetermined after smoke testing**; it is deferred until an approved full-scale run.

## Reproducibility

- Training command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m src.models.run_thiam2021_reproduction --config configs/thiam2021_smoke.yaml --fold 1 --epochs 2`
- Finalization command: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 '/Users/liangyingxin/Desktop/BioVid_A /src/models/run_thiam2021_reproduction.py' --config configs/thiam2021_smoke.yaml --fold 1 --epochs 2 --finalize-existing`
- Seed: `20261025`
- Model checkpoint SHA-256: `b416d4dffcab60274b1b361409361bd8514f0fef3eaa13a969007007630141a0`
- Raw-signal collection SHA-256: `b0d460d5e0048070ff832f5572dbf216033b127da59efcccd6bcb17e52e89837`
- Samples metadata SHA-256: `5e98de3957cc998e18c2d6d65b699ce613a484e0d2a1fb8afaec3f3877ac61fe`
- Fold assignment SHA-256: `d01b58e80bb8c8cfb36e6c43a02c2b33d19525c9fcc9e41d44407f60af8510a3`
- Paper PDF SHA-256: `eb00d25c3b59945867532c7ac7a8ff659a80dd348381bca218c4cc4b18893b0c`
- Full dependency versions, split roles, standardization values, configuration, history, predictions, and subject-level metrics are stored in `experiments/thiam2021_smoke/`.

## Stop decision

The requested stop condition is met. No other outer fold and no additional seed was started.
