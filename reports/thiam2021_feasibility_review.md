# Thiam et al. (2021) supervised DDCAE reproduction feasibility review

Review date: 2026-10-25

Scope: supervised multimodal DDCAE with attention for BioVid Part A five-class classification. The self-supervised AVAE extension is out of scope. No full training was started during this review.

Primary source: Patrick Thiam et al., *Multi-Modal Pain Intensity Assessment Based on Physiological Signals: A Deep Learning Perspective*, Frontiers in Physiology 12:720464, DOI [10.3389/fphys.2021.720464](https://doi.org/10.3389/fphys.2021.720464). The local PDF is `references/Thiam_2021_Multimodal_Pain.pdf` (21 pages; SHA-256 `eb00d25c3b59945867532c7ac7a8ff659a80dd348381bca218c4cc4b18893b0c`).

## Decision

The repository can support an **adapted reproduction** of the supervised ECG/EDA/EMG DDCAE with gating and attention on the exact Phase 3 five subject-independent outer folds. It cannot support a fully faithful reproduction of the published experiment from the currently available Part A clips.

The two protocol-level differences are unavoidable:

1. The paper evaluates 87 Leave-One-Subject-Out runs (page 13, Section 4.1), whereas the requested comparison must reuse the five Phase 3 `StratifiedGroupKFold` outer folds.
2. The paper augments each training trial by shifting a 4.5 s window backward and forward in 250 ms increments, up to 1 s in each direction, producing nine windows (page 11, Section 3.5). A complete implementation needs at least 6.5 s around the canonical 4 s post-onset window. The local Part A files contain only 5.5 s. They likely begin about 3 s after the elicitation trigger, but the precise trigger definition is unknown. They therefore cannot supply all nine paper windows, especially the forward shifts.

A third fidelity risk is preprocessing. The paper describes filtering recorded signals and then segmenting them. Only clipped Part A trials are available locally, so reconstructing the paper filters from `biosignals_raw` applies long-memory filters to short 5.5 s clips and introduces boundary behavior that would not occur when filtering continuous recordings.

## Paper-specified method

### Inputs and preprocessing

Evidence: page 11, Section 3.5; page 12, Figure 6.

- BioVid physiological inputs used by the supervised model: EDA, ECG, and EMG. The objective weights reported for BioVid contain three reconstruction terms, confirming three modalities.
- Sampling rate reduced to 256 Hz before modeling.
- EDA: third-order low-pass Butterworth, cutoff 0.2 Hz.
- EMG: fourth-order band-pass Butterworth, 20-250 Hz.
- ECG: third-order band-pass Butterworth, 0.1-250 Hz, followed by piecewise detrending via subtraction of a fitted fifth-degree polynomial.
- Canonical evaluation window: 4.5 s, starting 4 s after elicitation onset, giving 1,152 samples per modality.
- Training-only augmentation: shift the 4.5 s segmentation window backward and forward in 250 ms steps, with maximum shift 1 s in either direction; this yields nine times the initial training samples. Evaluation uses only the canonical 4 s-shifted window.

The paper says that EDA was filtered with a cutoff of 0.2 Hz but does not explicitly state whether that is a low-pass or high-pass filter in the prose. The described order and cutoff, the earlier authors' pipeline, and the slow nature of EDA make low-pass the defensible interpretation; this remains a reproduction assumption.

### Encoder, decoder, attention, gate, and classifier

Evidence: pages 5-7, Figures 1-3 and Equations 1-11; page 12, Table 1.

- One denoising convolutional autoencoder per modality.
- Gaussian corruption during training: mean 0, standard deviation 0.1.
- ELU activation in convolutional and fully connected hidden layers.
- Encoder per modality:
  - two Conv1D/MaxPool operations with 8 filters;
  - two Conv1D/MaxPool operations with 16 filters;
  - two Conv1D/MaxPool operations with 32 filters;
  - flatten;
  - fully connected latent representation of 256 units.
- Max pooling uses size 2 and stride 2. The decoder mirrors this with upsampling size/stride 2.
- Modality-specific convolution kernel: EDA 3; ECG and EMG 11; stride 1.
- One-dimensional spatial attention follows every encoder convolution. It concatenates channel-axis average and maximum maps, applies a Conv1D with kernel 3/stride 1 and sigmoid, and multiplies the resulting position mask into all feature channels.
- Decoder: fully connected 576 units; reshape; two Conv1D/Upsampling operations at each of 32, 16, and 8 filters; final one-channel Conv1D.
- The 576 decoder units strongly imply a reshape to length 18 x 32 because six factor-2 pooling operations reduce 1,152 samples to 18. The paper does not state this shape explicitly.
- Gate: each 256-dimensional latent vector is transformed with a modality-specific tanh layer; transformed vectors are concatenated and passed through a feature-wise softmax; the corresponding 256-element section weights each original modality latent vector; the weighted modality vectors are summed.
- Classifier: fully connected 128, dropout 0.25, fully connected `c`; for this task `c=5`, followed by softmax probabilities.

### Objective and optimization

Evidence: page 6, Equations 6-9; pages 12-13, Section 4.1.

- Each decoder uses mean squared reconstruction error plus L2 parameter penalty `lambda = 1e-3`.
- Classification uses categorical cross-entropy.
- Joint end-to-end objective for BioVid: `0.2 * E_EDA + 0.2 * E_ECG + 0.2 * E_EMG + 0.4 * CE`.
- Adam, fixed learning rate `1e-5`.
- 100 epochs.
- Five-class BioVid batch size 100.
- TensorFlow, Keras, and scikit-learn were used by the authors.

### Published five-class result

Evidence: pages 14-15, Figure 7C and Table 4.

| model | accuracy, mean (SD) | F1, mean (SD) |
|---|---:|---:|
| gated DDCAE without attention | 33.31% (9.35%) | 30.31% (9.30%) |
| gated DDCAE with attention | 35.44% (8.66%) | 31.04% (7.91%) |

These are averages over 87 LOSO experiments. They are not directly comparable with pooled out-of-fold metrics from the fixed five-fold protocol.

The paper's Table 2 writes multiclass accuracy as the macro average of one-vs-rest `(TP+TN)/(TP+TN+FP+FN)`. That formula cannot produce the reported 35.44% for a balanced five-class problem (its minimum is 60%). The result table therefore appears to use ordinary sample accuracy despite the printed formula. This reproduction will report ordinary accuracy and flag the inconsistency.

The paper reports only Accuracy and macro-averaged F1 for the five-class classification task. It does not report balanced accuracy, MAE, QWK, per-class recall, confusion matrices, adjacent-pair AUC, calibration, selective prediction, conformal sets, or subject-level confidence intervals.

## Explicitly unspecified or ambiguous details

The following details are not recoverable from the article and are material to an exact implementation:

- convolution padding convention and whether biases are enabled;
- exact order of convolution, ELU, attention, and pooling operations;
- explicit decoder reshape and whether decoder convolutions occur before or after upsampling;
- final decoder activation;
- weight initialization, Adam beta/epsilon values, random seed, shuffling, and determinism;
- signal amplitude normalization or standardization before adding Gaussian noise;
- filter implementation details (causal vs zero-phase, SOS vs transfer function, boundary padding) and whether filters were fit/applied to continuous sessions before clipping;
- exact definition of "piecewise" fifth-degree ECG detrending;
- reconstruction MSE reduction (per point, per sample, or summed) and exact L2 scope;
- whether the L2 term in Equation 6 was implemented through optimizer regularizers or added explicitly;
- validation split, model selection, checkpoint selection, or early stopping. The paper states 100 epochs and does not describe early stopping;
- whether all nine augmented windows were materialized or sampled, and their exact handling near recording boundaries;
- trial-level predictions, per-subject outputs, and fold assignments beyond LOSO.

These details make a from-paper implementation an **independent implementation of the reported architecture**, nested within an **adapted evaluation protocol**.

## Local-data compatibility

Verified against `reports/data_audit.md`, `reports/data_protocol_audit.md`, `starting_point/samples.csv`, and the actual signal files:

- 87 subjects, 8,700 trials, exactly 20 trials per subject and class.
- Exact label mapping: BL1=0, PA1=1, PA2=2, PA3=3, PA4=4.
- Raw files: 2,816 samples at 512 Hz, about 5.5 s.
- Available usable paper modalities: `gsr` (EDA), `ecg`, and `emg_trapezius`.
- `emg_corrugator` and `emg_zygomaticus` are all zero and are excluded; they are not the paper's BioVid EMG input.
- Phase 3 outer folds are complete and subject-disjoint: folds 1-3 contain 17 test subjects/1,700 trials; folds 4-5 contain 18 test subjects/1,800 trials.
- Apple Silicon MPS is available through PyTorch 2.8.0; CPU fallback is possible.

### Canonical-window assumption

The local audit indicates that released Part A clips likely start about 3 s after the elicitation trigger. To approximate the paper's canonical 4 s-shifted evaluation window without shortening it, the adapted pipeline will crop samples `[512:2816]` from each 512 Hz clip, then apply the paper preprocessing/downsampling to produce 1,152 samples. Because the exact trigger definition is unknown, this is an explicit assumption, not a verified faithful alignment.

For preprocessing correctness, filtering should occur at 512 Hz before the factor-2 downsampling and before the 4.5 s crop. This respects the paper's 250 Hz upper cutoff, which would be invalid after downsampling to a 128 Hz Nyquist frequency.

### Training augmentation decision

The smoke test will use only the canonical window. This is a smoke-test-only cost control, not the proposed final experiment. A later full adapted experiment has two defensible options:

1. canonical-window-only training, which changes the published augmentation but keeps every train/test trial aligned; or
2. use only the backward shifts available inside the local 5.5 s clip, which creates five rather than nine windows and asymmetrically changes augmentation.

Neither option is faithful. Padding, extrapolation, shortening the model input, dropping modalities, or pretending that duplicated windows are forward shifts is not acceptable.

## Leakage-safe adapted protocol

For each fixed Phase 3 outer fold:

1. reserve fold-local calibration subjects using the deterministic Phase 10 subject split;
2. split the remaining model-development subjects again by subject into fit and early-stopping validation roles;
3. compute any amplitude standardization statistics on fit subjects only;
4. train/checkpoint on fit subjects and select the epoch on validation-subject loss;
5. fit temperature and conformal thresholds on calibration subjects only;
6. evaluate once on the untouched outer-test subjects.

The smoke test will exercise one outer fold for a small number of epochs and will stop. It will report runtime, loss behavior, MPS/CPU use, memory, test metrics, reliability diagnostics, and projected cost for five folds and one/three seeds. Smoke metrics are engineering diagnostics and must not be treated as a reproduction result.

## Code availability review

The paper's Data Availability Statement says that contributions are included in the article/supplementary material and directs further inquiries to the corresponding author; it does not provide a repository. Searches by exact title, DOI, `DDCAE`, the 2020 ICPRAM title, author name, and GitHub did not identify an author-maintained implementation. The earlier conference paper is available from [SciTePress](https://www.scitepress.org/Papers/2020/88961/88961.pdf), but no official code link was found. Unrelated or later third-party repositories are not treated as reusable official implementations.

Conclusion: implement from the paper with traceable assumptions; label the code as an independent implementation, not official author code.
