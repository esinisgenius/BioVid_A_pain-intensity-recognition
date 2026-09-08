# Phase 2 Feature Specification

Audit date: 2026-09-08

Status: proposed specification for review. This file defines the controlled feature extraction plan only. No batch feature extraction, baseline modeling, SHAP, feature ranking, modality ranking, or deep learning has been run.

## Scope and Decisions

Primary input:

- `biosignals_filtered/<subject_name>/<sample_name>_bio.csv`
- Sampling rate: 512 Hz
- Window length: 2,816 rows, about 5.5 seconds

Included channels:

- ECG: `ecg`
- EDA/GSR: `gsr`
- Trapezius EMG: `emg_trapezius`

Excluded channels and fields:

- `temperature`: excluded from the main feature table by Phase 1 review decision.
- `emg_corrugator`: excluded because it is all zero in local raw and filtered data.
- `emg_zygomaticus`: excluded because it is all zero in local raw and filtered data.
- `trial_number`, `sample_id`, `sample_name`, file paths, and filename-derived sequence values: metadata only; never model features.
- `subject_id` and `subject_name`: grouping/audit metadata only; never model features.
- `class_id` and `class_name`: labels only; never feature columns.

Feature sets:

- `domain_informed`: primary analysis feature set, using physiology-informed definitions and conservative short-window assumptions.
- `generic_automated`: optional supplementary feature set, using a small custom tsfresh configuration. These features are not interpreted as physiological mechanisms.
- `shared`: feature provenance marker for highly overlapping domain-informed and generic-automated summaries. Shared features may remain in both exported sets for provenance, but Phase 3 must not treat the two sets as fully independent evidence.

Implementation dependency note:

- The current local Python environment does not have `neurokit2`, `tsfresh`, or `scipy` installed. Pipeline implementation should add or document these dependencies before extraction.
- NeuroKit2 documentation supports ECG, EDA, and EMG processing pipelines, including ECG cleaning/R-peaks/rate/quality, EDA cleaning/phasic/SCR peak detection, and EMG processing/activation workflows.
- tsfresh documentation confirms that `ComprehensiveFCParameters` is the default if no settings are provided; this project will not use that default. It will use an explicit small custom feature calculator dictionary.

References checked:

- NeuroKit2 ECG functions: https://neuropsychology.github.io/NeuroKit/functions/ecg.html
- NeuroKit2 EDA functions: https://neuropsychology.github.io/NeuroKit/functions/eda.html
- NeuroKit2 EMG activation documentation: https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/emg/emg_activation.html
- tsfresh feature calculators: https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
- tsfresh feature extraction settings: https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html

## Methodological Rules

1. Treat each CSV as one event-related trial/window.
2. Do not use any feature that requires subject identity, filename order, class name, class id, global sample id, or trial number.
3. Do not use features that require long stationary windows when only about 5.5 seconds are available.
4. Do not use traditional long-window HRV features as primary ECG features.
5. All feature failures must be recorded explicitly. Samples should not be silently dropped.
6. If a feature cannot be computed for a sample, the extraction report must record the reason and either produce a documented missing value for later fold-local imputation or exclude that feature globally after review.
7. Learned preprocessing and any later imputation/scaling must be fit inside training folds only. Deterministic per-sample signal processing is allowed at extraction time.
8. Feature metadata must distinguish `physiological-derived feature` from `generic signal/statistical summary`; waveform summaries such as ECG RMS/std/peak-to-peak are not to be interpreted as physiological markers.

## Domain-Informed Feature Specification

### ECG

Implemented method after Phase 2 full-data robustness check:

- Beat-derived ECG features use median-centered filtered ECG, a fixed 5-20 Hz Butterworth bandpass, and NeuroKit2 `ecg_peaks(method="neurokit")`.
- ECG waveform-summary features remain based on NeuroKit2-cleaned filtered ECG rather than the 5-20 Hz bandpassed signal, because the bandpass materially changes amplitude summaries.
- Require a minimum valid R-peak count for beat-derived features. Proposed threshold: at least 3 detected R-peaks, giving at least 2 RR intervals. Samples below threshold should be flagged.

| feature name | modality | set | method / library | definition | physiological or signal-processing meaning | 5.5 sec suitable? | limitations | enters main analysis? |
|---|---|---|---|---|---|---|---|---|
| `ecg_rpeak_count` | ECG | `domain_informed` | NeuroKit2 `ecg_peaks` or equivalent | Number of detected R-peaks in the 5.5 sec window | Beat occurrence count; coarse short-window cardiac activity | Yes | Sensitive to peak detection errors and ectopic/artifact peaks | Yes |
| `ecg_rate_mean_bpm` | ECG | `domain_informed` | NeuroKit2 `signal_rate` from R-peaks | Mean interpolated heart rate in beats per minute | Short-window central tendency of cardiac rate | Yes, with peak-count flag | Not full HRV; unreliable if too few R-peaks | Yes |
| `ecg_rate_median_bpm` | ECG | `domain_informed` | NeuroKit2 `signal_rate` from R-peaks | Median interpolated heart rate | Robust short-window cardiac rate | Yes, with peak-count flag | Same limitations as rate mean | Yes |
| `ecg_rate_min_bpm` | ECG | `domain_informed` | NeuroKit2 `signal_rate` from R-peaks | Minimum interpolated heart rate | Lower bound of short-window rate | Borderline | Can be unstable at window edges | Yes, quality-gated |
| `ecg_rate_max_bpm` | ECG | `domain_informed` | NeuroKit2 `signal_rate` from R-peaks | Maximum interpolated heart rate | Upper bound of short-window rate | Borderline | Can be unstable at window edges | Yes, quality-gated |
| `ecg_rate_range_bpm` | ECG | `domain_informed` | Derived from rate max - rate min | Short-window heart-rate range | Coarse within-window rate change | Borderline | Not an HRV substitute; edge interpolation can dominate | Yes, quality-gated |
| `ecg_rate_slope_bpm_per_s` | ECG | `domain_informed` | Linear regression of interpolated rate over seconds | Slope of short-window heart rate | Direction of cardiac rate change after stimulus | Yes | Sensitive to interpolation and few beats | Yes |
| `ecg_rr_mean_ms` | ECG | `domain_informed` | R-peak intervals | Mean RR interval in milliseconds | Beat-to-beat interval level | Yes, with at least 2 RR intervals | Not a long-window HRV measure | Yes |
| `ecg_rr_median_ms` | ECG | `domain_informed` | R-peak intervals | Median RR interval in milliseconds | Robust beat interval level | Yes, with at least 2 RR intervals | Unstable with very few intervals | Yes |
| `ecg_quality_mean` | ECG | `domain_informed` | NeuroKit2 `ecg_quality` or equivalent | Mean ECG quality score across the window | Signal quality / artifact indicator | Yes | Quality scale depends on algorithm; should not be overinterpreted physiologically | Yes |
| `ecg_clean_std` | ECG | `domain_informed` | Cleaned ECG signal | Standard deviation of cleaned ECG amplitude | ECG amplitude variability / signal energy proxy | Yes | Electrode placement and preprocessing can dominate | Yes |
| `ecg_clean_rms` | ECG | `domain_informed` | Cleaned ECG signal | Root mean square of cleaned ECG amplitude | Short-window signal power proxy | Yes | Not directly physiological without quality context | Yes |
| `ecg_clean_peak_to_peak` | ECG | `domain_informed` | Cleaned ECG signal | Max minus min cleaned ECG amplitude | Amplitude span | Yes | Strongly affected by artifacts and electrode scaling | Yes |

ECG candidate features excluded before implementation:

| excluded feature family | reason for exclusion |
|---|---|
| LF/HF, VLF, ULF, frequency-domain HRV | Requires much longer windows and sufficient beat series; 5.5 sec is methodologically unsuitable. |
| SDNN, RMSSD, SDSD, pNN20/pNN50 as traditional HRV | Very few RR intervals per trial; estimates would be unstable and easily overinterpreted. |
| Poincare SD1/SD2, sample entropy, DFA, fractal/nonlinear HRV | Requires longer beat sequences; not credible for 5.5 sec trials. |
| QRS/QT/ST morphology and detailed delineation features | NeuroKit2 can delineate waves, but robust delineation over 5.5 sec filtered single-lead files must be validated first; exclude from initial main analysis. |
| ECG-derived respiration | Requires longer rhythm and respiratory cycles; no respiration channel verified locally. |

### EDA / GSR

Planned method:

- Use filtered `gsr` as the EDA/GSR signal.
- Use NeuroKit2-style EDA cleaning and optional phasic/SCR peak detection only for short-window-compatible features.
- Apply a deterministic per-window SCR amplitude filter after NeuroKit2 detection: retain detected SCR amplitudes only when `SCR_Amplitude >= max(0.01, 0.01 * window_range)`. This removes near-zero numerical peak artifacts without using pain labels or dataset-wide statistics.
- Avoid sympathetic frequency-domain indices and long-window decomposition claims.

| feature name | modality | set | method / library | definition | physiological or signal-processing meaning | 5.5 sec suitable? | limitations | enters main analysis? |
|---|---|---|---|---|---|---|---|---|
| `eda_scl_mean` | EDA/GSR | `domain_informed` | Direct filtered GSR | Mean skin conductance level in the window | Tonic level proxy within trial | Yes | Absolute level varies strongly by subject/electrodes; needs subject-independent handling | Yes |
| `eda_scl_median` | EDA/GSR | `domain_informed` | Direct filtered GSR | Median skin conductance level | Robust tonic level proxy | Yes | Same subject-level scaling concern | Yes |
| `eda_scl_std` | EDA/GSR | `domain_informed` | Direct filtered GSR | Standard deviation of GSR | Short-window variability | Yes | Can reflect noise or stimulus response | Yes |
| `eda_scl_min` | EDA/GSR | `domain_informed` | Direct filtered GSR | Minimum GSR value | Lower bound of conductance in trial | Yes | Subject scaling and artifacts | Yes |
| `eda_scl_max` | EDA/GSR | `domain_informed` | Direct filtered GSR | Maximum GSR value | Upper bound of conductance in trial | Yes | Subject scaling and artifacts | Yes |
| `eda_scl_range` | EDA/GSR | `domain_informed` | Max - min | Conductance excursion | Magnitude of short-window response/change | Yes | Does not distinguish tonic drift from SCR | Yes |
| `eda_scl_auc` | EDA/GSR | `domain_informed` | Trapezoidal integral over seconds | Area under the GSR curve | Total conductance exposure over the trial | Yes | Correlated with mean because length is fixed | Yes |
| `eda_scl_slope_per_s` | EDA/GSR | `domain_informed` | Linear regression over seconds | Slope of GSR over trial | Direction/magnitude of conductance change | Yes | Linear summary may miss nonlinear SCR shape | Yes |
| `eda_scl_delta` | EDA/GSR | `domain_informed` | Last value - first value | Net conductance change | Simple response direction | Yes | Sensitive to edge artifacts | Yes |
| `eda_scr_count` | EDA/GSR | `domain_informed` | NeuroKit2 `eda_phasic` + `eda_peaks`, filtered by `max(0.01, 1% window range)` amplitude threshold | Number of retained SCR peaks in the window | Phasic sympathetic arousal response count | Borderline but defensible | SCR latency/recovery may be truncated; coverage must be audited | Yes, quality-gated |
| `eda_scr_amplitude_max` | EDA/GSR | `domain_informed` | NeuroKit2 SCR peak info after deterministic amplitude filtering | Maximum retained SCR amplitude | Largest phasic response magnitude | Borderline | Set to 0 if no SCR is retained; coverage must be audited | Yes, quality-gated |
| `eda_scr_amplitude_sum` | EDA/GSR | `domain_informed` | Sum of retained SCR amplitudes after deterministic amplitude filtering | Total retained phasic response magnitude | Borderline | Set to 0 if no SCR is retained; coverage must be audited | Yes, quality-gated |

EDA candidate features excluded before implementation:

| excluded feature family | reason for exclusion |
|---|---|
| EDA sympathetic frequency-domain indices | NeuroKit2 documentation notes the Posada method requires at least 64 seconds; local windows are about 5.5 seconds. |
| LF/HF-style EDA spectral balance | Not appropriate for 5.5 sec windows and not part of the primary BioVid physiological scope. |
| cvxEDA/SparsEDA-derived full decomposition parameters | Decomposition can be dependency-heavy and short-window unstable; not suitable for the initial controlled pipeline. |
| Full SCR recovery time and half-recovery time | Responses may extend beyond the 5.5 sec trial window, causing systematic truncation. |
| EDA autocorrelation at long lags | A 4 sec lag leaves little independent overlap in a 5.5 sec signal; unstable for main analysis. |
| Change-point counts | Experimental and threshold-sensitive; could generate uninterpretable short-window artifacts. |

### Trapezius EMG

Planned method:

- Use filtered `emg_trapezius` only.
- Use NeuroKit2-style EMG amplitude/envelope and activation detection where available.
- Prefer amplitude, energy, and activation summaries over high-dimensional frequency or nonlinear features.
- Define activation without labels or dataset-wide statistics: absolute EMG is smoothed with a 50 ms moving average; activation is `envelope > median(envelope) + 3 * MAD(envelope)` within the same trial; retained activation segments must last at least 50 ms.

| feature name | modality | set | method / library | definition | physiological or signal-processing meaning | 5.5 sec suitable? | limitations | enters main analysis? |
|---|---|---|---|---|---|---|---|---|
| `emg_trapezius_mav` | Trapezius EMG | `domain_informed` | Absolute filtered EMG | Mean absolute value | Average muscle activity amplitude | Yes | Sensitive to preprocessing and baseline muscle tension | Yes |
| `emg_trapezius_rms` | Trapezius EMG | `domain_informed` | Filtered EMG | Root mean square amplitude | Standard EMG amplitude/energy proxy | Yes | Electrode placement and artifacts can dominate | Yes |
| `emg_trapezius_std` | Trapezius EMG | `domain_informed` | Filtered EMG | Standard deviation | Amplitude variability | Yes | Can reflect noise bursts | Yes |
| `emg_trapezius_iemg` | Trapezius EMG | `domain_informed` | Sum absolute amplitude, optionally normalized by duration | Integrated EMG over the trial | Total muscle activation magnitude | Yes | Correlated with MAV because length is fixed | Yes |
| `emg_trapezius_energy` | Trapezius EMG | `domain_informed` | Sum of squared amplitude | Short-window signal energy | Muscle activity power proxy | Yes | Correlated with RMS because length is fixed | Yes |
| `emg_trapezius_peak_to_peak` | Trapezius EMG | `domain_informed` | Max - min | Amplitude span | Burst/artifact magnitude | Yes | Outlier-sensitive | Yes |
| `emg_trapezius_waveform_length` | Trapezius EMG | `domain_informed` | Sum of absolute first differences | Waveform complexity/activity | Common time-domain EMG descriptor | Yes | Sensitive to high-frequency noise and filtering | Yes |
| `emg_trapezius_zero_crossings` | Trapezius EMG | `domain_informed` | Count sign changes after small deadband threshold | Oscillatory activity/noise descriptor | Yes | Threshold must be fixed and documented; can be noisy | Yes |
| `emg_trapezius_activation_fraction` | Trapezius EMG | `domain_informed` | NeuroKit2 `emg_activation` or deterministic threshold on amplitude envelope | Fraction of samples classified active | Duration of muscle activation within trial | Yes | Thresholding can be unstable; requires extraction quality flag | Yes, quality-gated |
| `emg_trapezius_activation_count` | Trapezius EMG | `domain_informed` | NeuroKit2 `emg_activation` or equivalent | Number of activation segments | Burst count | Borderline | Very short windows can fragment bursts; threshold-sensitive | Yes, quality-gated |
| `emg_trapezius_activation_mean_duration_s` | Trapezius EMG | `domain_informed` | Activation segments | Mean activation duration in seconds | Typical burst duration | Borderline | Undefined if no activations; needs zero/missing policy | Yes, quality-gated |

EMG candidate features excluded before implementation:

| excluded feature family | reason for exclusion |
|---|---|
| Mean/median frequency, spectral fatigue indicators | 5.5 sec windows may support a rough PSD, but fatigue-style interpretation requires longer, more controlled contractions. |
| EMG entropy/fractal/recurrence features | Too complex and unstable for initial short-window interpretable analysis. |
| Cross-channel facial EMG interactions | `emg_corrugator` and `emg_zygomaticus` are all zero locally and are formally excluded. |
| Learned activation thresholds fit across all subjects | Would introduce leakage if estimated globally; initial thresholds must be deterministic or fit inside training folds. |

## Generic Automated tsfresh Supplement

Purpose:

- Provide a small, controlled, non-comprehensive time-series feature set for comparison against the `domain_informed` set.
- These features are marked as `generic_automated` and should not be interpreted as physiological mechanisms.
- They may be useful for checking whether generic signal shape summaries add predictive signal or expose instability across subjects.
- Features that duplicate or closely overlap with domain-informed summaries are marked with provenance `shared` in machine-readable metadata.

Implementation rule:

- Do not call tsfresh without `default_fc_parameters`.
- Do not use `ComprehensiveFCParameters`.
- Do not use automated relevance filtering on the full dataset, because that would use labels globally and leak test-subject information.
- Use a small explicit custom dictionary. If relevance filtering is ever used later, it must be inside training folds only and separately reviewed.

Proposed custom tsfresh calculators:

| feature name pattern | modalities | set | method / library | definition | signal-processing meaning | 5.5 sec suitable? | limitations | enters main analysis? |
|---|---|---|---|---|---|---|---|---|
| `<modality>__tsfresh_mean` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `mean` | Arithmetic mean of the signal | Central tendency | Yes | Duplicates some domain features; generic only | Supplement only |
| `<modality>__tsfresh_median` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `median` | Median of the signal | Robust central tendency | Yes | Generic; may duplicate domain features | Supplement only |
| `<modality>__tsfresh_standard_deviation` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `standard_deviation` | Standard deviation of the signal | Dispersion | Yes | Sensitive to amplitude scaling | Supplement only |
| `<modality>__tsfresh_minimum` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `minimum` | Minimum value | Lower amplitude bound | Yes | Outlier-sensitive | Supplement only |
| `<modality>__tsfresh_maximum` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `maximum` | Maximum value | Upper amplitude bound | Yes | Outlier-sensitive | Supplement only |
| `<modality>__tsfresh_root_mean_square` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `root_mean_square` | Root mean square of signal values | Signal magnitude/power proxy | Yes | Generic; can duplicate EMG/ECG RMS | Supplement only |
| `<modality>__tsfresh_abs_energy` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `abs_energy` | Sum of squared signal values | Energy over fixed-length trial | Yes | Correlated with RMS because length is constant | Supplement only |
| `<modality>__tsfresh_absolute_sum_of_changes` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `absolute_sum_of_changes` | Sum of absolute consecutive differences | Total roughness/change | Yes | Sensitive to noise and filtering | Supplement only |
| `<modality>__tsfresh_mean_change` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `mean_change` | Mean first difference; equivalent to net change divided by number of intervals | Directional drift/change | Yes | Small values can be numerically fragile | Supplement only |
| `<modality>__tsfresh_linear_trend_slope` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `linear_trend`, `attr="slope"` | Least-squares slope over sample index | Monotonic trend | Yes | Uses sample index, not timestamps; acceptable because all windows have same sampling | Supplement only |
| `<modality>__tsfresh_skewness` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `skewness` | Adjusted Fisher-Pearson skewness | Distribution asymmetry | Yes | Unstable for near-constant signals | Supplement only |
| `<modality>__tsfresh_kurtosis` | ECG, EDA/GSR, Trapezius EMG | `generic_automated` | tsfresh `kurtosis` | Distribution tail/heaviness statistic | Outlier/burst sensitivity | Yes | Can be unstable for near-constant signals | Supplement only |

Expected generic feature count:

- 12 generic features per modality.
- 3 modalities.
- 36 `generic_automated` features total.

tsfresh candidate features excluded before implementation:

| excluded feature family | reason for exclusion |
|---|---|
| `ComprehensiveFCParameters` | Generates too many features and is explicitly out of scope. |
| `EfficientFCParameters` as a whole | Still too broad for this controlled specification. |
| Label-based `extract_relevant_features` on the full dataset | Would leak label information across subject-independent folds. |
| FFT aggregates, Welch density, Fourier entropy | Frequency-domain summaries over 5.5 sec windows are not robust enough for initial generic supplement. |
| CWT coefficients | Too many parameterized features and not interpretable enough for first controlled pass. |
| Matrix profile | Short windows and parameter sensitivity; not useful as an initial interpretable supplement. |
| Sample entropy and nonlinear complexity features | Unstable in short windows and near-constant signals. |
| Autocorrelation over multiple lags | Parameter-heavy; lag selection would be arbitrary in 5.5 sec windows. |
| First/last location of extrema | Could encode within-trial response timing, but is highly artifact-sensitive and less physiologically interpretable; exclude initially. |
| Longest strike above/below mean | Threshold depends on each sample's mean; can be hard to interpret across subjects and modalities. |
| Number crossing arbitrary thresholds | Threshold selection is modality-specific; use EMG zero crossings only in the domain-informed EMG set. |

## Proposed Feature Counts Entering Phase 3

If this specification is approved:

| modality | `domain_informed` features | `generic_automated` supplement features | notes |
|---|---:|---:|---|
| ECG | 13 | 12 | Domain ECG features avoid long-window HRV and complex morphology. |
| EDA/GSR | 12 | 12 | SCR features are quality-gated because 5.5 sec can truncate responses. |
| Trapezius EMG | 11 | 12 | Activation features are quality-gated because thresholds can be unstable. |
| Total | 36 | 36 | Main and supplement stay separate. |

## Planned Metadata Fields for Later Pipeline

Metadata fields retained for audit/grouping/joins, not for modeling:

- `subject_id`
- `subject_name`
- `class_id`
- `class_name`
- `sample_id`
- `sample_name`
- `source_dir`
- `sampling_rate_hz`
- `n_samples`
- `duration_s`

Feature metadata fields to generate later:

- `feature_name`
- `feature_set`: `domain_informed` or `generic_automated`
- `modality`
- `source_channel`
- `method`
- `library`
- `definition`
- `units`
- `window_suitability`
- `limitations`
- `enters_main_analysis`
- `quality_gated`
- `missing_policy`

## Required Tests for Later Pipeline

The extraction implementation must include tests or validation scripts confirming:

1. All 8,700 samples are represented exactly once.
2. Subject and label mappings match `starting_point/samples.csv`.
3. Output contains no duplicate sample rows.
4. Output feature values contain no NaN/Inf, or every exception is explicitly recorded with a reviewed missing/exclusion policy.
5. No `temperature`, `emg_corrugator`, or `emg_zygomaticus` feature appears.
6. No leakage field appears in the model feature matrix: `subject_id`, `subject_name`, `sample_id`, `sample_name`, `class_id`, `class_name`, `trial_number`, file paths, or filename sequence values.
7. Running the extractor twice on the same inputs produces identical feature values and metadata.
8. Feature metadata contains every output feature exactly once.

## Review Questions Before Implementation

1. Approve the conservative ECG policy: include short-window rate/count/quality/amplitude summaries, exclude traditional HRV and detailed morphology for now.
2. Approve the EDA policy: include level/slope/AUC plus quality-gated SCR count/amplitude, exclude EDA sympathetic spectral indices and full recovery features.
3. Approve the EMG policy: include time-domain amplitude/energy/waveform/activation features, exclude frequency fatigue and nonlinear features.
4. Approve the tsfresh supplement as a 36-feature custom set, kept separate from the 36-feature `domain_informed` set.
