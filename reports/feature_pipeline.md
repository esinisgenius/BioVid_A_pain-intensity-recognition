# Phase 2 Feature Extraction Pipeline

Run date: 2026-09-08

Status: completed. This phase generated a controlled sample-level feature table and validation artifacts only. No LOSO baseline, SHAP, feature ranking, modality ranking, or modeling was run.

## Inputs

Primary source:

- `biosignals_filtered/<subject_name>/<sample_name>_bio.csv`

Included channels:

- ECG: `ecg`
- EDA/GSR: `gsr`
- Trapezius EMG: `emg_trapezius`

Excluded from extraction:

- `temperature`
- `emg_corrugator`
- `emg_zygomaticus`
- `trial_number`
- filename sequence/index fields
- file paths as features

Metadata retained for grouping/audit only:

- `subject_id`
- `subject_name`
- `sample_id`
- `sample_name`
- `class_id`
- `class_name`
- `source_dir`
- `sampling_rate_hz`
- `n_samples`
- `duration_s`

These metadata columns are not model feature columns.

## Output Artifacts

- `artifacts/feature_table_filtered.parquet`: final retained 8700-row feature table.
- `artifacts/feature_table_filtered.csv`: CSV copy of the final retained feature table.
- `artifacts/feature_table_filtered_all_candidates.parquet`: candidate table before degeneracy filtering.
- `artifacts/feature_metadata.json`: machine-readable metadata for all retained features.
- `artifacts/feature_stats_all_candidates.csv`: per-feature statistics before filtering.
- `artifacts/feature_stats_filtered.csv`: per-feature statistics after filtering.
- `artifacts/feature_extraction_diagnostics.csv`: sample-level extraction diagnostics.
- `artifacts/feature_exclusions.csv`: feature-level exclusions from degeneracy filtering.
- `artifacts/feature_pipeline_summary.json`: machine-readable summary and reproducibility hash.

Implementation files:

- `configs/feature_pipeline.yaml`
- `src/data/load_samples.py`
- `src/features/feature_definitions.py`
- `src/features/extract_trial_features.py`
- `tests/test_phase2_features.py`

## Feature Provenance and Shared Features

The final table keeps two feature sets separate:

- `domain_informed`: physiology-informed primary features.
- `generic_automated`: explicitly configured tsfresh supplement.

Because several summaries overlap conceptually, provenance is also tracked:

- `domain_informed`: domain feature without close generic duplicate.
- `generic_automated`: generic feature without close domain duplicate.
- `shared`: highly overlapping domain/generic summary.

Shared features are retained with provenance rather than silently merged. Phase 3 must not treat `domain_informed` and `generic_automated` as completely independent evidence.

Final retained feature counts:

| grouping | count |
|---|---:|
| Total retained model features | 72 |
| `domain_informed` | 36 |
| `generic_automated` | 36 |
| provenance `domain_informed` | 23 |
| provenance `generic_automated` | 23 |
| provenance `shared` | 26 |

Counts by modality:

| modality | retained features |
|---|---:|
| ECG | 25 |
| EDA/GSR | 24 |
| Trapezius EMG | 23 |

## Interpretation Type

Feature metadata distinguishes:

- `physiological-derived feature`
- `generic signal/statistical summary`

Waveform summaries such as ECG RMS, ECG standard deviation, ECG peak-to-peak, EMG RMS, EMG standard deviation, EMG energy, and similar tsfresh summaries are marked as generic signal/statistical summaries. They should not be interpreted as physiological markers.

## ECG Extraction

Method:

- Beat-derived ECG features use the revised production pipeline selected after full-data Phase 2 robustness checking:
  - median-center the filtered ECG trial;
  - apply a fixed 5-20 Hz Butterworth bandpass;
  - detect R-peaks with NeuroKit2 `ecg_peaks(method="neurokit")`;
  - derive HR/RR features from detected R-peak intervals.
- ECG waveform-summary features remain based on NeuroKit2-cleaned filtered ECG, not the 5-20 Hz bandpassed signal.
- Samples with fewer than 3 detected R-peaks are flagged as R-peak detection failures.

ECG quality summary:

- Beat pipeline processing success rate: 100%.
- R-peak detection success rate: 100%.
- Failed R-peak samples: 0 of 8700.
- R-peak count range: 3 to 10.
- ECG quality-score success rate: 99.94%; quality-score failures are recorded separately and do not override successful R-peak detection.

The final feature table contains no ECG beat-derived missing values or R-peak failures. Any future quality-score handling should treat `ecg_quality_mean` as a signal-quality covariate, not as a physiological marker.

### ECG R-Peak Failure Sanity Check

Additional Phase 2 sanity artifacts:

- `artifacts/ecg_rpeak_failure_audit.csv`
- `artifacts/ecg_rpeak_failure_robustness_audit.csv`
- `artifacts/ecg_rpeak_failure_robustness_summary.csv`

Before the full-data robustness comparison, the original production pipeline A had 5 failed ECG R-peak detection trials. All 5 belonged to subject `79` / `112016_m_25`. Pipeline A found 0 R-peaks for each failed filtered ECG trial. Running the same A detector on the corresponding raw ECG also found 0 R-peaks.

| subject_id | subject_name | sample_id | sample_name | class | source | R-peaks | ECG min | ECG max | ECG std | peak-to-peak | likely cause |
|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 79 | `112016_m_25` | 7826 | `112016_m_25-PA1-023` | PA1 | filtered | 0 | -290.19 | 2104.53 | 318.26 | 2394.72 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7826 | `112016_m_25-PA1-023` | PA1 | raw | 0 | 22697.10 | 25202.80 | 321.31 | 2505.70 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7863 | `112016_m_25-PA3-012` | PA3 | filtered | 0 | -333.59 | 2163.10 | 341.34 | 2496.70 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7863 | `112016_m_25-PA3-012` | PA3 | raw | 0 | 23945.60 | 26362.90 | 322.52 | 2417.30 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7865 | `112016_m_25-PA3-018` | PA3 | filtered | 0 | -318.28 | 2043.29 | 286.96 | 2361.57 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7865 | `112016_m_25-PA3-018` | PA3 | raw | 0 | 22909.60 | 25257.00 | 285.91 | 2347.40 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7878 | `112016_m_25-PA3-065` | PA3 | filtered | 0 | -320.85 | 2165.49 | 308.92 | 2486.34 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7878 | `112016_m_25-PA3-065` | PA3 | raw | 0 | 23687.60 | 26434.70 | 358.02 | 2747.10 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7883 | `112016_m_25-PA4-009` | PA4 | filtered | 0 | -395.65 | 2002.65 | 318.46 | 2398.30 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |
| 79 | `112016_m_25` | 7883 | `112016_m_25-PA4-009` | PA4 | raw | 0 | 24285.20 | 26758.60 | 331.01 | 2473.40 | detector/parameter failure on noisy or atypical ECG; raw signal still has nonzero amplitude |

Interpretation:

- These ECG windows are not empty, all-zero, or near-flat; raw ECG has substantial amplitude variation and filtered ECG has comparable short-window variability.
- Raw ECG shows a large positive DC offset in these trials, while filtered ECG is centered differently. The detector still fails in both views.
- This evidence pointed more toward detector/parameter sensitivity on atypical or noisy ECG morphology than toward a simple corrupted/empty data file.
- These trials were not deleted.

### ECG R-Peak Robustness Check for Subject 79

Scope:

- Only the 5 failed trials from subject `79` / `112016_m_25` were checked.
- The goal was to test whether failure is recoverable with reasonable preprocessing, not to tune a detector specifically for these 5 samples.
- No model, ranking, or baseline result was changed by this check.

Compared schemes:

1. Current NeuroKit2 `ecg_process` on original signal.
2. Current NeuroKit2 `ecg_process` after median centering / DC removal.
3. Current NeuroKit2 `ecg_process` after linear detrending.
4. Fixed 5-20 Hz Butterworth bandpass followed by NeuroKit2 `ecg_peaks` with several common detectors: `neurokit`, `pantompkins1985`, `hamilton2002`, and `elgendi2010`.

Summary across the 5 failed trials:

| source | preprocessing | detector | recovered plausible trials | R-peak count range | detector errors |
|---|---|---|---:|---:|---:|
| filtered | original | current `ecg_process` | 0 / 5 | 0-0 | 5 |
| filtered | median center | current `ecg_process` | 0 / 5 | 0-0 | 5 |
| filtered | linear detrend | current `ecg_process` | 0 / 5 | 0-0 | 5 |
| filtered | 5-20 Hz bandpass | `neurokit` | 5 / 5 | 3-4 | 0 |
| filtered | 5-20 Hz bandpass | `pantompkins1985` | 5 / 5 | 3-4 | 0 |
| filtered | 5-20 Hz bandpass | `hamilton2002` | 5 / 5 | 3-4 | 0 |
| filtered | 5-20 Hz bandpass | `elgendi2010` | 5 / 5 | 4-4 | 0 |
| raw | original | current `ecg_process` | 0 / 5 | 0-0 | 5 |
| raw | median center | current `ecg_process` | 0 / 5 | 0-0 | 5 |
| raw | linear detrend | current `ecg_process` | 0 / 5 | 0-0 | 5 |
| raw | 5-20 Hz bandpass | `neurokit` | 5 / 5 | 3-4 | 0 |
| raw | 5-20 Hz bandpass | `pantompkins1985` | 5 / 5 | 3-4 | 0 |
| raw | 5-20 Hz bandpass | `hamilton2002` | 5 / 5 | 3-4 | 0 |
| raw | 5-20 Hz bandpass | `elgendi2010` | 5 / 5 | 4-4 | 0 |

Interpretation:

- The failures are recoverable after a standard fixed ECG bandpass, for both filtered and raw files.
- DC removal or linear detrending alone does not fix the current detector failure.
- Raw and filtered bandpass results are highly similar, suggesting the underlying ECG-like rhythm is present rather than absent.
- Some detector outputs may include suspicious short RR intervals in individual trials, so this is not enough to silently patch the feature table.
- The evidence now points to detector/preprocessing sensitivity rather than unrecoverable data corruption.

### Full-Data ECG Pipeline Comparison

Additional Phase 2 artifacts:

- `artifacts/ecg_pipeline_comparison.csv`
- `artifacts/ecg_pipeline_comparison_summary.json`
- `artifacts/ecg_pipeline_comparison_by_subject.csv`

Compared pipelines:

- A: original production pipeline, NeuroKit2 `ecg_process` on the filtered ECG trial.
- B: median-center filtered ECG, apply fixed 5-20 Hz bandpass, then use NeuroKit2 `ecg_peaks(method="neurokit")` as the main detector.

This comparison used all 8700 ECG trials. Pain labels were not used to choose the pipeline, and no classification accuracy was computed.

Detection summary:

| metric | A | B |
|---|---:|---:|
| R-peak detection success rate | 99.94% | 100.00% |
| Failed trials | 5 | 0 |
| R-peak count min | 0 | 3 |
| R-peak count median | 6 | 6 |
| R-peak count 95th percentile | 8 | 8 |
| R-peak count 99th percentile | 9 | 9 |
| R-peak count max | 10 | 10 |
| HR mean abnormal proportion | 0.057% | 0.000% |
| HR min abnormal proportion | 0.057% | 0.000% |
| HR max abnormal proportion | 0.057% | 0.000% |
| RR mean abnormal proportion | 0.057% | 0.000% |
| RR median abnormal proportion | 0.057% | 0.000% |

A/B consistency among the 8695 trials where A and B both succeeded:

| metric | value |
|---|---:|
| Exact R-peak count agreement | 96.39% |
| R-peak count agreement within 1 peak | 99.99% |
| Mean B - A R-peak count difference | 0.0008 |
| Mean absolute R-peak count difference | 0.0362 |

HR/RR feature correlations among trials where both A and B succeeded:

| feature | Pearson r |
|---|---:|
| R-peak count | 0.9834 |
| Mean HR | 0.9976 |
| Median HR | 0.9986 |
| Minimum HR | 0.9993 |
| Maximum HR | 0.9528 |
| HR range | 0.9098 |
| HR slope | 0.9506 |
| Mean RR | 0.9970 |
| Median RR | 0.9975 |

Subject-level failure distribution:

- Under A, only subject `79` / `112016_m_25` had R-peak failures: 5 of 100 trials.
- Under B, no subject had R-peak failures.
- Subject 79's 5 original failures all recovered under B.
- No trial that succeeded under A failed under B.

B false-detection/extreme-value check:

- B introduced no HR/RR plausibility failures under the configured audit thresholds.
- B's R-peak count range was 3-10, matching the expected short-window range for these 5.5 second recordings.
- The B count distribution closely matched A for the already successful trials.

Waveform-summary check:

- 5-20 Hz bandpass materially changes ECG waveform amplitude summaries.
- Correlation between original cleaned waveform summaries and bandpassed summaries was moderate to high: std `r=0.8095`, RMS `r=0.7996`, peak-to-peak `r=0.9035`.
- Median B/A amplitude-summary ratios were about 0.68 for std/RMS/peak-to-peak.
- Therefore B was adopted only for beat-derived ECG features, not for ECG waveform-summary features.

Decision:

- B replaces A as the unified ECG beat-derived feature pipeline.
- The sample-level feature table and metadata were regenerated after this decision.
- ECG waveform-summary features were not switched to bandpassed-waveform summaries.

## EDA/GSR Extraction

Method:

- Direct filtered GSR summaries for SCL level/change features.
- NeuroKit2 `eda_process` for SCR candidates.
- SCR amplitudes are retained only when:

```text
SCR_Amplitude >= max(0.01, 0.01 * window_gsr_range)
```

This threshold is computed within each trial only. It uses no pain labels, no subject labels, and no dataset-wide statistics.

EDA SCR validity audit:

| feature | zero proportion | non-zero coverage | unique values | min | max | retained |
|---|---:|---:|---:|---:|---:|---|
| `eda_scr_count` | 0.5310 | 0.4690 | 5 | 0.0 | 4.0 | yes |
| `eda_scr_amplitude_max` | 0.5310 | 0.4690 | 4081 | 0.0 | 2.4534 | yes |
| `eda_scr_amplitude_sum` | 0.5310 | 0.4690 | 4081 | 0.0 | 3.2635 | yes |

SCR detection processing success rate was 95.49%. The retained SCR features did not degenerate into constants or near-all-zero features under the configured policy, so they remain in the final feature table with quality-gated metadata.

## Trapezius EMG Extraction

Time-domain features:

- MAV
- RMS
- standard deviation
- integrated EMG
- energy
- peak-to-peak amplitude
- waveform length
- zero crossings

Activation threshold:

```text
envelope = moving_average(abs(emg_trapezius), window=50 ms)
threshold = median(envelope) + 3 * MAD(envelope)
active = envelope > threshold
minimum retained activation segment = 50 ms
```

This is computed independently within each trial. It does not use pain labels, subject labels, class labels, or dataset-wide statistics, so it does not introduce cross-subject leakage.

Activation features:

- `emg_trapezius_activation_fraction`
- `emg_trapezius_activation_count`
- `emg_trapezius_activation_mean_duration_s`

### EMG Activation Sanity Check

Additional Phase 2 sanity artifacts:

- `artifacts/emg_activation_sanity_by_class.csv`
- `artifacts/emg_activation_sanity_sample.csv`

Meaning of the threshold:

- The filtered trapezius EMG is rectified with `abs(emg_trapezius)`.
- The rectified signal is smoothed with a 50 ms moving average to form a short-window amplitude envelope.
- The threshold is the within-trial median envelope plus three within-trial median absolute deviations.
- This makes activation a relative burst indicator within each trial. It is not an absolute EMG amplitude measure and should not be interpreted as absolute muscle force or absolute physiological activation.

Class-level distribution:

| class | activation fraction mean | fraction median | fraction min-max | count mean | count median | count min-max | mean duration mean (s) | duration median (s) | duration min-max (s) |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| BL1 | 0.1817 | 0.1658 | 0.0000-0.4982 | 9.49 | 10 | 0-28 | 0.0957 | 0.0855 | 0.0000-1.1738 |
| PA1 | 0.1775 | 0.1575 | 0.0000-0.4972 | 9.15 | 9 | 0-27 | 0.0954 | 0.0858 | 0.0000-1.3516 |
| PA2 | 0.1790 | 0.1566 | 0.0000-0.4964 | 9.17 | 9 | 0-26 | 0.0961 | 0.0856 | 0.0000-0.9824 |
| PA3 | 0.1752 | 0.1626 | 0.0000-0.4972 | 8.98 | 9 | 0-25 | 0.0985 | 0.0880 | 0.0000-1.2090 |
| PA4 | 0.1676 | 0.1539 | 0.0000-0.4972 | 8.38 | 8 | 0-28 | 0.1039 | 0.0874 | 0.0000-1.6172 |

Global activation feature sanity:

| feature | zero proportion | min | max | unique values | sanity judgment |
|---|---:|---:|---:|---:|---|
| `emg_trapezius_activation_fraction` | 0.0832 | 0.0000 | 0.4982 | 1293 | Not degenerate; not mostly zero; no all-window activation. |
| `emg_trapezius_activation_count` | 0.0832 | 0.0000 | 28.0000 | 29 | Not degenerate; broad but bounded event-count range. |
| `emg_trapezius_activation_mean_duration_s` | 0.0832 | 0.0000 | 1.6172 | 3144 | Not degenerate; zero only when no activation segment is detected. |

Conclusion:

- The activation threshold does not cause large-scale collapse to 0.
- The activation fraction does not saturate near 1; the maximum is about 0.50.
- These features represent relative burst activity within each trial, not absolute EMG amplitude.

## Generic Automated Supplement

The pipeline uses an explicit small set of tsfresh calculators. It does not use `ComprehensiveFCParameters`, `EfficientFCParameters`, or full-dataset label-based relevance filtering.

Per modality, 12 generic features were generated:

- mean
- median
- standard deviation
- minimum
- maximum
- root mean square
- absolute energy
- absolute sum of changes
- mean change
- linear trend slope
- skewness
- kurtosis

Total generic automated features: 36.

## Degeneracy Policy and Exclusions

Features are excluded if any of the following is true:

- missing proportion > 0
- unique value count <= 1
- zero proportion >= 0.995
- standard deviation <= 1e-12

Result:

- No final candidate feature was removed by this degeneracy policy.
- `artifacts/feature_exclusions.csv` contains only the header.
- No retained feature has a near-constant/degenerate warning in `artifacts/feature_stats_filtered.csv`.

Features excluded before extraction because of scope or methodological unsuitability remain documented in `reports/feature_spec.md`.

## Validation

Tests run:

```text
PYTHONPATH=. pytest -q
```

Result:

```text
5 passed
```

Validated:

- 8700 samples represented exactly once.
- Subject and label mapping matches `starting_point/samples.csv`.
- No duplicate sample rows.
- All model feature values are finite.
- No `temperature`, `emg_corrugator`, or `emg_zygomaticus` feature appears.
- No leakage feature appears in the model feature matrix.
- Retained feature metadata exactly matches feature table columns.
- Feature table CSV hash matches `feature_pipeline_summary.json`.

Current feature table CSV SHA-256:

```text
fcca0c940671d6ad8976da86d36e29277e37087a982df1403775b312551c07e1
```

## Phase 3 Guidance

Phase 3 can compare:

- `domain_informed` only;
- `generic_automated` only;
- combined features with shared provenance handled explicitly.

Because 26 retained features are marked `shared`, any comparison must be framed as a controlled feature-family comparison, not as independent physiological-vs-generic evidence.
