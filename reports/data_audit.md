# BioVid Part A Physiological Data Audit

Audit date: 2026-09-08

Scope: Phase 1 only. This audit inspected the local repository documentation and the local BioVid Part A physiological data. No facial/video data, modeling, feature extraction, SHAP, or large experiments were used.

## Authoritative Local Sources

- Project research/execution rules: `AGENTS.md`
- Dataset readme: `starting_point/Readme.txt`
- Sample index and labels: `starting_point/samples.csv`
- Biomedical signals, raw: `biosignals_raw/<subject_name>/<sample_name>_bio.csv`
- Biomedical signals, filtered: `biosignals_filtered/<subject_name>/<sample_name>_bio.csv`
- Thermal stimulator temperature series: `temperature/<subject_name>/<sample_name>_temp.csv`
- Existing design note: `docs/superpowers/specs/2026-06-04-biovid-feature-interaction-design.md`

The repository also contains zipped copies of the three data directories and agreement PDFs. The audit used the extracted local CSV files and `starting_point/samples.csv`.

## Repository Structure

Top-level contents relevant to this project:

- `AGENTS.md`
- `starting_point/`
  - `Readme.txt`
  - `samples.csv`
  - `Agreement.pdf`
- `biosignals_raw/`
- `biosignals_filtered/`
- `temperature/`
- `docs/superpowers/specs/2026-06-04-biovid-feature-interaction-design.md`

No source-code pipeline structure such as `src/`, `configs/`, `tests/`, `artifacts/`, or `experiments/` is currently present.

## Sample Index and Labels

`starting_point/samples.csv` is tab-delimited with columns:

```text
subject_id, subject_name, class_id, class_name, sample_id, sample_name
```

Confirmed totals:

- Samples: 8,700
- Subjects: 87
- Samples per subject: 100
- Classes per subject: 5
- Trials per subject per class: 20
- Dataset balance: exactly balanced by class and by subject/class cell

Confirmed label mapping from local `Readme.txt` and `samples.csv`:

| class_id | class_name | Meaning |
|---:|---|---|
| 0 | BL1 | baseline / no pain |
| 1 | PA1 | lowest pain stimulus intensity |
| 2 | PA2 | pain stimulus intensity level 2 |
| 3 | PA3 | pain stimulus intensity level 3 |
| 4 | PA4 | highest pain stimulus intensity |

`sample_id` is globally unique and monotonic from 1 to 8700. `sample_name` is also unique.

## Subjects

Subject identifiers are recoverable from `subject_id` and `subject_name`.

- `subject_id`: integers 1 to 87
- `subject_name`: strings matching `NNNNNN_<sex>_<age>`, for example `071309_w_21`
- Parsed sex markers in `subject_name`: 44 `m`, 43 `w`
- Parsed ages in `subject_name`: min 20, max 65, mean about 41.2

Subject keys:

```text
1:071309_w_21, 2:071313_m_41, 3:071614_m_20, 4:071709_w_23, 5:071814_w_23, 6:071911_w_24, 7:072414_m_23, 8:072514_m_27, 9:072609_w_23, 10:072714_m_23, 11:073109_w_28, 12:073114_m_25, 13:080209_w_26, 14:080309_m_29, 15:080314_w_25, 16:080609_w_27, 17:080614_m_24, 18:080709_m_24, 19:080714_m_23, 20:081014_w_27, 21:081609_w_40, 22:081617_m_27, 23:081714_m_36, 24:082014_w_24, 25:082109_m_53, 26:082208_w_45, 27:082315_w_60, 28:082414_m_64, 29:082714_m_22, 30:082809_m_26, 31:082814_w_46, 32:082909_m_47, 33:083009_w_42, 34:083013_w_47, 35:083109_m_60, 36:083114_w_55, 37:091809_w_43, 38:091814_m_37, 39:091914_m_46, 40:092009_m_54, 41:092014_m_56, 42:092509_w_51, 43:092514_m_50, 44:092714_m_64, 45:092808_m_51, 46:092813_w_24, 47:100117_w_36, 48:100214_m_50, 49:100417_m_44, 50:100509_w_43, 51:100514_w_51, 52:100909_w_65, 53:100914_m_39, 54:101015_w_43, 55:101114_w_37, 56:101209_w_61, 57:101216_m_40, 58:101309_m_48, 59:101514_w_36, 60:101609_m_36, 61:101809_m_59, 62:101814_m_58, 63:101908_m_61, 64:101916_m_40, 65:102008_w_22, 66:102214_w_36, 67:102309_m_61, 68:102316_w_50, 69:102414_w_58, 70:102514_w_40, 71:110614_m_42, 72:110810_m_62, 73:110909_m_29, 74:111313_m_64, 75:111409_w_63, 76:111609_m_65, 77:111914_w_63, 78:112009_w_43, 79:112016_m_25, 80:112209_m_51, 81:112310_m_20, 82:112610_w_60, 83:112809_w_23, 84:112909_w_20, 85:112914_w_51, 86:120514_w_56, 87:120614_w_61
```

Demographic use is not assumed. Per `AGENTS.md`, sex/age should only be used later if statistically justified and reviewed.

## File Naming and Trial Structure

All sample names match:

```text
<subject_name>-<class_name>-<trial_number>
```

Biomedical signal files:

```text
biosignals_raw/<subject_name>/<sample_name>_bio.csv
biosignals_filtered/<subject_name>/<sample_name>_bio.csv
```

Temperature files:

```text
temperature/<subject_name>/<sample_name>_temp.csv
```

Confirmed trial numbering:

- BL1 files use trial numbers 081 to 100.
- PA1, PA2, PA3, and PA4 files use 20 trial numbers each drawn from 001 to 080.
- Pain trial numbers are reused across pain classes. Therefore `trial_number` alone is not a globally unique trial key.
- The safe trial key is `sample_name`, or the tuple `(subject_id, subject_name, class_id, class_name, trial_number)`.

## Available Physiological Modalities and Channels

Biomedical CSV schema is identical in raw and filtered directories:

```text
time, gsr, ecg, emg_trapezius, emg_corrugator, emg_zygomaticus
```

Temperature CSV schema:

```text
time, temperature
```

Confirmed usable physiological channels from local data:

- GSR/EDA: `gsr`
- ECG: `ecg`
- EMG trapezius: `emg_trapezius`
- Thermal stimulator temperature: `temperature`

Channels present but all-zero in every biomedical CSV:

- `emg_corrugator`
- `emg_zygomaticus`

These all-zero channels should not be treated as valid discriminative physiological signals without an explicit later decision. They may be retained only as documented missing/unusable channels or dropped with a recorded exclusion reason.

## Raw, Filtered, and Temperature Coverage

Each extracted data directory has 87 subject folders and 8,700 CSV files matching `samples.csv`.

| Directory | CSV files | Subject dirs | Missing samples | Extra samples | Non-CSV files |
|---|---:|---:|---:|---:|---|
| `biosignals_raw` | 8,700 | 87 | 0 | 0 | `.DS_Store` |
| `biosignals_filtered` | 8,700 | 87 | 0 | 0 | `.DS_Store` |
| `temperature` | 8,700 | 87 | 0 | 0 | `.DS_Store` |

Raw and filtered biomedical files have identical timestamps for all matched samples.

## Sampling Rate and Window Duration

Local `Readme.txt` states:

- `biosignals_raw`: 512 Hz, timestamps in microseconds relative to the first frame of the corresponding Pike video.
- `biosignals_filtered`: 512 Hz, timestamps in microseconds relative to the first frame of the corresponding Pike video.
- `temperature`: thermal stimulator temperature time series.

Confirmed from CSV files:

| Data source | Data rows per file | Timestamp interval | Effective sampling rate | Span from first to last sample | Approx. trial/window duration |
|---|---:|---:|---:|---:|---:|
| `biosignals_raw` | 2,816 | 1,953 or 1,954 microseconds | 512 Hz | 5,498,046 or 5,498,047 microseconds | 5.5 seconds |
| `biosignals_filtered` | 2,816 | 1,953 or 1,954 microseconds | 512 Hz | 5,498,046 or 5,498,047 microseconds | 5.5 seconds |
| `temperature` | 176 | 31,250 microseconds | 32 Hz | 5,468,750 microseconds | 5.5 seconds |

The timestamp interval variation in biomedical files is consistent with integer microsecond representation of 512 Hz sampling.

## Missing, Corrupted, or Inconsistent Data

Confirmed by full-file audit:

- No missing sample files in raw biomedical signals.
- No missing sample files in filtered biomedical signals.
- No missing sample files in temperature.
- No extra sample CSVs beyond `samples.csv`.
- No duplicate `sample_name`.
- No duplicate `sample_id`.
- No empty CSV files.
- No NaN values.
- No infinite values.
- All biomedical files have exactly 2,816 data rows.
- All temperature files have exactly 176 data rows.
- All raw and filtered biomedical files share the same column names.
- Raw and filtered biomedical timestamps match for all 8,700 samples.

Known benign clutter:

- `.DS_Store` appears in `biosignals_raw/`, `biosignals_filtered/`, and `temperature/`; future file discovery must ignore non-CSV files.

Confirmed abnormal/unusable channels:

- `emg_corrugator` is all zero in raw and filtered data.
- `emg_zygomaticus` is all zero in raw and filtered data.

## Preprocessing State

Local `Readme.txt` states:

- `biosignals_raw`: biomedical signals without preprocessing.
- `biosignals_filtered`: biomedical signals after preprocessing, with details described in Werner et al. 2014.

The local repository does not include a detailed local preprocessing recipe beyond the reference in `Readme.txt`. Therefore:

- Filtered signals are available and complete.
- Raw signals are available and complete.
- The exact filtering operations should be treated as externally referenced and not fully specified by local files.
- Main Phase 2 feature extraction should start from `biosignals_filtered` only if that choice is explicitly documented.
- Any robustness check using `biosignals_raw` should be separate and not silently mixed with filtered features.

## Balance and Repeated Trials

The label distribution is exactly balanced:

| class_name | class_id | samples | samples per subject |
|---|---:|---:|---:|
| BL1 | 0 | 1,740 | 20 |
| PA1 | 1 | 1,740 | 20 |
| PA2 | 2 | 1,740 | 20 |
| PA3 | 3 | 1,740 | 20 |
| PA4 | 4 | 1,740 | 20 |

Repeated trials exist by design: each subject has 20 samples per class.

## Leakage Risks

Subject leakage risks:

- Random sample/window splits would leak subject identity because each subject contributes 100 trials.
- `subject_id`, `subject_name`, parsed sex, parsed age, and the prefix embedded in `subject_name` must not be used as model features.
- Subject-derived metadata may only be used for grouping, audit, and carefully reviewed exploratory analysis.

Trial/index leakage risks:

- `sample_id` is globally ordered by subject and must not be used as a feature.
- `sample_name` embeds subject and class and must not be used as a model feature.
- `class_name` and `class_id` are labels and must not be present in feature columns.
- `trial_number` should not be used as a feature because BL1 uses 081-100 while pain files use 001-080, which can leak baseline vs pain.
- File paths encode subject and class and must not be transformed into features.

Preprocessing leakage risks:

- Scaling, imputation, feature selection, PCA, correlation clustering, outlier thresholds, model tuning, class balancing, and learned preprocessing must be fit inside each training fold only.
- Any feature clustering or interaction selection must be recomputed on training subjects only.
- If subject-level normalization is considered later, it must be reviewed carefully because test-subject calibration data may be unavailable or may change the scientific question.

Raw/filtered leakage or duplication risks:

- Raw and filtered biomedical files are paired views of the same trial. They should not be mixed as independent samples.
- If both are used in a robustness analysis, outputs must identify the source (`raw` vs `filtered`) and avoid duplicate-trial inflation.

## Phase 2 Input Decisions

Reviewed decisions recorded on 2026-09-08:

1. Use `biosignals_filtered` as the primary biomedical input for Phase 2.
2. Keep `biosignals_raw` only for later robustness checks.
3. Do not include `temperature` in the main Phase 2 feature table.
4. Formally exclude `emg_corrugator` and `emg_zygomaticus` from model features because both channels are all zero in the local raw and filtered data.

## Current Blocking or Review Items

No blocking ambiguity was found for labels, subject identity, or sample file coverage. Phase 2 can begin under the input decisions above.

Remaining guardrail:

- Parsed sex/age from `subject_name` should remain audit-only metadata and be excluded from modeling unless separately reviewed and statistically justified.
