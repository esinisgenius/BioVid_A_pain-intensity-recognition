# BioVid Part A Execution Plan

Scope: This plan follows `AGENTS.md` and keeps the project focused on physiological contribution heterogeneity, pain-level dependency, cross-subject stability, and reproducibility. It does not change the research questions or reposition the project as a global modality-ranking study.

## Phase 1 - Audit

Status: completed in this round.

Goal:

- Verify local repository documentation and physiological data structure before modeling.
- Confirm labels, subjects, modalities/channels, file coverage, sampling rates, trial structure, and leakage risks.

New/modified files:

- `reports/data_audit.md`
- `reports/execution_plan.md`

Dependencies:

- `AGENTS.md`
- `starting_point/Readme.txt`
- `starting_point/samples.csv`
- `biosignals_raw/`
- `biosignals_filtered/`
- `temperature/`

Validation:

- Full coverage check of all 8,700 expected samples against raw, filtered, and temperature directories.
- Full schema and row-count check for all CSV files.
- Label and subject consistency checks from `samples.csv`.
- Timestamp/sampling checks.
- NaN/Inf/empty-file checks.

Outputs:

- Reproducible written audit.
- Leakage-risk inventory.
- Review items before Phase 2.

Checkpoint for review:

- Completed on 2026-09-08: use filtered biomedical signals as primary input; keep raw only for robustness; do not include temperature in the main feature table; exclude all-zero `emg_corrugator` and `emg_zygomaticus`.

## Phase 2 - Clean Feature Pipeline

Goal:

- Build a deterministic physiological feature table with one row per trial and explicit metadata columns.
- Keep identity/label/path fields separate from model features.
- Document every feature by modality and definition.

Expected new/modified files:

- `src/data/load_samples.py`
- `src/data/validate_dataset.py`
- `src/features/extract_trial_features.py`
- `src/features/feature_definitions.py`
- `configs/feature_pipeline.yaml`
- `artifacts/feature_table_filtered.parquet` or `.csv`
- `artifacts/feature_manifest.json`
- `artifacts/exclusions.json`
- `tests/test_label_mapping.py`
- `tests/test_feature_dimensions.py`
- `tests/test_no_identity_features.py`

Dependencies:

- Phase 1 review decisions.
- Confirmed `samples.csv` schema.
- Confirmed CSV signal schemas.
- Primary input: `biosignals_filtered`.
- Robustness-only input: `biosignals_raw`.
- Excluded from the main feature table: `temperature`, `emg_corrugator`, `emg_zygomaticus`.

Necessary tests/validation:

- `class_id` to `class_name` mapping is exactly `{0: BL1, 1: PA1, 2: PA2, 3: PA3, 4: PA4}`.
- Every feature row maps to exactly one `sample_name`.
- Every subject/class cell has 20 rows before exclusions.
- Exclusions are explicit, not silent.
- Feature columns contain no `subject_id`, `subject_name`, `sample_id`, `sample_name`, file paths, class labels, or trial numbers.
- Feature extraction is deterministic across repeated runs.

Outputs:

- A reproducible feature table: `subject -> trial/window -> pain label -> modality -> features`.
- Feature manifest with modality, definition, units when known, domain category, and preprocessing source.
- Exclusion record documenting `temperature` as out of main-table scope and `emg_corrugator` / `emg_zygomaticus` as all-zero channels.

Checkpoint for review:

- Review feature list and exclusion log before any baseline modeling.
- Decide whether temperature features enter the main feature table or a separate auxiliary table.

## Phase 3 - Subject-Independent Baseline

Goal:

- Establish clean grouped/LOSO baselines only after the feature table is validated.
- Use models as analysis tools, not as leaderboard optimization.

Expected new/modified files:

- `src/evaluation/splits.py`
- `src/models/baselines.py`
- `src/evaluation/metrics.py`
- `configs/baseline_grouped_cv.yaml`
- `configs/baseline_loso.yaml`
- `experiments/baselines/`
- `reports/baseline_results.md`
- `tests/test_subject_split_integrity.py`
- `tests/test_preprocessing_fit_scope.py`

Dependencies:

- Phase 2 feature table and manifest.
- Review approval that labels, subjects, and leakage controls are settled.

Necessary tests/validation:

- No subject appears in both train and test.
- Preprocessing is fit only on training subjects within each fold.
- Results include macro F1, balanced accuracy, per-class recall, per-class precision, and confusion matrix.
- All reported numbers include task definition, labels, split protocol, model, feature set, and averaging method.

Outputs:

- Subject-independent baseline metrics.
- Per-pain-level baseline performance.
- Machine-readable experiment configs and metrics.

Checkpoint for review:

- Confirm baseline protocol for subsequent modality and attribution work: GroupKFold for iteration and LOSO or repeated grouped CV for final analysis.

## Phase 4 - Modality Contribution Analysis

Goal:

- Quantify modality contribution by pain level without claiming a universal best modality.
- Compare unimodal, multimodal, and leave-one-modality-out settings.

Expected new/modified files:

- `src/analysis/modality_ablation.py`
- `configs/modality_ablation.yaml`
- `experiments/modality_ablation/`
- `reports/modality_contribution.md`

Dependencies:

- Phase 3 baseline protocol and preprocessing pipeline.
- Finalized modality inclusion list from Phase 2.

Necessary tests/validation:

- Same folds and preprocessing rules across full and ablated variants.
- Report `Delta performance(modality removal | pain level)`, especially per-class recall deltas.
- Bootstrap or fold-level uncertainty for ablation effects where feasible.

Outputs:

- Modality contribution tables by pain level.
- Confusion matrices and pain-level ablation figures.

Checkpoint for review:

- Review whether observed modality effects are stable enough to justify feature-level attribution.

## Phase 5 - Feature Attribution

Goal:

- Explain which features within each modality drive model decisions at different pain levels.
- Use at least two complementary attribution methods if computationally feasible.

Expected new/modified files:

- `src/explainability/permutation_importance.py`
- `src/explainability/linear_coefficients.py`
- `src/explainability/shap_analysis.py` if feasible and justified
- `configs/feature_attribution.yaml`
- `experiments/feature_attribution/`
- `reports/feature_pain_level_analysis.md`

Dependencies:

- Phase 3/4 finalized models and folds.
- Stable feature manifest from Phase 2.

Necessary tests/validation:

- Attribution is computed using held-out subjects or fold-consistent test data where appropriate.
- Attributions are aggregated hierarchically: modality -> feature -> pain level -> subject stability.
- No raw SHAP magnitude comparisons across incompatible models without explicit caveat.

Outputs:

- Feature importance/attribution by pain level.
- Feature x pain-level heatmaps.
- Machine-readable attribution records.

Checkpoint for review:

- Decide which attribution results are credible enough for cross-subject stability analysis.

## Phase 6 - Cross-Subject Stability

Goal:

- Quantify whether population-level modality/feature importance generalizes across held-out subjects.
- Separate mean predictive contribution from consistency/stability.

Expected new/modified files:

- `src/analysis/subject_stability.py`
- `configs/subject_stability.yaml`
- `experiments/subject_stability/`
- `reports/subject_stability.md`

Dependencies:

- Phase 5 attribution records.
- Subject-independent evaluation artifacts.

Necessary tests/validation:

- Per-subject importance is derived without training on that subject in the relevant fold.
- Stability metrics include candidates such as top-k frequency, rank agreement, variance/IQR, coefficient of variation, and proportion of positive contribution.
- Bootstrap confidence intervals where feasible.
- Subjects with insufficient valid samples are flagged rather than overinterpreted.

Outputs:

- Mean importance vs stability scatter.
- Feature consistency tables.
- Subject x modality contribution heatmap.
- Categorization into high/low average importance and high/low stability groups.

Checkpoint for review:

- Review proposed stability thresholds and whether categorical labels are justified by uncertainty.

## Phase 7 - Joint Pain x Subject Heterogeneity

Goal:

- Summarize whether the same physiological features remain important for the same pain intensities across subjects.
- Avoid producing thousands of uninterpretable plots.

Expected new/modified files:

- `src/analysis/joint_heterogeneity.py`
- `configs/joint_heterogeneity.yaml`
- `experiments/joint_heterogeneity/`
- `reports/joint_heterogeneity.md`

Dependencies:

- Phase 6 stability outputs.
- Reviewed feature and modality attribution summaries.

Necessary tests/validation:

- Joint summaries are traceable to subject-independent attribution records.
- Any subject clustering is explicitly exploratory.
- No demographic subgroup interpretation unless separately reviewed and statistically justified.

Outputs:

- Pattern summaries: globally important and stable; globally important but subject-dependent; pain-level-specific and stable; pain-level-specific and subject-dependent.
- Compact figures suitable for research reporting.

Checkpoint for review:

- Decide which joint patterns are robust enough for the final research narrative.

## Phase 8 - Research Synthesis

Goal:

- Write the final research summary while preserving the `AGENTS.md` guardrails.
- Separate robust findings, tentative findings, negative results, limitations, and possible paper contribution.

Expected new/modified files:

- `reports/research_findings.md`
- `reports/figures/`
- Final machine-readable experiment index under `experiments/`

Dependencies:

- Completed and reviewed Phase 3 through Phase 7 outputs.

Necessary tests/validation:

- Every claim links back to a reproducible artifact.
- No claims of causal biomarkers, clinical diagnostic validity, deployment readiness, or universal superiority.
- Literature comparisons are only made when task definition and evaluation protocol are matched.

Outputs:

- Final research summary.
- Short list of robust, tentative, and negative findings.
- Paper narrative centered on heterogeneity, stability, and generalizability.

Checkpoint for review:

- Review final claims before any paper-style writing or submission-oriented polishing.

## Immediate Recommendation

The next best step is Phase 2 implementation:

1. Build a filtered-only feature pipeline for `gsr`, `ecg`, and `emg_trapezius`.
2. Keep labels, subject identifiers, paths, and trial numbers as metadata only.
3. Write an explicit exclusion record for `temperature`, `emg_corrugator`, and `emg_zygomaticus`.
4. Add tests for label mapping, subject split integrity, feature dimensions, deterministic extraction, and identity-feature exclusion.
