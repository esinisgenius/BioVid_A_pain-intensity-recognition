# BioVid Feature Interaction Study Design

## Research Aim

This study investigates interpretable physiological feature contributions for pain intensity prediction on BioVid Part A. The goal is not simply to compare modalities, but to identify which individual features and feature interactions are useful, stable, and clinically efficient under subject-independent evaluation.

Primary research question:

> Can stable feature interactions improve subject-independent ordinal pain intensity prediction while reducing the number of required modalities and features?

Secondary research questions:

- Which features contribute globally across all pain levels?
- Which features and interactions contribute to specific ordinal pain boundaries?
- Are within-modality interactions sufficient for strong performance, or do cross-modality interactions add meaningful value?
- What is the smallest interpretable feature/modality set whose performance is statistically close to the best model?

## Dataset Scope

Use BioVid Heat Pain Database Part A from this workspace.

Primary data:

- `starting_point/samples.csv`
- `biosignals_filtered/*/*_bio.csv`
- Optional later robustness check: `biosignals_raw/*/*_bio.csv`
- Optional auxiliary input: `temperature/*/*_temp.csv`

Labels:

- `class_id = 0`: BL1, baseline/no pain
- `class_id = 1`: PA1, lowest pain intensity
- `class_id = 2`: PA2
- `class_id = 3`: PA3
- `class_id = 4`: PA4, highest pain intensity

Initial scope:

- Use filtered biomedical signals first.
- Start with EDA/GSR and ECG.
- Add EMG and temperature only after the EDA/ECG pipeline is validated.

## Prediction Task

The primary task is ordinal pain intensity prediction:

```text
y in {0, 1, 2, 3, 4}
```

Use ordinal boundary decomposition for pain-level-specific analysis:

```text
Boundary 1: y >= 1    BL1 vs PA1-PA4
Boundary 2: y >= 2    BL1-PA1 vs PA2-PA4
Boundary 3: y >= 3    BL1-PA2 vs PA3-PA4
Boundary 4: y >= 4    BL1-PA3 vs PA4
```

The final prediction remains a five-level pain intensity prediction. The boundary models are used both for ordinal prediction and for interpreting which features matter at different pain transitions.

## Evaluation Protocol

Use subject-independent evaluation only. Random sample-level splits are disallowed because they leak subject identity.

Phase 1:

- Grouped 5-fold cross-validation by `subject_id`
- Used for fast iteration and model comparison

Phase 2:

- Leave-one-subject-out evaluation or repeated grouped CV
- Used only for final candidate models

Primary metrics:

- MAE, because ordinal distance matters
- Quadratic weighted kappa
- Balanced accuracy
- Macro-F1

Boundary-level metrics:

- AUC
- Balanced accuracy
- Macro-F1

Statistical comparison:

- Compare paired fold-level results between model variants.
- Use bootstrap confidence intervals or Wilcoxon signed-rank tests for performance deltas.
- Treat small non-significant differences as evidence for preferring the simpler clinical model.

## Feature Extraction

Each trial CSV becomes one row in a feature matrix.

EDA/GSR candidate features:

- Mean, standard deviation, min, max, range
- AUC
- Linear slope
- Peak count
- Peak amplitude statistics
- Recovery-related statistics when robustly detectable

ECG candidate features:

- Mean, standard deviation, range
- R-peak-derived heart rate features if peak detection is reliable
- RR mean
- RR standard deviation
- RMSSD
- SDNN
- Heart-rate change or slope

EMG candidate features, later phase:

- RMS
- Mean absolute value
- Variance
- Energy
- Waveform length
- Zero crossing count
- Peak-to-peak amplitude

Temperature candidate features, optional:

- Mean, max, min, range
- AUC
- Slope
- Duration above selected thresholds

All feature extraction rules must be deterministic and documented. Features that fail quality checks on many samples should be dropped rather than imputed aggressively.

## Redundancy Reduction By Feature Clustering

Feature clustering is used to reduce redundancy and constrain the interaction search space.

Primary clustering method:

- Standardize features within each training fold.
- Compute feature-feature Spearman correlations on training data only.
- Define distance as:

```text
distance(i, j) = 1 - abs(Spearman rho(i, j))
```

- Use hierarchical clustering with average linkage.
- Start with distance threshold `0.3`, corresponding roughly to `|rho| >= 0.7`.
- Run sensitivity checks at thresholds `0.2` and `0.4`.

Cluster representative:

- Use the cluster medoid as the default representative.
- The medoid is the feature with the smallest average distance to other features in the cluster.
- Avoid label-based representative selection in the main analysis to reduce feature selection bias.

Important leakage rule:

- Clustering must be recomputed inside each training fold.
- Test-subject data must never influence clustering, representative selection, interaction generation, or feature selection.

## Interaction Discovery

Interaction discovery is separated into three levels.

Level 1: Main effects

- Individual selected feature representatives only.
- No interaction terms.

Level 2: Within-modality interactions

- Pairwise interactions between representative features from the same modality.
- Clinical motivation: a single modality with internal interactions may be preferable to adding more sensors.

Level 3: Cross-modality interactions

- Pairwise interactions between representative features from different modalities.
- Scientific motivation: ECG, EMG, or temperature may add value only through specific combinations with EDA features.

Interaction generation rules:

- Generate interactions only among cluster representatives in the main analysis.
- Include original main effects whenever interactions are included.
- Standardize continuous features before multiplying interaction terms.
- Track whether each interaction is within-modality or cross-modality.

Example interaction families:

```text
EDA_feature_a x EDA_feature_b
ECG_feature_a x ECG_feature_b
EDA_feature_a x ECG_feature_b
EDA_feature_a x EMG_feature_b
ECG_feature_a x EMG_feature_b
```

## Model Families

Start with transparent models before moving to stronger black-box models.

Primary interpretable models:

- Elastic Net logistic/ordinal decomposition models with generated interaction terms
- Explainable Boosting Machine if available and stable in the local environment

Performance comparison models:

- Random Forest
- XGBoost or LightGBM if installable

The primary evidence should not depend on a single model family. A feature or interaction is more credible if it appears under both a sparse linear model and a tree/boosting model.

## Model Variants To Compare

Minimal initial EDA/ECG experiment:

```text
M1: EDA main effects
M2: EDA main effects + EDA within-modality interactions
M3: ECG main effects
M4: ECG main effects + ECG within-modality interactions
M5: EDA + ECG main effects
M6: EDA + ECG main effects + within-modality interactions
M7: EDA + ECG main effects + cross-modality interactions
M8: EDA + ECG main effects + within-modality + cross-modality interactions
```

Later expanded experiment:

```text
M9: All modalities main effects
M10: All modalities main effects + selected within-modality interactions
M11: All modalities main effects + selected cross-modality interactions
M12: Minimal selected clinical model
```

## Stability Criteria

A feature or interaction should not be called important based on one split.

Selection stability:

- In grouped 5-fold CV, a feature/interaction is stable if selected in at least 4 of 5 folds.
- In LOSO, a feature/interaction is stable if selected in at least 60% of folds.

Direction stability:

- For sparse linear models, coefficient signs should be consistent across selected folds.
- If signs frequently flip, report the feature as unstable even if selected often.

Performance stability:

- Adding an interaction family should improve mean performance and show consistent fold-level improvements.
- If performance is equivalent, prefer the simpler model with fewer modalities and fewer features.

## Pain-Level-Specific Contribution

Contribution must be reported globally and by ordinal boundary.

For each boundary:

```text
y >= 1
y >= 2
y >= 3
y >= 4
```

Report:

- Stable selected main effects
- Stable selected within-modality interactions
- Stable selected cross-modality interactions
- Boundary-specific performance gain over main effects
- Direction of effect when interpretable

Example target table:

```text
Boundary | Stable term | Type | Modality | Selection frequency | Direction | Delta AUC
y>=1     | EDA_AUC     | main | EDA      | 5/5                 | positive  | +0.03
y>=3     | EDA_x_ECG   | cross| EDA-ECG  | 4/5                 | positive  | +0.04
```

This is the main interpretability output of the study.

## Association Rule Analysis

Association rule mining is allowed only as an auxiliary interpretability layer.

Purpose:

- Translate stable model-discovered features or interactions into human-readable rules.

Allowed use:

- Restrict rule mining to stable selected features/interactions.
- Discretize features into low/medium/high bins using training data only.
- Mine rules separately for ordinal boundaries.
- Validate rules on held-out subjects.

Disallowed use:

- Do not mine all possible features and report the most exciting rules without validation.
- Do not treat support/confidence/lift as proof of predictive contribution.

Report:

- Support
- Confidence
- Lift
- Held-out boundary precision or balanced accuracy
- Rule stability across folds

## Minimal Clinical Model Selection

The final recommended model should be chosen by performance and clinical burden.

Selection order:

1. Prefer fewer modalities.
2. Prefer fewer features.
3. Prefer fewer interactions.
4. Prefer performance statistically close to the best model.

Possible conclusion types:

- EDA-only with within-modality interactions is sufficient.
- ECG adds value only for specific pain boundaries.
- Cross-modality interactions improve high-pain detection but not low-pain detection.
- EMG or temperature does not justify extra sensor burden under subject-independent evaluation.

## Risks And Controls

Risk: Subject leakage

- Control: all splitting, clustering, selection, scaling, and interaction generation happen inside training folds.

Risk: Interaction search becomes p-hacking

- Control: cluster-constrained interaction candidates, stability thresholds, and held-out validation.

Risk: Too many experiments

- Control: validate EDA/ECG first, then expand.

Risk: Feature extraction uncertainty from filtered data

- Control: use official filtered signals as primary data and document the limitation. Optionally run raw-signal robustness later with an explicitly defined preprocessing pipeline.

Risk: ECG peak features are unreliable

- Control: include ECG signal-summary features first, add peak-derived HRV only after quality checks.

## Deliverables

Phase 1 deliverables:

- Feature extraction script
- Feature matrix for filtered EDA/ECG
- Grouped 5-fold baseline results
- Main-effects vs interaction model comparison
- Initial stability table

Phase 2 deliverables:

- Pain-boundary-specific contribution tables
- Minimal clinical model comparison
- Association-rule explanation table for stable terms
- Final LOSO or repeated grouped-CV results

Phase 3 optional deliverables:

- EMG and temperature expansion
- Raw-signal robustness analysis
- SHAP or EBM visualizations for selected final models

## Success Criteria

The experiment is successful if it can answer at least one of the following with subject-independent evidence:

- A small set of within-modality interactions matches or approaches multi-modality performance.
- ECG provides boundary-specific value beyond EDA rather than uniform global value.
- Stable interaction terms improve ordinal boundary prediction beyond main effects.
- The best clinically efficient model uses fewer modalities/features while remaining statistically close to the best-performing model.

