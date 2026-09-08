# AGENTS.md

## Project Mission

This repository studies **physiological pain recognition on BioVid Part A**, using the physiological data documented in this project.

The project **does not use facial/video data**.

The primary scientific goal is **not** to rediscover which modality or modality combination achieves the highest overall accuracy. Prior BioVid literature has already studied that question extensively.

Instead, this project asks:

> **How stable are physiological modality- and feature-level contributions to pain recognition across pain intensities and across subjects?**

The main research contribution should focus on **heterogeneity, stability, and generalizability of physiological contributions**, rather than another global modality ranking.

---

# 1. Scientific Positioning

## 1.1 What this project is NOT

Do not frame the main contribution as:

- finding the globally best physiological modality;
- finding the globally best modality combination;
- showing that multimodal input beats unimodal input;
- producing another overall feature-importance ranking;
- obtaining the highest possible BioVid classification accuracy;
- using feature selection only to improve predictive performance.

These analyses may be used as baselines or supporting evidence, but they are **not the main novelty**.

Existing work has already repeatedly compared physiological modalities and combinations on BioVid. Recent work has also performed feature-level and pain-level importance analysis.

Therefore, avoid claims such as:

> "EDA is the best modality."

or:

> "Feature X is the most important physiological marker of pain."

unless the claim is explicitly qualified by evaluation setting, pain level, subject population, and uncertainty.

---

## 1.2 Main scientific gap

Most prior modality- and feature-importance conclusions are reported at a **population-aggregated level**.

This project examines whether those conclusions remain stable when conditioned on:

1. **pain intensity**, and
2. **individual subject**.

The key distinction is:

> **Predictive importance is not necessarily the same as cross-subject consistency.**

A feature may have high mean importance while being important for only a subset of subjects.

Conversely, a moderately important feature may be consistently useful across many subjects.

This project should quantify that difference.

---

# 2. Primary Research Questions

## RQ1 — Pain-level heterogeneity

> **How does the predictive contribution of physiological modalities vary across pain intensity levels?**

The purpose is not to declare a single winner.

Measure whether a modality's contribution changes between baseline, low, moderate, and high pain.

Example outputs:

- modality contribution by pain level;
- per-class performance change after modality ablation;
- class-specific SHAP/permutation importance;
- confusion patterns, especially for intermediate pain levels.

---

## RQ2 — Feature-level explanation

> **Which physiological features within each modality drive discrimination at different pain intensity levels?**

Feature-level analysis is intended to explain **why a modality is useful**, not to imply that a feature can be acquired independently from its parent sensor.

Interpretation must remain hierarchical:

`modality -> derived physiological features -> pain-level contribution`

Do not claim that an individual feature can replace its sensing modality.

---

## RQ3 — Cross-subject stability

> **How stable are modality- and feature-level contributions across subjects?**

This is the main novelty candidate.

Quantify whether population-level feature/modality importance generalizes to individuals.

Possible measures include:

- variance of per-subject importance;
- coefficient of variation;
- rank stability;
- top-k inclusion frequency;
- Kendall/Spearman rank agreement;
- proportion of subjects for which a feature has positive contribution;
- bootstrap confidence intervals;
- between-subject vs within-subject variability.

A useful conceptual metric is:

`Consistency(feature) = number of subjects where feature is in top-k / number of evaluated subjects`

Do not use this exact definition blindly if a better statistically justified stability measure is available.

---

## RQ4 — Joint pain × subject heterogeneity

> **Do the same physiological features remain important for the same pain intensities across different subjects?**

This should be treated as a higher-order analysis after RQ1–RQ3 are working.

The goal is not to create thousands of uninterpretable subject-feature plots.

Instead, summarize patterns such as:

- globally important and cross-subject stable;
- globally important but highly subject-dependent;
- pain-level-specific but stable;
- pain-level-specific and subject-dependent.

---

# 3. Data Scope

Use **BioVid Part A physiological data only**.

Do not use:

- facial video;
- facial landmarks;
- facial action units;
- image embeddings;
- visual pain scores;
- any feature derived from facial/video data.

The repository documentation contains the authoritative description of the available physiological files, channels, sampling rates, labels, subject identifiers, trial structure, preprocessing state, and directory layout.

## Mandatory rule

**Inspect the repository documentation and actual files before assuming the schema.**

Do not hard-code modality names, column names, file paths, label mappings, or sampling rates based only on memory or public BioVid documentation.

If the repository documents ECG, EMG, EDA, temperature, or other physiological channels, record exactly what is available.

The primary planned modalities are:

- ECG;
- EMG;
- EDA.

Any additional physiological channel must be:

1. verified from the local documentation/data;
2. clearly documented;
3. treated separately from the main ECG/EMG/EDA analysis unless there is a justified reason to include it.

---

# 4. First Task: Repository and Data Audit

Before building models, create a reproducible audit.

Produce:

`reports/data_audit.md`

and, if useful:

`artifacts/data_audit.json`

The audit must include:

- authoritative data directories;
- file formats;
- subject count;
- subject IDs or anonymized subject keys;
- available physiological modalities/channels;
- sampling frequency per signal;
- trial/window duration;
- pain labels and their exact encoding;
- baseline labels;
- number of samples/trials per pain level;
- missing/corrupted files;
- preprocessing already present in the dataset;
- whether raw and filtered versions both exist;
- whether labels are balanced;
- whether repeated trials exist per subject/pain level;
- whether there are any variables that can leak subject identity or label information.

Do not proceed to large experiments if the audit reveals unresolved label mapping or subject-split ambiguity.

---

# 5. Leakage and Evaluation Rules

This is a **subject-dependent physiological dataset**, so leakage control is critical.

## 5.1 Default evaluation protocol

The default main evaluation must be **subject-independent**.

Prefer:

- Leave-One-Subject-Out (LOSO), or
- GroupKFold / stratified group split by subject,

depending on computational cost and class support.

All windows/trials from the same subject must remain in the same split.

Never randomly split windows from the same subject across train and test.

---

## 5.2 Preprocessing leakage

All learned preprocessing must be fit on the training fold only.

This includes:

- scaling;
- normalization;
- imputation;
- feature selection;
- PCA;
- learned filters;
- dimensionality reduction;
- outlier thresholds derived from population statistics;
- class balancing;
- feature ranking;
- hyperparameter selection.

For LOSO or GroupKFold:

`train preprocessing -> fit on training subjects -> transform held-out subject`

Do not compute global normalization statistics using the full dataset.

---

## 5.3 Feature extraction

Deterministic signal-processing transformations may be applied consistently, but if any transformation estimates parameters from data, estimate them on the training fold whenever possible.

Document every feature with:

- modality;
- mathematical/statistical definition;
- units;
- time domain / frequency domain / nonlinear category;
- preprocessing dependencies.

Avoid generating hundreds of arbitrary features without a physiological or methodological rationale.

---

# 6. Modeling Strategy

The model is a tool for scientific analysis, not the main contribution.

Prefer interpretable or explainable models as the primary analysis model.

Recommended starting point:

- Logistic Regression;
- Random Forest;
- XGBoost / LightGBM if already available;
- optionally a simple MLP as a nonlinear comparison.

Do not begin with a large Transformer, CNN, or complex deep multimodal architecture unless the simpler pipeline cannot answer the RQs.

The preferred workflow is:

1. establish a clean subject-independent baseline;
2. validate that features carry signal;
3. perform modality ablations;
4. perform pain-level analysis;
5. perform feature attribution;
6. quantify cross-subject stability;
7. only then consider more complex models.

---

# 7. Baseline Experiments

Baseline experiments are necessary but should not be presented as novelty.

At minimum, compare:

- each verified physiological modality individually;
- planned modality combinations;
- all available target physiological modalities together.

Report:

- macro F1;
- balanced accuracy;
- per-class recall;
- per-class precision;
- confusion matrix;
- optional AUROC where mathematically appropriate.

Do not rely on raw accuracy alone, especially for multiclass pain intensity.

For every reported number, state:

- task definition;
- pain labels included;
- split protocol;
- model;
- feature set;
- averaging method.

Do not compare results against literature without matching the task definition.

---

# 8. Pain-Level Analysis

Intermediate pain levels are particularly important.

Do not only report overall multiclass metrics.

For each pain level, investigate:

- true positive rate / recall;
- confusion with adjacent pain levels;
- modality ablation effect;
- feature attribution;
- uncertainty / confidence;
- cross-subject variation.

Where appropriate, compute:

`Delta performance(modality removal | pain level)`

Example:

`ΔRecall_P2 = Recall_full(P2) - Recall_without_EDA(P2)`

This is often more interpretable than a single global feature-importance score.

---

# 9. Feature-Level Analysis

Use at least two complementary attribution strategies if computationally feasible.

Possible approaches:

- SHAP;
- permutation importance;
- ablation/removal;
- model coefficients for standardized linear models.

Do not treat one attribution method as ground truth.

## Required hierarchy

Report results at:

1. modality level;
2. feature level within modality;
3. pain level;
4. subject-level stability.

Avoid comparing raw SHAP magnitudes between incompatible models without justification.

Avoid claiming causality.

Use language such as:

- predictive contribution;
- model attribution;
- discriminative utility;
- association with model decisions.

Do not say:

- physiological cause;
- causal biomarker;
- mechanism,

unless the study design supports it.

---

# 10. Cross-Subject Stability Analysis

This is a core deliverable.

For each important feature and modality, estimate both:

## 10.1 Mean predictive contribution

How important is it on average?

## 10.2 Stability / consistency

How consistently is it important across held-out subjects?

Potential analyses:

- per-subject feature rank;
- top-k frequency;
- rank correlation across subjects;
- interquartile range of importance;
- coefficient of variation;
- bootstrap confidence intervals;
- clustering subjects by attribution profile;
- mixed-effects analysis if statistically justified.

A strong result would distinguish categories such as:

### Type A
High average importance + high stability

Interpretation:
potentially generalizable physiological signal.

### Type B
High average importance + low stability

Interpretation:
strong population-level effect driven by a subset of subjects.

### Type C
Moderate average importance + high stability

Interpretation:
weak but robust cross-subject marker.

### Type D
Low average importance + low stability

Interpretation:
limited general utility.

Do not overinterpret unstable estimates from subjects with insufficient samples.

---

# 11. Subject Grouping

Do not invent demographic subgroups.

Only use age, sex, or other demographic variables if:

- they are explicitly present in the local BioVid documentation/data;
- sample sizes are sufficient;
- the statistical analysis is justified.

The preferred main concept is **inter-subject heterogeneity**, not demographic stereotyping.

If clustering subjects by physiological response or attribution profile, clustering must be exploratory and clearly labelled as such.

Do not assign clinical meaning to clusters without evidence.

---

# 12. Statistical Analysis

Where possible, attach uncertainty to conclusions.

Prefer:

- bootstrap confidence intervals;
- repeated grouped cross-validation;
- paired fold-level comparisons;
- nonparametric tests when assumptions are doubtful;
- effect sizes in addition to p-values.

Correct for multiple comparisons when testing many features or pain levels.

Do not run dozens of significance tests and only report significant results.

Maintain a machine-readable record of all tested hypotheses.

---

# 13. Reproducibility

Every experiment must be reproducible from code.

Use deterministic seeds where supported.

Store experiment configuration including:

- dataset version/path;
- included modalities;
- included features;
- label mapping;
- split protocol;
- random seed;
- preprocessing;
- model;
- hyperparameters;
- evaluation metrics;
- attribution method.

Recommended structure:

```text
configs/
src/
  data/
  features/
  models/
  evaluation/
  explainability/
  analysis/
experiments/
reports/
artifacts/
tests/
```

Do not reorganize the repository aggressively if an existing structure already exists.

Adapt to the current project first.

---

# 14. Required Deliverables

The work should eventually produce:

## A. Data audit

`reports/data_audit.md`

## B. Reproducible feature table

A table mapping:

`subject -> trial/window -> pain label -> modality -> features`

with no leakage.

## C. Subject-independent baseline results

Overall and per pain level.

## D. Modality contribution analysis

Including ablation and uncertainty.

## E. Feature × pain-level analysis

Identify which physiological features drive decisions at different pain levels.

## F. Cross-subject stability analysis

The main novelty candidate.

## G. Figures

Prefer interpretable figures such as:

- pain-level confusion matrix;
- modality ablation by pain level;
- feature importance heatmap: feature × pain level;
- feature consistency plot across subjects;
- mean importance vs cross-subject stability scatter;
- subject × modality contribution heatmap.

Do not generate dozens of redundant plots.

## H. Research summary

`reports/research_findings.md`

This document must separate:

- robust findings;
- tentative findings;
- negative results;
- limitations;
- possible paper contribution.

---

# 15. Expected Paper Narrative

The intended narrative is:

1. Prior literature already identifies strong physiological modalities and useful multimodal combinations on BioVid.
2. Population-level modality or feature rankings do not necessarily imply individual-level generalizability.
3. We therefore analyze whether physiological contributions remain stable across pain intensities and subjects.
4. We quantify:
   - modality contribution;
   - feature-level contribution;
   - pain-level dependency;
   - cross-subject stability.
5. We distinguish **average predictive importance** from **generalizable consistency**.
6. The implications concern:
   - personalized pain recognition;
   - sensor selection;
   - robust physiological pain modeling;
   - limitations of population-aggregated explanations.

---

# 16. Interpretation Guardrails

Never make claims stronger than the evidence.

Do not claim:

- clinical diagnostic validity;
- causal physiological mechanisms;
- universal pain biomarkers;
- deployment readiness;
- superiority over all prior BioVid methods.

BioVid experimental heat pain is not equivalent to all real-world clinical pain.

Make this limitation explicit.

Feature-level findings explain model behavior and predictive signal, not necessarily biological causality.

---

# 17. Coding Rules for Codex

When modifying the repository:

1. Inspect existing code and documentation first.
2. Prefer extending existing modules instead of rewriting working code.
3. Keep data paths configurable.
4. Never commit raw BioVid data.
5. Never copy sensitive/licensed data into reports.
6. Add tests for:
   - subject split integrity;
   - label mapping;
   - feature extraction dimensions;
   - leakage-prone preprocessing;
   - deterministic outputs where feasible.
7. Preserve intermediate experiment outputs in machine-readable form.
8. Do not silently drop subjects or trials.
9. Log exclusions with explicit reasons.
10. Do not fabricate missing metadata.

---

# 18. Execution Order

Follow this order unless repository constraints require a justified deviation.

## Phase 1 — Audit
- inspect docs;
- inspect data structure;
- verify modalities;
- verify labels;
- verify subject/trial structure;
- write `reports/data_audit.md`.

## Phase 2 — Clean feature pipeline
- implement or verify physiological preprocessing;
- implement feature extraction;
- validate feature table;
- add leakage tests.

## Phase 3 — Subject-independent baseline
- establish grouped/LOSO evaluation;
- run simple interpretable baseline;
- report multiclass and per-pain-level metrics.

## Phase 4 — Modality analysis
- unimodal;
- multimodal;
- modality removal;
- pain-level-specific ablation effects.

## Phase 5 — Feature attribution
- global attribution;
- class/pain-level attribution;
- feature ablation where feasible.

## Phase 6 — Subject stability
- derive per-subject importance;
- quantify stability;
- compare mean importance vs consistency.

## Phase 7 — Joint analysis
- pain × feature × subject patterns;
- only add demographic or subject clustering analysis if statistically justified.

## Phase 8 — Research synthesis
- write findings;
- identify what is actually novel;
- explicitly report null or contradictory findings.

---

# 19. Stop Conditions

Stop and report instead of guessing if:

- pain labels cannot be mapped confidently;
- subject IDs cannot be reliably recovered;
- train/test leakage cannot be ruled out;
- documentation conflicts with file contents;
- a modality is missing or incomplete;
- there are too few subjects for a requested subgroup analysis;
- an experiment would require facial/video data;
- a proposed result cannot be reproduced.

When blocked, write:

- what was verified;
- what is ambiguous;
- why it matters;
- the minimum decision required to continue.

---

# 20. Decision Principle

When choosing between:

A. a more complex model that may improve accuracy, and  
B. a cleaner analysis that better answers the research questions,

prefer **B**.

The purpose of this project is not leaderboard optimization.

The purpose is to determine:

> **whether physiological pain-recognition signals that look important at the population level remain important across pain levels and across individuals.**
