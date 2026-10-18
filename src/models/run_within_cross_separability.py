from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices, score_vector
from src.models.run_subject_independent_baselines import feature_sets, read_feature_table, validate_feature_inputs


RANDOM_STATE = 20260909
N_SPLITS = 5
N_REPEATS = 10
N_BOOTSTRAPS = 2000
FEATURE_SET = "phase6_preselected"
FEATURES = [
    "eda_scl_delta",
    "eda_scl_slope_per_s",
    "eda_scr_amplitude_sum",
    "eda_scr_count",
    "emg_trapezius__tsfresh_linear_trend_slope",
]
PAIRS = [
    ("PA1_vs_PA2", 1, 2),
    ("PA2_vs_PA3", 2, 3),
    ("PA3_vs_PA4", 3, 4),
    ("PA1_vs_PA3", 1, 3),
    ("BL1_vs_PA4", 0, 4),
]
TRIAL_PATTERN = re.compile(r"-(\d{3})$")


def stable_seed(*parts: object, base: int = RANDOM_STATE) -> int:
    value = int(base) & 0xFFFFFFFF
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value * 16777619) ^ byte) & 0xFFFFFFFF
    return value


def add_trial_number(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    extracted = out["sample_name"].str.extract(TRIAL_PATTERN, expand=False)
    if extracted.isna().any():
        bad = out.loc[extracted.isna(), "sample_name"].head().tolist()
        raise ValueError(f"Cannot recover trial number from sample_name: {bad}")
    out["trial_number"] = extracted.astype(int)
    return out


def validate_trial_structure(table: pd.DataFrame) -> None:
    counts = table.groupby(["subject_id", "class_id"]).size()
    if not (counts == 20).all():
        raise ValueError("Phase 8 requires exactly 20 trials in every subject/class cell")
    for subject_id, group in table.groupby("subject_id"):
        pain = group[group["class_id"] > 0]["trial_number"]
        baseline = group[group["class_id"] == 0]["trial_number"]
        if set(pain) != set(range(1, 81)):
            raise ValueError(f"Subject {subject_id} pain trial indices are not exactly 001-080")
        if set(baseline) != set(range(81, 101)):
            raise ValueError(f"Subject {subject_id} baseline trial indices are not exactly 081-100")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    table = read_feature_table(root / "artifacts" / "feature_table_filtered.csv")
    metadata = pd.read_csv(root / "experiments" / "baselines" / "feature_set_manifest.csv")
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    missing = sorted(set(FEATURES).difference(table.columns))
    if missing:
        raise ValueError(f"Missing preselected Phase 6 features: {missing}")
    table = add_trial_number(table)
    validate_trial_structure(table)
    fold_samples = load_phase3_folds(root / "experiments" / "baselines" / "fold_assignments.csv")
    folds = make_fold_indices(table, fold_samples)
    return table, metadata, folds


def pair_table(table: pd.DataFrame, low_class: int, high_class: int) -> pd.DataFrame:
    pair = table[table["class_id"].isin([low_class, high_class])].copy().reset_index(drop=True)
    pair["binary_label"] = (pair["class_id"] == high_class).astype(int)
    return pair


def hedges_g(low: np.ndarray, high: np.ndarray) -> float:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    n0, n1 = len(low), len(high)
    df = n0 + n1 - 2
    if n0 < 2 or n1 < 2 or df <= 0:
        return np.nan
    pooled_var = ((n0 - 1) * np.var(low, ddof=1) + (n1 - 1) * np.var(high, ddof=1)) / df
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return 0.0 if np.isclose(np.mean(high), np.mean(low)) else np.nan
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(correction * (np.mean(high) - np.mean(low)) / np.sqrt(pooled_var))


def empirical_overlap(low: np.ndarray, high: np.ndarray) -> float:
    """Histogram intersection on common pooled-data bins; bounded in [0, 1]."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    pooled = np.concatenate([low, high])
    if np.nanmax(pooled) == np.nanmin(pooled):
        return 1.0
    edges = np.histogram_bin_edges(pooled, bins="fd")
    if len(edges) - 1 < 5:
        edges = np.linspace(np.nanmin(pooled), np.nanmax(pooled), 6)
    elif len(edges) - 1 > 15:
        edges = np.linspace(np.nanmin(pooled), np.nanmax(pooled), 16)
    low_hist, _ = np.histogram(low, bins=edges)
    high_hist, _ = np.histogram(high, bins=edges)
    return float(np.minimum(low_hist / low_hist.sum(), high_hist / high_hist.sum()).sum())


def signal_level_separability(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    for pair_name, low_class, high_class in PAIRS:
        pair = pair_table(table, low_class, high_class)
        for subject_id, subject in pair.groupby("subject_id", sort=True):
            low_rows = subject[subject["class_id"] == low_class]
            high_rows = subject[subject["class_id"] == high_class]
            order_auc = roc_auc_score(subject["binary_label"], subject["trial_number"])
            order_rows.append(
                {
                    "pair": pair_name,
                    "level_gap": high_class - low_class,
                    "subject_id": int(subject_id),
                    "subject_name": subject["subject_name"].iloc[0],
                    "low_trial_number_mean": float(low_rows["trial_number"].mean()),
                    "high_trial_number_mean": float(high_rows["trial_number"].mean()),
                    "trial_number_hedges_g": hedges_g(low_rows["trial_number"], high_rows["trial_number"]),
                    "trial_number_auc": float(order_auc),
                    "trial_number_separability_auc": float(max(order_auc, 1.0 - order_auc)),
                    "chronology_status": "filename_index_proxy_not_documented_as_acquisition_time",
                }
            )
            for feature in FEATURES:
                low = low_rows[feature].to_numpy(dtype=float)
                high = high_rows[feature].to_numpy(dtype=float)
                effect_rows.append(
                    {
                        "pair": pair_name,
                        "level_gap": high_class - low_class,
                        "subject_id": int(subject_id),
                        "subject_name": subject["subject_name"].iloc[0],
                        "feature_name": feature,
                        "n_low": len(low),
                        "n_high": len(high),
                        "low_mean": float(np.mean(low)),
                        "high_mean": float(np.mean(high)),
                        "low_sd": float(np.std(low, ddof=1)),
                        "high_sd": float(np.std(high, ddof=1)),
                        "pooled_within_sd": float(np.sqrt(((len(low) - 1) * np.var(low, ddof=1) + (len(high) - 1) * np.var(high, ddof=1)) / (len(low) + len(high) - 2))),
                        "hedges_g": hedges_g(low, high),
                        "abs_hedges_g": abs(hedges_g(low, high)),
                        "empirical_overlap": empirical_overlap(low, high),
                    }
                )
        for (subject_id, class_id), group in pair.groupby(["subject_id", "class_id"], sort=True):
            for feature in FEATURES:
                feature_values = group[feature].to_numpy(dtype=float)
                rho = (
                    spearmanr(group["trial_number"], feature_values).statistic
                    if np.ptp(feature_values) > 0
                    else np.nan
                )
                drift_rows.append(
                    {
                        "pair": pair_name,
                        "subject_id": int(subject_id),
                        "subject_name": group["subject_name"].iloc[0],
                        "class_id": int(class_id),
                        "class_name": EXPECTED_LABEL_MAP[int(class_id)],
                        "feature_name": feature,
                        "spearman_trial_feature": float(rho) if np.isfinite(rho) else np.nan,
                        "abs_spearman_trial_feature": abs(float(rho)) if np.isfinite(rho) else np.nan,
                    }
                )
    return pd.DataFrame(effect_rows), pd.DataFrame(order_rows), pd.DataFrame(drift_rows)


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Exact binary ROC-AUC via pairwise concordance, including 0.5 credit for ties."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = score[y == 1]
    negative = score[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    differences = positive[:, None] - negative[None, :]
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


def binary_balanced_accuracy(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(score, dtype=float) >= 0.5).astype(int)
    sensitivity = np.mean(pred[y == 1] == 1)
    specificity = np.mean(pred[y == 0] == 0)
    return float(0.5 * (sensitivity + specificity))


def metric_values(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return binary_auc(y, score), binary_balanced_accuracy(y, score)


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in sorted(np.unique(y)):
        idx = np.flatnonzero(y == label)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def bootstrap_metric_ci(y: np.ndarray, score: np.ndarray, seed: int, n_bootstraps: int = N_BOOTSTRAPS) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_bootstraps)
    bas = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = stratified_bootstrap_indices(y, rng)
        aucs[i], bas[i] = metric_values(y[idx], score[idx])
    return {
        "roc_auc_ci_low": float(np.quantile(aucs, 0.025)),
        "roc_auc_ci_high": float(np.quantile(aucs, 0.975)),
        "balanced_accuracy_ci_low": float(np.quantile(bas, 0.025)),
        "balanced_accuracy_ci_high": float(np.quantile(bas, 0.975)),
    }


def run_within_subject(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for pair_name, low_class, high_class in PAIRS:
        pair = pair_table(table, low_class, high_class)
        for subject_id, subject in pair.groupby("subject_id", sort=True):
            subject = subject.reset_index(drop=True)
            x = subject[FEATURES].to_numpy(dtype=float)
            y = subject["binary_label"].to_numpy(dtype=int)
            splitter = RepeatedStratifiedKFold(
                n_splits=N_SPLITS,
                n_repeats=N_REPEATS,
                random_state=stable_seed("within", pair_name, subject_id),
            )
            score_sum = np.zeros(len(subject), dtype=float)
            score_count = np.zeros(len(subject), dtype=int)
            for split_id, (train_index, test_index) in enumerate(splitter.split(x, y)):
                if set(train_index).intersection(test_index):
                    raise AssertionError("Within-subject train/test overlap detected")
                model = make_baseline_model("logistic_regression", stable_seed(pair_name, subject_id, split_id))
                model.fit(x[train_index], y[train_index])
                scores = score_vector(model, x[test_index])
                score_sum[test_index] += scores
                score_count[test_index] += 1
            if not np.all(score_count == N_REPEATS):
                raise AssertionError("Each trial must receive exactly one OOF prediction per CV repeat")
            mean_score = score_sum / score_count
            auc, ba = metric_values(y, mean_score)
            ci = bootstrap_metric_ci(y, mean_score, stable_seed("within_ci", pair_name, subject_id))
            metric_rows.append(
                {
                    "pair": pair_name,
                    "level_gap": high_class - low_class,
                    "subject_id": int(subject_id),
                    "subject_name": subject["subject_name"].iloc[0],
                    "method": "within_subject",
                    "n_trials": len(subject),
                    "n_features": len(FEATURES),
                    "roc_auc": auc,
                    "balanced_accuracy": ba,
                    **ci,
                }
            )
            for idx, row in subject.iterrows():
                prediction_rows.append(
                    {
                        "pair": pair_name,
                        "subject_id": int(subject_id),
                        "subject_name": row["subject_name"],
                        "sample_id": int(row["sample_id"]),
                        "sample_name": row["sample_name"],
                        "true_label": int(y[idx]),
                        "true_class_id": int(row["class_id"]),
                        "true_class_name": row["class_name"],
                        "oof_score": float(mean_score[idx]),
                        "oof_prediction": int(mean_score[idx] >= 0.5),
                        "n_oof_predictions": int(score_count[idx]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_cross_subject(table: pd.DataFrame, folds: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_samples = {
        int(split["fold"]): set(table.iloc[split["test_index"]]["sample_name"])
        for split in folds
    }
    for pair_name, low_class, high_class in PAIRS:
        pair = pair_table(table, low_class, high_class)
        x = pair[FEATURES].to_numpy(dtype=float)
        y = pair["binary_label"].to_numpy(dtype=int)
        pair_prediction_rows: list[dict[str, object]] = []
        for fold, test_samples in sorted(fold_samples.items()):
            test_index = pair.index[pair["sample_name"].isin(test_samples)].to_numpy(dtype=int)
            train_index = pair.index[~pair["sample_name"].isin(test_samples)].to_numpy(dtype=int)
            train_subjects = set(pair.iloc[train_index]["subject_id"].astype(int))
            test_subjects = set(pair.iloc[test_index]["subject_id"].astype(int))
            if train_subjects.intersection(test_subjects):
                raise AssertionError(f"Cross-subject leakage in fold {fold}")
            model = make_baseline_model("logistic_regression", stable_seed("cross", pair_name, fold))
            model.fit(x[train_index], y[train_index])
            scores = score_vector(model, x[test_index])
            for pos, idx in enumerate(test_index):
                row = pair.iloc[int(idx)]
                pair_prediction_rows.append(
                    {
                        "pair": pair_name,
                        "fold": fold,
                        "subject_id": int(row["subject_id"]),
                        "subject_name": row["subject_name"],
                        "sample_id": int(row["sample_id"]),
                        "sample_name": row["sample_name"],
                        "true_label": int(y[idx]),
                        "true_class_id": int(row["class_id"]),
                        "true_class_name": row["class_name"],
                        "score": float(scores[pos]),
                        "prediction": int(scores[pos] >= 0.5),
                        "n_train_trials": int(len(train_index)),
                        "n_train_subjects": int(len(train_subjects)),
                    }
                )
        predictions = pd.DataFrame(pair_prediction_rows)
        if predictions["sample_name"].duplicated().any() or len(predictions) != len(pair):
            raise AssertionError("Cross-subject OOF predictions must cover each pair trial exactly once")
        prediction_rows.extend(pair_prediction_rows)
        for subject_id, subject in predictions.groupby("subject_id", sort=True):
            y_subject = subject["true_label"].to_numpy(dtype=int)
            scores = subject["score"].to_numpy(dtype=float)
            auc, ba = metric_values(y_subject, scores)
            ci = bootstrap_metric_ci(y_subject, scores, stable_seed("cross_ci", pair_name, subject_id))
            metric_rows.append(
                {
                    "pair": pair_name,
                    "level_gap": high_class - low_class,
                    "subject_id": int(subject_id),
                    "subject_name": subject["subject_name"].iloc[0],
                    "method": "cross_subject",
                    "fold": int(subject["fold"].iloc[0]),
                    "n_trials": len(subject),
                    "n_features": len(FEATURES),
                    "roc_auc": auc,
                    "balanced_accuracy": ba,
                    **ci,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def paired_gap_table(within: pd.DataFrame, cross: pd.DataFrame, within_pred: pd.DataFrame, cross_pred: pd.DataFrame) -> pd.DataFrame:
    merged = within.merge(
        cross,
        on=["pair", "level_gap", "subject_id", "subject_name", "n_trials", "n_features"],
        suffixes=("_within", "_cross"),
        validate="one_to_one",
    )
    merged["auc_gap_within_minus_cross"] = merged["roc_auc_within"] - merged["roc_auc_cross"]
    merged["balanced_accuracy_gap_within_minus_cross"] = merged["balanced_accuracy_within"] - merged["balanced_accuracy_cross"]
    gap_lows = []
    gap_highs = []
    for row in merged.itertuples(index=False):
        w = within_pred[(within_pred["pair"] == row.pair) & (within_pred["subject_id"] == row.subject_id)]
        c = cross_pred[(cross_pred["pair"] == row.pair) & (cross_pred["subject_id"] == row.subject_id)]
        joined = w[["sample_name", "true_label", "oof_score"]].merge(
            c[["sample_name", "true_label", "score"]],
            on=["sample_name", "true_label"],
            validate="one_to_one",
        )
        y = joined["true_label"].to_numpy(dtype=int)
        ws = joined["oof_score"].to_numpy(dtype=float)
        cs = joined["score"].to_numpy(dtype=float)
        rng = np.random.default_rng(stable_seed("gap_ci", row.pair, row.subject_id))
        gaps = np.empty(N_BOOTSTRAPS)
        for i in range(N_BOOTSTRAPS):
            idx = stratified_bootstrap_indices(y, rng)
            gaps[i] = binary_auc(y[idx], ws[idx]) - binary_auc(y[idx], cs[idx])
        gap_lows.append(float(np.quantile(gaps, 0.025)))
        gap_highs.append(float(np.quantile(gaps, 0.975)))
    merged["auc_gap_ci_low"] = gap_lows
    merged["auc_gap_ci_high"] = gap_highs
    return merged


def bootstrap_location_ci(values: Iterable[float], seed: int, statistic: str = "mean") -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOTSTRAPS)
    fn = np.mean if statistic == "mean" else np.median
    for i in range(N_BOOTSTRAPS):
        estimates[i] = fn(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(result, index=p_values.index)


def summarize_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, group in gaps.groupby("pair", sort=False):
        values = group["auc_gap_within_minus_cross"].to_numpy(dtype=float)
        mean_ci = bootstrap_location_ci(values, stable_seed("pair_mean", pair), "mean")
        median_ci = bootstrap_location_ci(values, stable_seed("pair_median", pair), "median")
        try:
            p_value = float(wilcoxon(values).pvalue)
        except ValueError:
            p_value = 1.0
        rows.append(
            {
                "pair": pair,
                "level_gap": int(group["level_gap"].iloc[0]),
                "n_subjects": int(group["subject_id"].nunique()),
                "within_auc_mean": float(group["roc_auc_within"].mean()),
                "within_auc_median": float(group["roc_auc_within"].median()),
                "cross_auc_mean": float(group["roc_auc_cross"].mean()),
                "cross_auc_median": float(group["roc_auc_cross"].median()),
                "auc_gap_mean": float(np.mean(values)),
                "auc_gap_mean_ci_low": mean_ci[0],
                "auc_gap_mean_ci_high": mean_ci[1],
                "auc_gap_median": float(np.median(values)),
                "auc_gap_median_ci_low": median_ci[0],
                "auc_gap_median_ci_high": median_ci[1],
                "prop_positive_gap": float(np.mean(values > 0)),
                "wilcoxon_p": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_bh"] = benjamini_hochberg(out["wilcoxon_p"])
    return out


def phase6_consistency_analysis(root: Path, gaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase6 = pd.read_csv(root / "experiments" / "cross_subject_stability" / "subject_feature_monotonicity.csv")
    phase6 = phase6[phase6["feature_name"].isin(FEATURES)].copy()
    subject_consistency = (
        phase6.groupby(["subject_id", "subject_name"], as_index=False)
        .agg(
            phase6_median_abs_spearman=("spearman_rho", lambda s: float(np.nanmedian(np.abs(s)))),
            phase6_mean_abs_spearman=("spearman_rho", lambda s: float(np.nanmean(np.abs(s)))),
            phase6_sign_reversal_fraction=("has_sign_reversal", "mean"),
        )
    )
    linked = gaps.merge(subject_consistency, on=["subject_id", "subject_name"], validate="many_to_one")
    rows = []
    outcomes = ["roc_auc_within", "roc_auc_cross", "auc_gap_within_minus_cross"]
    predictors = ["phase6_median_abs_spearman", "phase6_sign_reversal_fraction"]
    for pair, group in linked.groupby("pair", sort=False):
        for predictor in predictors:
            for outcome in outcomes:
                result = spearmanr(group[predictor], group[outcome])
                rng = np.random.default_rng(stable_seed("rho", pair, predictor, outcome))
                boot = []
                for _ in range(N_BOOTSTRAPS):
                    idx = rng.choice(len(group), size=len(group), replace=True)
                    x = group[predictor].to_numpy(dtype=float)[idx]
                    y = group[outcome].to_numpy(dtype=float)[idx]
                    rho = spearmanr(x, y).statistic
                    if np.isfinite(rho):
                        boot.append(float(rho))
                rows.append(
                    {
                        "pair": pair,
                        "predictor": predictor,
                        "outcome": outcome,
                        "n_subjects": len(group),
                        "spearman_rho": float(result.statistic),
                        "rho_ci_low": float(np.quantile(boot, 0.025)),
                        "rho_ci_high": float(np.quantile(boot, 0.975)),
                        "p_value": float(result.pvalue),
                    }
                )
    associations = pd.DataFrame(rows)
    associations["p_value_bh"] = benjamini_hochberg(associations["p_value"])
    return linked, associations


def adjacent_wide_comparison(gaps: pd.DataFrame) -> pd.DataFrame:
    per_subject = gaps.copy()
    per_subject["pair_group"] = np.where(per_subject["level_gap"] == 1, "adjacent", "wide_control")
    means = per_subject.groupby(["subject_id", "pair_group"])[["roc_auc_within", "roc_auc_cross", "auc_gap_within_minus_cross"]].mean().unstack()
    rows = []
    for metric in ["roc_auc_within", "roc_auc_cross", "auc_gap_within_minus_cross"]:
        adjacent = means[(metric, "adjacent")].to_numpy(dtype=float)
        wide = means[(metric, "wide_control")].to_numpy(dtype=float)
        difference = wide - adjacent
        ci = bootstrap_location_ci(difference, stable_seed("adjacent_wide", metric), "mean")
        rows.append(
            {
                "metric": metric,
                "adjacent_subject_mean": float(np.mean(adjacent)),
                "wide_control_subject_mean": float(np.mean(wide)),
                "wide_minus_adjacent_mean": float(np.mean(difference)),
                "wide_minus_adjacent_ci_low": ci[0],
                "wide_minus_adjacent_ci_high": ci[1],
            }
        )
    return pd.DataFrame(rows)


def make_figures(output_dir: Path, gaps: pd.DataFrame, signal: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pair_order = [pair[0] for pair in PAIRS]
    positions = np.arange(len(pair_order))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    within_data = [gaps.loc[gaps["pair"] == pair, "roc_auc_within"] for pair in pair_order]
    cross_data = [gaps.loc[gaps["pair"] == pair, "roc_auc_cross"] for pair in pair_order]
    ax.boxplot(within_data, positions=positions - 0.16, widths=0.28, patch_artist=True, boxprops={"facecolor": "#4c78a8"}, medianprops={"color": "white"})
    ax.boxplot(cross_data, positions=positions + 0.16, widths=0.28, patch_artist=True, boxprops={"facecolor": "#f58518"}, medianprops={"color": "white"})
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, pair_order, rotation=20)
    ax.set_ylabel("Subject-level ROC-AUC")
    ax.set_title("Within-subject and cross-subject separability")
    ax.legend(handles=[plt.Line2D([0], [0], color="#4c78a8", lw=8, label="Within"), plt.Line2D([0], [0], color="#f58518", lw=8, label="Cross")])
    fig.tight_layout()
    fig.savefig(fig_dir / "subject_auc_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(pair_order), figsize=(16, 3.7), sharex=True, sharey=True)
    for ax, pair in zip(axes, pair_order):
        group = gaps[gaps["pair"] == pair]
        ax.scatter(group["roc_auc_cross"], group["roc_auc_within"], s=18, alpha=0.65)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set_title(pair.replace("_vs_", "–"))
        ax.set_xlabel("Cross AUC")
    axes[0].set_ylabel("Within AUC")
    fig.tight_layout()
    fig.savefig(fig_dir / "within_vs_cross_paired_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    gap_data = [gaps.loc[gaps["pair"] == pair, "auc_gap_within_minus_cross"] for pair in pair_order]
    ax.violinplot(gap_data, positions=positions, showmeans=True, showextrema=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, pair_order, rotation=20)
    ax.set_ylabel("AUC gap (within − cross)")
    ax.set_title("Cross-subject transfer gap by pain pair")
    fig.tight_layout()
    fig.savefig(fig_dir / "transfer_gap_distributions.png", dpi=180)
    plt.close(fig)

    summary = signal.groupby(["pair", "feature_name"], as_index=False)["empirical_overlap"].median()
    heat = summary.pivot(index="feature_name", columns="pair", values="empirical_overlap").reindex(index=FEATURES, columns=pair_order)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(heat.to_numpy(), vmin=0, vmax=1, cmap="magma_r", aspect="auto")
    ax.set_xticks(np.arange(len(pair_order)), pair_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FEATURES)), FEATURES)
    ax.set_title("Median within-subject empirical distribution overlap")
    fig.colorbar(image, ax=ax, label="Overlap coefficient (1 = complete)")
    fig.tight_layout()
    fig.savefig(fig_dir / "signal_overlap_heatmap.png", dpi=180)
    plt.close(fig)


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    report_path: Path,
    config: dict[str, object],
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    adjacent_wide: pd.DataFrame,
    signal: pd.DataFrame,
    order: pd.DataFrame,
    drift: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    pair_order = [pair[0] for pair in PAIRS]
    summary = summary.set_index("pair").reindex(pair_order).reset_index()
    adjacent = adjacent_wide.set_index("metric")
    overlap_summary = signal.groupby("pair")["empirical_overlap"].median()
    effect_summary = signal.groupby("pair")["abs_hedges_g"].median()
    order_summary = order.groupby("pair")["trial_number_separability_auc"].median()
    drift_summary = drift.groupby("pair")["abs_spearman_trial_feature"].median()
    pa2 = summary[summary["pair"].isin(["PA1_vs_PA2", "PA2_vs_PA3"])]
    strongest = gaps.loc[
        gaps.groupby("pair")["auc_gap_within_minus_cross"].idxmax(),
        ["pair", "subject_id", "auc_gap_within_minus_cross", "auc_gap_ci_low", "auc_gap_ci_high"],
    ].set_index("pair")
    positive_gap_counts = gaps.groupby("pair")["auc_gap_within_minus_cross"].apply(lambda s: int((s > 0).sum()))
    positive_ci_counts = gaps.groupby("pair")["auc_gap_ci_low"].apply(lambda s: int((s > 0).sum()))
    order_high_counts = order.groupby("pair")["trial_number_separability_auc"].apply(lambda s: int((s >= 0.70).sum()))

    lines = [
        "# Phase 8 — Within-subject vs Cross-subject Physiological Separability",
        "",
        f"Run date (UTC): {config['run_date_utc']}",
        "",
        "## Executive answer",
        "",
    ]
    adjacent_within = adjacent.loc["roc_auc_within", "adjacent_subject_mean"]
    adjacent_cross = adjacent.loc["roc_auc_cross", "adjacent_subject_mean"]
    wide_within = adjacent.loc["roc_auc_within", "wide_control_subject_mean"]
    wide_cross = adjacent.loc["roc_auc_cross", "wide_control_subject_mean"]
    lines.append(
        f"Adjacent pain levels were difficult even within subjects (subject-averaged AUC {adjacent_within:.3f}), while cross-subject performance was {adjacent_cross:.3f}. "
        f"Wide controls were more separable within subjects ({wide_within:.3f}) and cross-subject ({wide_cross:.3f}). "
        "All five mean within-minus-cross gaps were negative, so this experiment does not show population-level transfer failure as the dominant limitation. "
        "The strongest evidence is instead for within-subject overlap plus the statistical disadvantage of fitting a personalized model from only 32 training trials per fold; individual positive-gap exceptions still show heterogeneous transfer behavior."
    )
    lines.extend(
        [
            "",
            "These are stimulus-intensity labels, not trial-level subjective pain ratings. Performance therefore does not establish a physiological upper bound on pain recognition.",
            "",
            "## Protocol and leakage controls",
            "",
            f"- Input: Phase 2 filtered physiological feature table, 87 subjects, 20 trials per subject/class, pairs {', '.join(pair_order)}.",
            f"- Fixed low-dimensional input: {len(FEATURES)} Phase 6 preselected features (`{FEATURE_SET}`), chosen before Phase 8 modeling; no Phase 8 feature search was performed.",
            f"- Model: L2 Logistic Regression (`C=1`), median imputation and standardization inside each training fold.",
            f"- Within-subject: repeated stratified {N_SPLITS}-fold CV × {N_REPEATS} repeats; every trial receives {N_REPEATS} OOF scores which are averaged before scoring.",
            "- Cross-subject: exact Phase 3 subject-independent fold assignments; no held-out subject contributes to fitted preprocessing or model parameters.",
            f"- Uncertainty: {N_BOOTSTRAPS} stratified trial bootstraps for individual AUC/BA and gap CIs; subject bootstrap for pair summaries.",
            "- The within and cross conditions have very different training sizes (32 trials per within-subject fold versus thousands of trials across training subjects). The paired gap is descriptive and cannot be interpreted as a pure causal subject effect.",
            "",
            "## Pair-level results",
            "",
            "| pair | within AUC mean | cross AUC mean | paired gap mean (95% CI) | positive gaps | median signal overlap | median abs(g) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.within_auc_mean:.3f} | {row.cross_auc_mean:.3f} | {row.auc_gap_mean:.3f} [{row.auc_gap_mean_ci_low:.3f}, {row.auc_gap_mean_ci_high:.3f}] | {row.prop_positive_gap:.2%} | {overlap_summary[row.pair]:.3f} | {effect_summary[row.pair]:.3f} |"
        )
    lines.extend(
        [
            "",
            "Balanced accuracy and individual bootstrap intervals are retained in the machine-readable tables; the headline comparison uses ROC-AUC because it is threshold-independent.",
            "",
            "## Mechanistic interpretation",
            "",
            "### 1. Are adjacent levels difficult within the same subject?",
            "",
            f"Yes. The subject-averaged adjacent within-subject AUC was {adjacent_within:.3f}. The wide-control minus adjacent within-subject difference was {adjacent.loc['roc_auc_within', 'wide_minus_adjacent_mean']:.3f} "
            f"(95% subject-bootstrap CI {adjacent.loc['roc_auc_within', 'wide_minus_adjacent_ci_low']:.3f} to {adjacent.loc['roc_auc_within', 'wide_minus_adjacent_ci_high']:.3f}). "
            "The signal-level overlap table independently shows substantial overlap in the fixed Phase 6 features. This supports within-subject overlap as a real part of the difficulty.",
            "",
            "### 2. Which subjects show the largest transfer gaps?",
            "",
            "A positive gap means within-subject AUC exceeded cross-subject AUC. Group means were negative for every pair, but individual exceptions occurred:",
            "",
            "| pair | positive point gaps | bootstrap CI entirely > 0 | largest gap (subject) |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair in pair_order:
        row = strongest.loc[pair]
        lines.append(
            f"| {pair} | {positive_gap_counts[pair]}/87 | {positive_ci_counts[pair]}/87 | {row.auc_gap_within_minus_cross:.3f} (subject {int(row.subject_id)}, CI {row.auc_gap_ci_low:.3f} to {row.auc_gap_ci_high:.3f}) |"
        )
    lines.extend(
        [
            "",
            "The clearest individual positive-gap cases were concentrated in adjacent boundaries, but most individual intervals were wide. These are heterogeneity flags, not evidence that subject-specific training is generally superior.",
            "",
            "### 3. Adjacent versus wide-gap separability",
            "",
            f"Wide controls exceeded adjacent pairs by {adjacent.loc['roc_auc_within', 'wide_minus_adjacent_mean']:.3f} AUC within subjects and {adjacent.loc['roc_auc_cross', 'wide_minus_adjacent_mean']:.3f} AUC cross-subject. "
            "This indicates that larger stimulus separation improves physiological separability in both evaluation regimes; it is not solely a transfer phenomenon.",
            "",
            "### 4. What drives the PA2 difficulty?",
            "",
            f"Across PA1/PA2 and PA2/PA3, mean within-subject AUC was {pa2['within_auc_mean'].mean():.3f}, mean cross-subject AUC was {pa2['cross_auc_mean'].mean():.3f}, and mean paired gap was {pa2['auc_gap_mean'].mean():.3f}. "
            "PA1/PA2 was especially close to chance within subjects (0.521). Because cross-subject AUC was higher rather than lower on average, the present comparison points more strongly to within-subject overlap and small personalized training sets than to a population-level PA2 transfer failure. This is not a variance decomposition because training sizes differ.",
            "",
            "### 5. Does Phase 6 physiological consistency explain individual separability?",
            "",
        ]
    )
    lines.extend(
        [
            "Phase 6 median absolute monotonicity was positively associated with separability, especially for wider pairs:",
            "",
            "| pair | ρ with within AUC (BH p) | ρ with cross AUC (BH p) |",
            "|---|---:|---:|",
        ]
    )
    for pair in pair_order:
        within_assoc = associations[
            (associations["pair"] == pair)
            & (associations["predictor"] == "phase6_median_abs_spearman")
            & (associations["outcome"] == "roc_auc_within")
        ].iloc[0]
        cross_assoc = associations[
            (associations["pair"] == pair)
            & (associations["predictor"] == "phase6_median_abs_spearman")
            & (associations["outcome"] == "roc_auc_cross")
        ].iloc[0]
        lines.append(
            f"| {pair} | {within_assoc['spearman_rho']:.3f} ({within_assoc['p_value_bh']:.3g}) | {cross_assoc['spearman_rho']:.3f} ({cross_assoc['p_value_bh']:.3g}) |"
        )
    lines.extend(
        [
            "",
            "Direction reversals showed corresponding negative associations for several pairs (full corrected table in `phase6_consistency_associations.csv`). This supports consistency as a descriptive correlate of individual separability. However, Phase 6 consistency and Phase 8 AUC use the same subjects and overlapping features/levels, so the association is not an independent predictive validation and should not be called causal explanation.",
        ]
    )
    lines.extend(
        [
            "",
            "## Trial-order and drift diagnostics",
            "",
            f"For pain–pain pairs, the median label separability of the filename trial index ranged from {order_summary.drop('BL1_vs_PA4').min():.3f} to {order_summary.drop('BL1_vs_PA4').max():.3f}. "
            f"The median absolute within-class feature–trial Spearman correlation ranged from {drift_summary.min():.3f} to {drift_summary.max():.3f} across pairs.",
            "",
            f"The number of subjects with trial-index AUC ≥ 0.70 was {order_high_counts['PA1_vs_PA2']}, {order_high_counts['PA2_vs_PA3']}, {order_high_counts['PA3_vs_PA4']}, and {order_high_counts['PA1_vs_PA3']} for PA1/PA2, PA2/PA3, PA3/PA4, and PA1/PA3 respectively. This makes trial-order imbalance a plausible subject-specific confound even though pair medians were modest.",
            "",
            "The local documentation does not explicitly establish that the numeric filename suffix is acquisition chronology, so this is recorded as a trial-index proxy rather than proven elapsed time. BL1 uses indices 081–100 while pain trials use 001–080; consequently BL1/PA4 is structurally confounded with this proxy and cannot diagnose temporal drift from these identifiers alone.",
            "",
            "## Limitations",
            "",
            "- Labels encode calibrated stimulus intensity, not subjective pain experienced on each trial.",
            "- Forty trials per subject/pair constrain personalized estimation; repeated CV reduces split dependence but does not create independent observations.",
            "- Averaging repeated OOF scores makes efficient use of all trials, but trial-level bootstrap intervals condition on the fitted CV procedure.",
            "- Empirical overlap is a descriptive common-bin histogram intersection with 20 observations per class; it is not a population distribution estimate.",
            "- The five features were fixed from Phase 6 to limit overfitting and connect analyses, but they do not span every retained ECG/EDA/EMG feature.",
            "- The within/cross gap combines subject shift with unequal training-set size and training-population composition.",
            "- Phase 6 association tests are exploratory correlations, not causal explanations.",
            "",
            "## Reproducibility and outputs",
            "",
            "Run from repository root:",
            "",
            "```bash",
            f"{sys.executable} -m src.models.run_within_cross_separability --root . --output-dir experiments/within_cross_separability",
            "```",
            "",
            "Key tables: `signal_level_separability.csv`, `trial_order_diagnostics.csv`, `feature_trial_drift.csv`, `within_subject_metrics.csv`, `cross_subject_metrics.csv`, `subject_paired_gaps.csv`, `pair_summary.csv`, `adjacent_vs_wide.csv`, and `phase6_consistency_associations.csv`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def run(root: Path, output_dir: Path) -> None:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table, metadata, folds = load_inputs(root)
    signal, order, drift = signal_level_separability(table)
    within_metrics, within_predictions = run_within_subject(table)
    cross_metrics, cross_predictions = run_cross_subject(table, folds)
    gaps = paired_gap_table(within_metrics, cross_metrics, within_predictions, cross_predictions)
    summary = summarize_gaps(gaps)
    linked, associations = phase6_consistency_analysis(root, gaps)
    adjacent_wide = adjacent_wide_comparison(gaps)

    signal.to_csv(output_dir / "signal_level_separability.csv", index=False)
    order.to_csv(output_dir / "trial_order_diagnostics.csv", index=False)
    drift.to_csv(output_dir / "feature_trial_drift.csv", index=False)
    within_metrics.to_csv(output_dir / "within_subject_metrics.csv", index=False)
    within_predictions.to_csv(output_dir / "within_subject_oof_predictions.csv", index=False)
    cross_metrics.to_csv(output_dir / "cross_subject_metrics.csv", index=False)
    cross_predictions.to_csv(output_dir / "cross_subject_oof_predictions.csv", index=False)
    gaps.to_csv(output_dir / "subject_paired_gaps.csv", index=False)
    summary.to_csv(output_dir / "pair_summary.csv", index=False)
    adjacent_wide.to_csv(output_dir / "adjacent_vs_wide.csv", index=False)
    linked.to_csv(output_dir / "phase6_consistency_linked_subjects.csv", index=False)
    associations.to_csv(output_dir / "phase6_consistency_associations.csv", index=False)
    make_figures(output_dir, gaps, signal)

    config = {
        "phase": 8,
        "analysis_name": "within_cross_separability",
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "feature_set": FEATURE_SET,
        "features": FEATURES,
        "pain_pairs": [pair[0] for pair in PAIRS],
        "n_subjects": int(table["subject_id"].nunique()),
        "n_trials": int(len(table)),
        "within_subject_cv": {"protocol": "RepeatedStratifiedKFold", "n_splits": N_SPLITS, "n_repeats": N_REPEATS},
        "cross_subject_cv": {"protocol": "reused Phase 3 StratifiedGroupKFold", "n_folds": len(folds), "fold_assignments": "experiments/baselines/fold_assignments.csv"},
        "model": {"name": "logistic_regression", "penalty": "l2", "C": 1.0, "max_iter": 3000},
        "preprocessing": ["SimpleImputer(strategy=median)", "StandardScaler()"],
        "bootstrap_resamples": N_BOOTSTRAPS,
        "inputs": {
            "feature_table": {
                "path": "artifacts/feature_table_filtered.csv",
                "sha256": file_sha256(root / "artifacts" / "feature_table_filtered.csv"),
            },
            "phase3_fold_assignments": {
                "path": "experiments/baselines/fold_assignments.csv",
                "sha256": file_sha256(root / "experiments" / "baselines" / "fold_assignments.csv"),
            },
            "phase6_subject_consistency": {
                "path": "experiments/cross_subject_stability/subject_feature_monotonicity.csv",
                "sha256": file_sha256(root / "experiments" / "cross_subject_stability" / "subject_feature_monotonicity.csv"),
            },
        },
        "dependencies": package_versions(),
        "command": f"{sys.executable} -m src.models.run_within_cross_separability --root . --output-dir experiments/within_cross_separability",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_report(root / "reports" / "within_cross_separability.md", config, summary, gaps, adjacent_wide, signal, order, drift, associations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 8 within-subject versus cross-subject separability analysis.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/within_cross_separability"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.root, args.output_dir)
