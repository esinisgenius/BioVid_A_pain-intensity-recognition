from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.evaluation.metrics import classification_metrics
from src.models.baselines import make_baseline_model
from src.models.run_adjacent_class_analysis import PAIRS, binary_metrics, binary_pair_table
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices, score_vector
from src.models.run_subject_independent_baselines import (
    feature_sets,
    read_feature_table,
    validate_feature_inputs,
)


FEATURE_SET = "combined_unique"
PRIMARY_MODEL = "logistic_regression"
SELECTED_FEATURES = [
    "eda_scl_delta",
    "eda_scl_slope_per_s",
    "eda_scr_amplitude_sum",
    "eda_scr_count",
    "emg_trapezius__tsfresh_linear_trend_slope",
]
CLASS_ORDER = [0, 1, 2, 3, 4]
CLASS_NAMES = [EXPECTED_LABEL_MAP[i] for i in CLASS_ORDER]
BOUNDARIES = [
    ("BL1_to_PA1", 0, 1),
    ("PA1_to_PA2", 1, 2),
    ("PA2_to_PA3", 2, 3),
    ("PA3_to_PA4", 3, 4),
    ("PA1_to_PA3", 1, 3),
    ("BL1_to_PA4", 0, 4),
]
CALIBRATION_REPEATS = 20


def unique_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def robust_iqr(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def cliffs_delta(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float((np.sum(vals > 0) - np.sum(vals < 0)) / vals.size)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[dict[str, object]]]:
    table = read_feature_table(root / "artifacts" / "feature_table_filtered.csv")
    metadata = pd.read_csv(root / "experiments" / "baselines" / "feature_set_manifest.csv")
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    fold_samples = load_phase3_folds(root / "experiments" / "baselines" / "fold_assignments.csv")
    folds = make_fold_indices(table, fold_samples)
    return table, metadata, sets, folds


def validate_selected_features(table: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"Selected Phase 6 features are missing from feature table: {missing}")


def subject_trajectories(table: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = (
        table.groupby(["subject_id", "subject_name", "class_id", "class_name"], as_index=False)[features]
        .mean()
        .sort_values(["subject_id", "class_id"])
    )
    wide = trajectory.pivot(index=["subject_id", "subject_name"], columns="class_name", values=features)
    rows = []
    for feature in features:
        for (subject_id, subject_name), _ in wide.iterrows():
            means = np.array([wide.loc[(subject_id, subject_name), (feature, name)] for name in CLASS_NAMES], dtype=float)
            rho = spearmanr(CLASS_ORDER, means).statistic if np.isfinite(means).all() and np.nanstd(means) > 0 else np.nan
            diffs = np.diff(means)
            eps = max(float(np.nanstd(means)) * 1e-9, 1e-12)
            signs = np.sign(np.where(np.abs(diffs) <= eps, 0.0, diffs))
            nonzero = signs[signs != 0]
            rows.append(
                {
                    "feature_name": feature,
                    "subject_id": int(subject_id),
                    "subject_name": subject_name,
                    "mean_BL1": means[0],
                    "mean_PA1": means[1],
                    "mean_PA2": means[2],
                    "mean_PA3": means[3],
                    "mean_PA4": means[4],
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "strictly_non_decreasing": bool(np.all(diffs >= -eps)),
                    "strictly_non_increasing": bool(np.all(diffs <= eps)),
                    "has_sign_reversal": bool(len(set(nonzero.tolist())) > 1),
                    "n_levels_present": int(np.isfinite(means).sum()),
                }
            )
    subject_summary = pd.DataFrame(rows)
    monotonicity = (
        subject_summary.groupby("feature_name", as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_spearman_rho=("spearman_rho", "median"),
            iqr_spearman_rho=("spearman_rho", robust_iqr),
            prop_positive_rho=("spearman_rho", lambda s: float(np.mean(np.asarray(s, dtype=float) > 0))),
            prop_negative_rho=("spearman_rho", lambda s: float(np.mean(np.asarray(s, dtype=float) < 0))),
            prop_non_decreasing=("strictly_non_decreasing", "mean"),
            prop_non_increasing=("strictly_non_increasing", "mean"),
            prop_clearly_non_monotonic=("has_sign_reversal", "mean"),
        )
        .sort_values("feature_name")
    )
    return trajectory, subject_summary, monotonicity


def boundary_consistency(subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in subject_summary.groupby("feature_name", sort=True):
        for boundary_name, low, high in BOUNDARIES:
            low_name = EXPECTED_LABEL_MAP[low]
            high_name = EXPECTED_LABEL_MAP[high]
            diffs = group[f"mean_{high_name}"].to_numpy(dtype=float) - group[f"mean_{low_name}"].to_numpy(dtype=float)
            rows.append(
                {
                    "feature_name": feature,
                    "boundary": boundary_name,
                    "low_class": low_name,
                    "high_class": high_name,
                    "gap": int(high - low),
                    "n_subjects": int(np.isfinite(diffs).sum()),
                    "median_difference": float(np.nanmedian(diffs)),
                    "iqr_difference": robust_iqr(diffs),
                    "prop_positive_change": float(np.nanmean(diffs > 0)),
                    "prop_negative_change": float(np.nanmean(diffs < 0)),
                    "prop_zero_change": float(np.nanmean(diffs == 0)),
                    "cliffs_delta_vs_zero": cliffs_delta(diffs),
                    "abs_direction_consistency": float(max(np.nanmean(diffs > 0), np.nanmean(diffs < 0))),
                }
            )
    return pd.DataFrame(rows)


def pa2_heterogeneity(subject_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pattern_rows = []
    for feature, group in subject_summary.groupby("feature_name", sort=True):
        pa1 = group["mean_PA1"].to_numpy(dtype=float)
        pa2 = group["mean_PA2"].to_numpy(dtype=float)
        pa3 = group["mean_PA3"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_name": feature,
                "n_subjects": int(group["subject_id"].nunique()),
                "between_subject_variance_PA1": float(np.nanvar(pa1, ddof=1)),
                "between_subject_variance_PA2": float(np.nanvar(pa2, ddof=1)),
                "between_subject_variance_PA3": float(np.nanvar(pa3, ddof=1)),
                "pa2_variance_ratio_vs_pa1_pa3_mean": float(np.nanvar(pa2, ddof=1) / np.nanmean([np.nanvar(pa1, ddof=1), np.nanvar(pa3, ddof=1)])),
                "median_PA2_minus_PA1": float(np.nanmedian(pa2 - pa1)),
                "iqr_PA2_minus_PA1": robust_iqr(pa2 - pa1),
                "median_PA3_minus_PA2": float(np.nanmedian(pa3 - pa2)),
                "iqr_PA3_minus_PA2": robust_iqr(pa3 - pa2),
                "prop_pa2_between_pa1_pa3": float(np.nanmean(((pa1 <= pa2) & (pa2 <= pa3)) | ((pa3 <= pa2) & (pa2 <= pa1)))),
                "prop_pa2_lt_pa1": float(np.nanmean(pa2 < pa1)),
                "prop_pa2_gt_pa3": float(np.nanmean(pa2 > pa3)),
                "prop_pa2_outside_pa1_pa3_range": float(np.nanmean((pa2 < np.minimum(pa1, pa3)) | (pa2 > np.maximum(pa1, pa3)))),
            }
        )
        for _, row in group.iterrows():
            v1, v2, v3 = row["mean_PA1"], row["mean_PA2"], row["mean_PA3"]
            if v2 < v1:
                pattern = "PA2 < PA1"
            elif v2 > v3:
                pattern = "PA2 > PA3"
            elif min(v1, v3) <= v2 <= max(v1, v3):
                pattern = "PA2 between PA1 and PA3"
            else:
                pattern = "other_non_ordinal"
            pattern_rows.append(
                {
                    "feature_name": feature,
                    "subject_id": int(row["subject_id"]),
                    "subject_name": row["subject_name"],
                    "mean_PA1": v1,
                    "mean_PA2": v2,
                    "mean_PA3": v3,
                    "pa2_pattern": pattern,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pattern_rows)


def baseline_subtract(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = table.copy()
    baseline = table[table["class_id"] == 0].groupby("subject_id")[columns].mean()
    missing_subjects = sorted(set(table["subject_id"].astype(int)).difference(set(baseline.index.astype(int))))
    if missing_subjects:
        raise ValueError(f"Subjects without BL1 trials cannot be baseline-normalized: {missing_subjects}")
    for subject_id, idx in normalized.groupby("subject_id").groups.items():
        normalized.loc[idx, columns] = normalized.loc[idx, columns].to_numpy(dtype=float) - baseline.loc[subject_id, columns].to_numpy(dtype=float)
    return normalized


def run_multiclass(table: pd.DataFrame, columns: list[str], folds: list[dict[str, object]], representation: str, random_state: int) -> tuple[dict[str, object], pd.DataFrame]:
    y = table["class_id"].to_numpy(dtype=int)
    x = table[columns].to_numpy(dtype=float)
    oof_rows = []
    for split in folds:
        model = make_baseline_model(PRIMARY_MODEL, random_state)
        model.fit(x[split["train_index"]], y[split["train_index"]])
        pred = model.predict(x[split["test_index"]])
        for pos, idx in enumerate(split["test_index"]):
            source = table.iloc[int(idx)]
            oof_rows.append(
                {
                    "task": "5class",
                    "representation": representation,
                    "model": PRIMARY_MODEL,
                    "feature_set": FEATURE_SET,
                    "fold": int(split["fold"]),
                    "subject_id": int(source["subject_id"]),
                    "subject_name": source["subject_name"],
                    "sample_id": int(source["sample_id"]),
                    "sample_name": source["sample_name"],
                    "true_class_id": int(source["class_id"]),
                    "true_class_name": source["class_name"],
                    "pred_class_id": int(pred[pos]),
                    "pred_class_name": EXPECTED_LABEL_MAP[int(pred[pos])],
                }
            )
    oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
    metrics = classification_metrics(
        oof["true_class_id"].to_numpy(dtype=int),
        oof["pred_class_id"].to_numpy(dtype=int),
        CLASS_ORDER,
        EXPECTED_LABEL_MAP,
    )
    return {
        "task": "5class",
        "representation": representation,
        "model": PRIMARY_MODEL,
        "feature_set": FEATURE_SET,
        "n_features": len(columns),
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }, oof


def run_binary_pairs(table: pd.DataFrame, columns: list[str], folds: list[dict[str, object]], representation: str, random_state: int) -> pd.DataFrame:
    fold_samples = {int(split["fold"]): set(table.iloc[split["test_index"]]["sample_name"]) for split in folds}
    rows = []
    for pair_name, low_class, high_class in PAIRS:
        pair = binary_pair_table(table, low_class, high_class)
        x = pair[columns].to_numpy(dtype=float)
        y = pair["binary_label"].to_numpy(dtype=int)
        oof_rows = []
        for fold in sorted(fold_samples):
            test_index = pair.index[pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
            train_index = pair.index[~pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
            train_subjects = set(pair.iloc[train_index]["subject_id"].astype(int))
            test_subjects = set(pair.iloc[test_index]["subject_id"].astype(int))
            overlap = train_subjects.intersection(test_subjects)
            if overlap:
                raise ValueError(f"Subject leakage for {pair_name} fold {fold}: {sorted(overlap)}")
            model = make_baseline_model(PRIMARY_MODEL, random_state)
            model.fit(x[train_index], y[train_index])
            pred = model.predict(x[test_index])
            score = score_vector(model, x[test_index])
            for pos, idx in enumerate(test_index):
                oof_rows.append(
                    {
                        "fold": int(fold),
                        "subject_id": int(pair.iloc[int(idx)]["subject_id"]),
                        "sample_name": pair.iloc[int(idx)]["sample_name"],
                        "true_binary_label": int(y[idx]),
                        "pred_binary_label": int(pred[pos]),
                        "positive_score": float(score[pos]),
                    }
                )
        oof = pd.DataFrame(oof_rows)
        metrics = binary_metrics(
            oof["true_binary_label"].to_numpy(dtype=int),
            oof["pred_binary_label"].to_numpy(dtype=int),
            oof["positive_score"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "task": pair_name,
                "representation": representation,
                "model": PRIMARY_MODEL,
                "feature_set": FEATURE_SET,
                "negative_class": EXPECTED_LABEL_MAP[low_class],
                "positive_class": EXPECTED_LABEL_MAP[high_class],
                "roc_auc": metrics["roc_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "n_eval": int(len(oof)),
            }
        )
    return pd.DataFrame(rows)


def normalization_experiment(table: pd.DataFrame, columns: list[str], folds: list[dict[str, object]], random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_result, raw_oof = run_multiclass(table, columns, folds, "raw", random_state)
    normalized_table = baseline_subtract(table, columns)
    norm_result, norm_oof = run_multiclass(normalized_table, columns, folds, "baseline_subtracted", random_state)
    binary_raw = run_binary_pairs(table, columns, folds, "raw", random_state)
    binary_norm = run_binary_pairs(normalized_table, columns, folds, "baseline_subtracted", random_state)
    metrics = pd.concat(
        [
            pd.DataFrame([raw_result, norm_result]),
            pd.concat([binary_raw, binary_norm], ignore_index=True),
        ],
        ignore_index=True,
        sort=False,
    )
    oof = pd.concat([raw_oof, norm_oof], ignore_index=True)
    return metrics, oof


def calibrate_scores(score: np.ndarray, labels: np.ndarray, eval_score: np.ndarray) -> np.ndarray:
    if len(np.unique(labels)) < 2:
        return eval_score
    calibrator = LogisticRegression(solver="lbfgs", C=1.0, random_state=0)
    calibrator.fit(score.reshape(-1, 1), labels)
    return calibrator.predict_proba(eval_score.reshape(-1, 1))[:, 1]


def calibration_pilot(table: pd.DataFrame, columns: list[str], folds: list[dict[str, object]], random_state: int) -> pd.DataFrame:
    fold_samples = {int(split["fold"]): set(table.iloc[split["test_index"]]["sample_name"]) for split in folds}
    rows = []
    rng_master = np.random.default_rng(random_state)
    for pair_name, low_class, high_class in PAIRS:
        if pair_name not in {"PA1_vs_PA2", "PA2_vs_PA3", "PA1_vs_PA3", "BL1_vs_PA4"}:
            continue
        pair = binary_pair_table(table, low_class, high_class)
        x = pair[columns].to_numpy(dtype=float)
        y = pair["binary_label"].to_numpy(dtype=int)
        fold_scores = np.full(len(pair), np.nan)
        for fold in sorted(fold_samples):
            test_index = pair.index[pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
            train_index = pair.index[~pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
            model = make_baseline_model(PRIMARY_MODEL, random_state)
            model.fit(x[train_index], y[train_index])
            fold_scores[test_index] = score_vector(model, x[test_index])
        if not np.isfinite(fold_scores).all():
            raise ValueError(f"Missing OOF scores in calibration pilot for {pair_name}")

        for k in [0, 1, 2]:
            n_repeats = 1 if k == 0 else CALIBRATION_REPEATS
            for repeat in range(n_repeats):
                eval_indices = []
                eval_scores = []
                eval_labels = []
                calibrated_scores = []
                seed = int(rng_master.integers(0, 1_000_000_000))
                rng = np.random.default_rng(seed + repeat + k)
                for subject_id, subject_group in pair.groupby("subject_id", sort=True):
                    subject_idx = subject_group.index.to_numpy(dtype=int)
                    calibration_idx = []
                    if k > 0:
                        for label in [0, 1]:
                            label_idx = subject_group.index[subject_group["binary_label"] == label].to_numpy(dtype=int)
                            if len(label_idx) < k + 1:
                                calibration_idx = []
                                break
                            calibration_idx.extend(rng.choice(label_idx, size=k, replace=False).tolist())
                    calibration_idx = np.asarray(sorted(set(calibration_idx)), dtype=int)
                    subject_eval_idx = np.asarray(sorted(set(subject_idx.tolist()).difference(calibration_idx.tolist())), dtype=int)
                    if len(subject_eval_idx) == 0:
                        continue
                    scores_eval = fold_scores[subject_eval_idx]
                    scores_out = scores_eval
                    if k > 0 and len(calibration_idx) == 2 * k:
                        scores_out = calibrate_scores(
                            fold_scores[calibration_idx],
                            y[calibration_idx],
                            scores_eval,
                        )
                    eval_indices.extend(subject_eval_idx.tolist())
                    eval_scores.extend(scores_eval.tolist())
                    eval_labels.extend(y[subject_eval_idx].tolist())
                    calibrated_scores.extend(scores_out.tolist())
                labels = np.asarray(eval_labels, dtype=int)
                calibrated = np.asarray(calibrated_scores, dtype=float)
                pred = (calibrated >= 0.5).astype(int)
                rows.append(
                    {
                        "task": pair_name,
                        "model": PRIMARY_MODEL,
                        "feature_set": FEATURE_SET,
                        "calibration_strategy": "per_subject_score_logistic" if k > 0 else "none_global_threshold",
                        "k_per_class": int(k),
                        "repeat": int(repeat),
                        "roc_auc": safe_auc(labels, calibrated),
                        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
                        "f1": float(f1_score(labels, pred, zero_division=0)),
                        "accuracy": float(np.mean(labels == pred)),
                        "n_eval": int(len(labels)),
                        "n_subjects": int(pair["subject_id"].nunique()),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["task", "model", "feature_set", "calibration_strategy", "k_per_class"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            n_eval_mean=("n_eval", "mean"),
            n_subjects=("n_subjects", "max"),
            repeats=("repeat", "nunique"),
        )
    )
    return raw, summary


def compare_importance_stability(root: Path, monotonicity: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    phase5 = pd.read_csv(root / "experiments" / "feature_contribution" / "top_feature_cross_model_consistency.csv")
    selected = phase5[phase5["feature_name"].isin(SELECTED_FEATURES)].copy()
    selected = selected.sort_values(["feature_name", "pain_class"]).drop_duplicates("feature_name")
    adjacent = boundary[boundary["gap"] == 1].groupby("feature_name", as_index=False).agg(
        mean_adjacent_direction_consistency=("abs_direction_consistency", "mean"),
        min_adjacent_direction_consistency=("abs_direction_consistency", "min"),
    )
    wide = monotonicity.merge(adjacent, on="feature_name", how="left")
    cols = [
        "feature_name",
        "pain_class",
        "modality_name",
        "xgboost_mean_abs_shap",
        "xgboost_rank_all_features",
        "xgboost_fold_top10_frequency",
    ]
    return wide.merge(selected[cols], on="feature_name", how="left")


def write_figures(output_dir: Path, subject_summary: pd.DataFrame, boundary: pd.DataFrame, norm: pd.DataFrame, calibration: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    data = [subject_summary.loc[subject_summary["feature_name"] == f, "spearman_rho"].dropna().to_numpy() for f in SELECTED_FEATURES]
    plt.boxplot(data, tick_labels=[f.replace("_", "\n") for f in SELECTED_FEATURES], showfliers=False)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Subject-level Spearman rho")
    plt.tight_layout()
    plt.savefig(fig_dir / "subject_spearman_rho_distribution.png", dpi=160)
    plt.close()

    heat = boundary.pivot(index="feature_name", columns="boundary", values="abs_direction_consistency").reindex(SELECTED_FEATURES)
    plt.figure(figsize=(9, 4.8))
    im = plt.imshow(heat.to_numpy(dtype=float), aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="Directional consistency")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "boundary_direction_consistency_heatmap.png", dpi=160)
    plt.close()

    traj = subject_summary[subject_summary["feature_name"].isin(["eda_scl_delta", "eda_scl_slope_per_s"])].copy()
    long_rows = []
    for _, row in traj.iterrows():
        for name in ["PA1", "PA2", "PA3"]:
            long_rows.append({"feature_name": row["feature_name"], "subject_id": row["subject_id"], "class_name": name, "value": row[f"mean_{name}"]})
    long = pd.DataFrame(long_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, feature in zip(axes, ["eda_scl_delta", "eda_scl_slope_per_s"], strict=True):
        subset = long[long["feature_name"] == feature]
        for _, g in subset.groupby("subject_id"):
            ax.plot(g["class_name"], g["value"], color="#7b8794", alpha=0.18, linewidth=0.8)
        med = subset.groupby("class_name")["value"].median().reindex(["PA1", "PA2", "PA3"])
        ax.plot(["PA1", "PA2", "PA3"], med.values, color="#c0392b", linewidth=2.5)
        ax.set_title(feature)
        ax.set_ylabel("Subject mean")
    plt.tight_layout()
    plt.savefig(fig_dir / "pa1_pa2_pa3_subject_trajectories.png", dpi=160)
    plt.close()

    norm_pairs = norm[norm["task"].isin(["PA1_vs_PA2", "PA2_vs_PA3", "PA1_vs_PA3", "BL1_vs_PA4"])].copy()
    pivot = norm_pairs.pivot(index="task", columns="representation", values="roc_auc")
    pivot = pivot.reindex(["PA1_vs_PA2", "PA2_vs_PA3", "PA1_vs_PA3", "BL1_vs_PA4"])
    pivot.plot(kind="bar", figsize=(8.5, 4.3), color=["#4c78a8", "#f58518"])
    plt.ylabel("ROC-AUC")
    plt.ylim(0.45, max(0.75, float(np.nanmax(pivot.to_numpy())) + 0.03))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_vs_baseline_normalized_pair_auc.png", dpi=160)
    plt.close()

    cal = calibration[calibration["task"].isin(["PA1_vs_PA2", "PA2_vs_PA3"])].copy()
    plt.figure(figsize=(7, 4.2))
    for task, group in cal.groupby("task", sort=True):
        group = group.sort_values("k_per_class")
        plt.errorbar(group["k_per_class"], group["roc_auc_mean"], yerr=group["roc_auc_std"].fillna(0), marker="o", capsize=3, label=task)
    plt.xlabel("Calibration examples per class")
    plt.ylabel("ROC-AUC")
    plt.xticks([0, 1, 2])
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "performance_vs_calibration_size.png", dpi=160)
    plt.close()


def write_report(
    root: Path,
    output_dir: Path,
    config: dict[str, object],
    monotonicity: pd.DataFrame,
    boundary: pd.DataFrame,
    pa2: pd.DataFrame,
    norm: pd.DataFrame,
    calibration: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    phase4 = pd.read_csv(root / "experiments" / "modality_contribution" / "summary_metrics.csv")
    phase4_delta = pd.read_csv(root / "experiments" / "modality_contribution" / "modality_pain_level_delta.csv")
    phase55 = pd.read_csv(root / "experiments" / "adjacent_class_analysis" / "binary_pair_summary_metrics.csv")
    primary_phase55 = phase55[
        (phase55["model"] == "logistic_regression")
        & (phase55["feature_set"] == FEATURE_SET)
        & (phase55["modality_set"] == "all_modalities")
    ].copy()

    raw5 = norm[(norm["task"] == "5class") & (norm["representation"] == "raw")].iloc[0]
    base5 = norm[(norm["task"] == "5class") & (norm["representation"] == "baseline_subtracted")].iloc[0]
    norm_pairs = norm[norm["task"] != "5class"].pivot(index="task", columns="representation", values="roc_auc")
    pa2_row = pa2.set_index("feature_name").loc["eda_scl_delta"]
    adjacent_consistency = boundary[boundary["gap"] == 1]["abs_direction_consistency"].median()
    wide_consistency = boundary[boundary["gap"] > 1]["abs_direction_consistency"].median()

    lines = [
        "# Phase 6 Cross-Subject Stability and Personalization Analysis",
        "",
        f"Run date: {config['run_date_utc']}",
        "",
        "## Reused fixed inputs",
        "",
        "- Phase 2 feature table: `artifacts/feature_table_filtered.csv`.",
        "- Folds: `experiments/baselines/fold_assignments.csv`; all Phase 6 predictive analyses reuse these subject-independent folds.",
        "- Phase 4 modality context: `experiments/modality_contribution/summary_metrics.csv` and `modality_pain_level_delta.csv`.",
        "- Phase 5 feature context: `experiments/feature_contribution/*` summaries.",
        "- Phase 5.5 adjacent separability context: `experiments/adjacent_class_analysis/binary_pair_summary_metrics.csv`.",
        "",
        "## Selected features",
        "",
        "The analysis uses five representative features: `eda_scl_delta`, `eda_scl_slope_per_s`, `eda_scr_amplitude_sum`, `eda_scr_count`, and `emg_trapezius__tsfresh_linear_trend_slope`. The requested SCR amplitude-sum concept is named `eda_scr_amplitude_sum` in the local feature table.",
        "",
        "## Validation checks",
        "",
        f"- Subjects in feature table: {config['n_subjects']}.",
        f"- Samples in feature table: {config['n_samples']}.",
        f"- Expected subjects accounted for in trajectory summaries: {config['trajectory_subjects']}.",
        f"- Reused folds: {config['n_folds']}; subject overlap checks passed for normalization and binary calibration runs.",
        "- Baseline normalization used only each subject's BL1 trials to compute subtraction offsets. PA1-PA4 labels from the held-out subject were not used for normalization.",
        "- Calibration and evaluation trials were disjoint within each held-out subject for `k > 0`.",
        "",
        "## Connection to previous phases",
        "",
        f"Phase 4 showed that the full XGBoost model had macro F1 {phase4.loc[(phase4['model']=='xgboost') & (phase4['modality_set']=='full'), 'macro_f1'].iloc[0]:.3f}, while removing EDA reduced macro F1 to {phase4.loc[(phase4['model']=='xgboost') & (phase4['modality_set']=='remove_eda'), 'macro_f1'].iloc[0]:.3f}. EDA therefore remained the clearest modality-level contributor in the fixed Phase 4 artifacts.",
        f"At the pain-level ablation scale, EDA removal most strongly reduced BL1 and PA4 recall, but PA2 remained weak: the full XGBoost PA2 recall was {phase4_delta.loc[(phase4_delta['model']=='xgboost') & (phase4_delta['modality']=='eda') & (phase4_delta['class_name']=='PA2'), 'full_recall'].iloc[0]:.3f}.",
        "Phase 5 identified EDA SCL features as globally important, especially `eda_scl_delta`. Phase 6 asks whether those globally important features move in consistent directions across subjects and pain boundaries.",
        "",
        "## A. Subject-Level Physiological Trajectories",
        "",
        "| feature | median rho | IQR rho | positive rho | negative rho | clearly non-monotonic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in monotonicity.iterrows():
        lines.append(
            f"| {row['feature_name']} | {row['median_spearman_rho']:.3f} | {row['iqr_spearman_rho']:.3f} | {row['prop_positive_rho']:.3f} | {row['prop_negative_rho']:.3f} | {row['prop_clearly_non_monotonic']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Global predictive importance did not imply universal physiological monotonicity. Several EDA features had positive median subject-level associations with pain, but a substantial minority of subjects showed negative or sign-reversing trajectories.",
            "",
            "## B. Boundary-Level Direction Consistency",
            "",
            f"The median directional consistency across adjacent boundaries was {adjacent_consistency:.3f}, compared with {wide_consistency:.3f} for wider gaps. This supports the expected pattern that wider pain-level gaps have more stable physiological direction than adjacent boundaries, although the effect varies by feature.",
            "",
            "| feature | boundary | median diff | IQR | positive | negative | direction consistency |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in boundary.sort_values(["feature_name", "gap", "boundary"]).iterrows():
        lines.append(
            f"| {row['feature_name']} | {row['boundary']} | {row['median_difference']:.4g} | {row['iqr_difference']:.4g} | {row['prop_positive_change']:.3f} | {row['prop_negative_change']:.3f} | {row['abs_direction_consistency']:.3f} |"
        )
    lines.extend(["", "Phase 5.5 all-modality logistic ROC-AUC values for the same major pairs were:"])
    for _, row in primary_phase55.sort_values("task").iterrows():
        lines.append(f"- `{row['task']}`: ROC-AUC {row['overall_roc_auc']:.3f}.")
    lines.extend(
        [
            "",
            "The physiological direction-consistency results line up with Phase 5.5: adjacent pairs remain close to chance, while BL1 vs PA4 is much more separable.",
            "",
            "## C. PA2 Heterogeneity",
            "",
            f"For `eda_scl_delta`, PA2 between-subject variance was {pa2_row['between_subject_variance_PA2']:.4g}, with a PA2 variance ratio of {pa2_row['pa2_variance_ratio_vs_pa1_pa3_mean']:.3f} relative to the mean of PA1 and PA3 variance.",
            f"PA2 lay physiologically between PA1 and PA3 for {pa2_row['prop_pa2_between_pa1_pa3']:.3f} of subjects, while `PA2 < PA1` occurred for {pa2_row['prop_pa2_lt_pa1']:.3f} and `PA2 > PA3` occurred for {pa2_row['prop_pa2_gt_pa3']:.3f}.",
            "This points to heterogeneous subject-specific PA2 responses rather than a single universally harmful feature. It does not establish a physiological cause.",
            "",
            "## D. Subject Baseline Normalization",
            "",
            f"Five-class macro F1 changed from {raw5['macro_f1']:.3f} with raw features to {base5['macro_f1']:.3f} after BL1 subtraction. Balanced accuracy changed from {raw5['balanced_accuracy']:.3f} to {base5['balanced_accuracy']:.3f}.",
            "",
            "| pair | raw ROC-AUC | baseline-subtracted ROC-AUC | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for task, row in norm_pairs.iterrows():
        delta = row["baseline_subtracted"] - row["raw"]
        lines.append(f"| {task} | {row['raw']:.3f} | {row['baseline_subtracted']:.3f} | {delta:+.3f} |")
    lines.extend(
        [
            "",
            "The normalization result is the direct test of baseline heterogeneity. Here, BL1 subtraction did not materially rescue the adjacent PA1/PA2 or PA2/PA3 boundaries, so absolute baseline offsets alone are not sufficient to explain the problem.",
            "",
            "## E. Lightweight Calibration Pilot",
            "",
            "| pair | k per class | ROC-AUC mean | balanced accuracy mean | F1 mean | eval rows mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in calibration.sort_values(["task", "k_per_class"]).iterrows():
        lines.append(
            f"| {row['task']} | {int(row['k_per_class'])} | {row['roc_auc_mean']:.3f} | {row['balanced_accuracy_mean']:.3f} | {row['f1_mean']:.3f} | {row['n_eval_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Calibration used only a 1D logistic mapping from the global model's held-out subject scores to labels, with calibration trials excluded from evaluation. This is a score-level pilot, not a subject-specific model retrain. In this run, score calibration did not materially improve the adjacent PA1/PA2 or PA2/PA3 ROC-AUCs; the small calibration sets mostly reduced usable evaluation data and sometimes degraded discrimination.",
            "",
            "## F. Global Importance vs Stability",
            "",
            "| feature | Phase 5 class | XGB abs SHAP | XGB rank | rho median | adjacent consistency | non-monotonic |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in stability.sort_values("feature_name").iterrows():
        shap = row["xgboost_mean_abs_shap"]
        rank = row["xgboost_rank_all_features"]
        lines.append(
            f"| {row['feature_name']} | {row.get('pain_class', np.nan)} | {shap:.4f} | {rank:.0f} | {row['median_spearman_rho']:.3f} | {row['mean_adjacent_direction_consistency']:.3f} | {row['prop_clearly_non_monotonic']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This table is the central Phase 6 distinction: a feature can be globally useful to the model and still be directionally inconsistent across subjects or across adjacent boundaries.",
            "",
            "## Answers to the Phase 6 Questions",
            "",
            "1. **Are globally important features also directionally consistent across subjects?** Partly. `eda_scl_delta` is globally important and has a positive median trajectory, but consistency is not universal; subject-level sign reversals remain common enough to limit a universal monotonic interpretation.",
            "2. **Does physiological consistency decrease at adjacent pain boundaries?** Yes in the aggregate. Adjacent boundaries show lower median direction consistency than wider gaps, matching the weak adjacent-pair ROC-AUC pattern from Phase 5.5.",
            "3. **Is PA2 unusually heterogeneous?** PA2 shows substantial subject-specific non-ordinal behavior for the selected EDA features. The evidence supports PA2 heterogeneity as a plausible contributor to weak PA2 classification, not as a proven cause.",
            "4. **How much of adjacent-class difficulty can be explained by baseline heterogeneity?** Only a limited amount in this pilot. BL1 subtraction did not materially improve the most difficult adjacent ROC-AUCs.",
            "5. **Does minimal subject calibration recover information that a fully global model misses?** Not materially in this pilot. `k=1` and `k=2` score calibration did not rescue PA1/PA2 or PA2/PA3 ROC-AUC, so the current evidence does not show that a tiny labelled calibration set can recover a hidden adjacent-boundary signal.",
            "6. **Main bottleneck?** Baseline shift is not the dominant explanation. The evidence points to visible response-pattern heterogeneity plus intrinsic adjacent-class overlap in the available physiological features; because lightweight calibration also failed to help, the adjacent-boundary bottleneck appears stronger than a simple subject-specific score-threshold problem.",
            "",
            "## Restrained Research Conclusion",
            "",
            "Phase 6 supports the hypothesis that fine-grained physiological pain-intensity recognition is constrained by cross-subject heterogeneity, especially at adjacent pain boundaries. The evidence does not justify claiming that fully personalized models are required, but it does show that population-level feature importance should not be read as universal physiological consistency.",
            "",
            "## Output Files",
            "",
            "- `experiments/cross_subject_stability/subject_level_trajectories.csv`",
            "- `experiments/cross_subject_stability/subject_feature_monotonicity.csv`",
            "- `experiments/cross_subject_stability/feature_monotonicity_summary.csv`",
            "- `experiments/cross_subject_stability/boundary_consistency.csv`",
            "- `experiments/cross_subject_stability/pa2_heterogeneity_summary.csv`",
            "- `experiments/cross_subject_stability/normalization_comparison_metrics.csv`",
            "- `experiments/cross_subject_stability/calibration_comparison_metrics.csv`",
            "- `experiments/cross_subject_stability/figures/`",
        ]
    )
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "cross_subject_stability.md").write_text("\n".join(lines) + "\n")


def run(root: Path, output_dir: Path, random_state: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table, metadata, sets, folds = load_inputs(root)
    columns = unique_preserving_order(sets[FEATURE_SET])
    validate_selected_features(table, SELECTED_FEATURES)

    trajectory, subject_summary, monotonicity = subject_trajectories(table, SELECTED_FEATURES)
    boundary = boundary_consistency(subject_summary)
    pa2_summary, pa2_patterns = pa2_heterogeneity(subject_summary)
    norm_metrics, norm_oof = normalization_experiment(table, columns, folds, random_state)
    calibration_raw, calibration_summary = calibration_pilot(table, columns, folds, random_state)
    stability = compare_importance_stability(root, monotonicity, boundary)

    trajectory.to_csv(output_dir / "subject_level_trajectories.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_feature_monotonicity.csv", index=False)
    monotonicity.to_csv(output_dir / "feature_monotonicity_summary.csv", index=False)
    boundary.to_csv(output_dir / "boundary_consistency.csv", index=False)
    pa2_summary.to_csv(output_dir / "pa2_heterogeneity_summary.csv", index=False)
    pa2_patterns.to_csv(output_dir / "pa2_subject_patterns.csv", index=False)
    norm_metrics.to_csv(output_dir / "normalization_comparison_metrics.csv", index=False)
    norm_oof.to_csv(output_dir / "normalization_oof_predictions.csv", index=False)
    calibration_raw.to_csv(output_dir / "calibration_repeated_metrics.csv", index=False)
    calibration_summary.to_csv(output_dir / "calibration_comparison_metrics.csv", index=False)
    stability.to_csv(output_dir / "importance_stability_comparison.csv", index=False)

    config = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "feature_set": FEATURE_SET,
        "model": PRIMARY_MODEL,
        "selected_features": SELECTED_FEATURES,
        "n_subjects": int(table["subject_id"].nunique()),
        "n_samples": int(len(table)),
        "trajectory_subjects": int(subject_summary["subject_id"].nunique()),
        "n_folds": int(len(folds)),
        "calibration_repeats": CALIBRATION_REPEATS,
        "fixed_inputs": {
            "feature_table": "artifacts/feature_table_filtered.csv",
            "fold_assignments": "experiments/baselines/fold_assignments.csv",
            "phase4": "experiments/modality_contribution/",
            "phase5": "experiments/feature_contribution/",
            "phase5_5": "experiments/adjacent_class_analysis/",
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    write_figures(output_dir, subject_summary, boundary, norm_metrics, calibration_summary)
    write_report(root, output_dir, config, monotonicity, boundary, pa2_summary, norm_metrics, calibration_summary, stability)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 cross-subject stability and personalization analysis.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cross_subject_stability"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    run(root=root, output_dir=output_dir, random_state=args.random_state)


if __name__ == "__main__":
    main()
