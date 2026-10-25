from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, roc_auc_score

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs


CLASS_IDS = np.arange(5, dtype=int)
CLASS_NAMES = [EXPECTED_LABEL_MAP[int(i)] for i in CLASS_IDS]
PROBABILITY_COLUMNS = [f"proba_{name}" for name in CLASS_NAMES]
EPS = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, EPS, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    logits = np.log(normalize_probabilities(probabilities)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, y_true: np.ndarray, bounds: tuple[float, float]) -> float:
    probabilities = normalize_probabilities(probabilities)
    y_true = np.asarray(y_true, dtype=int)

    def objective(temperature: float) -> float:
        return float(log_loss(y_true, apply_temperature(probabilities, temperature), labels=CLASS_IDS))

    result = minimize_scalar(objective, method="bounded", bounds=bounds, options={"xatol": 1e-6})
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x)


def ece_score(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    probabilities = normalize_probabilities(probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(confidence, edges[1:-1], right=True), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(CLASS_IDS), dtype=float)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((normalize_probabilities(probabilities) - one_hot) ** 2, axis=1)))


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int) -> dict[str, float]:
    probabilities = normalize_probabilities(probabilities)
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(np.mean(prediction == y_true)),
        "ece": ece_score(y_true, probabilities, n_bins),
        "brier": multiclass_brier(y_true, probabilities),
        "nll": float(log_loss(y_true, probabilities, labels=CLASS_IDS)),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def reliability_bins(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int) -> pd.DataFrame:
    probabilities = normalize_probabilities(probabilities)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.digitize(confidence, edges[1:-1], right=True), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = assignments == bin_id
        rows.append(
            {
                "bin": bin_id + 1,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    return -np.sum(probabilities * np.log(probabilities), axis=1) / math.log(probabilities.shape[1])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not 0 < alpha < 1:
        raise ValueError("Conformal scores must be non-empty and alpha must be in (0, 1)")
    rank = min(scores.size, int(math.ceil((scores.size + 1) * (1.0 - alpha))))
    return float(np.partition(scores, rank - 1)[rank - 1])


def standard_conformal_sets(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (1.0 - normalize_probabilities(probabilities)) <= (float(threshold) + 1e-12)


def ordinal_conformal_intervals(probabilities: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    expected = normalize_probabilities(probabilities) @ CLASS_IDS.astype(float)
    lower = np.maximum(0, np.ceil(expected - radius - 1e-12)).astype(int)
    upper = np.minimum(int(CLASS_IDS.max()), np.floor(expected + radius + 1e-12)).astype(int)
    return lower, upper


def interval_membership(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    grid = CLASS_IDS.reshape(1, -1)
    return (grid >= lower.reshape(-1, 1)) & (grid <= upper.reshape(-1, 1))


def split_model_and_calibration_subjects(
    train_subjects: Iterable[int], calibration_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    subjects = np.asarray(sorted({int(value) for value in train_subjects}), dtype=int)
    if len(subjects) < 4:
        raise ValueError("At least four training subjects are required for an inner subject split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects)
    n_calibration = max(1, int(round(len(subjects) * calibration_fraction)))
    n_calibration = min(n_calibration, len(subjects) - 1)
    calibration = sorted(shuffled[:n_calibration].astype(int).tolist())
    model_train = sorted(shuffled[n_calibration:].astype(int).tolist())
    if set(model_train).intersection(calibration):
        raise AssertionError("Inner subject overlap")
    return model_train, calibration


def selection_metrics(y_true: np.ndarray, prediction: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    accepted = np.asarray(accepted, dtype=bool)
    if not accepted.any():
        return {"n_accepted": 0, "accuracy": np.nan, "risk": np.nan, "mae": np.nan, "severe_error_rate": np.nan}
    errors = np.abs(prediction[accepted] - y_true[accepted])
    accuracy = float(np.mean(errors == 0))
    return {
        "n_accepted": int(accepted.sum()),
        "accuracy": accuracy,
        "risk": 1.0 - accuracy,
        "mae": float(errors.mean()),
        "severe_error_rate": float(np.mean(errors >= 2)),
    }


def aurc(y_true: np.ndarray, prediction: np.ndarray, uncertainty: np.ndarray) -> float:
    order = np.argsort(np.asarray(uncertainty), kind="mergesort")
    errors = (prediction[order] != y_true[order]).astype(float)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(cumulative_risk, coverage))


def subject_cluster_bootstrap(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    subjects = np.asarray(sorted(frame["subject_id"].unique()), dtype=int)
    grouped = {subject: frame.index[frame["subject_id"] == subject].to_numpy() for subject in subjects}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        positions = np.concatenate([grouped[int(subject)] for subject in sampled])
        values.append(float(metric(frame.loc[positions])))
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(metric(frame)),
        "ci_low": float(np.quantile(values, tail)),
        "ci_high": float(np.quantile(values, 1.0 - tail)),
    }


def _probabilities(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)


def build_fold_predictions(
    table: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    calibration_fraction: float,
    seed: int,
    temperature_bounds: tuple[float, float],
    alpha_levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = table[columns].to_numpy(dtype=float)
    y = table["class_id"].to_numpy(dtype=int)
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for split in folds:
        fold = int(split["fold"])
        model_subjects, calibration_subjects = split_model_and_calibration_subjects(
            split["train_subjects"], calibration_fraction, seed + fold
        )
        model_mask = table["subject_id"].isin(model_subjects).to_numpy()
        calibration_mask = table["subject_id"].isin(calibration_subjects).to_numpy()
        model_index = np.flatnonzero(model_mask)
        calibration_index = np.flatnonzero(calibration_mask)
        test_index = np.asarray(split["test_index"], dtype=int)
        test_subjects = sorted(table.iloc[test_index]["subject_id"].astype(int).unique().tolist())
        if set(model_subjects) & set(calibration_subjects) or set(model_subjects) & set(test_subjects) or set(calibration_subjects) & set(test_subjects):
            raise ValueError(f"Subject leakage in nested split for fold {fold}")
        for role, subjects in [
            ("model_training", model_subjects),
            ("calibration", calibration_subjects),
            ("outer_test", test_subjects),
        ]:
            for subject in subjects:
                split_rows.append({"fold": fold, "subject_id": subject, "role": role})

        model = make_baseline_model("xgboost", seed)
        model.fit(x[model_index], y[model_index])
        calibration_raw = normalize_probabilities(model.predict_proba(x[calibration_index]))
        test_raw = normalize_probabilities(model.predict_proba(x[test_index]))
        temperature = fit_temperature(calibration_raw, y[calibration_index], temperature_bounds)
        calibration_scaled = apply_temperature(calibration_raw, temperature)
        test_scaled = apply_temperature(test_raw, temperature)
        calibration_expected = calibration_scaled @ CLASS_IDS.astype(float)
        standard_scores = 1.0 - calibration_scaled[np.arange(len(calibration_index)), y[calibration_index]]
        ordinal_scores = np.abs(y[calibration_index].astype(float) - calibration_expected)

        for alpha in alpha_levels:
            standard_threshold = conformal_quantile(standard_scores, alpha)
            ordinal_radius = conformal_quantile(ordinal_scores, alpha)
            standard_set = standard_conformal_sets(test_scaled, standard_threshold)
            lower, upper = ordinal_conformal_intervals(test_scaled, ordinal_radius)
            threshold_rows.append(
                {
                    "fold": fold,
                    "alpha": alpha,
                    "temperature": temperature,
                    "n_model_training": len(model_index),
                    "n_calibration": len(calibration_index),
                    "n_outer_test": len(test_index),
                    "standard_threshold": standard_threshold,
                    "ordinal_radius": ordinal_radius,
                }
            )
            for pos, idx in enumerate(test_index):
                for class_id, class_name in enumerate(CLASS_NAMES):
                    prediction_rows.append(
                        {
                            "record_type": "conformal",
                            "fold": fold,
                            "row_index": int(idx),
                            "sample_name": table.iloc[idx]["sample_name"],
                            "subject_id": int(table.iloc[idx]["subject_id"]),
                            "true_class_id": int(y[idx]),
                            "alpha": alpha,
                            "class_id": class_id,
                            "class_name": class_name,
                            "standard_in_set": bool(standard_set[pos, class_id]),
                            "ordinal_in_interval": bool(lower[pos] <= class_id <= upper[pos]),
                            "ordinal_lower": int(lower[pos]),
                            "ordinal_upper": int(upper[pos]),
                        }
                    )

        prediction = test_scaled.argmax(axis=1)
        for pos, idx in enumerate(test_index):
            row = {
                "record_type": "prediction",
                "fold": fold,
                "row_index": int(idx),
                "sample_name": table.iloc[idx]["sample_name"],
                "subject_id": int(table.iloc[idx]["subject_id"]),
                "subject_name": table.iloc[idx]["subject_name"],
                "true_class_id": int(y[idx]),
                "true_class_name": EXPECTED_LABEL_MAP[int(y[idx])],
                "pred_class_id": int(prediction[pos]),
                "pred_class_name": EXPECTED_LABEL_MAP[int(prediction[pos])],
                "temperature": temperature,
            }
            for class_id, class_name in enumerate(CLASS_NAMES):
                row[f"raw_{class_name}"] = float(test_raw[pos, class_id])
                row[f"calibrated_{class_name}"] = float(test_scaled[pos, class_id])
            prediction_rows.append(row)
    all_rows = pd.DataFrame(prediction_rows)
    predictions = all_rows[all_rows["record_type"] == "prediction"].dropna(axis=1, how="all").copy()
    conformal_long = all_rows[all_rows["record_type"] == "conformal"].dropna(axis=1, how="all").copy()
    predictions = predictions.sort_values(["fold", "subject_id", "row_index"]).reset_index(drop=True)
    conformal_long = conformal_long.sort_values(["fold", "subject_id", "row_index", "alpha", "class_id"]).reset_index(drop=True)
    return predictions, conformal_long, pd.DataFrame(split_rows), pd.DataFrame(threshold_rows)


def calibration_outputs(predictions: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bin_frames = []
    for probability_type, prefix in [("raw", "raw"), ("temperature_scaled", "calibrated")]:
        probabilities = _probabilities(predictions, prefix)
        y = predictions["true_class_id"].to_numpy(dtype=int)
        groups: list[tuple[str, object, np.ndarray]] = [("overall", "all", np.ones(len(predictions), dtype=bool))]
        groups += [("fold", int(value), predictions["fold"].eq(value).to_numpy()) for value in sorted(predictions["fold"].unique())]
        groups += [("pain_level", EXPECTED_LABEL_MAP[int(value)], predictions["true_class_id"].eq(value).to_numpy()) for value in CLASS_IDS]
        groups += [("subject", int(value), predictions["subject_id"].eq(value).to_numpy()) for value in sorted(predictions["subject_id"].unique())]
        for group_type, group_value, mask in groups:
            metric = calibration_metrics(y[mask], probabilities[mask], n_bins)
            rows.append({"probability_type": probability_type, "group_type": group_type, "group": group_value, "n": int(mask.sum()), **metric})
        bins = reliability_bins(y, probabilities, n_bins)
        bins.insert(0, "probability_type", probability_type)
        bin_frames.append(bins)

        for class_id, class_name in enumerate(CLASS_NAMES):
            class_prob = probabilities[:, class_id]
            class_y = y == class_id
            edges = np.linspace(0, 1, n_bins + 1)
            assignments = np.minimum(np.digitize(class_prob, edges[1:-1], right=True), n_bins - 1)
            class_rows = []
            for bin_id in range(n_bins):
                mask = assignments == bin_id
                class_rows.append(
                    {
                        "probability_type": probability_type,
                        "class_name": class_name,
                        "bin": bin_id + 1,
                        "n": int(mask.sum()),
                        "mean_probability": float(class_prob[mask].mean()) if mask.any() else np.nan,
                        "observed_frequency": float(class_y[mask].mean()) if mask.any() else np.nan,
                    }
                )
            bin_frames.append(pd.DataFrame(class_rows))
    return pd.DataFrame(rows), pd.concat(bin_frames, ignore_index=True, sort=False)


def selective_outputs(
    predictions: pd.DataFrame, coverage_grid: list[float], random_repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = predictions["true_class_id"].to_numpy(dtype=int)
    pred = predictions["pred_class_id"].to_numpy(dtype=int)
    probabilities = _probabilities(predictions, "calibrated")
    scores = {
        "max_probability": 1.0 - probabilities.max(axis=1),
        "predictive_entropy": predictive_entropy(probabilities),
    }
    curve_rows = []
    rejected_rows = []
    summary_rows = []
    for score_name, uncertainty in scores.items():
        summary_rows.append(
            {
                "method": score_name,
                "aurc": aurc(y, pred, uncertainty),
                "error_detection_auroc": float(roc_auc_score(pred != y, uncertainty)),
            }
        )
        for fold in sorted(predictions["fold"].unique()):
            fold_mask = predictions["fold"].eq(fold).to_numpy()
            order_local = np.argsort(uncertainty[fold_mask], kind="mergesort")
            positions = np.flatnonzero(fold_mask)[order_local]
            for coverage in coverage_grid:
                n_accept = max(1, int(math.ceil(coverage * len(positions))))
                accepted = np.zeros(len(predictions), dtype=bool)
                accepted[positions[:n_accept]] = True
                metrics = selection_metrics(y, pred, accepted)
                curve_rows.append({"fold": int(fold), "method": score_name, "coverage_target": coverage, "coverage": float(accepted[fold_mask].mean()), **metrics})
                rejected = predictions.loc[fold_mask & ~accepted, "true_class_name"].value_counts()
                for class_name in CLASS_NAMES:
                    rejected_rows.append(
                        {
                            "fold": int(fold),
                            "method": score_name,
                            "coverage_target": coverage,
                            "class_name": class_name,
                            "n_rejected": int(rejected.get(class_name, 0)),
                            "rejected_fraction": float(rejected.get(class_name, 0) / max(1, (~accepted & fold_mask).sum())),
                        }
                    )

    rng = np.random.default_rng(seed)
    for fold in sorted(predictions["fold"].unique()):
        fold_positions = np.flatnonzero(predictions["fold"].eq(fold).to_numpy())
        for coverage in coverage_grid:
            metrics_list = []
            n_accept = max(1, int(math.ceil(coverage * len(fold_positions))))
            for _ in range(random_repetitions):
                accepted = np.zeros(len(predictions), dtype=bool)
                accepted[rng.choice(fold_positions, size=n_accept, replace=False)] = True
                metrics_list.append(selection_metrics(y, pred, accepted))
            curve_rows.append(
                {
                    "fold": int(fold),
                    "method": "random_rejection",
                    "coverage_target": coverage,
                    "coverage": n_accept / len(fold_positions),
                    **{key: float(np.mean([row[key] for row in metrics_list])) for key in ["n_accepted", "accuracy", "risk", "mae", "severe_error_rate"]},
                }
            )
    curve = pd.DataFrame(curve_rows)
    return curve, pd.DataFrame(rejected_rows), pd.DataFrame(summary_rows)


def conformal_outputs(conformal_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["fold", "row_index", "sample_name", "subject_id", "true_class_id", "alpha"]
    grouped = conformal_long.groupby(key, sort=False)
    sample_rows = []
    for values, group in grouped:
        fold, row_index, sample_name, subject_id, true_class_id, alpha = values
        standard_members = group.sort_values("class_id")["standard_in_set"].to_numpy(dtype=bool)
        lower = int(group["ordinal_lower"].iloc[0])
        upper = int(group["ordinal_upper"].iloc[0])
        standard_ids = CLASS_IDS[standard_members]
        standard_contiguous = bool(len(standard_ids) <= 1 or np.all(np.diff(standard_ids) == 1))
        for method, covered, width, singleton, contiguous in [
            ("standard_classification", bool(standard_members[int(true_class_id)]), int(standard_members.sum()), int(standard_members.sum()) == 1, standard_contiguous),
            ("ordinal_contiguous", lower <= int(true_class_id) <= upper, upper - lower + 1, lower == upper, True),
        ]:
            sample_rows.append(
                {
                    "fold": int(fold),
                    "row_index": int(row_index),
                    "sample_name": sample_name,
                    "subject_id": int(subject_id),
                    "true_class_id": int(true_class_id),
                    "true_class_name": EXPECTED_LABEL_MAP[int(true_class_id)],
                    "alpha": float(alpha),
                    "nominal_coverage": 1.0 - float(alpha),
                    "method": method,
                    "covered": bool(covered),
                    "width": int(width),
                    "singleton": bool(singleton),
                    "contiguous": bool(contiguous),
                    "lower": lower if method == "ordinal_contiguous" else np.nan,
                    "upper": upper if method == "ordinal_contiguous" else np.nan,
                }
            )
    samples = pd.DataFrame(sample_rows)
    rows = []
    for (method, alpha, group_type, group_value), group in _conformal_groups(samples):
        rows.append(
            {
                "method": method,
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "group_type": group_type,
                "group": group_value,
                "n": len(group),
                "empirical_coverage": float(group["covered"].mean()),
                "mean_width": float(group["width"].mean()),
                "median_width": float(group["width"].median()),
                "singleton_rate": float(group["singleton"].mean()),
                "singleton_error_rate": float((~group.loc[group["singleton"], "covered"]).mean()) if group["singleton"].any() else np.nan,
                "contiguous_rate": float(group["contiguous"].mean()),
            }
        )
    return samples, pd.DataFrame(rows)


def _conformal_groups(samples: pd.DataFrame):
    for (method, alpha), base in samples.groupby(["method", "alpha"], sort=True):
        yield (method, alpha, "overall", "all"), base
        for value, group in base.groupby("true_class_name", sort=False):
            yield (method, alpha, "pain_level", value), group
        for value, group in base.groupby("subject_id", sort=True):
            yield (method, alpha, "subject", int(value)), group


def uncertainty_outputs(predictions: pd.DataFrame) -> pd.DataFrame:
    probabilities = _probabilities(predictions, "calibrated")
    working = predictions[["fold", "subject_id", "true_class_id", "true_class_name", "pred_class_id"]].copy()
    working["max_probability"] = probabilities.max(axis=1)
    working["predictive_entropy"] = predictive_entropy(probabilities)
    working["correct"] = working["true_class_id"].eq(working["pred_class_id"])
    rows = []
    for group_type, column in [("pain_level", "true_class_name"), ("subject", "subject_id")]:
        for value, group in working.groupby(column, sort=True):
            rows.append(
                {
                    "group_type": group_type,
                    "group": value,
                    "n": len(group),
                    "accuracy": float(group["correct"].mean()),
                    "mean_max_probability": float(group["max_probability"].mean()),
                    "median_max_probability": float(group["max_probability"].median()),
                    "mean_predictive_entropy": float(group["predictive_entropy"].mean()),
                    "median_predictive_entropy": float(group["predictive_entropy"].median()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_outputs(
    predictions: pd.DataFrame,
    conformal_samples: pd.DataFrame,
    n_bins: int,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for probability_type, prefix in [("raw", "raw"), ("temperature_scaled", "calibrated")]:
        for metric_name, function in [
            ("ece", lambda d, p=prefix: ece_score(d["true_class_id"].to_numpy(int), _probabilities(d, p), n_bins)),
            ("brier", lambda d, p=prefix: multiclass_brier(d["true_class_id"].to_numpy(int), _probabilities(d, p))),
            ("nll", lambda d, p=prefix: float(log_loss(d["true_class_id"].to_numpy(int), _probabilities(d, p), labels=CLASS_IDS))),
        ]:
            ci = subject_cluster_bootstrap(predictions, function, repetitions, seed, confidence_level)
            rows.append({"analysis": "calibration", "method": probability_type, "metric": metric_name, **ci})
    y = predictions["true_class_id"].to_numpy(int)
    pred = predictions["pred_class_id"].to_numpy(int)
    probabilities = _probabilities(predictions, "calibrated")
    predictions = predictions.copy()
    predictions["uncertainty_max_probability"] = 1.0 - probabilities.max(axis=1)
    predictions["uncertainty_predictive_entropy"] = predictive_entropy(probabilities)
    for method in ["max_probability", "predictive_entropy"]:
        function = lambda d, m=method: aurc(
            d["true_class_id"].to_numpy(int), d["pred_class_id"].to_numpy(int), d[f"uncertainty_{m}"].to_numpy(float)
        )
        ci = subject_cluster_bootstrap(predictions, function, repetitions, seed + 1, confidence_level)
        rows.append({"analysis": "selective_prediction", "method": method, "metric": "aurc", **ci})
    del y, pred
    for (method, alpha), frame in conformal_samples.groupby(["method", "alpha"], sort=True):
        for metric_name, function in [
            ("empirical_coverage", lambda d: float(d["covered"].mean())),
            ("mean_width", lambda d: float(d["width"].mean())),
            ("singleton_rate", lambda d: float(d["singleton"].mean())),
        ]:
            ci = subject_cluster_bootstrap(frame.reset_index(drop=True), function, repetitions, seed + 2, confidence_level)
            rows.append({"analysis": "conformal", "method": method, "alpha": alpha, "metric": metric_name, **ci})
    return pd.DataFrame(rows)


def plot_outputs(
    output_dir: Path,
    reliability: pd.DataFrame,
    calibration: pd.DataFrame,
    selective: pd.DataFrame,
    uncertainty: pd.DataFrame,
    conformal_summary: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    overall_bins = reliability[reliability["class_name"].isna()]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="ideal")
    for probability_type, group in overall_bins.groupby("probability_type"):
        valid = group.dropna(subset=["mean_confidence", "accuracy"])
        ax.plot(valid["mean_confidence"], valid["accuracy"], marker="o", label=probability_type)
    ax.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy", title="Top-label reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "reliability_diagram_overall.png", dpi=180)
    plt.close(fig)

    class_bins = reliability[reliability["class_name"].notna()]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True, sharey=True)
    for ax, class_name in zip(axes, CLASS_NAMES):
        ax.plot([0, 1], [0, 1], "--", color="0.6")
        for probability_type, group in class_bins[class_bins["class_name"] == class_name].groupby("probability_type"):
            valid = group.dropna(subset=["mean_probability", "observed_frequency"])
            ax.plot(valid["mean_probability"], valid["observed_frequency"], marker=".", label=probability_type)
        ax.set_title(class_name)
    axes[0].set_ylabel("Observed frequency")
    for ax in axes:
        ax.set_xlabel("Predicted probability")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "reliability_diagrams_per_class.png", dpi=180)
    plt.close(fig)

    aggregate = selective.groupby(["method", "coverage_target"], as_index=False).agg(risk=("risk", "mean"), mae=("mae", "mean"), severe_error_rate=("severe_error_rate", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for method, group in aggregate.groupby("method"):
        for ax, metric, label in zip(axes, ["risk", "mae", "severe_error_rate"], ["Error risk", "MAE", "Severe error rate"]):
            ax.plot(group["coverage_target"], group[metric], marker="o", label=method)
            ax.set(xlabel="Coverage", ylabel=label)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "risk_coverage_curves.png", dpi=180)
    plt.close(fig)

    pain = uncertainty[uncertainty["group_type"] == "pain_level"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(pain["group"].astype(str), pain["mean_max_probability"])
    axes[0].set(ylabel="Mean max probability", title="Confidence by pain level")
    axes[1].bar(pain["group"].astype(str), pain["mean_predictive_entropy"])
    axes[1].set(ylabel="Mean normalized entropy", title="Uncertainty by pain level")
    fig.tight_layout()
    fig.savefig(figure_dir / "uncertainty_by_pain_level.png", dpi=180)
    plt.close(fig)

    overall = conformal_summary[conformal_summary["group_type"] == "overall"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for method, group in overall.groupby("method"):
        ordered = group.sort_values("empirical_coverage")
        ax.plot(ordered["mean_width"], ordered["empirical_coverage"], marker="o", label=method)
        for _, row in ordered.iterrows():
            ax.annotate(f"{row['nominal_coverage']:.0%}", (row["mean_width"], row["empirical_coverage"]), fontsize=8)
    ax.set(xlabel="Mean number of labels", ylabel="Empirical coverage", title="Coverage-width trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "conformal_coverage_width.png", dpi=180)
    plt.close(fig)


def dependency_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML", "xgboost"]
    return {package: importlib.metadata.version(package) for package in packages}


def write_report(
    path: Path,
    config: dict[str, object],
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selective: pd.DataFrame,
    selective_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    conformal_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    overall_cal = calibration[calibration["group_type"] == "overall"]
    conformal = conformal_summary[conformal_summary["group_type"] == "overall"]
    agg_selective = selective.groupby(["method", "coverage_target"], as_index=False).agg(accuracy=("accuracy", "mean"), mae=("mae", "mean"), severe_error_rate=("severe_error_rate", "mean"))
    pain_uncertainty = uncertainty[uncertainty["group_type"] == "pain_level"]
    lines = [
        "# Phase 10 — Prediction Reliability, Uncertainty, and Ordinal Intervals",
        "",
        f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Executive finding",
        "",
        "This phase is a reliability feasibility study. The conclusions below distinguish probability reliability, ranking of error risk, and prediction-set coverage; none is treated as evidence of a causal physiological mechanism or clinical validity.",
        "",
        "## Leakage-safe protocol",
        "",
        "- Reused the exact Phase 3 five-fold outer subject assignments.",
        f"- Within every outer fold, {config['split']['calibration_subject_fraction']:.0%} of outer-training subjects were assigned to calibration; the remaining subjects trained the fixed Population-only XGBoost model.",
        "- Temperature, conformal thresholds, and ordinal radii were fitted only on fold-local calibration subjects. Outer-test labels were used only for final evaluation.",
        "- Selective curves rank outer-test samples using probability-derived uncertainty only. Coverage cutoffs are label-free empirical quantiles within each outer test fold; no outer-test label selected a threshold.",
        "- Confidence intervals resample subjects as clusters, preserving their repeated trials.",
        "",
        "## Probability calibration",
        "",
        "| probabilities | accuracy | ECE | multiclass Brier | NLL | mean confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall_cal.iterrows():
        lines.append(f"| {row['probability_type']} | {row['accuracy']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} | {row['nll']:.4f} | {row['mean_confidence']:.4f} |")
    lines.extend(["", "Subject-clustered 95% confidence intervals:", "", "| probabilities | metric | estimate | 95% CI |", "|---|---|---:|---:|"])
    for _, row in bootstrap[bootstrap["analysis"] == "calibration"].iterrows():
        lines.append(f"| {row['method']} | {row['metric']} | {row['estimate']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    temperatures = thresholds.groupby("fold")["temperature"].first()
    lines.extend(["", f"Fold temperatures ranged from {temperatures.min():.3f} to {temperatures.max():.3f} (median {temperatures.median():.3f}).", "", "Figures: `figures/reliability_diagram_overall.png` and `figures/reliability_diagrams_per_class.png`.", "", "## Uncertainty and selective prediction", "", "| uncertainty | AURC | error-detection AUROC |", "|---|---:|---:|"])
    for _, row in selective_summary.iterrows():
        lines.append(f"| {row['method']} | {row['aurc']:.4f} | {row['error_detection_auroc']:.4f} |")
    lines.extend(["", "Mean fold performance at selected coverage levels:", "", "| method | coverage | accepted accuracy | MAE | severe error rate |", "|---|---:|---:|---:|---:|"])
    for _, row in agg_selective[agg_selective["coverage_target"].isin([0.5, 0.7, 0.9, 1.0])].iterrows():
        lines.append(f"| {row['method']} | {row['coverage_target']:.0%} | {row['accuracy']:.4f} | {row['mae']:.4f} | {row['severe_error_rate']:.4f} |")
    lines.extend(["", "Rejected-sample class counts and proportions are recorded in `selective_rejected_class_distribution.csv`; accepted-only improvements must be read together with those rejection profiles.", "", "Pain-level uncertainty:", "", "| pain level | accuracy | mean max probability | mean entropy |", "|---|---:|---:|---:|"])
    for _, row in pain_uncertainty.iterrows():
        lines.append(f"| {row['group']} | {row['accuracy']:.4f} | {row['mean_max_probability']:.4f} | {row['mean_predictive_entropy']:.4f} |")
    lines.extend(["", "## Conformal prediction", "", "| method | target coverage | empirical coverage | mean width | median width | singleton rate | singleton error rate | contiguity |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for _, row in conformal.iterrows():
        lines.append(f"| {row['method']} | {row['nominal_coverage']:.0%} | {row['empirical_coverage']:.4f} | {row['mean_width']:.3f} | {row['median_width']:.1f} | {row['singleton_rate']:.4f} | {row['singleton_error_rate']:.4f} | {row['contiguous_rate']:.4f} |")
    lines.extend(
        [
            "",
            "The standard comparator is split conformal classification using nonconformity `1 - p(y)`. The ordinal comparator is a split-conformal residual interval around the probability-weighted expected level, using `|y - E[level]|`; it is a baseline construction, not a novel algorithm.",
            "",
            "### Guarantee scope",
            "",
            "Classical split-conformal marginal coverage requires exchangeability of calibration and future examples. Trials are clustered within subjects here, so trial-level exchangeability is doubtful. The subject-disjoint construction prevents identity leakage, but it does not establish finite-sample per-subject or per-class coverage. Per-subject and per-class results are diagnostics only; subject-clustered bootstrap intervals quantify sampling uncertainty but do not restore a conformal guarantee.",
            "",
            "## Failure cases and limitations",
            "",
            "- The fold-local calibration split reduces the data available to train XGBoost, so these predictions are not numerically identical to the earlier model trained on every outer-training subject.",
            "- Temperature scaling changes probability sharpness but not the argmax class, so it cannot improve five-class accuracy.",
            "- A useful selective system needs both low accepted risk and acceptable rejection burden; class-dependent rejection can reduce apparent risk by disproportionately abstaining on intermediate levels.",
            "- Marginal conformal coverage can coexist with serious subject-level undercoverage and wide, low-resolution sets.",
            "- BioVid experimental heat pain does not establish clinical diagnostic or deployment validity.",
            "",
            "## Decision on boundary-aware selective resolution",
            "",
            "Proceed only if uncertainty materially orders error risk beyond random rejection, while at least one conformal baseline exposes a practically important coverage-resolution or subject-heterogeneity gap. Otherwise, these standard baselines already answer the reliability question and further method complexity is not justified.",
            "",
            "Machine-readable per-subject, per-class, fold-level, selective, and conformal diagnostics are stored beside this report in `experiments/prediction_reliability/`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, config_path: Path, command: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = root / config["inputs"]["feature_table"]
    metadata_path = root / config["inputs"]["feature_metadata"]
    fold_path = root / config["inputs"]["fold_assignments"]
    table = pd.read_csv(feature_path)
    metadata = pd.read_json(metadata_path)
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    columns = sets[config["feature_set"]]
    folds = make_fold_indices(table, load_phase3_folds(fold_path))
    seed = int(config["split"]["random_state"])
    alpha_levels = [float(value) for value in config["conformal"]["alpha_levels"]]

    predictions, conformal_long, split_manifest, thresholds = build_fold_predictions(
        table,
        columns,
        folds,
        float(config["split"]["calibration_subject_fraction"]),
        seed,
        tuple(float(value) for value in config["calibration"]["temperature_bounds"]),
        alpha_levels,
    )
    calibration, reliability = calibration_outputs(predictions, int(config["calibration"]["ece_bins"]))
    selective, rejected, selective_summary = selective_outputs(
        predictions,
        [float(value) for value in config["selective_prediction"]["coverage_grid"]],
        int(config["selective_prediction"]["random_repetitions"]),
        seed,
    )
    conformal_samples, conformal_summary = conformal_outputs(conformal_long)
    uncertainty = uncertainty_outputs(predictions)
    bootstrap = bootstrap_outputs(
        predictions,
        conformal_samples,
        int(config["calibration"]["ece_bins"]),
        int(config["bootstrap"]["repetitions"]),
        float(config["bootstrap"]["confidence_level"]),
        seed,
    )

    predictions.to_csv(output_dir / "outer_test_predictions.csv", index=False)
    split_manifest.to_csv(output_dir / "nested_subject_split_manifest.csv", index=False)
    thresholds.to_csv(output_dir / "fold_calibration_conformal_thresholds.csv", index=False)
    calibration.to_csv(output_dir / "calibration_metrics.csv", index=False)
    reliability.to_csv(output_dir / "reliability_bins.csv", index=False)
    selective.to_csv(output_dir / "selective_prediction_metrics.csv", index=False)
    rejected.to_csv(output_dir / "selective_rejected_class_distribution.csv", index=False)
    selective_summary.to_csv(output_dir / "selective_summary.csv", index=False)
    uncertainty.to_csv(output_dir / "uncertainty_by_pain_and_subject.csv", index=False)
    conformal_samples.to_csv(output_dir / "conformal_sample_results.csv", index=False)
    conformal_summary.to_csv(output_dir / "conformal_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "subject_clustered_bootstrap_ci.csv", index=False)
    plot_outputs(output_dir, reliability, calibration, selective, uncertainty, conformal_summary)

    run_manifest = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "random_seed": seed,
        "input_sha256": {
            config["inputs"]["feature_table"]: sha256_file(feature_path),
            config["inputs"]["feature_metadata"]: sha256_file(metadata_path),
            config["inputs"]["fold_assignments"]: sha256_file(fold_path),
            str(config_path.relative_to(root)): sha256_file(config_path),
        },
        "config": config,
        "n_features": len(columns),
        "features": columns,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_report(root / config["outputs"]["report"], config, calibration, bootstrap, selective, selective_summary, uncertainty, conformal_summary, thresholds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/prediction_reliability.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    command = f"{sys.executable} -m src.models.run_prediction_reliability --root {root} --config {config_path.relative_to(root)}"
    run(root, config_path, command)


if __name__ == "__main__":
    main()
