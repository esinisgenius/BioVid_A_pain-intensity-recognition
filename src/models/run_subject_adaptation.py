from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_subject_independent_baselines import read_feature_table


PAIR_CLASS_IDS = {
    "PA1_vs_PA2": (1, 2),
    "PA2_vs_PA3": (2, 3),
    "PA3_vs_PA4": (3, 4),
    "PA1_vs_PA3": (1, 3),
    "BL1_vs_PA4": (0, 4),
}
METHODS = ("population_only", "individual_only", "population_plus_adaptation")


def stable_seed(*parts: object, base: int = 20261025) -> int:
    value = int(base) & 0xFFFFFFFF
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value * 16777619) ^ byte) & 0xFFFFFFFF
    return value


def load_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Phase 9 configuration must be a mapping")
    required = {"features", "pain_pairs", "sample_efficiency", "models", "inputs"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Phase 9 configuration missing keys: {missing}")
    return config


def configured_pairs(config: dict[str, object]) -> list[tuple[str, int, int]]:
    pairs = []
    label_to_id = {name: class_id for class_id, name in EXPECTED_LABEL_MAP.items()}
    for low_name, high_name in config["pain_pairs"]:
        name = f"{low_name}_vs_{high_name}"
        if name not in PAIR_CLASS_IDS or low_name not in label_to_id or high_name not in label_to_id:
            raise ValueError(f"Unsupported or unknown pain pair: {low_name}/{high_name}")
        pairs.append((name, label_to_id[low_name], label_to_id[high_name]))
    return pairs


def pair_table(table: pd.DataFrame, low_class: int, high_class: int) -> pd.DataFrame:
    pair = table[table["class_id"].isin([low_class, high_class])].copy().reset_index(drop=True)
    pair["binary_label"] = (pair["class_id"] == high_class).astype(int)
    return pair


def make_adaptation_split(
    subject: pd.DataFrame,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional adaptation/evaluation indices, stratified without replacement."""
    if k < 0:
        raise ValueError("k must be non-negative")
    rng = np.random.default_rng(seed)
    adaptation: list[int] = []
    for label in (0, 1):
        positions = np.flatnonzero(subject["binary_label"].to_numpy(dtype=int) == label)
        if len(positions) <= k:
            raise ValueError(f"Need at least k+1 trials for label {label}; found {len(positions)} at k={k}")
        if k:
            adaptation.extend(rng.choice(positions, size=k, replace=False).tolist())
    adaptation_idx = np.asarray(sorted(adaptation), dtype=int)
    evaluation_idx = np.asarray(sorted(set(range(len(subject))).difference(adaptation_idx)), dtype=int)
    if set(adaptation_idx).intersection(evaluation_idx):
        raise AssertionError("Adaptation and evaluation trials overlap")
    return adaptation_idx, evaluation_idx


def make_lr(c_value: float, max_iter: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    random_state=int(seed),
                ),
            ),
        ]
    )


def positive_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def population_transform(model: Pipeline, x: np.ndarray) -> np.ndarray:
    imputed = model.named_steps["imputer"].transform(x)
    return model.named_steps["scaler"].transform(imputed)


def fit_residual_correction(
    base_adaptation_scores: np.ndarray,
    adaptation_features: np.ndarray,
    labels: np.ndarray,
    l2_penalty: float,
    max_iter: int,
) -> np.ndarray:
    """Fit a penalized feature-level correction around a fixed population logit.

    Unlike Phase 6 score calibration, the population coefficient is fixed at one and
    subject-specific coefficients can alter the feature-space boundary. The penalty
    shrinks every correction coefficient, including the intercept, toward zero.
    """
    labels = np.asarray(labels, dtype=float)
    base = logit(np.clip(np.asarray(base_adaptation_scores, dtype=float), 1e-6, 1 - 1e-6))
    design = np.column_stack([np.ones(len(labels)), np.asarray(adaptation_features, dtype=float)])
    penalty = float(l2_penalty)

    def objective(delta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = base + design @ delta
        probabilities = expit(linear)
        loss = np.logaddexp(0.0, linear).sum() - labels @ linear
        value = float(loss + 0.5 * penalty * np.dot(delta, delta))
        gradient = design.T @ (probabilities - labels) + penalty * delta
        return value, gradient

    result = minimize(
        fun=lambda value: objective(value)[0],
        x0=np.zeros(design.shape[1], dtype=float),
        jac=lambda value: objective(value)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Residual correction failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def apply_residual_correction(
    base_scores: np.ndarray,
    transformed_features: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    base = logit(np.clip(np.asarray(base_scores, dtype=float), 1e-6, 1 - 1e-6))
    design = np.column_stack([np.ones(len(base)), np.asarray(transformed_features, dtype=float)])
    return expit(base + design @ np.asarray(delta, dtype=float))


def metric_values(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return (
        float(roc_auc_score(labels, scores)),
        float(balanced_accuracy_score(labels, scores >= 0.5)),
    )


def validate_outer_fold(pair: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_subjects = set(pair.iloc[train_idx]["subject_id"].astype(int))
    test_subjects = set(pair.iloc[test_idx]["subject_id"].astype(int))
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise AssertionError(f"Outer-fold subject leakage: {sorted(overlap)}")


def run_sample_efficiency(
    table: pd.DataFrame,
    folds: list[dict[str, object]],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = list(config["features"])
    missing = sorted(set(features).difference(table.columns))
    if missing:
        raise ValueError(f"Missing Phase 9 features: {missing}")
    k_values = [int(value) for value in config["sample_efficiency"]["k_per_class"]]
    repeats = int(config["sample_efficiency"]["repeats"])
    base_seed = int(config["random_state"])
    population_cfg = config["models"]["population"]
    individual_cfg = config["models"]["individual"]
    adaptation_cfg = config["models"]["adaptation"]
    fold_samples = {
        int(split["fold"]): set(table.iloc[split["test_index"]]["sample_name"])
        for split in folds
    }
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for pair_name, low_class, high_class in configured_pairs(config):
        pair = pair_table(table, low_class, high_class)
        x_all = pair[features].to_numpy(dtype=float)
        y_all = pair["binary_label"].to_numpy(dtype=int)
        for fold, test_samples in sorted(fold_samples.items()):
            test_idx = pair.index[pair["sample_name"].isin(test_samples)].to_numpy(dtype=int)
            train_idx = pair.index[~pair["sample_name"].isin(test_samples)].to_numpy(dtype=int)
            validate_outer_fold(pair, train_idx, test_idx)
            population = make_lr(
                population_cfg["C"],
                population_cfg["max_iter"],
                stable_seed("population", pair_name, fold, base=base_seed),
            )
            population.fit(x_all[train_idx], y_all[train_idx])
            fold_test = pair.iloc[test_idx].copy()
            fold_test["pair_index"] = test_idx
            for subject_id, subject in fold_test.groupby("subject_id", sort=True):
                subject = subject.sort_values("sample_name").reset_index(drop=True)
                subject_positions = subject["pair_index"].to_numpy(dtype=int)
                subject_x = x_all[subject_positions]
                subject_y = y_all[subject_positions]
                subject_population_scores = positive_scores(population, subject_x)
                subject_population_x = population_transform(population, subject_x)
                for k in k_values:
                    n_repeats = 1 if k == 0 else repeats
                    for repeat in range(n_repeats):
                        split_seed = stable_seed("split", pair_name, subject_id, k, repeat, base=base_seed)
                        adaptation_idx, evaluation_idx = make_adaptation_split(subject, k, split_seed)
                        adaptation_names = set(subject.iloc[adaptation_idx]["sample_name"])
                        evaluation_names = set(subject.iloc[evaluation_idx]["sample_name"])
                        if adaptation_names.intersection(evaluation_names):
                            raise AssertionError("Target-subject adaptation/evaluation leakage")
                        split_rows.append(
                            {
                                "pair": pair_name,
                                "fold": fold,
                                "subject_id": int(subject_id),
                                "subject_name": subject["subject_name"].iloc[0],
                                "k_per_class": k,
                                "repeat": repeat,
                                "split_seed": split_seed,
                                "n_adaptation": len(adaptation_idx),
                                "n_evaluation": len(evaluation_idx),
                                "adaptation_samples_sha256": hashlib.sha256("\n".join(sorted(adaptation_names)).encode()).hexdigest(),
                                "evaluation_samples_sha256": hashlib.sha256("\n".join(sorted(evaluation_names)).encode()).hexdigest(),
                            }
                        )
                        method_scores: dict[str, np.ndarray] = {
                            "population_only": subject_population_scores[evaluation_idx]
                        }
                        if k > 0:
                            individual = make_lr(
                                individual_cfg["C"],
                                individual_cfg["max_iter"],
                                stable_seed("individual", pair_name, subject_id, k, repeat, base=base_seed),
                            )
                            individual.fit(subject_x[adaptation_idx], subject_y[adaptation_idx])
                            method_scores["individual_only"] = positive_scores(individual, subject_x[evaluation_idx])
                            delta = fit_residual_correction(
                                subject_population_scores[adaptation_idx],
                                subject_population_x[adaptation_idx],
                                subject_y[adaptation_idx],
                                adaptation_cfg["l2_penalty"],
                                adaptation_cfg["max_iter"],
                            )
                            method_scores["population_plus_adaptation"] = apply_residual_correction(
                                subject_population_scores[evaluation_idx],
                                subject_population_x[evaluation_idx],
                                delta,
                            )
                        for method, scores in method_scores.items():
                            for local_pos, score in zip(evaluation_idx, scores):
                                row = subject.iloc[int(local_pos)]
                                prediction_rows.append(
                                    {
                                        "pair": pair_name,
                                        "level_gap": high_class - low_class,
                                        "fold": fold,
                                        "subject_id": int(subject_id),
                                        "subject_name": row["subject_name"],
                                        "k_per_class": k,
                                        "repeat": repeat,
                                        "split_seed": split_seed,
                                        "method": method,
                                        "sample_id": int(row["sample_id"]),
                                        "sample_name": row["sample_name"],
                                        "true_label": int(row["binary_label"]),
                                        "true_class_id": int(row["class_id"]),
                                        "true_class_name": row["class_name"],
                                        "score": float(score),
                                        "prediction": int(score >= 0.5),
                                        "n_population_train_trials": len(train_idx),
                                        "n_population_train_subjects": int(pair.iloc[train_idx]["subject_id"].nunique()),
                                        "n_adaptation_trials": len(adaptation_idx),
                                    }
                                )

    predictions = pd.DataFrame(prediction_rows)
    split_manifest = pd.DataFrame(split_rows)
    metric_rows = []
    keys = ["pair", "level_gap", "fold", "subject_id", "subject_name", "k_per_class", "repeat", "method"]
    for values, group in predictions.groupby(keys, sort=False):
        auc, ba = metric_values(group["true_label"], group["score"])
        metric_rows.append(
            dict(
                zip(keys, values),
                roc_auc=auc,
                balanced_accuracy=ba,
                n_evaluation=len(group),
                n_adaptation_trials=int(group["n_adaptation_trials"].iloc[0]),
            )
        )
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics, split_manifest


def paired_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["pair", "level_gap", "fold", "subject_id", "subject_name", "k_per_class", "repeat"]
    wide = metrics.pivot(index=identifiers, columns="method", values=["roc_auc", "balanced_accuracy"])
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    eligible = wide[wide["k_per_class"] > 0].copy()
    required = [
        "roc_auc_population_only",
        "roc_auc_individual_only",
        "roc_auc_population_plus_adaptation",
        "balanced_accuracy_population_only",
        "balanced_accuracy_individual_only",
        "balanced_accuracy_population_plus_adaptation",
    ]
    if eligible[required].isna().any().any():
        raise AssertionError("Methods are not evaluated on every matched nonzero-k split")
    eligible["adaptation_gain_auc"] = eligible["roc_auc_population_plus_adaptation"] - eligible["roc_auc_population_only"]
    eligible["adaptation_gain_balanced_accuracy"] = eligible["balanced_accuracy_population_plus_adaptation"] - eligible["balanced_accuracy_population_only"]
    eligible["individual_minus_population_auc"] = eligible["roc_auc_individual_only"] - eligible["roc_auc_population_only"]
    eligible["adapted_minus_individual_auc"] = eligible["roc_auc_population_plus_adaptation"] - eligible["roc_auc_individual_only"]
    return eligible


def subject_clustered_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    n_bootstraps: int,
    base_seed: int,
) -> pd.DataFrame:
    """Average repeats within subject, then bootstrap subjects as independent clusters."""
    subject_values = (
        frame.groupby(group_columns + ["subject_id"], as_index=False)[value_column]
        .mean()
        .dropna(subset=[value_column])
    )
    rows = []
    for group_key, group in subject_values.groupby(group_columns, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group[value_column].to_numpy(dtype=float)
        rng = np.random.default_rng(stable_seed("bootstrap", *group_key, value_column, base=base_seed))
        estimates = np.empty(n_bootstraps, dtype=float)
        for idx in range(n_bootstraps):
            estimates[idx] = np.mean(rng.choice(values, size=len(values), replace=True))
        rows.append(
            {
                **dict(zip(group_columns, group_key)),
                "metric": value_column,
                "subject_mean": float(np.mean(values)),
                "subject_median": float(np.median(values)),
                "ci_low": float(np.quantile(estimates, 0.025)),
                "ci_high": float(np.quantile(estimates, 0.975)),
                "n_subjects": len(values),
            }
        )
    return pd.DataFrame(rows)


def summarize_results(metrics: pd.DataFrame, gains: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_bootstraps = int(config["bootstrap"]["n_resamples"])
    seed = int(config["random_state"])
    performance_parts = []
    for metric in ("roc_auc", "balanced_accuracy"):
        performance_parts.append(
            subject_clustered_summary(
                metrics,
                ["pair", "level_gap", "k_per_class", "method"],
                metric,
                n_bootstraps,
                seed,
            )
        )
    performance = pd.concat(performance_parts, ignore_index=True)
    gain_parts = []
    for metric in ("adaptation_gain_auc", "adaptation_gain_balanced_accuracy", "individual_minus_population_auc", "adapted_minus_individual_auc"):
        gain_parts.append(
            subject_clustered_summary(gains, ["pair", "level_gap", "k_per_class"], metric, n_bootstraps, seed)
        )
    gain_summary = pd.concat(gain_parts, ignore_index=True)
    practical = float(config["inference"]["practical_auc_gain"])
    subject_gains = gains.groupby(["pair", "level_gap", "subject_id", "subject_name", "k_per_class"], as_index=False).agg(
        adaptation_gain_auc=("adaptation_gain_auc", "mean"),
        adaptation_gain_auc_sd=("adaptation_gain_auc", "std"),
        adaptation_gain_balanced_accuracy=("adaptation_gain_balanced_accuracy", "mean"),
        individual_minus_population_auc=("individual_minus_population_auc", "mean"),
        adapted_minus_individual_auc=("adapted_minus_individual_auc", "mean"),
        repeats=("repeat", "nunique"),
    )
    subject_gains["improved"] = subject_gains["adaptation_gain_auc"] > 0
    subject_gains["negative_transfer"] = subject_gains["adaptation_gain_auc"] < 0
    subject_gains["practically_meaningful_gain"] = subject_gains["adaptation_gain_auc"] >= practical
    return performance, gain_summary, subject_gains


def improvement_rates(subject_gains: pd.DataFrame) -> pd.DataFrame:
    return subject_gains.groupby(["pair", "level_gap", "k_per_class"], as_index=False).agg(
        n_subjects=("subject_id", "nunique"),
        improvement_rate=("improved", "mean"),
        negative_transfer_rate=("negative_transfer", "mean"),
        practical_gain_rate=("practically_meaningful_gain", "mean"),
        mean_gain_auc=("adaptation_gain_auc", "mean"),
        median_gain_auc=("adaptation_gain_auc", "median"),
    )


def phase6_gain_associations(root: Path, subject_gains: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / config["inputs"]["phase6_consistency"]
    phase6 = pd.read_csv(path)
    features = set(config["features"])
    phase6 = phase6[phase6["feature_name"].isin(features)]
    consistency = phase6.groupby(["subject_id", "subject_name"], as_index=False).agg(
        phase6_median_abs_spearman=("spearman_rho", lambda values: float(np.median(np.abs(values)))),
        phase6_sign_reversal_rate=("has_sign_reversal", "mean"),
    )
    linked = subject_gains.merge(consistency, on=["subject_id", "subject_name"], validate="many_to_one")
    rows = []
    for (pair, k), group in linked.groupby(["pair", "k_per_class"], sort=False):
        for predictor in ("phase6_median_abs_spearman", "phase6_sign_reversal_rate"):
            result = spearmanr(group[predictor], group["adaptation_gain_auc"])
            rows.append(
                {
                    "pair": pair,
                    "k_per_class": k,
                    "predictor": predictor,
                    "outcome": "adaptation_gain_auc",
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n_subjects": len(group),
                    "interpretation": "exploratory_nonindependent_same_cohort",
                }
            )
    associations = pd.DataFrame(rows)
    associations["p_value_bh"] = benjamini_hochberg(associations["p_value"])
    return linked, associations


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return pd.Series(output, index=values.index)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def make_figures(output_dir: Path, performance: pd.DataFrame, subject_gains: pd.DataFrame) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pair_order = list(PAIR_CLASS_IDS)
    method_labels = {
        "population_only": "Population-only",
        "individual_only": "Individual-only",
        "population_plus_adaptation": "Population + adaptation",
    }
    auc = performance[performance["metric"] == "roc_auc"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for axis, pair in zip(axes.flat, pair_order):
        subset = auc[auc["pair"] == pair]
        for method in METHODS:
            values = subset[subset["method"] == method].sort_values("k_per_class")
            if values.empty:
                continue
            axis.plot(values["k_per_class"], values["subject_mean"], marker="o", label=method_labels[method])
            axis.fill_between(values["k_per_class"], values["ci_low"], values["ci_high"], alpha=0.15)
        axis.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_title(pair.replace("_vs_", " / "))
        axis.set_xlabel("Adaptation trials per class (k)")
        axis.set_ylabel("Subject-mean ROC-AUC")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / "sample_efficiency_learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for axis, pair in zip(axes.flat, pair_order):
        subset = subject_gains[subject_gains["pair"] == pair]
        data = [subset.loc[subset["k_per_class"] == k, "adaptation_gain_auc"] for k in (1, 2, 5, 10)]
        axis.boxplot(data, tick_labels=[1, 2, 5, 10], showfliers=False)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(pair.replace("_vs_", " / "))
        axis.set_xlabel("k per class")
        axis.set_ylabel("Adapted − population AUC")
        axis.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "subject_adaptation_gain_distributions.png", dpi=180)
    plt.close(fig)


def build_run_metadata(root: Path, config_path: Path, output_dir: Path, config: dict[str, object], table: pd.DataFrame, folds: list[dict[str, object]]) -> dict[str, object]:
    inputs = {}
    for name, relative in config["inputs"].items():
        path = root / relative
        inputs[name] = {"path": str(relative), "sha256": file_sha256(path)}
    return {
        **config,
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "n_subjects": int(table["subject_id"].nunique()),
        "n_trials": len(table),
        "n_outer_folds": len(folds),
        "outer_protocol": "reused Phase 3 StratifiedGroupKFold assignments",
        "input_hashes": inputs,
        "config_file": {"path": str(config_path.relative_to(root)), "sha256": file_sha256(config_path)},
        "dependencies": package_versions(),
        "command": f"{sys.executable} -m src.models.run_subject_adaptation --root . --config configs/subject_adaptation.yaml --output-dir experiments/subject_adaptation",
        "output_dir": str(output_dir.relative_to(root)),
    }


def run(root: Path, config_path: Path, output_dir: Path) -> None:
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    feature_path = root / config["inputs"]["feature_table"]
    fold_path = root / config["inputs"]["fold_assignments"]
    table = read_feature_table(feature_path)
    fold_samples = load_phase3_folds(fold_path)
    folds = make_fold_indices(table, fold_samples)
    predictions, metrics, split_manifest = run_sample_efficiency(table, folds, config)
    gains = paired_gain_table(metrics)
    performance, gain_summary, subject_gains = summarize_results(metrics, gains, config)
    rates = improvement_rates(subject_gains)
    linked, associations = phase6_gain_associations(root, subject_gains, config)

    predictions.to_csv(output_dir / "trial_predictions.csv", index=False)
    metrics.to_csv(output_dir / "subject_repeat_metrics.csv", index=False)
    split_manifest.to_csv(output_dir / "adaptation_split_manifest.csv", index=False)
    gains.to_csv(output_dir / "paired_method_comparisons.csv", index=False)
    performance.to_csv(output_dir / "performance_summary.csv", index=False)
    gain_summary.to_csv(output_dir / "adaptation_gain_summary.csv", index=False)
    subject_gains.to_csv(output_dir / "subject_adaptation_gains.csv", index=False)
    rates.to_csv(output_dir / "improvement_negative_transfer_rates.csv", index=False)
    linked.to_csv(output_dir / "phase6_consistency_linked_gains.csv", index=False)
    associations.to_csv(output_dir / "phase6_consistency_gain_associations.csv", index=False)
    make_figures(output_dir, performance, subject_gains)
    metadata = build_run_metadata(root, config_path, output_dir, config, table, folds)
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 9 sample-efficient subject adaptation.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/subject_adaptation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/subject_adaptation"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.root, arguments.config, arguments.output_dir)
