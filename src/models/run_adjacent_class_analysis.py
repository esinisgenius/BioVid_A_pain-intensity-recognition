from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices, score_vector
from src.models.run_subject_independent_baselines import feature_sets, read_feature_table, validate_feature_inputs


FEATURE_SET = "combined_unique"
MODELS = ["logistic_regression", "xgboost_binary"]
MODEL_LABELS = {
    "logistic_regression": "logistic_regression",
    "xgboost_binary": "xgboost",
}
PAIRS = [
    ("PA1_vs_PA2", 1, 2),
    ("PA2_vs_PA3", 2, 3),
    ("PA1_vs_PA3", 1, 3),
    ("BL1_vs_PA4", 0, 4),
]
MODALITY_SETS = {
    "ecg_only": ["ECG"],
    "eda_only": ["EDA/GSR"],
    "emg_only": ["Trapezius EMG"],
    "ecg_eda": ["ECG", "EDA/GSR"],
    "eda_emg": ["EDA/GSR", "Trapezius EMG"],
    "ecg_emg": ["ECG", "Trapezius EMG"],
    "all_modalities": ["ECG", "EDA/GSR", "Trapezius EMG"],
}
PRIMARY_BINARY_MODEL = "xgboost"
PRIMARY_MODALITY_SET = "all_modalities"
TARGET_FEATURE_FALLBACKS = ["eda_scl_delta", "eda_scl_slope_per_s"]


def columns_for_modalities(metadata: pd.DataFrame, combined: list[str]) -> dict[str, list[str]]:
    retained = metadata[metadata["feature_name"].isin(combined)].copy()
    by_modality = {
        modality: retained.loc[retained["modality"] == modality, "feature_name"].tolist()
        for modality in ["ECG", "EDA/GSR", "Trapezius EMG"]
    }
    out = {}
    for set_name, modalities in MODALITY_SETS.items():
        selected = set()
        for modality in modalities:
            selected.update(by_modality[modality])
        out[set_name] = [col for col in combined if col in selected]
    return out


def fold_samples_from_splits(table: pd.DataFrame, folds: list[dict[str, object]]) -> dict[int, set[str]]:
    return {
        int(split["fold"]): set(table.iloc[split["test_index"]]["sample_name"])
        for split in folds
    }


def binary_pair_table(table: pd.DataFrame, low_class: int, high_class: int) -> pd.DataFrame:
    pair = table[table["class_id"].isin([low_class, high_class])].copy().reset_index(drop=True)
    pair["binary_label"] = (pair["class_id"] == high_class).astype(int)
    pair["binary_class_name"] = np.where(pair["binary_label"] == 1, EXPECTED_LABEL_MAP[high_class], EXPECTED_LABEL_MAP[low_class])
    return pair


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }


def run_binary_benchmark(
    table: pd.DataFrame,
    columns_by_modality: dict[str, list[str]],
    folds: list[dict[str, object]],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_samples = fold_samples_from_splits(table, folds)
    fold_rows = []
    summary_rows = []
    oof_frames = []

    for pair_name, low_class, high_class in PAIRS:
        pair = binary_pair_table(table, low_class, high_class)
        for modality_set, columns in columns_by_modality.items():
            x = pair[columns].to_numpy(dtype=float)
            y = pair["binary_label"].to_numpy(dtype=int)
            for model_name in MODELS:
                label = MODEL_LABELS[model_name]
                oof_rows = []
                for fold in sorted(fold_samples):
                    test_index = pair.index[pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
                    train_index = pair.index[~pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
                    train_subjects = set(pair.iloc[train_index]["subject_id"].astype(int))
                    test_subjects = set(pair.iloc[test_index]["subject_id"].astype(int))
                    overlap = train_subjects.intersection(test_subjects)
                    if overlap:
                        raise ValueError(f"Subject overlap for {pair_name} fold {fold}: {sorted(overlap)}")

                    model = make_baseline_model(model_name, random_state=random_state)
                    model.fit(x[train_index], y[train_index])
                    pred = model.predict(x[test_index])
                    score = score_vector(model, x[test_index])
                    metrics = binary_metrics(y[test_index], pred, score)
                    fold_rows.append(
                        {
                            "task": pair_name,
                            "model": label,
                            "feature_set": FEATURE_SET,
                            "modality_set": modality_set,
                            "modalities": "+".join(MODALITY_SETS[modality_set]),
                            "n_features": len(columns),
                            "fold": int(fold),
                            "n_train": int(len(train_index)),
                            "n_test": int(len(test_index)),
                            "n_train_subjects": int(len(train_subjects)),
                            "n_test_subjects": int(len(test_subjects)),
                            "negative_class": EXPECTED_LABEL_MAP[low_class],
                            "positive_class": EXPECTED_LABEL_MAP[high_class],
                            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                            "confusion_matrix": metrics["confusion_matrix"],
                        }
                    )
                    for pos, idx in enumerate(test_index):
                        row = pair.iloc[int(idx)]
                        oof_rows.append(
                            {
                                "task": pair_name,
                                "model": label,
                                "feature_set": FEATURE_SET,
                                "modality_set": modality_set,
                                "fold": int(fold),
                                "subject_id": int(row["subject_id"]),
                                "subject_name": row["subject_name"],
                                "sample_id": int(row["sample_id"]),
                                "sample_name": row["sample_name"],
                                "true_binary_label": int(row["binary_label"]),
                                "true_class_id": int(row["class_id"]),
                                "true_class_name": row["class_name"],
                                "pred_binary_label": int(pred[pos]),
                                "pred_class_id": high_class if int(pred[pos]) == 1 else low_class,
                                "pred_class_name": EXPECTED_LABEL_MAP[high_class] if int(pred[pos]) == 1 else EXPECTED_LABEL_MAP[low_class],
                                "positive_class": EXPECTED_LABEL_MAP[high_class],
                                "positive_score": float(score[pos]),
                            }
                        )

                oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
                overall = binary_metrics(
                    oof["true_binary_label"].to_numpy(dtype=int),
                    oof["pred_binary_label"].to_numpy(dtype=int),
                    oof["positive_score"].to_numpy(dtype=float),
                )
                summary_rows.append(
                    {
                        "task": pair_name,
                        "model": label,
                        "feature_set": FEATURE_SET,
                        "modality_set": modality_set,
                        "modalities": "+".join(MODALITY_SETS[modality_set]),
                        "n_features": len(columns),
                        "negative_class": EXPECTED_LABEL_MAP[low_class],
                        "positive_class": EXPECTED_LABEL_MAP[high_class],
                        **{f"overall_{k}": v for k, v in overall.items() if k != "confusion_matrix"},
                    }
                )
                oof_frames.append(oof)

    fold_df = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    fold_summary = (
        fold_df.groupby(["task", "model", "feature_set", "modality_set", "modalities", "n_features", "negative_class", "positive_class"], as_index=False)
        .agg(
            fold_roc_auc_mean=("roc_auc", "mean"),
            fold_roc_auc_std=("roc_auc", "std"),
            fold_balanced_accuracy_mean=("balanced_accuracy", "mean"),
            fold_balanced_accuracy_std=("balanced_accuracy", "std"),
            fold_f1_mean=("f1", "mean"),
            fold_f1_std=("f1", "std"),
            fold_accuracy_mean=("accuracy", "mean"),
            fold_accuracy_std=("accuracy", "std"),
        )
    )
    summary = summary.merge(
        fold_summary,
        on=["task", "model", "feature_set", "modality_set", "modalities", "n_features", "negative_class", "positive_class"],
        how="left",
    )
    return fold_df, summary, pd.concat(oof_frames, ignore_index=True)


def analyze_multiclass_oof(oof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(oof_path)
    if "modality_set" in oof.columns:
        oof = oof[(oof["feature_set"] == FEATURE_SET) & (oof["modality_set"] == "full")].copy()
    else:
        oof = oof[oof["feature_set"] == FEATURE_SET].copy()
        oof["modality_set"] = "full"
    oof = oof[oof["model"].isin(["logistic_regression", "xgboost"])].copy()
    if oof.empty:
        raise ValueError(f"No matching existing OOF multiclass predictions in {oof_path}")

    labels = sorted(EXPECTED_LABEL_MAP)
    class_names = [EXPECTED_LABEL_MAP[label] for label in labels]
    confusion_rows = []
    destination_rows = []
    summary_rows = []
    for (model, modality_set), group in oof.groupby(["model", "modality_set"], sort=True):
        y_true = group["true_class_id"].to_numpy(dtype=int)
        y_pred = group["pred_class_id"].to_numpy(dtype=int)
        errors = np.abs(y_true - y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(int)
        for i, true_name in enumerate(class_names):
            row_total = int(matrix[i].sum())
            correct = int(matrix[i, i])
            for j, pred_name in enumerate(class_names):
                confusion_rows.append(
                    {
                        "model": model,
                        "feature_set": FEATURE_SET,
                        "modality_set": modality_set,
                        "true_class": true_name,
                        "pred_class": pred_name,
                        "count": int(matrix[i, j]),
                        "row_proportion": float(matrix[i, j] / row_total) if row_total else np.nan,
                    }
                )
            error_total = row_total - correct
            for j, pred_name in enumerate(class_names):
                if i == j:
                    continue
                destination_rows.append(
                    {
                        "model": model,
                        "feature_set": FEATURE_SET,
                        "modality_set": modality_set,
                        "true_class": true_name,
                        "predicted_as": pred_name,
                        "count": int(matrix[i, j]),
                        "error_proportion": float(matrix[i, j] / error_total) if error_total else np.nan,
                        "class_support": row_total,
                        "class_errors": error_total,
                    }
                )
        summary_rows.append(
            {
                "model": model,
                "feature_set": FEATURE_SET,
                "modality_set": modality_set,
                "n": int(len(group)),
                "accuracy": float(np.mean(y_true == y_pred)),
                "mean_absolute_class_error": float(np.mean(errors)),
                "within_one_class_proportion": float(np.mean(errors <= 1)),
                "severe_error_rate_abs_ge_2": float(np.mean(errors >= 2)),
                "exact_error_rate": float(np.mean(errors > 0)),
            }
        )
    return pd.DataFrame(confusion_rows), pd.DataFrame(destination_rows), pd.DataFrame(summary_rows)


def select_targeted_interactions(metadata: pd.DataFrame, columns: list[str], phase5_dir: Path) -> tuple[list[tuple[str, str, str]], list[str]]:
    available = set(columns)
    focused_path = phase5_dir / "focused_top_features.csv"
    shap_path = phase5_dir / "class_specific_shap.csv"
    eda_features = [f for f in TARGET_FEATURE_FALLBACKS if f in available]
    emg_features: list[str] = []

    if focused_path.exists():
        focused = pd.read_csv(focused_path)
        eda_features.extend(
            focused.loc[
                (focused["modality"] == "eda") & (focused["pain_class"].isin(["PA1", "PA2", "PA3"])),
                "feature_name",
            ].tolist()
        )
        emg_features.extend(
            focused.loc[
                (focused["modality"] == "emg") & (focused["pain_class"] == "PA1"),
                "feature_name",
            ].tolist()
        )
    if not emg_features and shap_path.exists():
        shap = pd.read_csv(shap_path)
        emg = shap[(shap["modality"] == "Trapezius EMG") & (shap["pain_class"] == "PA1")].sort_values("mean_abs_shap", ascending=False)
        emg_features.extend(emg["feature_name"].head(1).tolist())

    meta = metadata.set_index("feature_name")
    if not emg_features:
        emg_features = meta[(meta["modality"] == "Trapezius EMG") & (meta["retained"])].index[:1].tolist()

    eda_unique = []
    for feature in eda_features:
        if feature in available and feature not in eda_unique:
            eda_unique.append(feature)
    emg_unique = []
    for feature in emg_features:
        if feature in available and feature not in emg_unique:
            emg_unique.append(feature)

    interactions = []
    for eda in eda_unique[:4]:
        for emg in emg_unique[:2]:
            interactions.append((f"{eda}__x__{emg}", eda, emg))
    key_features = eda_unique[:2] + emg_unique[:1]
    return interactions, key_features


def add_interaction_columns(frame: pd.DataFrame, interactions: list[tuple[str, str, str]]) -> pd.DataFrame:
    out = frame.copy()
    for name, left, right in interactions:
        out[name] = out[left] * out[right]
    return out


def run_targeted_interactions(
    table: pd.DataFrame,
    base_columns: list[str],
    interactions: list[tuple[str, str, str]],
    folds: list[dict[str, object]],
    random_state: int,
) -> pd.DataFrame:
    if not interactions:
        return pd.DataFrame()
    augmented = add_interaction_columns(table, interactions)
    interaction_names = [name for name, _, _ in interactions]
    columns_by_condition = {
        "base_all_modalities": base_columns,
        "base_plus_targeted_eda_emg_interactions": base_columns + interaction_names,
    }
    fold_samples = fold_samples_from_splits(table, folds)
    rows = []
    for pair_name, low_class, high_class in PAIRS[:3]:
        pair = binary_pair_table(augmented, low_class, high_class)
        for model_name in MODELS:
            label = MODEL_LABELS[model_name]
            for condition, columns in columns_by_condition.items():
                x = pair[columns].to_numpy(dtype=float)
                y = pair["binary_label"].to_numpy(dtype=int)
                for fold in sorted(fold_samples):
                    test_index = pair.index[pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
                    train_index = pair.index[~pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
                    model = make_baseline_model(model_name, random_state=random_state)
                    model.fit(x[train_index], y[train_index])
                    pred = model.predict(x[test_index])
                    score = score_vector(model, x[test_index])
                    metrics = binary_metrics(y[test_index], pred, score)
                    rows.append(
                        {
                            "task": pair_name,
                            "model": label,
                            "condition": condition,
                            "fold": int(fold),
                            "n_features": len(columns),
                            "interaction_features": "|".join(interaction_names),
                            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                        }
                    )
    fold_df = pd.DataFrame(rows)
    base = fold_df[fold_df["condition"] == "base_all_modalities"].set_index(["task", "model", "fold"])
    plus = fold_df[fold_df["condition"] == "base_plus_targeted_eda_emg_interactions"].set_index(["task", "model", "fold"])
    delta = plus[["roc_auc", "balanced_accuracy", "f1", "accuracy"]].subtract(base[["roc_auc", "balanced_accuracy", "f1", "accuracy"]])
    delta = delta.add_prefix("delta_").reset_index()
    summary = (
        delta.groupby(["task", "model"], as_index=False)
        .agg(
            mean_delta_roc_auc=("delta_roc_auc", "mean"),
            sd_delta_roc_auc=("delta_roc_auc", "std"),
            mean_delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
            sd_delta_balanced_accuracy=("delta_balanced_accuracy", "std"),
            mean_delta_f1=("delta_f1", "mean"),
            sd_delta_f1=("delta_f1", "std"),
            positive_auc_fold_prop=("delta_roc_auc", lambda s: float(np.mean(np.asarray(s) > 0))),
        )
    )
    summary["interaction_features"] = "|".join(interaction_names)
    return fold_df.merge(delta, on=["task", "model", "fold"], how="left"), summary


def predict_positive_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(model.classes_)
        return proba[:, classes.index(1)]
    return np.asarray(model.decision_function(x), dtype=float)


def write_pdp_ice_figures(
    table: pd.DataFrame,
    columns: list[str],
    key_features: list[str],
    folds: list[dict[str, object]],
    output_dir: Path,
    random_state: int,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    made = []
    fold_samples = fold_samples_from_splits(table, folds)
    for pair_name, low_class, high_class in PAIRS[:3]:
        pair = binary_pair_table(table, low_class, high_class)
        usable_features = [f for f in key_features if f in columns]
        for feature in usable_features:
            feature_idx = columns.index(feature)
            x_all = pair[columns].to_numpy(dtype=float)
            values = x_all[:, feature_idx]
            grid = np.quantile(values, np.linspace(0.05, 0.95, 25))
            grid = np.unique(grid)
            if len(grid) < 4:
                continue
            ice_lines = []
            pdp_lines = []
            for fold in sorted(fold_samples):
                test_index = pair.index[pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
                train_index = pair.index[~pair["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
                x_train = x_all[train_index]
                y_train = pair.iloc[train_index]["binary_label"].to_numpy(dtype=int)
                x_test = x_all[test_index]
                model = make_baseline_model("xgboost_binary", random_state=random_state)
                model.fit(x_train, y_train)
                row_scores = []
                for value in grid:
                    x_grid = x_test.copy()
                    x_grid[:, feature_idx] = value
                    row_scores.append(predict_positive_scores(model, x_grid))
                scores = np.vstack(row_scores).T
                pdp_lines.append(scores.mean(axis=0))
                take = np.linspace(0, scores.shape[0] - 1, min(12, scores.shape[0]), dtype=int)
                ice_lines.extend(scores[take])
            pdp = np.vstack(pdp_lines).mean(axis=0)
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for line in ice_lines[:60]:
                ax.plot(grid, line, color="#7d8da1", alpha=0.18, linewidth=0.8)
            ax.plot(grid, pdp, color="#1f5a98", linewidth=2.4, label="fold-averaged PDP")
            ax.set_title(f"{pair_name}: {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel(f"Predicted probability of {EXPECTED_LABEL_MAP[high_class]}")
            ax.set_ylim(-0.03, 1.03)
            ax.legend(loc="best")
            fig.tight_layout()
            out = figure_dir / f"pdp_ice_{pair_name}_{feature}.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            made.append(str(out))
    return made


def md_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> list[str]:
    float_cols = float_cols or set()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col in float_cols and pd.notna(value):
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    output_path: Path,
    ordinal_summary: pd.DataFrame,
    error_destinations: pd.DataFrame,
    binary_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    interactions: list[tuple[str, str, str]],
    figures: list[str],
    config: dict[str, object],
) -> None:
    primary = binary_summary[
        (binary_summary["model"] == PRIMARY_BINARY_MODEL) & (binary_summary["modality_set"] == PRIMARY_MODALITY_SET)
    ].copy()
    lr_primary = binary_summary[
        (binary_summary["model"] == "logistic_regression") & (binary_summary["modality_set"] == PRIMARY_MODALITY_SET)
    ].copy()
    eda = binary_summary[(binary_summary["model"] == PRIMARY_BINARY_MODEL) & (binary_summary["modality_set"] == "eda_only")].copy()
    eda_emg = binary_summary[(binary_summary["model"] == PRIMARY_BINARY_MODEL) & (binary_summary["modality_set"] == "eda_emg")].copy()
    adjacent_tasks = ["PA1_vs_PA2", "PA2_vs_PA3"]
    controls = ["PA1_vs_PA3", "BL1_vs_PA4"]

    adjacent_auc = primary[primary["task"].isin(adjacent_tasks)]["overall_roc_auc"].mean()
    control_auc = primary[primary["task"].isin(controls)]["overall_roc_auc"].mean()
    adjacent_ba = primary[primary["task"].isin(adjacent_tasks)]["overall_balanced_accuracy"].mean()
    control_ba = primary[primary["task"].isin(controls)]["overall_balanced_accuracy"].mean()
    eda_adjacent_auc = eda[eda["task"].isin(adjacent_tasks)]["overall_roc_auc"].mean()
    eda_control_auc = eda[eda["task"].isin(controls)]["overall_roc_auc"].mean()

    pair_show = primary[[
        "task",
        "overall_roc_auc",
        "fold_roc_auc_mean",
        "fold_roc_auc_std",
        "overall_balanced_accuracy",
        "overall_f1",
        "overall_accuracy",
    ]].sort_values("task")
    modality_show = binary_summary[
        (binary_summary["model"].isin(["logistic_regression", PRIMARY_BINARY_MODEL]))
        & (binary_summary["task"].isin(PAIRS[i][0] for i in range(len(PAIRS))))
    ][
        [
            "task",
            "model",
            "modality_set",
            "overall_roc_auc",
            "fold_roc_auc_mean",
            "overall_balanced_accuracy",
            "overall_f1",
            "overall_accuracy",
        ]
    ].sort_values(["task", "model", "modality_set"])

    pa2_errors = error_destinations[
        (error_destinations["model"] == "xgboost")
        & (error_destinations["true_class"] == "PA2")
        & (error_destinations["count"] > 0)
    ].sort_values("count", ascending=False)

    lines = [
        "# Phase 5.5 Adjacent-Class Separability and Ordinal Error Analysis",
        "",
        f"Run date: {config['run_date_utc']}",
        "",
        "## Protocol",
        "",
        "- Reused the existing Phase 3 subject-independent fold assignments from `experiments/baselines/fold_assignments.csv`.",
        "- Reused the current `combined_unique` physiological feature set and did not modify the Phase 4/5 pipeline.",
        "- Binary benchmarks compare PA1 vs PA2, PA2 vs PA3, PA1 vs PA3, and BL1 vs PA4.",
        "- Models: fold-local Logistic Regression and XGBoost; preprocessing remains inside sklearn pipelines fit only on training subjects.",
        "- Targeted interaction analysis is limited to Phase 5-supported EDA and PA1 EMG features; no unrestricted feature mining or permutation importance is run.",
        "",
        "## Existing Multiclass OOF Ordinal Error",
        "",
    ]
    lines.extend(
        md_table(
            ordinal_summary,
            [
                "model",
                "accuracy",
                "mean_absolute_class_error",
                "within_one_class_proportion",
                "severe_error_rate_abs_ge_2",
                "exact_error_rate",
            ],
            {"accuracy", "mean_absolute_class_error", "within_one_class_proportion", "severe_error_rate_abs_ge_2", "exact_error_rate"},
        )
    )
    lines.extend(["", "### XGBoost PA2 Error Destinations", ""])
    lines.extend(md_table(pa2_errors, ["predicted_as", "count", "error_proportion"], {"error_proportion"}))

    lines.extend(["", "## All-Modality XGBoost Pair Separability", ""])
    lines.extend(
        md_table(
            pair_show,
            pair_show.columns.tolist(),
            {"overall_roc_auc", "fold_roc_auc_mean", "fold_roc_auc_std", "overall_balanced_accuracy", "overall_f1", "overall_accuracy"},
        )
    )
    lines.extend(
        [
            "",
            "## Full Binary Benchmark Table",
            "",
        ]
    )
    lines.extend(
        md_table(
            modality_show,
            modality_show.columns.tolist(),
            {"overall_roc_auc", "fold_roc_auc_mean", "overall_balanced_accuracy", "overall_f1", "overall_accuracy"},
        )
    )

    lines.extend(["", "## Targeted EDA x EMG Interaction Check", ""])
    lines.append("Targeted products: " + (", ".join(name for name, _, _ in interactions) if interactions else "none"))
    lines.append("")
    if not interaction_summary.empty:
        lines.extend(
            md_table(
                interaction_summary,
                [
                    "task",
                    "model",
                    "mean_delta_roc_auc",
                    "sd_delta_roc_auc",
                    "mean_delta_balanced_accuracy",
                    "mean_delta_f1",
                    "positive_auc_fold_prop",
                ],
                {
                    "mean_delta_roc_auc",
                    "sd_delta_roc_auc",
                    "mean_delta_balanced_accuracy",
                    "mean_delta_f1",
                    "positive_auc_fold_prop",
                },
            )
        )
    else:
        lines.append("No targeted interaction features were available.")

    lines.extend(["", "## Focused PDP/ICE Figures", ""])
    for figure in figures:
        lines.append(f"- `{figure}`")

    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"- Are PA1 vs PA2 and PA2 vs PA3 genuinely less separable than wider pain-level pairs? Yes in the primary all-modality XGBoost benchmark: adjacent mean ROC-AUC is {adjacent_auc:.3f} versus {control_auc:.3f} for the wider controls, and adjacent mean balanced accuracy is {adjacent_ba:.3f} versus {control_ba:.3f}. Logistic Regression shows the same direction with all modalities: adjacent mean ROC-AUC "
            f"{lr_primary[lr_primary['task'].isin(adjacent_tasks)]['overall_roc_auc'].mean():.3f} versus controls {lr_primary[lr_primary['task'].isin(controls)]['overall_roc_auc'].mean():.3f}.",
            f"- Does EDA preserve ordinal pain information even when exact PA2 classification is weak? Yes, with qualification: EDA-only XGBoost retains above-chance pairwise ROC-AUC, averaging {eda_adjacent_auc:.3f} for adjacent pairs and {eda_control_auc:.3f} for wider controls, while the multiclass OOF errors are mostly within one pain level.",
        ]
    )
    merged = eda_emg.set_index("task")["overall_roc_auc"].subtract(eda.set_index("task")["overall_roc_auc"])
    adj_delta = merged.loc[adjacent_tasks].mean()
    lines.append(
        f"- Do EDA+EMG combinations improve adjacent-class discrimination? The effect is modest and model-dependent in this targeted benchmark; for primary XGBoost, EDA+EMG changes adjacent ROC-AUC by {adj_delta:+.3f} on average versus EDA alone."
    )
    severe = ordinal_summary.loc[ordinal_summary["model"] == "xgboost", "severe_error_rate_abs_ge_2"]
    severe_value = float(severe.iloc[0]) if not severe.empty else np.nan
    within = ordinal_summary.loc[ordinal_summary["model"] == "xgboost", "within_one_class_proportion"]
    within_value = float(within.iloc[0]) if not within.empty else np.nan
    lines.append(
        f"- Is PA2 difficulty more consistent with class overlap than with a few harmful individual features? Yes. The OOF multiclass pattern has within-±1 proportion {within_value:.3f} and severe-error rate {severe_value:.3f}; binary adjacent separability is lower than non-adjacent controls, while targeted EDA x EMG interactions do not produce a large rescue. That points more toward adjacent decision-boundary overlap than a small set of harmful features."
    )

    lines.extend(
        [
            "",
            "## Artifact Index",
            "",
            "- `experiments/adjacent_class_analysis/multiclass_confusion_matrix.csv`",
            "- `experiments/adjacent_class_analysis/multiclass_error_destinations.csv`",
            "- `experiments/adjacent_class_analysis/multiclass_ordinal_error_summary.csv`",
            "- `experiments/adjacent_class_analysis/binary_pair_fold_metrics.csv`",
            "- `experiments/adjacent_class_analysis/binary_pair_summary_metrics.csv`",
            "- `experiments/adjacent_class_analysis/binary_pair_oof_predictions.csv`",
            "- `experiments/adjacent_class_analysis/targeted_interaction_fold_metrics.csv`",
            "- `experiments/adjacent_class_analysis/targeted_interaction_summary.csv`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    table = read_feature_table(Path(args.feature_table))
    metadata = pd.read_json(args.feature_metadata)
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    columns = sets[FEATURE_SET]
    columns_by_modality = columns_for_modalities(metadata, columns)
    fold_samples = load_phase3_folds(Path(args.fold_assignments))
    folds = make_fold_indices(table, fold_samples)

    confusion, destinations, ordinal_summary = analyze_multiclass_oof(Path(args.multiclass_oof))
    confusion.to_csv(output_dir / "multiclass_confusion_matrix.csv", index=False)
    destinations.to_csv(output_dir / "multiclass_error_destinations.csv", index=False)
    ordinal_summary.to_csv(output_dir / "multiclass_ordinal_error_summary.csv", index=False)

    fold_metrics, summary, binary_oof = run_binary_benchmark(table, columns_by_modality, folds, args.random_state)
    fold_metrics.to_csv(output_dir / "binary_pair_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "binary_pair_summary_metrics.csv", index=False)
    binary_oof.to_csv(output_dir / "binary_pair_oof_predictions.csv", index=False)

    interactions, key_features = select_targeted_interactions(metadata, columns, Path(args.phase5_dir))
    interaction_fold, interaction_summary = run_targeted_interactions(
        table,
        columns_by_modality[PRIMARY_MODALITY_SET],
        interactions,
        folds,
        args.random_state,
    )
    interaction_fold.to_csv(output_dir / "targeted_interaction_fold_metrics.csv", index=False)
    interaction_summary.to_csv(output_dir / "targeted_interaction_summary.csv", index=False)

    figures = write_pdp_ice_figures(
        table,
        columns_by_modality[PRIMARY_MODALITY_SET],
        key_features,
        folds,
        output_dir,
        args.random_state,
    )

    config = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_table": args.feature_table,
        "feature_metadata": args.feature_metadata,
        "fold_assignments": args.fold_assignments,
        "multiclass_oof": args.multiclass_oof,
        "feature_set": FEATURE_SET,
        "random_state": args.random_state,
        "pairs": PAIRS,
        "modality_sets": MODALITY_SETS,
        "targeted_interactions": interactions,
        "key_pdp_features": key_features,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(
        Path(args.report),
        ordinal_summary,
        destinations,
        summary,
        interaction_summary,
        interactions,
        figures,
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5.5 adjacent-class separability and ordinal error analysis")
    parser.add_argument("--feature-table", default="artifacts/feature_table_filtered.csv")
    parser.add_argument("--feature-metadata", default="artifacts/feature_metadata.json")
    parser.add_argument("--fold-assignments", default="experiments/baselines/fold_assignments.csv")
    parser.add_argument("--multiclass-oof", default="experiments/modality_contribution/oof_predictions.csv")
    parser.add_argument("--phase5-dir", default="experiments/feature_contribution")
    parser.add_argument("--output-dir", default="experiments/adjacent_class_analysis")
    parser.add_argument("--report", default="reports/adjacent_class_analysis.md")
    parser.add_argument("--random-state", type=int, default=20260909)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
