from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.evaluation.metrics import classification_metrics, fold_summary
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs


FEATURE_SET = "combined_unique"
MODELS = ["xgboost", "random_forest", "rbf_svc", "logistic_regression"]
PRIMARY_MODEL = "xgboost"
LABELS = sorted(EXPECTED_LABEL_MAP)
CLASS_NAMES = [EXPECTED_LABEL_MAP[label] for label in LABELS]
MODALITY_MAP = {
    "ECG": "ecg",
    "EDA/GSR": "eda",
    "Trapezius EMG": "emg",
}
MODALITY_SETS = {
    "full": ["ECG", "EDA/GSR", "Trapezius EMG"],
    "remove_ecg": ["EDA/GSR", "Trapezius EMG"],
    "remove_eda": ["ECG", "Trapezius EMG"],
    "remove_emg": ["ECG", "EDA/GSR"],
    "ecg_only": ["ECG"],
    "eda_only": ["EDA/GSR"],
    "emg_only": ["Trapezius EMG"],
}
ABLATIONS = {
    "ecg": "remove_ecg",
    "eda": "remove_eda",
    "emg": "remove_emg",
}


def columns_for_sets(metadata: pd.DataFrame, combined: list[str]) -> dict[str, list[str]]:
    retained = metadata[metadata["feature_name"].isin(combined)].copy()
    by_modality = {
        modality: retained.loc[retained["modality"] == modality, "feature_name"].tolist()
        for modality in MODALITY_MAP
    }
    out = {}
    for set_name, modalities in MODALITY_SETS.items():
        cols = []
        for modality in modalities:
            cols.extend(by_modality[modality])
        out[set_name] = [col for col in combined if col in set(cols)]
    return out


def run_one(
    table: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    model_name: str,
    modality_set: str,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    x = table[columns].to_numpy(dtype=float)
    y = table["class_id"].to_numpy(dtype=int)
    oof_rows = []
    fold_metrics = []

    for split in folds:
        model = make_baseline_model(model_name, random_state=random_state)
        train_index = split["train_index"]
        test_index = split["test_index"]
        model.fit(x[train_index], y[train_index])
        pred = model.predict(x[test_index])
        proba = model.predict_proba(x[test_index]) if hasattr(model, "predict_proba") else None
        model_classes = [int(c) for c in model.classes_] if proba is not None else []
        fold_result = classification_metrics(y[test_index], pred, LABELS, EXPECTED_LABEL_MAP)
        fold_metrics.append(
            {
                "fold": int(split["fold"]),
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "n_train_subjects": int(len(split["train_subjects"])),
                "n_test_subjects": int(len(split["test_subjects"])),
                "test_subjects": split["test_subjects"],
                "macro_f1": fold_result["macro_f1"],
                "balanced_accuracy": fold_result["balanced_accuracy"],
                "per_class": fold_result["per_class"],
                "confusion_matrix": fold_result["confusion_matrix"],
            }
        )
        for pos, idx in enumerate(test_index):
            source = table.iloc[int(idx)]
            row = {
                "task": "5class",
                "model": model_name,
                "feature_set": FEATURE_SET,
                "modality_set": modality_set,
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
            if proba is not None:
                for class_pos, class_id in enumerate(model_classes):
                    row[f"proba_{EXPECTED_LABEL_MAP[class_id]}"] = float(proba[pos, class_pos])
            oof_rows.append(row)

    oof = pd.DataFrame(oof_rows).sort_values(["model", "modality_set", "fold", "subject_id", "sample_id"]).reset_index(drop=True)
    overall = classification_metrics(
        oof["true_class_id"].to_numpy(dtype=int),
        oof["pred_class_id"].to_numpy(dtype=int),
        LABELS,
        EXPECTED_LABEL_MAP,
    )
    return {
        "task": "5class",
        "model": model_name,
        "feature_set": FEATURE_SET,
        "modality_set": modality_set,
        "modalities": MODALITY_SETS[modality_set],
        "n_features": len(columns),
        "features": columns,
        "overall": overall,
        "fold_metrics": fold_metrics,
        "fold_summary": {
            "macro_f1": fold_summary(fold_metrics, "macro_f1"),
            "balanced_accuracy": fold_summary(fold_metrics, "balanced_accuracy"),
        },
    }, oof


def per_class_rows(result: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "model": result["model"],
            "modality_set": result["modality_set"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "support": row["support"],
        }
        for row in result["overall"]["per_class"]
    ]


def write_metric_tables(output_dir: Path, results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    per_rows = []
    fold_rows = []
    conf_rows = []
    for result in results:
        base = {
            "task": "5class",
            "model": result["model"],
            "feature_set": FEATURE_SET,
            "modality_set": result["modality_set"],
            "modalities": "+".join(result["modalities"]),
            "n_features": result["n_features"],
        }
        summary_rows.append(
            {
                **base,
                "macro_f1": result["overall"]["macro_f1"],
                "balanced_accuracy": result["overall"]["balanced_accuracy"],
                "fold_macro_f1_mean": result["fold_summary"]["macro_f1"]["mean"],
                "fold_macro_f1_std": result["fold_summary"]["macro_f1"]["std"],
                "fold_balanced_accuracy_mean": result["fold_summary"]["balanced_accuracy"]["mean"],
                "fold_balanced_accuracy_std": result["fold_summary"]["balanced_accuracy"]["std"],
            }
        )
        per_rows.extend([{**base, **row} for row in result["overall"]["per_class"]])
        for split in result["fold_metrics"]:
            fold_rows.append({**base, **{k: v for k, v in split.items() if k not in ["per_class", "confusion_matrix"]}})
            for cls in split["per_class"]:
                fold_rows.append(
                    {
                        **base,
                        "fold": split["fold"],
                        "n_train": split["n_train"],
                        "n_test": split["n_test"],
                        "n_train_subjects": split["n_train_subjects"],
                        "n_test_subjects": split["n_test_subjects"],
                        "test_subjects": split["test_subjects"],
                        "metric_scope": "class",
                        "class_id": cls["class_id"],
                        "class_name": cls["class_name"],
                        "precision": cls["precision"],
                        "recall": cls["recall"],
                        "f1": cls["f1"],
                        "support": cls["support"],
                    }
                )
        matrix = np.asarray(result["overall"]["confusion_matrix"], dtype=int)
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                conf_rows.append({**base, "true_class": true_name, "pred_class": pred_name, "count": int(matrix[i, j])})

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_rows)
    fold = pd.DataFrame(fold_rows)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    fold.to_csv(output_dir / "fold_metrics_long.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(output_dir / "confusion_matrices_long.csv", index=False)
    return summary, per_class, fold


def build_delta_tables(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    results: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    pain_rows = []
    fold_rows = []
    summary_idx = summary.set_index(["model", "modality_set"])
    per_idx = per_class.set_index(["model", "modality_set", "class_name"])
    by_result = {(r["model"], r["modality_set"]): r for r in results}

    for model in MODELS:
        for modality, removal_set in ABLATIONS.items():
            full = summary_idx.loc[(model, "full")]
            removed = summary_idx.loc[(model, removal_set)]
            overall_rows.append(
                {
                    "model": model,
                    "modality": modality,
                    "full_modality_set": "full",
                    "ablated_modality_set": removal_set,
                    "delta_macro_f1": full["macro_f1"] - removed["macro_f1"],
                    "delta_balanced_accuracy": full["balanced_accuracy"] - removed["balanced_accuracy"],
                    "full_macro_f1": full["macro_f1"],
                    "ablated_macro_f1": removed["macro_f1"],
                    "full_balanced_accuracy": full["balanced_accuracy"],
                    "ablated_balanced_accuracy": removed["balanced_accuracy"],
                }
            )
            for class_name in CLASS_NAMES:
                full_cls = per_idx.loc[(model, "full", class_name)]
                removed_cls = per_idx.loc[(model, removal_set, class_name)]
                pain_rows.append(
                    {
                        "model": model,
                        "modality": modality,
                        "class_name": class_name,
                        "delta_precision": full_cls["precision"] - removed_cls["precision"],
                        "delta_recall": full_cls["recall"] - removed_cls["recall"],
                        "delta_f1": full_cls["f1"] - removed_cls["f1"],
                        "full_precision": full_cls["precision"],
                        "ablated_precision": removed_cls["precision"],
                        "full_recall": full_cls["recall"],
                        "ablated_recall": removed_cls["recall"],
                        "full_f1": full_cls["f1"],
                        "ablated_f1": removed_cls["f1"],
                    }
                )

            full_result = by_result[(model, "full")]
            removed_result = by_result[(model, removal_set)]
            for full_fold, removed_fold in zip(full_result["fold_metrics"], removed_result["fold_metrics"], strict=True):
                if int(full_fold["fold"]) != int(removed_fold["fold"]):
                    raise ValueError("Fold mismatch while computing deltas")
                fold_rows.append(
                    {
                        "model": model,
                        "modality": modality,
                        "fold": int(full_fold["fold"]),
                        "metric_scope": "overall",
                        "class_name": "ALL",
                        "delta_macro_f1": full_fold["macro_f1"] - removed_fold["macro_f1"],
                        "delta_balanced_accuracy": full_fold["balanced_accuracy"] - removed_fold["balanced_accuracy"],
                    }
                )
                full_by_class = {row["class_name"]: row for row in full_fold["per_class"]}
                removed_by_class = {row["class_name"]: row for row in removed_fold["per_class"]}
                for class_name in CLASS_NAMES:
                    full_cls = full_by_class[class_name]
                    removed_cls = removed_by_class[class_name]
                    fold_rows.append(
                        {
                            "model": model,
                            "modality": modality,
                            "fold": int(full_fold["fold"]),
                            "metric_scope": "class",
                            "class_name": class_name,
                            "delta_precision": full_cls["precision"] - removed_cls["precision"],
                            "delta_recall": full_cls["recall"] - removed_cls["recall"],
                            "delta_f1": full_cls["f1"] - removed_cls["f1"],
                        }
                    )

    return pd.DataFrame(overall_rows), pd.DataFrame(pain_rows), pd.DataFrame(fold_rows)


def classify_consistency(model_positive: float, fold_positive: float) -> str:
    if model_positive >= 0.75 and fold_positive >= 0.65:
        return "cross-model robust positive"
    if model_positive <= 0.25 and fold_positive <= 0.35:
        return "cross-model robust negative"
    if (model_positive >= 0.50 and fold_positive >= 0.50) or (model_positive <= 0.50 and fold_positive <= 0.50):
        return "moderately consistent"
    return "model-dependent / unstable"


def consistency_table(pain_delta: pd.DataFrame, fold_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    class_fold = fold_delta[fold_delta["metric_scope"] == "class"].copy()
    for (modality, class_name), group in pain_delta.groupby(["modality", "class_name"], sort=False):
        fold_group = class_fold[(class_fold["modality"] == modality) & (class_fold["class_name"] == class_name)]
        model_delta = group.set_index("model")["delta_f1"].to_dict()
        model_positive = float((group["delta_f1"] > 0).mean())
        fold_positive = float((fold_group["delta_f1"] > 0).mean())
        rows.append(
            {
                "modality": modality,
                "class_name": class_name,
                "mean_delta_f1": float(group["delta_f1"].mean()),
                "median_delta_f1": float(group["delta_f1"].median()),
                "min_delta_f1": float(group["delta_f1"].min()),
                "max_delta_f1": float(group["delta_f1"].max()),
                "xgboost_delta_f1": float(model_delta.get("xgboost", np.nan)),
                "random_forest_delta_f1": float(model_delta.get("random_forest", np.nan)),
                "rbf_svc_delta_f1": float(model_delta.get("rbf_svc", np.nan)),
                "logistic_regression_delta_f1": float(model_delta.get("logistic_regression", np.nan)),
                "positive_model_proportion": model_positive,
                "positive_fold_proportion": fold_positive,
                "consistency": classify_consistency(model_positive, fold_positive),
            }
        )
    return pd.DataFrame(rows)


def heatmap(table: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    primary = table[table["model"] == PRIMARY_MODEL]
    pivot = primary.pivot(index="modality", columns="class_name", values=metric).loc[["ecg", "eda", "emg"], CLASS_NAMES]
    vmax = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 0.001)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks(np.arange(3), labels=["ECG", "EDA", "EMG"])
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def consistency_figure(consistency: pd.DataFrame, output_path: Path) -> None:
    plot = consistency.copy()
    order = {"ecg": 0, "eda": 1, "emg": 2}
    plot["y"] = plot["modality"].map(order)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    sizes = 130 + 420 * plot["positive_model_proportion"]
    colors = plot["mean_delta_f1"]
    vmax = max(abs(float(colors.min())), abs(float(colors.max())), 0.001)
    scatter = ax.scatter(
        plot["class_name"].map({name: idx for idx, name in enumerate(CLASS_NAMES)}),
        plot["y"],
        s=sizes,
        c=colors,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
    )
    for _, row in plot.iterrows():
        ax.text(
            CLASS_NAMES.index(row["class_name"]),
            row["y"],
            f"{row['positive_fold_proportion']:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks([0, 1, 2], labels=["ECG", "EDA", "EMG"])
    ax.set_title(
        "Cross-model contribution consistency\n"
        "color = mean ΔF1, label = positive fold share, size = positive model share",
        fontsize=11,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.84)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def md_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---:" if col in formats else "---" for col in columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(formats[col].format(value) if col in formats else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_path: Path,
    summary: pd.DataFrame,
    overall_delta: pd.DataFrame,
    pain_delta: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    primary_overall = overall_delta[overall_delta["model"] == PRIMARY_MODEL].copy()
    primary_pain = pain_delta[pain_delta["model"] == PRIMARY_MODEL].copy()
    all_overall = overall_delta.groupby("modality", as_index=False).agg(
        mean_delta_macro_f1=("delta_macro_f1", "mean"),
        mean_delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
        positive_model_proportion=("delta_macro_f1", lambda x: float((x > 0).mean())),
    )
    xgb_pivot = primary_pain.pivot(index="class_name", columns="modality", values="delta_f1").loc[CLASS_NAMES, ["ecg", "eda", "emg"]]
    eda_rank = xgb_pivot.idxmax(axis=1)
    negative_primary = primary_pain[primary_pain["delta_f1"] < 0].sort_values("delta_f1")
    unstable = consistency[consistency["consistency"] == "model-dependent / unstable"].copy()
    robust = consistency[consistency["consistency"].str.contains("robust", regex=False)].copy()

    lines = [
        "# Phase 4 Cross-Model Modality Contribution Analysis",
        "",
        f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "Task: subject-independent 5-class classification (`BL1`, `PA1`, `PA2`, `PA3`, `PA4`) using BioVid Part A physiological features only.",
        "",
        "## Protocol",
        "",
        "- Main analysis model: `XGBoost + combined_unique`.",
        "- Robustness models: Random Forest, RBF-SVM, Logistic Regression.",
        "- Folds: exact Phase 3 subject-independent folds from `experiments/baselines/fold_assignments.csv`.",
        "- Learned preprocessing remains inside each sklearn Pipeline and is fit only on training subjects in each fold.",
        "- Contribution is defined as `performance(full ECG+EDA+EMG) - performance(without modality)`.",
        "- No SHAP, feature ranking, subject clustering, demographic analysis, CNN attribution, tuning, or binary attribution is run here.",
        "",
        "## XGBoost Overall Contribution",
        "",
    ]
    lines.extend(
        md_table(
            primary_overall[["modality", "delta_macro_f1", "delta_balanced_accuracy", "full_macro_f1", "ablated_macro_f1"]],
            ["modality", "delta_macro_f1", "delta_balanced_accuracy", "full_macro_f1", "ablated_macro_f1"],
            {col: "{:.4f}" for col in ["delta_macro_f1", "delta_balanced_accuracy", "full_macro_f1", "ablated_macro_f1"]},
        )
    )
    lines.extend(["", "## Cross-Model Overall Contribution", ""])
    lines.extend(
        md_table(
            all_overall,
            ["modality", "mean_delta_macro_f1", "mean_delta_balanced_accuracy", "positive_model_proportion"],
            {col: "{:.4f}" for col in ["mean_delta_macro_f1", "mean_delta_balanced_accuracy", "positive_model_proportion"]},
        )
    )
    lines.extend(["", "## XGBoost Pain-Level ΔF1", ""])
    xgb_f1 = primary_pain.pivot(index="modality", columns="class_name", values="delta_f1").loc[["ecg", "eda", "emg"], CLASS_NAMES].reset_index()
    lines.extend(md_table(xgb_f1, ["modality"] + CLASS_NAMES, {name: "{:.4f}" for name in CLASS_NAMES}))
    lines.extend(["", "## XGBoost Pain-Level ΔRecall", ""])
    xgb_recall = primary_pain.pivot(index="modality", columns="class_name", values="delta_recall").loc[["ecg", "eda", "emg"], CLASS_NAMES].reset_index()
    lines.extend(md_table(xgb_recall, ["modality"] + CLASS_NAMES, {name: "{:.4f}" for name in CLASS_NAMES}))
    lines.extend(["", "## Cross-Model Consistency", ""])
    compact = consistency[[
        "modality",
        "class_name",
        "mean_delta_f1",
        "xgboost_delta_f1",
        "random_forest_delta_f1",
        "rbf_svc_delta_f1",
        "logistic_regression_delta_f1",
        "positive_model_proportion",
        "positive_fold_proportion",
        "consistency",
    ]]
    lines.extend(
        md_table(
            compact,
            compact.columns.tolist(),
            {
                "mean_delta_f1": "{:.4f}",
                "xgboost_delta_f1": "{:.4f}",
                "random_forest_delta_f1": "{:.4f}",
                "rbf_svc_delta_f1": "{:.4f}",
                "logistic_regression_delta_f1": "{:.4f}",
                "positive_model_proportion": "{:.2f}",
                "positive_fold_proportion": "{:.2f}",
            },
        )
    )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `experiments/modality_contribution/figures/xgboost_modality_pain_delta_f1_heatmap.png`",
            "- `experiments/modality_contribution/figures/xgboost_modality_pain_delta_recall_heatmap.png`",
            "- `experiments/modality_contribution/figures/cross_model_contribution_consistency.png`",
            "",
            "## Interpretation",
            "",
        ]
    )
    eda_all = bool((eda_rank == "eda").all())
    if eda_all:
        lines.append("- EDA is the largest XGBoost ΔF1 contributor for every pain level.")
    else:
        parts = []
        for cls, mod in eda_rank.items():
            value = xgb_pivot.loc[cls, mod]
            qualifier = "least negative" if value < 0 else "largest positive"
            parts.append(f"{cls}={mod} ({qualifier}, ΔF1={value:+.4f})")
        lines.append("- EDA is not the largest XGBoost ΔF1 contributor for every pain level; by level, the highest XGBoost delta is " + ", ".join(parts) + ".")
    pa2 = primary_pain[primary_pain["class_name"] == "PA2"].sort_values("delta_f1", ascending=False)
    lines.append(
        "- PA2 shows "
        + (
            "a distinct pattern: "
            if pa2.iloc[0]["modality"] != primary_pain.groupby("modality")["delta_f1"].mean().idxmax()
            else "no clean reversal relative to the overall XGBoost pattern, but it remains one of the weakest and most sensitive classes: "
        )
        + ", ".join(f"{row.modality} ΔF1={row.delta_f1:+.4f}" for row in pa2.itertuples())
        + "."
    )
    if negative_primary.empty:
        lines.append("- Under XGBoost, no pain-level ΔF1 is negative after modality removal.")
    else:
        examples = "; ".join(f"{row.modality}/{row.class_name} ΔF1={row.delta_f1:+.4f}" for row in negative_primary.head(8).itertuples())
        lines.append(f"- Some XGBoost pain-level scores improve after removing a modality, most notably {examples}.")
    if not robust.empty:
        robust_examples = "; ".join(f"{row.modality}/{row.class_name}: {row.consistency}" for row in robust.itertuples())
        lines.append(f"- Cross-model robust cells: {robust_examples}.")
    else:
        lines.append("- No modality × pain-level cell meets the stricter cross-model robust criterion; most effects are modest and fold-sensitive.")
    if not unstable.empty:
        unstable_examples = "; ".join(f"{row.modality}/{row.class_name}" for row in unstable.itertuples())
        lines.append(f"- Model-dependent or unstable cells: {unstable_examples}.")
    lines.extend(
        [
            "- Population-level modality contribution does depend on model choice: direction is usually aligned for some endpoint classes, but intermediate pain levels have small deltas and frequent sign changes.",
            "- These outputs are sufficient to enter Phase 5 feature-level attribution, but Phase 5 should treat stable modality/pain cells as priorities and explicitly separate robust signals from model-dependent effects.",
            "",
            "## Machine-Readable Outputs",
            "",
            "- `experiments/modality_contribution/metrics.json`",
            "- `experiments/modality_contribution/summary_metrics.csv`",
            "- `experiments/modality_contribution/per_class_metrics.csv`",
            "- `experiments/modality_contribution/modality_overall_delta.csv`",
            "- `experiments/modality_contribution/modality_pain_level_delta.csv`",
            "- `experiments/modality_contribution/cross_model_consistency.csv`",
            "- `experiments/modality_contribution/fold_level_delta.csv`",
            "- `experiments/modality_contribution/oof_predictions.csv`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, random_state: int) -> None:
    output_dir = root / "experiments" / "modality_contribution"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(root / "artifacts" / "feature_table_filtered.csv")
    metadata = pd.read_json(root / "artifacts" / "feature_metadata.json")
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    set_columns = columns_for_sets(metadata, sets[FEATURE_SET])
    folds = make_fold_indices(table, load_phase3_folds(root / "experiments" / "baselines" / "fold_assignments.csv"))

    results = []
    oof_frames = []
    for model_name in MODELS:
        for modality_set, columns in set_columns.items():
            result, oof = run_one(table, columns, folds, model_name, modality_set, random_state)
            results.append(result)
            oof_frames.append(oof)
            print(f"finished {model_name}/{modality_set} ({len(columns)} features)", flush=True)

    oof_all = pd.concat(oof_frames, ignore_index=True)
    summary, per_class, _ = write_metric_tables(output_dir, results)
    overall_delta, pain_delta, fold_delta = build_delta_tables(summary, per_class, results)
    consistency = consistency_table(pain_delta, fold_delta)

    oof_all.to_csv(output_dir / "oof_predictions.csv", index=False)
    overall_delta.to_csv(output_dir / "modality_overall_delta.csv", index=False)
    pain_delta.to_csv(output_dir / "modality_pain_level_delta.csv", index=False)
    fold_delta.to_csv(output_dir / "fold_level_delta.csv", index=False)
    consistency.to_csv(output_dir / "cross_model_consistency.csv", index=False)

    heatmap(pain_delta, "delta_f1", "XGBoost modality contribution by pain level: ΔF1", figure_dir / "xgboost_modality_pain_delta_f1_heatmap.png")
    heatmap(pain_delta, "delta_recall", "XGBoost modality contribution by pain level: ΔRecall", figure_dir / "xgboost_modality_pain_delta_recall_heatmap.png")
    consistency_figure(consistency, figure_dir / "cross_model_contribution_consistency.png")

    config = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase3_folds": "experiments/baselines/fold_assignments.csv",
        "task": "5class",
        "classes": CLASS_NAMES,
        "feature_set": FEATURE_SET,
        "primary_model": PRIMARY_MODEL,
        "models": MODELS,
        "modality_sets": MODALITY_SETS,
        "contribution_definition": "full performance minus performance without the named modality",
        "random_state": random_state,
        "set_n_features": {name: len(cols) for name, cols in set_columns.items()},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"config": config, "results": results}, indent=2), encoding="utf-8")
    write_report(root / "reports" / "modality_contribution.md", summary, overall_delta, pain_delta, consistency)
    print("Wrote Phase 4 outputs to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--random-state", type=int, default=20260909)
    args = parser.parse_args()
    run(args.root.resolve(), args.random_state)


if __name__ == "__main__":
    main()
