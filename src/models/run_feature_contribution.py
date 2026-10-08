from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs


FEATURE_SET = "combined_unique"
PRIMARY_MODEL = "xgboost"
ROBUSTNESS_MODELS = ["logistic_regression"]
LABELS = sorted(EXPECTED_LABEL_MAP)
CLASS_NAMES = [EXPECTED_LABEL_MAP[label] for label in LABELS]
KEY_CELLS = [
    ("eda", "EDA/GSR", "BL1"),
    ("eda", "EDA/GSR", "PA2"),
    ("eda", "EDA/GSR", "PA3"),
    ("eda", "EDA/GSR", "PA4"),
    ("emg", "Trapezius EMG", "PA1"),
    ("ecg", "ECG", "BL1"),
]


def class_metric(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[class_id],
        zero_division=0,
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "support": int(support[0]),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def shap_array(model, x_imputed: np.ndarray, columns: list[str]) -> np.ndarray:
    booster = model.named_steps["classifier"].get_booster()
    dmatrix = xgb.DMatrix(x_imputed, feature_names=columns)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    arr = np.asarray(contribs, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected XGBoost SHAP contribution shape: {arr.shape}")
    if arr.shape[1] == len(LABELS):
        return arr[:, :, :-1]
    if arr.shape[2] == len(LABELS):
        return np.transpose(arr[:, :-1, :], (0, 2, 1))
    raise ValueError(f"Cannot infer class/feature axes from SHAP shape: {arr.shape}")


def xgb_gain_importance(model, columns: list[str]) -> dict[str, float]:
    booster = model.named_steps["classifier"].get_booster()
    raw = booster.get_score(importance_type="gain")
    out = {}
    for idx, feature in enumerate(columns):
        out[feature] = float(raw.get(feature, raw.get(f"f{idx}", 0.0)))
    return out


def collect_xgb_shap(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    output_dir: Path,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = table["class_id"].to_numpy(dtype=int)
    x = table[columns].to_numpy(dtype=float)
    meta = metadata.set_index("feature_name")
    fold_rows = []
    summary_rows = []
    global_rows = []

    for split in folds:
        model = make_baseline_model(PRIMARY_MODEL, random_state=random_state)
        train_index = split["train_index"]
        test_index = split["test_index"]
        model.fit(x[train_index], y[train_index])
        imputer = model.named_steps["imputer"]
        x_test = imputer.transform(x[test_index])
        shap_values = shap_array(model, x_test, columns)
        gain = xgb_gain_importance(model, columns)

        for feature_idx, feature in enumerate(columns):
            info = meta.loc[feature]
            global_rows.append(
                {
                    "model": PRIMARY_MODEL,
                    "feature_set": FEATURE_SET,
                    "fold": int(split["fold"]),
                    "feature_name": feature,
                    "modality": info["modality"],
                    "provenance": info["provenance"],
                    "gain_importance": gain[feature],
                }
            )
            values = x_test[:, feature_idx]
            for class_pos, class_id in enumerate(LABELS):
                class_name = EXPECTED_LABEL_MAP[class_id]
                sv = shap_values[:, class_pos, feature_idx]
                mask = y[test_index] == class_id
                sv_true = sv[mask]
                values_true = values[mask]
                row = {
                    "model": PRIMARY_MODEL,
                    "feature_set": FEATURE_SET,
                    "fold": int(split["fold"]),
                    "explained_class_id": int(class_id),
                    "pain_class": class_name,
                    "feature_name": feature,
                    "modality": info["modality"],
                    "provenance": info["provenance"],
                    "mean_abs_shap_all": float(np.mean(np.abs(sv))),
                    "mean_signed_shap_all": float(np.mean(sv)),
                    "median_signed_shap_all": float(np.median(sv)),
                    "q25_signed_shap_all": float(np.quantile(sv, 0.25)),
                    "q75_signed_shap_all": float(np.quantile(sv, 0.75)),
                    "positive_shap_prop_all": float(np.mean(sv > 0)),
                    "feature_value_shap_corr_all": safe_corr(values, sv),
                    "mean_abs_shap_true_class": float(np.mean(np.abs(sv_true))),
                    "mean_signed_shap_true_class": float(np.mean(sv_true)),
                    "median_signed_shap_true_class": float(np.median(sv_true)),
                    "q25_signed_shap_true_class": float(np.quantile(sv_true, 0.25)),
                    "q75_signed_shap_true_class": float(np.quantile(sv_true, 0.75)),
                    "positive_shap_prop_true_class": float(np.mean(sv_true > 0)),
                    "feature_value_shap_corr_true_class": safe_corr(values_true, sv_true),
                    "n_all": int(len(sv)),
                    "n_true_class": int(mask.sum()),
                }
                fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby(["model", "feature_set", "pain_class", "feature_name", "modality", "provenance"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap_all", "mean"),
            sd_abs_shap=("mean_abs_shap_all", "std"),
            mean_signed_shap=("mean_signed_shap_true_class", "mean"),
            sd_signed_shap=("mean_signed_shap_true_class", "std"),
            median_signed_shap=("median_signed_shap_true_class", "mean"),
            positive_shap_prop=("positive_shap_prop_true_class", "mean"),
            feature_value_shap_corr=("feature_value_shap_corr_true_class", "mean"),
            fold_positive_abs_top10_prop=("mean_abs_shap_all", lambda s: np.nan),
        )
        .reset_index(drop=True)
    )
    ranks = fold_df.copy()
    ranks["fold_rank_within_class"] = ranks.groupby(["fold", "pain_class"])["mean_abs_shap_all"].rank(
        ascending=False,
        method="min",
    )
    top10 = (
        ranks.assign(is_top10=lambda d: d["fold_rank_within_class"] <= 10)
        .groupby(["pain_class", "feature_name"])["is_top10"]
        .mean()
        .rename("fold_top10_frequency")
        .reset_index()
    )
    summary = summary.drop(columns=["fold_positive_abs_top10_prop"]).merge(
        top10,
        on=["pain_class", "feature_name"],
        how="left",
    )
    global_df = pd.DataFrame(global_rows)
    global_summary = (
        global_df.groupby(["model", "feature_set", "feature_name", "modality", "provenance"], as_index=False)
        .agg(
            mean_gain_importance=("gain_importance", "mean"),
            sd_gain_importance=("gain_importance", "std"),
            positive_gain_fold_prop=("gain_importance", lambda s: float(np.mean(np.asarray(s) > 0))),
        )
    )

    fold_df.to_csv(output_dir / "fold_level_feature_attribution.csv", index=False)
    summary.to_csv(output_dir / "class_specific_shap.csv", index=False)
    global_summary.to_csv(output_dir / "xgboost_global_feature_importance.csv", index=False)
    return fold_df, summary, global_summary


def select_target_features(shap_summary: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    rows = []
    for short_modality, modality, class_name in KEY_CELLS:
        subset = shap_summary[(shap_summary["modality"] == modality) & (shap_summary["pain_class"] == class_name)]
        subset = subset.sort_values(["mean_abs_shap", "fold_top10_frequency"], ascending=False).head(top_n)
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            rows.append(
                {
                    "modality": short_modality,
                    "modality_name": modality,
                    "pain_class": class_name,
                    "rank_in_cell": rank,
                    "feature_name": row["feature_name"],
                    "provenance": row["provenance"],
                    "mean_abs_shap": row["mean_abs_shap"],
                    "mean_signed_shap": row["mean_signed_shap"],
                    "positive_shap_prop": row["positive_shap_prop"],
                    "feature_value_shap_corr": row["feature_value_shap_corr"],
                }
            )
    return pd.DataFrame(rows)


def predict_fold_metrics(
    table: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    model_name: str,
    random_state: int,
) -> dict[tuple[int, str], dict[str, float]]:
    y = table["class_id"].to_numpy(dtype=int)
    x = table[columns].to_numpy(dtype=float)
    out = {}
    for split in folds:
        model = make_baseline_model(model_name, random_state=random_state)
        model.fit(x[split["train_index"]], y[split["train_index"]])
        pred = model.predict(x[split["test_index"]])
        for class_id, class_name in EXPECTED_LABEL_MAP.items():
            out[(int(split["fold"]), class_name)] = class_metric(y[split["test_index"]], pred, class_id)
    return out


def run_targeted_removal(
    table: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    targets: pd.DataFrame,
    output_dir: Path,
    random_state: int,
    models: list[str],
) -> pd.DataFrame:
    removal_specs = []
    for (modality, modality_name, pain_class), group in targets.groupby(["modality", "modality_name", "pain_class"]):
        removal_specs.append(
            {
                "removal_scope": "top_feature_group",
                "modality": modality,
                "modality_name": modality_name,
                "pain_class": pain_class,
                "removed_features": group.sort_values("rank_in_cell")["feature_name"].tolist(),
                "removed_feature_count": int(len(group)),
            }
        )

    rows = []
    for model_name in models:
        full_metrics = predict_fold_metrics(table, columns, folds, model_name, random_state)
        for spec in removal_specs:
            keep = [feature for feature in columns if feature not in set(spec["removed_features"])]
            ablated = predict_fold_metrics(table, keep, folds, model_name, random_state)
            for split in folds:
                fold = int(split["fold"])
                cls = spec["pain_class"]
                full = full_metrics[(fold, cls)]
                removed = ablated[(fold, cls)]
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": FEATURE_SET,
                        "fold": fold,
                        "modality": spec["modality"],
                        "modality_name": spec["modality_name"],
                        "pain_class": cls,
                        "removal_scope": spec["removal_scope"],
                        "removed_feature_count": spec["removed_feature_count"],
                        "removed_features": "|".join(spec["removed_features"]),
                        "delta_precision": full["precision"] - removed["precision"],
                        "delta_recall": full["recall"] - removed["recall"],
                        "delta_f1": full["f1"] - removed["f1"],
                        "full_f1": full["f1"],
                        "removed_f1": removed["f1"],
                        "full_recall": full["recall"],
                        "removed_recall": removed["recall"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "targeted_feature_removal_fold_level.csv", index=False)
    summary = (
        df.groupby(
            [
                "model",
                "feature_set",
                "modality",
                "modality_name",
                "pain_class",
                "removal_scope",
                "removed_feature_count",
                "removed_features",
            ],
            as_index=False,
        )
        .agg(
            mean_delta_f1=("delta_f1", "mean"),
            sd_delta_f1=("delta_f1", "std"),
            mean_delta_recall=("delta_recall", "mean"),
            sd_delta_recall=("delta_recall", "std"),
            positive_f1_fold_prop=("delta_f1", lambda s: float(np.mean(np.asarray(s) > 0))),
            positive_recall_fold_prop=("delta_recall", lambda s: float(np.mean(np.asarray(s) > 0))),
        )
    )
    summary.to_csv(output_dir / "targeted_feature_removal_summary.csv", index=False)
    return summary


def model_top_features(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    random_state: int,
) -> pd.DataFrame:
    y = table["class_id"].to_numpy(dtype=int)
    x = table[columns].to_numpy(dtype=float)
    meta = metadata.set_index("feature_name")
    rows = []
    for model_name in ["random_forest", "logistic_regression"]:
        for split in folds:
            model = make_baseline_model(model_name, random_state=random_state)
            model.fit(x[split["train_index"]], y[split["train_index"]])
            classifier = model.named_steps["classifier"]
            if model_name == "random_forest":
                importances = classifier.feature_importances_
                for feature, importance in zip(columns, importances):
                    info = meta.loc[feature]
                    for class_name in CLASS_NAMES:
                        rows.append(
                            {
                                "model": model_name,
                                "fold": int(split["fold"]),
                                "pain_class": class_name,
                                "feature_name": feature,
                                "modality": info["modality"],
                                "provenance": info["provenance"],
                                "importance": float(importance),
                                "signed_direction": np.nan,
                            }
                        )
            else:
                coefs = classifier.coef_
                for class_pos, class_id in enumerate(classifier.classes_):
                    class_name = EXPECTED_LABEL_MAP[int(class_id)]
                    for feature, coef in zip(columns, coefs[class_pos]):
                        info = meta.loc[feature]
                        rows.append(
                            {
                                "model": model_name,
                                "fold": int(split["fold"]),
                                "pain_class": class_name,
                                "feature_name": feature,
                                "modality": info["modality"],
                                "provenance": info["provenance"],
                                "importance": float(abs(coef)),
                                "signed_direction": float(np.sign(coef)),
                            }
                        )
    return pd.DataFrame(rows)


def cross_model_consistency(
    shap_summary: pd.DataFrame,
    removal_summary: pd.DataFrame,
    model_features: pd.DataFrame,
    targets: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    xgb_rank = shap_summary.copy()
    xgb_rank["rank"] = xgb_rank.groupby("pain_class")["mean_abs_shap"].rank(ascending=False, method="min")
    model_summary = (
        model_features.groupby(["model", "pain_class", "feature_name", "modality"], as_index=False)
        .agg(importance=("importance", "mean"), signed_direction=("signed_direction", "mean"))
    )
    model_summary["rank"] = model_summary.groupby(["model", "pain_class"])["importance"].rank(
        ascending=False,
        method="min",
    )
    removal_group = removal_summary[removal_summary["removal_scope"] == "top_feature_group"].copy()

    for _, target in targets.iterrows():
        feature = target["feature_name"]
        pain_class = target["pain_class"]
        modality_name = target["modality_name"]
        xgb = xgb_rank[(xgb_rank["pain_class"] == pain_class) & (xgb_rank["feature_name"] == feature)].iloc[0]
        row = {
            "pain_class": pain_class,
            "modality": target["modality"],
            "modality_name": modality_name,
            "feature_name": feature,
            "xgboost_mean_abs_shap": xgb["mean_abs_shap"],
            "xgboost_rank_all_features": xgb["rank"],
            "xgboost_signed_direction": float(np.sign(xgb["mean_signed_shap"])),
            "xgboost_fold_top10_frequency": xgb["fold_top10_frequency"],
        }
        for model_name in ["random_forest", "logistic_regression"]:
            match = model_summary[
                (model_summary["model"] == model_name)
                & (model_summary["pain_class"] == pain_class)
                & (model_summary["feature_name"] == feature)
            ]
            if not match.empty:
                row[f"{model_name}_rank_all_features"] = float(match.iloc[0]["rank"])
                row[f"{model_name}_top10"] = bool(match.iloc[0]["rank"] <= 10)
                row[f"{model_name}_signed_direction"] = match.iloc[0]["signed_direction"]
        for model_name in ["xgboost", "logistic_regression"]:
            match = removal_group[
                (removal_group["model"] == model_name)
                & (removal_group["pain_class"] == pain_class)
                & (removal_group["modality_name"] == modality_name)
            ]
            if not match.empty:
                row[f"{model_name}_top_feature_group_delta_f1"] = float(match.iloc[0]["mean_delta_f1"])
                row[f"{model_name}_top_feature_group_positive_fold_prop"] = float(match.iloc[0]["positive_f1_fold_prop"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "top_feature_cross_model_consistency.csv", index=False)
    return out


def write_figures(shap_summary: pd.DataFrame, removal_summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    top_features = (
        shap_summary.groupby("feature_name")["mean_abs_shap"]
        .max()
        .sort_values(ascending=False)
        .head(25)
        .index.tolist()
    )
    heat = shap_summary[shap_summary["feature_name"].isin(top_features)].pivot(
        index="feature_name",
        columns="pain_class",
        values="mean_abs_shap",
    )[CLASS_NAMES]
    plt.figure(figsize=(8, max(6, 0.25 * len(heat))))
    plt.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean |SHAP|")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.title("XGBoost feature attribution by pain class")
    plt.tight_layout()
    plt.savefig(figures / "feature_pain_level_shap_heatmap.png", dpi=200)
    plt.close()

    eda = shap_summary[shap_summary["modality"] == "EDA/GSR"].pivot(
        index="feature_name",
        columns="pain_class",
        values="mean_signed_shap",
    )[CLASS_NAMES]
    limit = float(np.nanmax(np.abs(eda.to_numpy())))
    plt.figure(figsize=(8, max(5, 0.28 * len(eda))))
    plt.imshow(eda.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    plt.colorbar(label="Mean signed SHAP on true-class rows")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(eda.index)), eda.index)
    plt.title("EDA signed SHAP direction by pain class")
    plt.tight_layout()
    plt.savefig(figures / "eda_signed_shap_by_pain_level.png", dpi=200)
    plt.close()

    plot_df = removal_summary[
        (removal_summary["model"] == PRIMARY_MODEL) & (removal_summary["removal_scope"] == "top_feature_group")
    ].copy()
    plot_df["cell"] = plot_df["modality"] + "/" + plot_df["pain_class"]
    plt.figure(figsize=(8, 4.5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.bar(plot_df["cell"], plot_df["mean_delta_f1"], color="#3b6ea8")
    plt.ylabel("Delta F1: full - feature group removed")
    plt.xticks(rotation=30, ha="right")
    plt.title("Targeted XGBoost feature-removal validation")
    plt.tight_layout()
    plt.savefig(figures / "targeted_feature_removal_delta_f1.png", dpi=200)
    plt.close()


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
    shap_summary: pd.DataFrame,
    targets: pd.DataFrame,
    removal_summary: pd.DataFrame,
    consistency: pd.DataFrame,
    config: dict[str, object],
) -> None:
    xgb_groups = removal_summary[
        (removal_summary["model"] == PRIMARY_MODEL) & (removal_summary["removal_scope"] == "top_feature_group")
    ].copy()
    target_summary = targets[targets["rank_in_cell"] <= 3].copy()
    target_summary["cell"] = target_summary["modality"] + "/" + target_summary["pain_class"]
    eda_targets = target_summary[target_summary["modality"] == "eda"]
    emg_targets = target_summary[(target_summary["modality"] == "emg") & (target_summary["pain_class"] == "PA1")]
    supported = consistency[
        (consistency["xgboost_fold_top10_frequency"] >= 0.6)
        & (consistency["xgboost_top_feature_group_delta_f1"].fillna(0) > 0)
    ].copy()

    lines = [
        "# Phase 5 Pain-Level-Specific Feature Attribution",
        "",
        f"Run date: {config['run_date_utc']}",
        "",
        "## Protocol",
        "",
        "- Main attribution model: `XGBoost + combined_unique`.",
        "- Folds: exact Phase 3 subject-independent folds from `experiments/baselines/fold_assignments.csv`.",
        "- Preprocessing: `SimpleImputer(strategy=\"median\")` remains inside each fold-local sklearn Pipeline and is fit only on training subjects.",
        "- SHAP source: XGBoost Booster native `pred_contribs=True` TreeSHAP values on held-out fold rows after fold-local imputation.",
        "- Cross-model sensitivity: Random Forest is used for top-feature overlap; Logistic Regression is used for top-feature overlap and targeted group-removal direction.",
        "- Interpretation: class-level contribution means predictive contribution in this 5-class setting, not physiological causality.",
        "",
        "## Focused XGBoost SHAP Features",
        "",
    ]
    cols = ["cell", "rank_in_cell", "feature_name", "provenance", "mean_abs_shap", "mean_signed_shap", "positive_shap_prop", "feature_value_shap_corr"]
    lines.extend(md_table(target_summary, cols, {"mean_abs_shap", "mean_signed_shap", "positive_shap_prop", "feature_value_shap_corr"}))

    lines.extend(["", "## Targeted Feature-Removal Validation", ""])
    show = xgb_groups[["modality", "pain_class", "removed_features", "mean_delta_f1", "sd_delta_f1", "mean_delta_recall", "positive_f1_fold_prop"]]
    lines.extend(md_table(show, show.columns.tolist(), {"mean_delta_f1", "sd_delta_f1", "mean_delta_recall", "positive_f1_fold_prop"}))

    lines.extend(["", "## Cross-Model Sensitivity", ""])
    show_consistency = consistency[
        [
            "modality",
            "pain_class",
            "feature_name",
            "xgboost_fold_top10_frequency",
            "random_forest_top10",
            "logistic_regression_top10",
            "xgboost_top_feature_group_delta_f1",
            "logistic_regression_top_feature_group_delta_f1",
        ]
    ].copy()
    lines.extend(
        md_table(
            show_consistency,
            show_consistency.columns.tolist(),
            {
                "xgboost_fold_top10_frequency",
                "xgboost_top_feature_group_delta_f1",
                "logistic_regression_top_feature_group_delta_f1",
            },
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation Summary",
            "",
            "### EDA cells",
            "",
        ]
    )
    for class_name in ["BL1", "PA2", "PA3", "PA4"]:
        features = eda_targets[eda_targets["pain_class"] == class_name].sort_values("rank_in_cell")
        names = ", ".join(features["feature_name"].tolist())
        removal = xgb_groups[(xgb_groups["modality"] == "eda") & (xgb_groups["pain_class"] == class_name)]
        delta = float(removal.iloc[0]["mean_delta_f1"]) if not removal.empty else np.nan
        lines.append(f"- `{class_name}`: driven most by {names}. Top-feature-group removal ΔF1={delta:.4f}.")

    pa2 = eda_targets[eda_targets["pain_class"] == "PA2"].copy()
    pa2_overlap_features = sorted(
        set(pa2["feature_name"]).intersection(
            set(eda_targets[eda_targets["pain_class"].isin(["BL1", "PA3", "PA4"])]["feature_name"])
        )
    )
    pa2_direction = pa2[["feature_name", "mean_signed_shap", "positive_shap_prop", "feature_value_shap_corr"]]
    lines.extend(
        [
            "",
            "### PA2 EDA pattern",
            "",
            f"- PA2 shares top EDA features with endpoint or higher-pain cells: {', '.join(pa2_overlap_features) if pa2_overlap_features else 'none among the top-five lists'}.",
            "- The negative Phase 4 EDA/PA2 performance contribution is therefore best treated as a multiclass decision-boundary trade-off or class-overlap pattern until Phase 6 tests subject-level stability.",
            "- PA2 signed SHAP directions for the top EDA features:",
        ]
    )
    lines.extend(md_table(pa2_direction, pa2_direction.columns.tolist(), {"mean_signed_shap", "positive_shap_prop", "feature_value_shap_corr"}))

    lines.extend(["", "### EMG and ECG", ""])
    emg_names = ", ".join(emg_targets.sort_values("rank_in_cell")["feature_name"].tolist())
    emg_delta = xgb_groups[(xgb_groups["modality"] == "emg") & (xgb_groups["pain_class"] == "PA1")]
    lines.append(
        f"- `EMG/PA1`: driven most by {emg_names}; top-feature-group removal ΔF1={float(emg_delta.iloc[0]['mean_delta_f1']):.4f}."
    )
    ecg_targets = target_summary[(target_summary["modality"] == "ecg") & (target_summary["pain_class"] == "BL1")]
    ecg_delta = xgb_groups[(xgb_groups["modality"] == "ecg") & (xgb_groups["pain_class"] == "BL1")]
    lines.append(
        f"- `ECG/BL1`: secondary, smaller effect; top features are {', '.join(ecg_targets.sort_values('rank_in_cell')['feature_name'].tolist())}; top-feature-group removal ΔF1={float(ecg_delta.iloc[0]['mean_delta_f1']):.4f}."
    )

    lines.extend(["", "### Evidence Strength", ""])
    if supported.empty:
        lines.append("- No focused finding currently has both high fold top-10 SHAP frequency and positive XGBoost group-removal support.")
    else:
        lines.append(
            "- Supported by both XGBoost SHAP rank stability and XGBoost group-removal direction: "
            + "; ".join(
                f"{row.modality}/{row.pain_class}/{row.feature_name}"
                for row in supported.itertuples(index=False)
            )
            + "."
        )
    unstable = consistency[
        (consistency["xgboost_fold_top10_frequency"] < 0.6)
        | (consistency["xgboost_top_feature_group_positive_fold_prop"].fillna(0) < 0.6 if "xgboost_top_feature_group_positive_fold_prop" in consistency else False)
    ]
    lines.append(
        "- Findings with low fold top-10 frequency or weak cross-model agreement should be carried into Phase 6 as explicit stability questions, not as final feature conclusions."
    )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `experiments/feature_contribution/class_specific_shap.csv`",
            "- `experiments/feature_contribution/fold_level_feature_attribution.csv`",
            "- `experiments/feature_contribution/xgboost_global_feature_importance.csv`",
            "- `experiments/feature_contribution/targeted_feature_removal_summary.csv`",
            "- `experiments/feature_contribution/targeted_feature_removal_fold_level.csv`",
            "- `experiments/feature_contribution/top_feature_cross_model_consistency.csv`",
            "- `experiments/feature_contribution/figures/feature_pain_level_shap_heatmap.png`",
            "- `experiments/feature_contribution/figures/eda_signed_shap_by_pain_level.png`",
            "- `experiments/feature_contribution/figures/targeted_feature_removal_delta_f1.png`",
            "",
            "## Phase 6 Readiness",
            "",
            "The Phase 5 outputs are sufficient to enter Phase 6 cross-subject stability. The next phase should compute subject-level attribution profiles and test whether the apparent class-specific EDA/EMG/ECG patterns are broadly shared or driven by subsets of subjects.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table", default="artifacts/feature_table_filtered.csv")
    parser.add_argument("--metadata", default="artifacts/feature_metadata.json")
    parser.add_argument("--folds", default="experiments/baselines/fold_assignments.csv")
    parser.add_argument("--output-dir", default="experiments/feature_contribution")
    parser.add_argument("--report", default="reports/feature_contribution.md")
    parser.add_argument("--random-state", type=int, default=20260909)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.feature_table)
    metadata = pd.DataFrame(json.loads(Path(args.metadata).read_text(encoding="utf-8")))
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    columns = sets[FEATURE_SET]
    fold_samples = load_phase3_folds(Path(args.folds))
    folds = make_fold_indices(table, fold_samples)

    fold_df, shap_summary, global_summary = collect_xgb_shap(
        table,
        metadata,
        columns,
        folds,
        output_dir,
        args.random_state,
    )
    targets = select_target_features(shap_summary, top_n=3)
    targets.to_csv(output_dir / "focused_top_features.csv", index=False)
    removal_summary = run_targeted_removal(
        table,
        columns,
        folds,
        targets,
        output_dir,
        args.random_state,
        [PRIMARY_MODEL, *ROBUSTNESS_MODELS],
    )
    model_features = model_top_features(table, metadata, columns, folds, args.random_state)
    model_features.to_csv(output_dir / "robustness_model_feature_rank_inputs.csv", index=False)
    consistency = cross_model_consistency(shap_summary, removal_summary, model_features, targets, output_dir)
    write_figures(shap_summary, removal_summary, output_dir)

    config = {
        "run_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "feature_table": args.feature_table,
        "feature_metadata": args.metadata,
        "folds": args.folds,
        "feature_set": FEATURE_SET,
        "primary_model": PRIMARY_MODEL,
        "robustness_models": ROBUSTNESS_MODELS,
        "n_features": len(columns),
        "n_samples": int(len(table)),
        "folds_reused": sorted(int(f["fold"]) for f in folds),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(Path(args.report), shap_summary, targets, removal_summary, consistency, config)


if __name__ == "__main__":
    main()
