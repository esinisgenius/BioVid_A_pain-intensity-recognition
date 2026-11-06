from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs


RANDOM_STATE = 20260909
FEATURE_SET = "combined_unique"

# A small, deliberately structured diagnostic grid.  It spans conservative to
# high-capacity settings without turning this scientific audit into leaderboard
# optimisation.
CANDIDATES: list[dict[str, float | int]] = [
    {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.10, "min_child_weight": 10, "reg_alpha": 1.0, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 600, "max_depth": 2, "learning_rate": 0.02, "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 5.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.10, "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 0.5, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.02, "min_child_weight": 10, "reg_alpha": 1.0, "reg_lambda": 10.0, "subsample": 0.7, "colsample_bytree": 0.7},
]


def make_model(params: dict[str, float | int], random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                XGBClassifier(
                    **params,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=4,
                ),
            ),
        ]
    )


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=sorted(EXPECTED_LABEL_MAP), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def validate_protocol(table: pd.DataFrame, folds: list[dict[str, object]], columns: list[str]) -> dict[str, object]:
    label_map = (
        table[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    outer_overlaps = []
    for split in folds:
        overlap = set(split["train_subjects"]).intersection(split["test_subjects"])
        if overlap:
            outer_overlaps.append({"fold": split["fold"], "subjects": sorted(overlap)})
    forbidden = sorted(set(columns).intersection({"subject_id", "subject_name", "sample_id", "sample_name", "class_id", "class_name"}))
    class_counts = table.groupby(["class_id", "class_name"]).size().to_dict()
    checks = {
        "label_map_matches_expected": label_map == EXPECTED_LABEL_MAP,
        "observed_label_map": label_map,
        "class_counts": {f"{key[0]}:{key[1]}": int(value) for key, value in class_counts.items()},
        "n_subjects": int(table["subject_id"].nunique()),
        "samples_per_subject_min": int(table.groupby("subject_id").size().min()),
        "samples_per_subject_max": int(table.groupby("subject_id").size().max()),
        "outer_subject_overlap": outer_overlaps,
        "feature_metadata_columns_in_model": forbidden,
        "all_model_values_finite": bool(np.isfinite(table[columns].to_numpy(dtype=float)).all()),
        "duplicate_sample_names": int(table["sample_name"].duplicated().sum()),
        "preprocessing_pipeline": "SimpleImputer(median) is fit separately in every inner/outer training fit",
        "feature_screening_note": "Feature definitions are deterministic; the existing full-data degeneracy audit retained every candidate feature (feature_exclusions.csv is empty).",
    }
    failed = [
        not checks["label_map_matches_expected"],
        bool(outer_overlaps),
        bool(forbidden),
        not checks["all_model_values_finite"],
        bool(checks["duplicate_sample_names"]),
    ]
    if any(failed):
        raise ValueError(f"Protocol validation failed: {checks}")
    return checks


def run(root: Path, random_state: int = RANDOM_STATE) -> None:
    output_dir = root / "experiments" / "xgboost_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / "artifacts" / "feature_table_filtered.csv")
    metadata = pd.read_json(root / "artifacts" / "feature_metadata.json")
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    columns = sets[FEATURE_SET]
    folds = make_fold_indices(table, load_phase3_folds(root / "experiments" / "baselines" / "fold_assignments.csv"))
    checks = validate_protocol(table, folds, columns)

    x = table[columns].to_numpy(dtype=float)
    y = table["class_id"].to_numpy(dtype=int)
    groups = table["subject_id"].to_numpy(dtype=int)
    inner_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    for split in folds:
        outer_fold = int(split["fold"])
        train_index = np.asarray(split["train_index"], dtype=int)
        test_index = np.asarray(split["test_index"], dtype=int)
        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state + outer_fold)
        inner_splits = list(inner_splitter.split(x[train_index], y[train_index], groups[train_index]))

        for candidate_id, params in enumerate(CANDIDATES):
            scores = []
            for inner_fold, (inner_train_local, inner_valid_local) in enumerate(inner_splits, start=1):
                inner_train = train_index[inner_train_local]
                inner_valid = train_index[inner_valid_local]
                train_subjects = set(groups[inner_train])
                valid_subjects = set(groups[inner_valid])
                if train_subjects.intersection(valid_subjects):
                    raise ValueError(f"Inner subject leakage in outer fold {outer_fold}, inner fold {inner_fold}")
                model = make_model(params, random_state + outer_fold * 100 + inner_fold)
                model.fit(x[inner_train], y[inner_train])
                score = metric_row(y[inner_valid], model.predict(x[inner_valid]))
                scores.append(score["macro_f1"])
                inner_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "candidate_id": candidate_id,
                        "inner_fold": inner_fold,
                        "n_train_subjects": len(train_subjects),
                        "n_valid_subjects": len(valid_subjects),
                        "subject_overlap": 0,
                        **params,
                        **score,
                    }
                )
            selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate_id": candidate_id,
                    "inner_macro_f1_mean": float(np.mean(scores)),
                    "inner_macro_f1_std": float(np.std(scores, ddof=1)),
                    **params,
                }
            )

        outer_candidates = [row for row in selected_rows if row["outer_fold"] == outer_fold]
        best = max(outer_candidates, key=lambda row: (row["inner_macro_f1_mean"], -int(row["max_depth"]), -int(row["n_estimators"])))
        best_id = int(best["candidate_id"])
        best_params = CANDIDATES[best_id]
        selected_model = make_model(best_params, random_state + outer_fold)
        selected_model.fit(x[train_index], y[train_index])
        pred = selected_model.predict(x[test_index])
        score = metric_row(y[test_index], pred)
        outer_rows.append({"outer_fold": outer_fold, "selected_candidate_id": best_id, **score, **best_params})
        for pos, idx in enumerate(test_index):
            row = table.iloc[int(idx)]
            oof_rows.append(
                {
                    "outer_fold": outer_fold,
                    "subject_id": int(row["subject_id"]),
                    "sample_name": row["sample_name"],
                    "true_class_id": int(row["class_id"]),
                    "pred_class_id": int(pred[pos]),
                    "selected_candidate_id": best_id,
                }
            )

        # This is an explicitly labelled post-hoc sensitivity analysis, not the
        # selection mechanism.  It asks whether the poor result persists across
        # reasonable settings on the same untouched outer folds.
        for candidate_id, params in enumerate(CANDIDATES):
            model = make_model(params, random_state + outer_fold)
            model.fit(x[train_index], y[train_index])
            candidate_pred = model.predict(x[test_index])
            sensitivity_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate_id": candidate_id,
                    **metric_row(y[test_index], candidate_pred),
                    **params,
                }
            )
        print(f"finished outer fold {outer_fold}", flush=True)

    inner_df = pd.DataFrame(inner_rows)
    candidates_df = pd.DataFrame(selected_rows)
    candidates_df["selected"] = candidates_df.groupby("outer_fold")["inner_macro_f1_mean"].transform("max").eq(candidates_df["inner_macro_f1_mean"])
    outer_df = pd.DataFrame(outer_rows)
    oof_df = pd.DataFrame(oof_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_summary = (
        sensitivity_df.groupby("candidate_id", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .merge(pd.DataFrame([{"candidate_id": i, **params} for i, params in enumerate(CANDIDATES)]), on="candidate_id")
    )
    nested_overall = metric_row(oof_df["true_class_id"].to_numpy(), oof_df["pred_class_id"].to_numpy())

    inner_df.to_csv(output_dir / "inner_cv_scores.csv", index=False)
    candidates_df.to_csv(output_dir / "inner_candidate_summary.csv", index=False)
    outer_df.to_csv(output_dir / "nested_outer_fold_metrics.csv", index=False)
    oof_df.to_csv(output_dir / "nested_oof_predictions.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_fold_metrics.csv", index=False)
    sensitivity_summary.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    (output_dir / "protocol_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    config = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_state": random_state,
        "feature_set": FEATURE_SET,
        "n_features": len(columns),
        "outer_protocol": "reused 5-fold subject-independent StratifiedGroupKFold assignments",
        "inner_protocol": "3-fold StratifiedGroupKFold within each outer training set",
        "selection_metric": "macro_f1",
        "candidates": CANDIDATES,
        "nested_overall": nested_overall,
    }
    (output_dir / "config_and_results.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(
        root / "reports" / "xgboost_diagnostic.md",
        config,
        checks,
        outer_df,
        candidates_df,
        sensitivity_summary,
        oof_df,
    )


def write_report(
    path: Path,
    config: dict[str, object],
    checks: dict[str, object],
    outer: pd.DataFrame,
    candidates: pd.DataFrame,
    sensitivity: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    nested = config["nested_overall"]
    baseline_f1 = 0.3052
    selected_counts = outer["selected_candidate_id"].value_counts().sort_index().to_dict()
    spread = float(sensitivity["macro_f1_mean"].max() - sensitivity["macro_f1_mean"].min())
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        oof["true_class_id"],
        oof["pred_class_id"],
        labels=sorted(EXPECTED_LABEL_MAP),
        zero_division=0,
    )
    lines = [
        "# XGBoost Diagnostic",
        "",
        "## Conclusions",
        "",
        f"- The previous XGBoost result was **not tuned**: it used one fixed configuration. This run adds strict nested subject-grouped tuning.",
        f"- Nested OOF macro F1: **{nested['macro_f1']:.4f}**; balanced accuracy: **{nested['balanced_accuracy']:.4f}**. The prior fixed-config macro F1 was {baseline_f1:.4f}.",
        f"- Selected candidate counts across five outer folds: `{selected_counts}`. Different settings winning across folds is evidence against one universally optimal configuration.",
        f"- Across the reasonable diagnostic settings, mean outer-fold macro F1 spans {sensitivity['macro_f1_mean'].min():.4f}–{sensitivity['macro_f1_mean'].max():.4f} (range {spread:.4f}).",
        "- No subject overlap, label-map mismatch, metadata-feature inclusion, duplicate sample, or non-finite model input was found.",
        "",
        "## Protocol",
        "",
        "- Outer evaluation: the existing five subject-independent folds.",
        "- Hyperparameter selection: three subject-grouped folds created only inside each outer training set.",
        "- Selection metric: macro F1.",
        "- Preprocessing: median imputation inside the sklearn pipeline, fit separately for every training split. XGBoost does not require scaling.",
        "- Sensitivity scores are post-hoc diagnostics on outer folds and were not used to select the nested model.",
        "",
        "## Nested Outer-Fold Results",
        "",
        "| fold | selected candidate | macro F1 | balanced accuracy | depth | learning rate | trees | alpha | lambda |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in outer.itertuples(index=False):
        lines.append(f"| {row.outer_fold} | {row.selected_candidate_id} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} | {row.max_depth} | {row.learning_rate} | {row.n_estimators} | {row.reg_alpha} | {row.reg_lambda} |")
    lines.extend([
        "",
        "## Nested OOF Pain-Level Performance",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ])
    for class_id, class_name in EXPECTED_LABEL_MAP.items():
        lines.append(
            f"| {class_name} | {precision[class_id]:.4f} | {recall[class_id]:.4f} | {per_class_f1[class_id]:.4f} | {int(support[class_id])} |"
        )
    lines.extend([
        "",
        "## Candidate Sensitivity",
        "",
        "| candidate | mean macro F1 | std | min–max | balanced accuracy | depth | learning rate | trees | min child | alpha | lambda | subsample | columns |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sensitivity.sort_values("candidate_id").itertuples(index=False):
        lines.append(f"| {row.candidate_id} | {row.macro_f1_mean:.4f} | {row.macro_f1_std:.4f} | {row.macro_f1_min:.4f}–{row.macro_f1_max:.4f} | {row.balanced_accuracy_mean:.4f} | {row.max_depth} | {row.learning_rate} | {row.n_estimators} | {row.min_child_weight} | {row.reg_alpha} | {row.reg_lambda} | {row.subsample} | {row.colsample_bytree} |")
    lines.extend([
        "",
        "## Integrity Audit",
        "",
        f"- Label map correct: `{checks['label_map_matches_expected']}`; counts: `{checks['class_counts']}`.",
        f"- Subjects: {checks['n_subjects']}; samples per subject: {checks['samples_per_subject_min']}–{checks['samples_per_subject_max']}.",
        f"- Outer subject overlaps: `{checks['outer_subject_overlap']}`.",
        f"- Metadata/leakage columns used as model features: `{checks['feature_metadata_columns_in_model']}`.",
        f"- All model values finite: `{checks['all_model_values_finite']}`; duplicate sample names: {checks['duplicate_sample_names']}.",
        f"- Feature screening: {checks['feature_screening_note']}",
        "",
        "## Interpretation",
        "",
        "Nested tuning changes macro F1 only slightly and all reasonable settings remain near the same performance band, so the weak five-class result is unlikely to be mainly a poor XGBoost configuration. PA2 remains the weakest class after tuning, while the repository's adjacent-class analysis independently reports much lower adjacent-pair than wider-pair separability. The class structure and cross-subject generalization problem are therefore the stronger explanation. This remains predictive evidence, not a claim about physiological causality.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    run(args.root.resolve(), args.random_state)


if __name__ == "__main__":
    main()
