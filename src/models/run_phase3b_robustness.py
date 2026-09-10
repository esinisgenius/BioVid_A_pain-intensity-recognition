from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.evaluation.metrics import classification_metrics, fold_summary
from src.models.baselines import make_baseline_model
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs


PHASE3_MODELS = ["logistic_regression", "random_forest"]
PHASE3B_MODELS = ["logistic_regression", "random_forest", "rbf_svc"]
FEATURE_SET = "combined_unique"


def load_phase3_folds(path: Path) -> dict[int, np.ndarray]:
    assignments = pd.read_csv(path)
    return {
        int(fold): values["sample_name"].to_numpy()
        for fold, values in assignments.groupby("fold", sort=True)
    }


def score_vector(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(model.classes_)
        pos = classes.index(1)
        return proba[:, pos]
    scores = model.decision_function(x)
    return np.asarray(scores, dtype=float)


def make_fold_indices(table: pd.DataFrame, fold_samples: dict[int, np.ndarray]) -> list[dict[str, object]]:
    sample_to_index = pd.Series(np.arange(len(table)), index=table["sample_name"]).to_dict()
    folds = []
    all_indices = set(range(len(table)))
    for fold, samples in sorted(fold_samples.items()):
        test_index = np.asarray([sample_to_index[name] for name in samples if name in sample_to_index], dtype=int)
        train_index = np.asarray(sorted(all_indices.difference(test_index.tolist())), dtype=int)
        train_subjects = set(table.iloc[train_index]["subject_id"].astype(int))
        test_subjects = set(table.iloc[test_index]["subject_id"].astype(int))
        overlap = train_subjects.intersection(test_subjects)
        if overlap:
            raise ValueError(f"Subject overlap in reused fold {fold}: {sorted(overlap)}")
        folds.append(
            {
                "fold": int(fold),
                "train_index": train_index,
                "test_index": test_index,
                "train_subjects": sorted(train_subjects),
                "test_subjects": sorted(test_subjects),
            }
        )
    return folds


def run_multiclass_svc(table: pd.DataFrame, columns: list[str], folds: list[dict[str, object]], random_state: int):
    y = table["class_id"].to_numpy(dtype=int)
    x = table[columns].to_numpy(dtype=float)
    labels = sorted(EXPECTED_LABEL_MAP)
    oof_rows = []
    fold_metrics = []
    for split in folds:
        model = make_baseline_model("rbf_svc", random_state)
        model.fit(x[split["train_index"]], y[split["train_index"]])
        pred = model.predict(x[split["test_index"]])
        fold_result = classification_metrics(y[split["test_index"]], pred, labels, EXPECTED_LABEL_MAP)
        fold_metrics.append(
            {
                "fold": split["fold"],
                "n_train": int(len(split["train_index"])),
                "n_test": int(len(split["test_index"])),
                "n_train_subjects": len(split["train_subjects"]),
                "n_test_subjects": len(split["test_subjects"]),
                "test_subjects": split["test_subjects"],
                "macro_f1": fold_result["macro_f1"],
                "balanced_accuracy": fold_result["balanced_accuracy"],
                "per_class": fold_result["per_class"],
                "confusion_matrix": fold_result["confusion_matrix"],
            }
        )
        for pos, idx in enumerate(split["test_index"]):
            row = table.iloc[int(idx)]
            oof_rows.append(
                {
                    "task": "5class",
                    "model": "rbf_svc",
                    "feature_set": FEATURE_SET,
                    "fold": split["fold"],
                    "subject_id": int(row["subject_id"]),
                    "subject_name": row["subject_name"],
                    "sample_id": int(row["sample_id"]),
                    "sample_name": row["sample_name"],
                    "true_class_id": int(row["class_id"]),
                    "true_class_name": row["class_name"],
                    "pred_class_id": int(pred[pos]),
                    "pred_class_name": EXPECTED_LABEL_MAP[int(pred[pos])],
                }
            )
    oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
    overall = classification_metrics(
        oof["true_class_id"].to_numpy(dtype=int),
        oof["pred_class_id"].to_numpy(dtype=int),
        labels,
        EXPECTED_LABEL_MAP,
    )
    return {
        "task": "5class",
        "model": "rbf_svc",
        "feature_set": FEATURE_SET,
        "n_features": len(columns),
        "overall": overall,
        "fold_metrics": fold_metrics,
        "fold_summary": {
            "macro_f1": fold_summary(fold_metrics, "macro_f1"),
            "balanced_accuracy": fold_summary(fold_metrics, "balanced_accuracy"),
        },
    }, oof


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }


def run_binary(table: pd.DataFrame, columns: list[str], folds: list[dict[str, object]], random_state: int):
    binary = table[table["class_id"].isin([0, 4])].copy().reset_index(drop=True)
    binary["binary_label"] = (binary["class_id"] == 4).astype(int)
    fold_samples = {
        split["fold"]: set(table.iloc[split["test_index"]]["sample_name"])
        for split in folds
    }
    y = binary["binary_label"].to_numpy(dtype=int)
    x = binary[columns].to_numpy(dtype=float)
    results = []
    oof_frames = []
    for model_name in PHASE3B_MODELS:
        oof_rows = []
        fold_metrics = []
        for fold in sorted(fold_samples):
            test_index = binary.index[binary["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
            train_index = binary.index[~binary["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
            train_subjects = set(binary.iloc[train_index]["subject_id"].astype(int))
            test_subjects = set(binary.iloc[test_index]["subject_id"].astype(int))
            overlap = train_subjects.intersection(test_subjects)
            if overlap:
                raise ValueError(f"Binary subject overlap in fold {fold}: {sorted(overlap)}")
            model = make_baseline_model(model_name, random_state)
            model.fit(x[train_index], y[train_index])
            pred = model.predict(x[test_index])
            scores = score_vector(model, x[test_index])
            fold_result = binary_metrics(y[test_index], pred, scores)
            fold_metrics.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "n_train_subjects": len(train_subjects),
                    "n_test_subjects": len(test_subjects),
                    "test_subjects": sorted(test_subjects),
                    **fold_result,
                }
            )
            for pos, idx in enumerate(test_index):
                row = binary.iloc[int(idx)]
                oof_rows.append(
                    {
                        "task": "binary_bl1_vs_pa4",
                        "model": model_name,
                        "feature_set": FEATURE_SET,
                        "fold": int(fold),
                        "subject_id": int(row["subject_id"]),
                        "subject_name": row["subject_name"],
                        "sample_id": int(row["sample_id"]),
                        "sample_name": row["sample_name"],
                        "true_binary_label": int(row["binary_label"]),
                        "true_class_id": int(row["class_id"]),
                        "true_class_name": row["class_name"],
                        "pred_binary_label": int(pred[pos]),
                        "pred_class_id": 4 if int(pred[pos]) == 1 else 0,
                        "pred_class_name": "PA4" if int(pred[pos]) == 1 else "BL1",
                        "score_pa4": float(scores[pos]),
                    }
                )
        oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
        overall = binary_metrics(
            oof["true_binary_label"].to_numpy(dtype=int),
            oof["pred_binary_label"].to_numpy(dtype=int),
            oof["score_pa4"].to_numpy(dtype=float),
        )
        results.append(
            {
                "task": "binary_bl1_vs_pa4",
                "model": model_name,
                "feature_set": FEATURE_SET,
                "n_features": len(columns),
                "positive_class": "PA4",
                "negative_class": "BL1",
                "overall": overall,
                "fold_metrics": fold_metrics,
                "fold_summary": {
                    metric: fold_summary(fold_metrics, metric)
                    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]
                },
            }
        )
        oof_frames.append(oof)
    return results, pd.concat(oof_frames, ignore_index=True)


def subject_accuracy(oof: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "5class":
        correct = oof["true_class_id"].eq(oof["pred_class_id"])
    else:
        correct = oof["true_binary_label"].eq(oof["pred_binary_label"])
    out = oof.assign(correct=correct).groupby(
        ["task", "model", "feature_set", "fold", "subject_id", "subject_name"], as_index=False
    ).agg(n=("correct", "size"), accuracy=("correct", "mean"))
    return out


def add_phase3_combined_results(root: Path) -> list[dict[str, object]]:
    phase3 = json.loads((root / "experiments" / "baselines" / "metrics.json").read_text())
    return [
        {
            "task": "5class",
            "model": row["model"],
            "feature_set": FEATURE_SET,
            "n_features": row["n_features"],
            "overall": row["overall"],
            "fold_metrics": row["fold_metrics"],
            "fold_summary": row["fold_summary"],
        }
        for row in phase3["results"]
        if row["model"] in PHASE3_MODELS and row["feature_set"] == FEATURE_SET
    ]


def phase3_combined_subject_accuracy(root: Path) -> pd.DataFrame:
    oof = pd.read_csv(root / "experiments" / "baselines" / "out_of_fold_predictions.csv")
    oof = oof[(oof["feature_set"] == FEATURE_SET) & (oof["model"].isin(PHASE3_MODELS))].copy()
    oof.insert(0, "task", "5class")
    return subject_accuracy(oof, "5class")


def write_long_outputs(output_dir: Path, results: list[dict[str, object]]) -> None:
    summary_rows = []
    per_class_rows = []
    fold_rows = []
    conf_rows = []
    for row in results:
        overall = row["overall"]
        base = {
            "task": row["task"],
            "model": row["model"],
            "feature_set": row["feature_set"],
            "n_features": row["n_features"],
        }
        if row["task"] == "5class":
            summary_rows.append({**base, "macro_f1": overall["macro_f1"], "balanced_accuracy": overall["balanced_accuracy"]})
            for cls in overall["per_class"]:
                per_class_rows.append({**base, **cls})
            labels = ["BL1", "PA1", "PA2", "PA3", "PA4"]
        else:
            summary_rows.append({**base, **{k: overall[k] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]}})
            labels = ["BL1", "PA4"]
        for split in row["fold_metrics"]:
            fold_rows.append({**base, **{k: v for k, v in split.items() if k not in ["per_class", "confusion_matrix"]}})
        matrix = np.asarray(overall["confusion_matrix"], dtype=int)
        for i, true in enumerate(labels):
            for j, pred in enumerate(labels):
                conf_rows.append({**base, "true_class": true, "pred_class": pred, "count": int(matrix[i, j])})
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "per_class_metrics_5class.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(output_dir / "confusion_matrices_long.csv", index=False)


def write_report(results: list[dict[str, object]], output_path: Path) -> None:
    five = [r for r in results if r["task"] == "5class"]
    binary = [r for r in results if r["task"] == "binary_bl1_vs_pa4"]
    lines = [
        "# Phase 3B Model Robustness and Binary Benchmark",
        "",
        f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "Phase 3 results are reused unchanged. This add-on uses the existing subject-independent folds and the `combined_unique` feature set only.",
        "",
        "## 5-Class Robustness",
        "",
        "| model | macro F1 | fold F1 mean ± std | balanced accuracy | fold bal. acc. mean ± std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in five:
        f1s = row["fold_summary"]["macro_f1"]
        bas = row["fold_summary"]["balanced_accuracy"]
        lines.append(
            f"| {row['model']} | {row['overall']['macro_f1']:.4f} | {f1s['mean']:.4f} ± {f1s['std']:.4f} | {row['overall']['balanced_accuracy']:.4f} | {bas['mean']:.4f} ± {bas['std']:.4f} |"
        )
    lines.extend(["", "### 5-Class Fold-Level Results", "", "| model | fold | macro F1 | balanced accuracy | test subjects |", "|---|---:|---:|---:|---:|"])
    for row in five:
        for split in row["fold_metrics"]:
            lines.append(
                f"| {row['model']} | {split['fold']} | {split['macro_f1']:.4f} | {split['balanced_accuracy']:.4f} | {split['n_test_subjects']} |"
            )
    lines.extend(["", "### 5-Class Per-Class Metrics", ""])
    for row in five:
        lines.extend([f"#### {row['model']}", "", "| class | precision | recall | F1 |", "|---|---:|---:|---:|"])
        for cls in row["overall"]["per_class"]:
            lines.append(f"| {cls['class_name']} | {cls['precision']:.4f} | {cls['recall']:.4f} | {cls['f1']:.4f} |")
        lines.append("")
    lines.extend(["## 5-Class Confusion Matrices", "", "Rows=true, columns=predicted, class order `BL1 PA1 PA2 PA3 PA4`.", ""])
    for row in five:
        lines.extend([f"### {row['model']}", "", "```text"])
        for matrix_row in row["overall"]["confusion_matrix"]:
            lines.append(" ".join(f"{v:4d}" for v in matrix_row))
        lines.extend(["```", ""])
    lines.extend(["## Binary BL1 vs PA4 Benchmark", "", "| model | accuracy | balanced accuracy | precision | recall | F1 | AUROC | fold F1 mean ± std |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in binary:
        fs = row["fold_summary"]["f1"]
        o = row["overall"]
        lines.append(
            f"| {row['model']} | {o['accuracy']:.4f} | {o['balanced_accuracy']:.4f} | {o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} | {o['auroc']:.4f} | {fs['mean']:.4f} ± {fs['std']:.4f} |"
        )
    lines.extend(["", "### Binary Fold-Level Results", "", "| model | fold | accuracy | balanced accuracy | F1 | AUROC | test subjects |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in binary:
        for split in row["fold_metrics"]:
            lines.append(
                f"| {row['model']} | {split['fold']} | {split['accuracy']:.4f} | {split['balanced_accuracy']:.4f} | {split['f1']:.4f} | {split['auroc']:.4f} | {split['n_test_subjects']} |"
            )
    lines.extend(["", "Binary confusion matrices use rows=true, columns=predicted, class order `BL1 PA4`.", ""])
    for row in binary:
        lines.extend([f"### {row['model']}", "", "```text"])
        for matrix_row in row["overall"]["confusion_matrix"]:
            lines.append(" ".join(f"{v:4d}" for v in matrix_row))
        lines.extend(["```", ""])
    lines.extend([
        "## Interpretation",
        "",
        "- The three 5-class models are at a similar performance level; RBF-SVC does not remove the central grading difficulty.",
        "- PA1, PA2, and PA3 remain weaker than BL1 and PA4 across linear, tree, and kernel models.",
        "- Subject-level performance remains heterogeneous across models, so the cross-subject stability question is still central.",
        "- BL1 vs PA4 is substantially easier than the 5-class grading task, as expected for a secondary literature-style benchmark.",
        "- These results support moving into Phase 4 contribution analysis, with Logistic Regression as the primary analysis model and RF/SVC as robustness checks rather than optimization targets.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, random_state: int) -> None:
    output_dir = root / "experiments" / "phase3b_model_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / "artifacts" / "feature_table_filtered.csv")
    metadata = pd.read_json(root / "artifacts" / "feature_metadata.json")
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    columns = sets[FEATURE_SET]
    folds = make_fold_indices(table, load_phase3_folds(root / "experiments" / "baselines" / "fold_assignments.csv"))

    svc_result, svc_oof = run_multiclass_svc(table, columns, folds, random_state)
    binary_results, binary_oof = run_binary(table, columns, folds, random_state)
    five_results = add_phase3_combined_results(root) + [svc_result]
    results = five_results + binary_results

    config = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase3_dependency": "Reuses experiments/baselines/fold_assignments.csv and Phase 3 LR/RF combined_unique metrics unchanged.",
        "feature_set": FEATURE_SET,
        "n_features": len(columns),
        "models": PHASE3B_MODELS,
        "svm_config": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "decision_function_shape": "ovr"},
        "binary_task": {"negative_class": "BL1", "positive_class": "PA4"},
        "random_state": random_state,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"config": config, "results": results}, indent=2), encoding="utf-8")
    svc_oof.to_csv(output_dir / "oof_predictions_5class_rbf_svc.csv", index=False)
    binary_oof.to_csv(output_dir / "oof_predictions_binary_bl1_vs_pa4.csv", index=False)
    pd.concat(
        [phase3_combined_subject_accuracy(root), subject_accuracy(svc_oof, "5class"), subject_accuracy(binary_oof, "binary")],
        ignore_index=True,
    ).to_csv(output_dir / "subject_level_oof_accuracy.csv", index=False)
    write_long_outputs(output_dir, results)
    write_report(results, root / "reports" / "phase3b_model_robustness.md")
    print("Wrote Phase 3B outputs to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--random-state", type=int, default=20260909)
    args = parser.parse_args()
    run(args.root.resolve(), args.random_state)


if __name__ == "__main__":
    main()
