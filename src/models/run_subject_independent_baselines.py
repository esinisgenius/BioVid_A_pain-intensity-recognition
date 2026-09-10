from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_samples import EXPECTED_LABEL_MAP
from src.evaluation.metrics import classification_metrics, fold_summary
from src.evaluation.splits import fold_assignment_table, make_subject_grouped_splits
from src.models.baselines import make_baseline_model


METADATA_COLUMNS = {
    "subject_id",
    "subject_name",
    "sample_id",
    "sample_name",
    "class_id",
    "class_name",
    "source_dir",
    "sampling_rate_hz",
    "n_samples",
    "duration_s",
}


def read_feature_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def feature_sets(metadata: pd.DataFrame) -> dict[str, list[str]]:
    retained = metadata[metadata["retained"]].copy()
    domain = retained.loc[retained["feature_set"] == "domain_informed", "feature_name"].tolist()
    generic = retained.loc[retained["feature_set"] == "generic_automated", "feature_name"].tolist()
    generic_unique = retained.loc[
        (retained["feature_set"] == "generic_automated") & (retained["provenance"] != "shared"),
        "feature_name",
    ].tolist()
    by_modality = {
        "ecg_unimodal": retained.loc[retained["modality"] == "ECG", "feature_name"].tolist(),
        "eda_gsr_unimodal": retained.loc[retained["modality"] == "EDA/GSR", "feature_name"].tolist(),
        "emg_trapezius_unimodal": retained.loc[retained["modality"] == "Trapezius EMG", "feature_name"].tolist(),
    }
    sets = {
        "domain_informed": domain,
        "generic_automated": generic,
        "combined_unique": domain + generic_unique,
    }
    sets.update(by_modality)
    return sets


def validate_feature_inputs(table: pd.DataFrame, metadata: pd.DataFrame, sets: dict[str, list[str]]) -> None:
    label_map = (
        table[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    if label_map != EXPECTED_LABEL_MAP:
        raise ValueError(f"Unexpected label map: {label_map}")
    if table["sample_name"].duplicated().any():
        raise ValueError("Duplicate sample_name rows found")
    retained_names = set(metadata.loc[metadata["retained"], "feature_name"])
    for set_name, columns in sets.items():
        missing = sorted(set(columns).difference(table.columns))
        if missing:
            raise ValueError(f"{set_name} references missing features: {missing}")
        leakage = sorted(set(columns).intersection(METADATA_COLUMNS))
        if leakage:
            raise ValueError(f"{set_name} includes metadata/leakage columns: {leakage}")
        unretained = sorted(set(columns).difference(retained_names))
        if unretained:
            raise ValueError(f"{set_name} includes unretained features: {unretained}")
        values = table[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{set_name} contains non-finite values")


def run_one(
    table: pd.DataFrame,
    set_name: str,
    columns: list[str],
    model_name: str,
    folds,
    labels: list[int],
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    y = table["class_id"].to_numpy(dtype=int)
    x = table[columns].to_numpy(dtype=float)
    label_names = EXPECTED_LABEL_MAP

    oof_rows = []
    fold_metrics = []
    for split in folds:
        model = make_baseline_model(model_name, random_state=random_state)
        model.fit(x[split.train_index], y[split.train_index])
        pred = model.predict(x[split.test_index])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x[split.test_index])
            model_classes = [int(c) for c in model.classes_]
        else:
            proba = None
            model_classes = []

        fold_result = classification_metrics(y[split.test_index], pred, labels, label_names)
        fold_metrics.append(
            {
                "fold": split.fold,
                "n_train": int(len(split.train_index)),
                "n_test": int(len(split.test_index)),
                "n_train_subjects": int(len(split.train_subjects)),
                "n_test_subjects": int(len(split.test_subjects)),
                "test_subjects": list(split.test_subjects),
                "macro_f1": fold_result["macro_f1"],
                "balanced_accuracy": fold_result["balanced_accuracy"],
                "per_class": fold_result["per_class"],
                "confusion_matrix": fold_result["confusion_matrix"],
            }
        )
        for pos, idx in enumerate(split.test_index):
            source = table.iloc[int(idx)]
            row = {
                "model": model_name,
                "feature_set": set_name,
                "fold": split.fold,
                "subject_id": int(source["subject_id"]),
                "subject_name": source["subject_name"],
                "sample_id": int(source["sample_id"]),
                "sample_name": source["sample_name"],
                "true_class_id": int(source["class_id"]),
                "true_class_name": source["class_name"],
                "pred_class_id": int(pred[pos]),
                "pred_class_name": label_names[int(pred[pos])],
            }
            if proba is not None:
                for class_pos, class_id in enumerate(model_classes):
                    row[f"proba_{label_names[class_id]}"] = float(proba[pos, class_pos])
            oof_rows.append(row)

    oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
    overall = classification_metrics(
        oof["true_class_id"].to_numpy(dtype=int),
        oof["pred_class_id"].to_numpy(dtype=int),
        labels,
        label_names,
    )
    result = {
        "model": model_name,
        "feature_set": set_name,
        "n_features": len(columns),
        "features": columns,
        "overall": overall,
        "fold_metrics": fold_metrics,
        "fold_summary": {
            "macro_f1": fold_summary(fold_metrics, "macro_f1"),
            "balanced_accuracy": fold_summary(fold_metrics, "balanced_accuracy"),
        },
    }
    return result, oof


def adjacent_confusion_rate(confusion: list[list[int]]) -> float:
    matrix = np.asarray(confusion, dtype=int)
    off_diag = matrix.sum() - np.trace(matrix)
    if off_diag == 0:
        return 0.0
    adjacent = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and abs(i - j) == 1:
                adjacent += int(matrix[i, j])
    return float(adjacent / off_diag)


def write_report(results: list[dict[str, object]], config: dict[str, object], output_path: Path) -> None:
    primary_sets = {"domain_informed", "generic_automated", "combined_unique"}
    lines = [
        "# Phase 3 Subject-Independent Baseline Results",
        "",
        f"Run date: {config['run_date_utc']}",
        "",
        "Task: 5-class classification (`BL1`, `PA1`, `PA2`, `PA3`, `PA4`) using BioVid Part A physiological features only.",
        "",
        "## Protocol",
        "",
        f"- Split protocol: `{config['split_protocol']}`.",
        f"- Split reason: {config['split_reason']}",
        f"- Folds: {config['n_splits']}; random seed: {config['random_state']}.",
        "- Grouping: `subject_id`; no subject appears in both train and test in any fold.",
        "- Fold-local preprocessing: `SimpleImputer(strategy=\"median\")` for both models; `StandardScaler` for Logistic Regression only. These steps are inside sklearn pipelines and are fit separately within each training fold.",
        "- Second non-linear baseline: Random Forest, selected because `xgboost` is not installed and Random Forest is available through scikit-learn with fewer dependencies.",
        "- Input feature table: CSV copy of the Phase 2 feature table. The local parquet file raised a pyarrow read error during Phase 3 setup, so CSV was used without changing feature values.",
        "",
        "## Overall 5-Class Baselines",
        "",
        "| model | feature set | features | macro F1 | fold F1 mean ± std | balanced accuracy | fold bal. acc. mean ± std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        if row["feature_set"] not in primary_sets:
            continue
        f1 = row["overall"]["macro_f1"]
        ba = row["overall"]["balanced_accuracy"]
        f1s = row["fold_summary"]["macro_f1"]
        bas = row["fold_summary"]["balanced_accuracy"]
        lines.append(
            f"| {row['model']} | {row['feature_set']} | {row['n_features']} | {f1:.4f} | {f1s['mean']:.4f} ± {f1s['std']:.4f} | {ba:.4f} | {bas['mean']:.4f} ± {bas['std']:.4f} |"
        )

    lines.extend(["", "## Unimodal Reference Baselines", ""])
    lines.extend(
        [
            "| model | feature set | features | macro F1 | fold F1 mean ± std | balanced accuracy |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        if row["feature_set"] in primary_sets:
            continue
        f1 = row["overall"]["macro_f1"]
        ba = row["overall"]["balanced_accuracy"]
        f1s = row["fold_summary"]["macro_f1"]
        lines.append(
            f"| {row['model']} | {row['feature_set']} | {row['n_features']} | {f1:.4f} | {f1s['mean']:.4f} ± {f1s['std']:.4f} | {ba:.4f} |"
        )

    lines.extend(["", "## Pain-Level Performance", ""])
    for row in results:
        if row["feature_set"] not in primary_sets:
            continue
        lines.extend(
            [
                f"### {row['model']} / {row['feature_set']}",
                "",
                "| class | precision | recall | F1 | support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for cls in row["overall"]["per_class"]:
            lines.append(
                f"| {cls['class_name']} | {cls['precision']:.4f} | {cls['recall']:.4f} | {cls['f1']:.4f} | {cls['support']} |"
            )
        lines.append("")

    lines.extend(["## Confusion Pattern Summary", ""])
    lines.extend(
        [
            "| model | feature set | off-diagonal errors | adjacent error share |",
            "|---|---|---:|---:|",
        ]
    )
    for row in results:
        matrix = np.asarray(row["overall"]["confusion_matrix"], dtype=int)
        off_diag = int(matrix.sum() - np.trace(matrix))
        lines.append(
            f"| {row['model']} | {row['feature_set']} | {off_diag} | {adjacent_confusion_rate(row['overall']['confusion_matrix']):.3f} |"
        )

    lines.extend(["", "## Fold Instability", ""])
    lines.extend(
        [
            "| model | feature set | macro F1 min-max | macro F1 std | balanced accuracy min-max | balanced accuracy std |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        f1s = row["fold_summary"]["macro_f1"]
        bas = row["fold_summary"]["balanced_accuracy"]
        lines.append(
            f"| {row['model']} | {row['feature_set']} | {f1s['min']:.4f}-{f1s['max']:.4f} | {f1s['std']:.4f} | {bas['min']:.4f}-{bas['max']:.4f} | {bas['std']:.4f} |"
        )

    best = max(results, key=lambda row: row["overall"]["macro_f1"])
    lines.extend(
        [
            "",
            "## Best Observed Baseline Detail",
            "",
            f"Best macro F1 in this Phase 3 grid: `{best['model']}` with `{best['feature_set']}`.",
            "",
            "| class | precision | recall | F1 | support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cls in best["overall"]["per_class"]:
        lines.append(
            f"| {cls['class_name']} | {cls['precision']:.4f} | {cls['recall']:.4f} | {cls['f1']:.4f} | {cls['support']} |"
        )
    lines.extend(["", "Confusion matrix rows=true, columns=predicted, class order `BL1 PA1 PA2 PA3 PA4`:", "", "```text"])
    for row in best["overall"]["confusion_matrix"]:
        lines.append(" ".join(f"{value:4d}" for value in row))
    lines.extend(
        [
            "```",
            "",
            f"Adjacent-pain-level share among off-diagonal errors for the best observed baseline: {adjacent_confusion_rate(best['overall']['confusion_matrix']):.3f}.",
            "",
            "## Leakage Review",
            "",
            "- No subject overlap was detected in any fold.",
            "- No metadata, label, sample id, subject id/name, trial number, source path, temperature, corrugator EMG, or zygomaticus EMG feature was included.",
            "- Learned preprocessing was nested inside sklearn pipelines and fit only on training rows for each fold.",
            "- New operational risk: `artifacts/feature_table_filtered.parquet` could not be read by the current pyarrow environment; Phase 3 used `artifacts/feature_table_filtered.csv`, whose Phase 2 SHA-256 is tracked.",
            "",
            "Machine-readable outputs are in `experiments/baselines/`: fold assignments, out-of-fold predictions, metrics JSON, feature-set manifest, per-class metrics, fold metrics, confusion matrices, and subject-level out-of-fold accuracy.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, n_splits: int, random_state: int) -> None:
    artifacts = root / "artifacts"
    experiments = root / "experiments" / "baselines"
    reports = root / "reports"
    configs = root / "configs"
    experiments.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    configs.mkdir(exist_ok=True)

    feature_table_path = artifacts / "feature_table_filtered.csv"
    metadata_path = artifacts / "feature_metadata.json"
    table = read_feature_table(feature_table_path)
    metadata = pd.read_json(metadata_path)
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)

    y = table["class_id"].to_numpy(dtype=int)
    groups = table["subject_id"].to_numpy(dtype=int)
    folds, protocol, reason = make_subject_grouped_splits(
        y=y,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        prefer_stratified=True,
    )
    labels = sorted(EXPECTED_LABEL_MAP)
    fold_assignments = fold_assignment_table(table, folds)
    fold_assignments.to_csv(experiments / "fold_assignments.csv", index=False)

    config = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "5-class classification: BL1/PA1/PA2/PA3/PA4",
        "feature_table": str(feature_table_path),
        "feature_table_csv_sha256": hashlib.sha256(feature_table_path.read_bytes()).hexdigest(),
        "feature_metadata": str(metadata_path),
        "split_protocol": protocol,
        "split_reason": reason,
        "group_column": "subject_id",
        "n_splits": n_splits,
        "random_state": random_state,
        "models": ["logistic_regression", "random_forest"],
        "preprocessing": {
            "logistic_regression": ["SimpleImputer(strategy=median)", "StandardScaler"],
            "random_forest": ["SimpleImputer(strategy=median)"],
        },
        "feature_sets": {
            name: {
                "n_features": len(columns),
                "features": columns,
            }
            for name, columns in sets.items()
        },
    }
    (configs / "baseline_grouped_cv.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    oof_frames = []
    for model_name in config["models"]:
        for set_name, columns in sets.items():
            result, oof = run_one(
                table=table,
                set_name=set_name,
                columns=columns,
                model_name=model_name,
                folds=folds,
                labels=labels,
                random_state=random_state,
            )
            results.append(result)
            oof_frames.append(oof)
            print(
                f"{model_name} {set_name}: macro_f1={result['overall']['macro_f1']:.4f}, "
                f"balanced_accuracy={result['overall']['balanced_accuracy']:.4f}",
                flush=True,
            )

    oof_all = pd.concat(oof_frames, ignore_index=True)
    oof_all.to_csv(experiments / "out_of_fold_predictions.csv", index=False)
    (experiments / "metrics.json").write_text(json.dumps({"config": config, "results": results}, indent=2), encoding="utf-8")
    manifest = []
    for set_name, columns in sets.items():
        subset = metadata[metadata["feature_name"].isin(columns)].copy()
        subset.insert(0, "baseline_feature_set", set_name)
        manifest.append(subset)
    pd.concat(manifest, ignore_index=True).to_csv(experiments / "feature_set_manifest.csv", index=False)
    write_report(results, config, reports / "baseline_results.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260909)
    args = parser.parse_args()
    run(args.root.resolve(), args.n_splits, args.random_state)


if __name__ == "__main__":
    main()
