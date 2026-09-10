from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_samples import EXPECTED_LABEL_MAP, signal_path
from src.evaluation.metrics import classification_metrics, fold_summary
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import (
    FEATURE_SET,
    binary_metrics,
    load_phase3_folds,
    make_fold_indices,
    phase3_combined_subject_accuracy,
    score_vector,
    subject_accuracy,
)
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs


FEATURE_MODELS = ["logistic_regression", "random_forest", "rbf_svc", "xgboost"]
CHANNELS = ["ecg", "gsr", "emg_trapezius"]
CLASS_NAMES_5 = ["BL1", "PA1", "PA2", "PA3", "PA4"]
CLASS_NAMES_BINARY = ["BL1", "PA4"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_existing_feature_results(root: Path) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    phase3 = json.loads((root / "experiments" / "baselines" / "metrics.json").read_text())
    phase3b = json.loads((root / "experiments" / "phase3b_model_robustness" / "metrics.json").read_text())
    five_results = []
    binary_results = []
    for row in phase3["results"]:
        if row["feature_set"] == FEATURE_SET and row["model"] in ["logistic_regression", "random_forest"]:
            copied = {k: row[k] for k in ["model", "feature_set", "n_features", "overall", "fold_metrics", "fold_summary"]}
            copied["task"] = "5class"
            copied["input_type"] = "engineered_features"
            five_results.append(copied)
    for row in phase3b["results"]:
        if row["task"] == "5class" and row["model"] == "rbf_svc":
            copied = dict(row)
            copied["input_type"] = "engineered_features"
            five_results.append(copied)
        elif row["task"] == "binary_bl1_vs_pa4" and row["model"] in ["logistic_regression", "random_forest", "rbf_svc"]:
            copied = dict(row)
            copied["input_type"] = "engineered_features"
            binary_results.append(copied)
    oof = pd.read_csv(root / "experiments" / "baselines" / "out_of_fold_predictions.csv")
    oof = oof[(oof["feature_set"] == FEATURE_SET) & (oof["model"].isin(["logistic_regression", "random_forest"]))].copy()
    oof.insert(0, "task", "5class")
    oof.insert(1, "input_type", "engineered_features")
    svc = pd.read_csv(root / "experiments" / "phase3b_model_robustness" / "oof_predictions_5class_rbf_svc.csv")
    if "input_type" not in svc.columns:
        svc.insert(1, "input_type", "engineered_features")
    binary_oof = pd.read_csv(root / "experiments" / "phase3b_model_robustness" / "oof_predictions_binary_bl1_vs_pa4.csv")
    if "input_type" not in binary_oof.columns:
        binary_oof.insert(1, "input_type", "engineered_features")
    return five_results, pd.concat([oof, svc], ignore_index=True), binary_results, binary_oof


def run_feature_model(
    table: pd.DataFrame,
    columns: list[str],
    folds: list[dict[str, object]],
    model_name: str,
    random_state: int,
    task: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    if task == "5class":
        subset = table.copy().reset_index(drop=True)
        y = subset["class_id"].to_numpy(dtype=int)
        labels = sorted(EXPECTED_LABEL_MAP)
    else:
        subset = table[table["class_id"].isin([0, 4])].copy().reset_index(drop=True)
        subset["binary_label"] = (subset["class_id"] == 4).astype(int)
        y = subset["binary_label"].to_numpy(dtype=int)
        labels = [0, 1]

    x = subset[columns].to_numpy(dtype=float)
    fold_samples = {
        split["fold"]: set(table.iloc[split["test_index"]]["sample_name"])
        for split in folds
    }
    oof_rows = []
    fold_metrics = []
    for fold in sorted(fold_samples):
        test_index = subset.index[subset["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
        train_index = subset.index[~subset["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
        train_subjects = set(subset.iloc[train_index]["subject_id"].astype(int))
        test_subjects = set(subset.iloc[test_index]["subject_id"].astype(int))
        overlap = train_subjects.intersection(test_subjects)
        if overlap:
            raise ValueError(f"Subject overlap for {model_name}/{task}/fold {fold}: {sorted(overlap)}")
        model_key = "xgboost_binary" if task != "5class" and model_name == "xgboost" else model_name
        model = make_baseline_model(model_key, random_state)
        model.fit(x[train_index], y[train_index])
        pred = model.predict(x[test_index])
        if task == "5class":
            fold_result = classification_metrics(y[test_index], pred, labels, EXPECTED_LABEL_MAP)
            fold_metrics.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "n_train_subjects": len(train_subjects),
                    "n_test_subjects": len(test_subjects),
                    "test_subjects": sorted(test_subjects),
                    "macro_f1": fold_result["macro_f1"],
                    "balanced_accuracy": fold_result["balanced_accuracy"],
                    "per_class": fold_result["per_class"],
                    "confusion_matrix": fold_result["confusion_matrix"],
                }
            )
        else:
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
            row = subset.iloc[int(idx)]
            base = {
                "task": task,
                "input_type": "engineered_features",
                "model": model_name,
                "feature_set": FEATURE_SET,
                "fold": int(fold),
                "subject_id": int(row["subject_id"]),
                "subject_name": row["subject_name"],
                "sample_id": int(row["sample_id"]),
                "sample_name": row["sample_name"],
                "true_class_id": int(row["class_id"]),
                "true_class_name": row["class_name"],
            }
            if task == "5class":
                base.update(
                    {
                        "pred_class_id": int(pred[pos]),
                        "pred_class_name": EXPECTED_LABEL_MAP[int(pred[pos])],
                    }
                )
            else:
                base.update(
                    {
                        "true_binary_label": int(row["binary_label"]),
                        "pred_binary_label": int(pred[pos]),
                        "pred_class_id": 4 if int(pred[pos]) == 1 else 0,
                        "pred_class_name": "PA4" if int(pred[pos]) == 1 else "BL1",
                        "score_pa4": float(scores[pos]),
                    }
                )
            oof_rows.append(base)

    oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
    if task == "5class":
        overall = classification_metrics(
            oof["true_class_id"].to_numpy(dtype=int),
            oof["pred_class_id"].to_numpy(dtype=int),
            labels,
            EXPECTED_LABEL_MAP,
        )
        metric_summary = {
            "macro_f1": fold_summary(fold_metrics, "macro_f1"),
            "balanced_accuracy": fold_summary(fold_metrics, "balanced_accuracy"),
        }
    else:
        overall = binary_metrics(
            oof["true_binary_label"].to_numpy(dtype=int),
            oof["pred_binary_label"].to_numpy(dtype=int),
            oof["score_pa4"].to_numpy(dtype=float),
        )
        metric_summary = {
            metric: fold_summary(fold_metrics, metric)
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]
        }
    return {
        "task": task,
        "input_type": "engineered_features",
        "model": model_name,
        "feature_set": FEATURE_SET,
        "n_features": len(columns),
        "overall": overall,
        "fold_metrics": fold_metrics,
        "fold_summary": metric_summary,
    }, oof


def signal_cache(root: Path, table: pd.DataFrame, output_dir: Path) -> Path:
    cache_path = output_dir / "filtered_signal_tensor_ecg_gsr_trapezius.npz"
    if cache_path.exists():
        return cache_path
    source_dir = root / "biosignals_filtered"
    x = np.empty((len(table), 2816, 3), dtype=np.float32)
    for idx, row in table.iterrows():
        path = signal_path(source_dir, row["subject_name"], row["sample_name"])
        signals = pd.read_csv(path, sep="\t", usecols=CHANNELS)
        x[int(idx)] = signals[CHANNELS].to_numpy(dtype=np.float32)
        if (idx + 1) % 1000 == 0:
            print(f"cached {idx + 1} signal windows", flush=True)
    np.savez_compressed(
        cache_path,
        x=x,
        sample_name=table["sample_name"].to_numpy(),
        class_id=table["class_id"].to_numpy(dtype=np.int64),
        subject_id=table["subject_id"].to_numpy(dtype=np.int64),
    )
    return cache_path


def subject_accuracy_full(oof: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "5class":
        correct = oof["true_class_id"].eq(oof["pred_class_id"])
    else:
        correct = oof["true_binary_label"].eq(oof["pred_binary_label"])
    working = oof.assign(correct=correct)
    if "input_type" not in working.columns:
        working["input_type"] = "engineered_features"
    return working.groupby(
        ["task", "input_type", "model", "feature_set", "fold", "subject_id", "subject_name"],
        as_index=False,
    ).agg(n=("correct", "size"), accuracy=("correct", "mean"))


class TinyCNN1D(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_by_training(x: np.ndarray, train_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_index].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x[train_index].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return ((x - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)).astype(np.float32), mean, std


def train_cnn_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    n_classes: int,
    config: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    x_norm, mean, std = normalize_by_training(x, train_index)
    train_x = torch.from_numpy(np.transpose(x_norm[train_index], (0, 2, 1)))
    train_y = torch.from_numpy(y[train_index].astype(np.int64))
    test_x = torch.from_numpy(np.transpose(x_norm[test_index], (0, 2, 1)))
    dataset = TensorDataset(train_x, train_y)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=generator)
    model = TinyCNN1D(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses))})
    model.eval()
    preds = []
    scores = []
    with torch.no_grad():
        for start in range(0, len(test_x), int(config["batch_size"])):
            logits = model(test_x[start : start + int(config["batch_size"])].to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            scores.append(prob)
            preds.append(prob.argmax(axis=1))
    return np.concatenate(preds), np.vstack(scores), np.vstack([mean, std]), history


def run_cnn_task(
    table: pd.DataFrame,
    signals: np.ndarray,
    folds: list[dict[str, object]],
    task: str,
    config: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
    if task == "5class":
        subset = table.copy().reset_index(drop=True)
        x = signals
        y = subset["class_id"].to_numpy(dtype=np.int64)
        n_classes = 5
    else:
        mask = table["class_id"].isin([0, 4]).to_numpy()
        subset = table[mask].copy().reset_index(drop=True)
        x = signals[mask]
        y = (subset["class_id"].to_numpy(dtype=np.int64) == 4).astype(np.int64)
        n_classes = 2

    fold_samples = {
        split["fold"]: set(table.iloc[split["test_index"]]["sample_name"])
        for split in folds
    }
    labels = list(range(n_classes))
    oof_rows = []
    fold_metrics = []
    training_records = []
    for fold in sorted(fold_samples):
        set_seed(int(config["seed"]) + int(fold))
        test_index = subset.index[subset["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
        train_index = subset.index[~subset["sample_name"].isin(fold_samples[fold])].to_numpy(dtype=int)
        train_subjects = set(subset.iloc[train_index]["subject_id"].astype(int))
        test_subjects = set(subset.iloc[test_index]["subject_id"].astype(int))
        overlap = train_subjects.intersection(test_subjects)
        if overlap:
            raise ValueError(f"CNN subject overlap for {task}/fold {fold}: {sorted(overlap)}")
        pred, proba, norm_stats, history = train_cnn_fold(x, y, train_index, test_index, n_classes, config, device)
        if task == "5class":
            fold_result = classification_metrics(y[test_index], pred, labels, EXPECTED_LABEL_MAP)
            fold_metrics.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "n_train_subjects": len(train_subjects),
                    "n_test_subjects": len(test_subjects),
                    "test_subjects": sorted(test_subjects),
                    "macro_f1": fold_result["macro_f1"],
                    "balanced_accuracy": fold_result["balanced_accuracy"],
                    "per_class": fold_result["per_class"],
                    "confusion_matrix": fold_result["confusion_matrix"],
                }
            )
        else:
            fold_result = binary_metrics(y[test_index], pred, proba[:, 1])
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
        training_records.append(
            {
                "task": task,
                "fold": int(fold),
                "channel_mean": norm_stats[0].tolist(),
                "channel_std": norm_stats[1].tolist(),
                "history": history,
            }
        )
        for pos, idx in enumerate(test_index):
            row = subset.iloc[int(idx)]
            out = {
                "task": task,
                "input_type": "filtered_raw_signal",
                "model": "tiny_1d_cnn",
                "feature_set": "filtered_ecg_gsr_trapezius_waveform",
                "fold": int(fold),
                "subject_id": int(row["subject_id"]),
                "subject_name": row["subject_name"],
                "sample_id": int(row["sample_id"]),
                "sample_name": row["sample_name"],
                "true_class_id": int(row["class_id"]),
                "true_class_name": row["class_name"],
            }
            if task == "5class":
                out.update({"pred_class_id": int(pred[pos]), "pred_class_name": EXPECTED_LABEL_MAP[int(pred[pos])]})
                for class_id in range(5):
                    out[f"proba_{EXPECTED_LABEL_MAP[class_id]}"] = float(proba[pos, class_id])
            else:
                out.update(
                    {
                        "true_binary_label": int(y[idx]),
                        "pred_binary_label": int(pred[pos]),
                        "pred_class_id": 4 if int(pred[pos]) == 1 else 0,
                        "pred_class_name": "PA4" if int(pred[pos]) == 1 else "BL1",
                        "score_pa4": float(proba[pos, 1]),
                    }
                )
            oof_rows.append(out)
    oof = pd.DataFrame(oof_rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
    if task == "5class":
        overall = classification_metrics(
            oof["true_class_id"].to_numpy(dtype=int),
            oof["pred_class_id"].to_numpy(dtype=int),
            labels,
            EXPECTED_LABEL_MAP,
        )
        metric_summary = {
            "macro_f1": fold_summary(fold_metrics, "macro_f1"),
            "balanced_accuracy": fold_summary(fold_metrics, "balanced_accuracy"),
        }
    else:
        overall = binary_metrics(
            oof["true_binary_label"].to_numpy(dtype=int),
            oof["pred_binary_label"].to_numpy(dtype=int),
            oof["score_pa4"].to_numpy(dtype=float),
        )
        metric_summary = {
            metric: fold_summary(fold_metrics, metric)
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]
        }
    result = {
        "task": task,
        "input_type": "filtered_raw_signal",
        "model": "tiny_1d_cnn",
        "feature_set": "filtered_ecg_gsr_trapezius_waveform",
        "n_features": None,
        "input_shape": [2816, 3],
        "overall": overall,
        "fold_metrics": fold_metrics,
        "fold_summary": metric_summary,
    }
    return result, oof, training_records


def write_tables(output_dir: Path, results: list[dict[str, object]]) -> None:
    summary_rows = []
    per_class_rows = []
    fold_rows = []
    conf_rows = []
    for row in results:
        base = {
            "task": row["task"],
            "input_type": row["input_type"],
            "model": row["model"],
            "feature_set": row["feature_set"],
            "n_features": row["n_features"],
        }
        overall = row["overall"]
        if row["task"] == "5class":
            summary_rows.append({**base, "macro_f1": overall["macro_f1"], "balanced_accuracy": overall["balanced_accuracy"]})
            for cls in overall["per_class"]:
                per_class_rows.append({**base, **cls})
            labels = CLASS_NAMES_5
        else:
            summary_rows.append({**base, **{k: overall[k] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]}})
            labels = CLASS_NAMES_BINARY
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


def write_report(results: list[dict[str, object]], subject_df: pd.DataFrame, output_path: Path) -> None:
    five = [r for r in results if r["task"] == "5class"]
    binary = [r for r in results if r["task"] == "binary_bl1_vs_pa4"]
    lines = [
        "# Model Strengthening Benchmark",
        "",
        f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "This benchmark strengthens the methods layer only. It reuses the existing subject-independent folds and does not run feature attribution, feature ranking, modality ablation, clustering, or tuning.",
        "",
        "## 5-Class Results",
        "",
        "| input | model | macro F1 | fold F1 mean ± std | balanced accuracy | fold bal. acc. mean ± std |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in five:
        f1s = row["fold_summary"]["macro_f1"]
        bas = row["fold_summary"]["balanced_accuracy"]
        lines.append(
            f"| {row['input_type']} | {row['model']} | {row['overall']['macro_f1']:.4f} | {f1s['mean']:.4f} ± {f1s['std']:.4f} | {row['overall']['balanced_accuracy']:.4f} | {bas['mean']:.4f} ± {bas['std']:.4f} |"
        )
    lines.extend(["", "## 5-Class Per-Class F1", "", "| input | model | BL1 | PA1 | PA2 | PA3 | PA4 |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in five:
        per = {cls["class_name"]: cls["f1"] for cls in row["overall"]["per_class"]}
        lines.append(
            f"| {row['input_type']} | {row['model']} | {per['BL1']:.4f} | {per['PA1']:.4f} | {per['PA2']:.4f} | {per['PA3']:.4f} | {per['PA4']:.4f} |"
        )
    lines.extend(["", "## 5-Class Per-Class Precision / Recall / F1", ""])
    for row in five:
        lines.extend(
            [
                f"### {row['input_type']} / {row['model']}",
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
    lines.extend(["", "## Binary BL1 vs PA4", "", "| input | model | accuracy | balanced accuracy | precision | recall | F1 | AUROC | fold F1 mean ± std |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in binary:
        fs = row["fold_summary"]["f1"]
        o = row["overall"]
        lines.append(
            f"| {row['input_type']} | {row['model']} | {o['accuracy']:.4f} | {o['balanced_accuracy']:.4f} | {o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} | {o['auroc']:.4f} | {fs['mean']:.4f} ± {fs['std']:.4f} |"
        )
    lines.extend(["", "## Subject-Level Heterogeneity", "", "| task | input | model | min acc. | mean acc. | max acc. | subject acc. std |", "|---|---|---|---:|---:|---:|---:|"])
    for (task, input_type, model), group in subject_df.groupby(["task", "input_type", "model"], sort=False):
        lines.append(
            f"| {task} | {input_type} | {model} | {group['accuracy'].min():.4f} | {group['accuracy'].mean():.4f} | {group['accuracy'].max():.4f} | {group['accuracy'].std(ddof=1):.4f} |"
        )
    lines.extend(["", "## Confusion Matrices", ""])
    for row in five:
        lines.extend([f"### 5-class / {row['input_type']} / {row['model']}", "", "```text"])
        for matrix_row in row["overall"]["confusion_matrix"]:
            lines.append(" ".join(f"{value:4d}" for value in matrix_row))
        lines.extend(["```", ""])
    for row in binary:
        lines.extend([f"### BL1 vs PA4 / {row['input_type']} / {row['model']}", "", "```text"])
        for matrix_row in row["overall"]["confusion_matrix"]:
            lines.append(" ".join(f"{value:4d}" for value in matrix_row))
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- XGBoost does not materially exceed the LR/RF/SVM feature-based baselines in 5-class subject-independent grading.",
            "- The tiny 1D-CNN does not clearly outperform engineered-feature models under this fixed, small architecture.",
            "- PA1, PA2, and PA3 remain difficult across feature-based and raw-signal models, so the difficulty is not only a handcrafted-feature limitation.",
            "- BL1 vs PA4 remains much easier than 5-class grading, confirming that endpoint discrimination is not equivalent to pain-intensity grading.",
            "- Subject-level heterogeneity persists across model families and input representations.",
            "",
            "## Recommended Model Roles",
            "",
            "- Main explanation model: Logistic Regression with `combined_unique`, because it is competitive and easiest to interpret without turning the paper into model optimization.",
            "- Boosting robustness: XGBoost, as a fixed stronger tree-boosting check rather than a tuned leaderboard model.",
            "- Kernel robustness: RBF-SVC.",
            "- Representation sensitivity baseline: tiny 1D-CNN on filtered ECG/GSR/trapezius waveforms.",
            "- Phase 4/5 should use LR as primary and compare contribution conclusions against RF, XGBoost, and SVC for cross-model consistency where computationally feasible.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, random_state: int) -> None:
    set_seed(random_state)
    output_dir = root / "experiments" / "model_strengthening_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / "artifacts" / "feature_table_filtered.csv")
    metadata = pd.read_json(root / "artifacts" / "feature_metadata.json")
    sets = feature_sets(metadata)
    validate_feature_inputs(table, metadata, sets)
    columns = sets[FEATURE_SET]
    folds = make_fold_indices(table, load_phase3_folds(root / "experiments" / "baselines" / "fold_assignments.csv"))

    feature_results, feature_oof_5, binary_results, binary_existing_oof = read_existing_feature_results(root)
    xgb_5, xgb_5_oof = run_feature_model(table, columns, folds, "xgboost", random_state, "5class")
    feature_results.append(xgb_5)
    xgb_binary, xgb_binary_oof = run_feature_model(table, columns, folds, "xgboost", random_state, "binary_bl1_vs_pa4")
    binary_results.append(xgb_binary)

    cache_path = signal_cache(root, table, output_dir)
    cache = np.load(cache_path)
    signals = cache["x"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn_config = {
        "seed": random_state,
        "epochs": 15,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "device": str(device),
        "input_source": "biosignals_filtered",
        "input_shape": [2816, 3],
        "channels": CHANNELS,
        "architecture": [
            "Conv1d(3,16,kernel=15,stride=2,padding=7) + BatchNorm + ReLU + MaxPool1d(4)",
            "Conv1d(16,32,kernel=9,stride=2,padding=4) + BatchNorm + ReLU + MaxPool1d(4)",
            "Conv1d(32,64,kernel=7,stride=2,padding=3) + BatchNorm + ReLU + AdaptiveAvgPool1d(1)",
            "Dropout(0.2) + Linear(64,n_classes)",
        ],
        "normalization": "Per fold, channel mean/std estimated only from training subjects and applied to train/test windows.",
    }
    cnn_5, cnn_5_oof, cnn_5_training = run_cnn_task(table, signals, folds, "5class", cnn_config, device)
    cnn_binary, cnn_binary_oof, cnn_binary_training = run_cnn_task(
        table, signals, folds, "binary_bl1_vs_pa4", cnn_config, device
    )
    all_results = feature_results + [cnn_5] + binary_results + [cnn_binary]
    config = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase3_folds": "experiments/baselines/fold_assignments.csv",
        "feature_set": FEATURE_SET,
        "feature_models": FEATURE_MODELS,
        "xgboost_config": {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
        "cnn_config": cnn_config,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "cnn_architecture.json").write_text(json.dumps(cnn_config, indent=2), encoding="utf-8")
    (output_dir / "cnn_training_history.json").write_text(
        json.dumps({"five_class": cnn_5_training, "binary_bl1_vs_pa4": cnn_binary_training}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps({"config": config, "results": all_results}, indent=2), encoding="utf-8")
    pd.concat([feature_oof_5, xgb_5_oof, cnn_5_oof], ignore_index=True).to_csv(output_dir / "oof_predictions_5class.csv", index=False)
    pd.concat([binary_existing_oof, xgb_binary_oof, cnn_binary_oof], ignore_index=True).to_csv(
        output_dir / "oof_predictions_binary_bl1_vs_pa4.csv", index=False
    )
    subject_df = pd.concat(
        [
            phase3_combined_subject_accuracy(root).assign(input_type="engineered_features"),
            subject_accuracy_full(pd.read_csv(root / "experiments" / "phase3b_model_robustness" / "oof_predictions_5class_rbf_svc.csv"), "5class"),
            subject_accuracy_full(xgb_5_oof, "5class"),
            subject_accuracy_full(cnn_5_oof, "5class"),
            subject_accuracy_full(binary_existing_oof, "binary"),
            subject_accuracy_full(xgb_binary_oof, "binary"),
            subject_accuracy_full(cnn_binary_oof, "binary"),
        ],
        ignore_index=True,
    )
    subject_df.to_csv(output_dir / "subject_level_oof_accuracy.csv", index=False)
    write_tables(output_dir, all_results)
    write_report(all_results, subject_df, root / "reports" / "model_strengthening_benchmark.md")
    print("Wrote model strengthening benchmark outputs to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--random-state", type=int, default=20260909)
    args = parser.parse_args()
    run(args.root.resolve(), args.random_state)


if __name__ == "__main__":
    main()
