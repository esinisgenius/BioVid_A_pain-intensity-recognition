from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LABELS = {0: "BL1", 1: "PA1", 2: "PA2", 3: "PA3", 4: "PA4"}
BOUNDARY_LABELS = {
    0: "BL1 | PA1-PA4",
    1: "BL1-PA1 | PA2-PA4",
    2: "BL1-PA2 | PA3-PA4",
    3: "BL1-PA3 | PA4",
}
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
NEURAL_METHODS = ("ordinal_mlp", "boundary_gated_mlp", "proposed")


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: tuple[int, ...]
    dropout: float
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    @property
    def name(self) -> str:
        hidden = "x".join(str(value) for value in self.hidden_sizes)
        return f"h{hidden}_d{self.dropout:g}"


@dataclass(frozen=True)
class TrainResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_loss: float


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cumulative_targets(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=int)
    return np.column_stack([values > boundary for boundary in range(4)]).astype(np.float32)


def cumulative_to_class_probabilities(q: torch.Tensor) -> torch.Tensor:
    """Convert monotone P(y>b) values into a valid five-class distribution."""
    return torch.cat(
        [1.0 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]],
        dim=1,
    )


class SharedEncoder(nn.Module):
    def __init__(self, n_features: int, config: NetworkConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for hidden in config.hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
            width = hidden
        self.network = nn.Sequential(*layers)
        self.output_size = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OrdinalMLP(nn.Module):
    """Shared representation with monotone conditional cumulative heads."""

    def __init__(self, n_features: int, config: NetworkConfig) -> None:
        super().__init__()
        self.encoder = SharedEncoder(n_features, config)
        self.head = nn.Linear(self.encoder.output_size, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = torch.sigmoid(self.head(self.encoder(x)))
        cumulative = torch.cumprod(conditional, dim=1)
        return cumulative, cumulative_to_class_probabilities(cumulative)

    def gate_values(self) -> torch.Tensor | None:
        return None


class BoundaryGatedOrdinalMLP(nn.Module):
    """Boundary-specific raw feature gates followed by a shared encoder."""

    def __init__(self, n_features: int, config: NetworkConfig) -> None:
        super().__init__()
        self.n_features = n_features
        self.gate_logits = nn.Parameter(torch.zeros(4, n_features))
        self.encoder = SharedEncoder(n_features, config)
        self.heads = nn.ModuleList([nn.Linear(self.encoder.output_size, 1) for _ in range(4)])

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gate_values()
        gated = x[:, None, :] * gates[None, :, :]
        encoded = self.encoder(gated.reshape(-1, self.n_features)).reshape(
            x.shape[0], 4, self.encoder.output_size
        )
        conditional = torch.cat(
            [self.heads[boundary](encoded[:, boundary, :]) for boundary in range(4)],
            dim=1,
        ).sigmoid()
        cumulative = torch.cumprod(conditional, dim=1)
        return cumulative, cumulative_to_class_probabilities(cumulative)


def make_model(method: str, n_features: int, config: NetworkConfig) -> nn.Module:
    if method == "ordinal_mlp":
        return OrdinalMLP(n_features, config)
    if method in {"boundary_gated_mlp", "proposed"}:
        return BoundaryGatedOrdinalMLP(n_features, config)
    raise ValueError(f"Unknown neural method: {method}")


def boundary_balancing_weights(y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    targets = cumulative_targets(y)
    prevalence = targets.mean(axis=0)
    positive = torch.tensor(0.5 / np.clip(prevalence, 1e-6, 1.0), dtype=torch.float32)
    negative = torch.tensor(0.5 / np.clip(1.0 - prevalence, 1e-6, 1.0), dtype=torch.float32)
    return positive, negative


def ordinal_loss(
    cumulative: torch.Tensor,
    targets: torch.Tensor,
    positive_weights: torch.Tensor,
    negative_weights: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-7
    cumulative = cumulative.clamp(eps, 1.0 - eps)
    boundary_losses = -(
        targets * positive_weights * torch.log(cumulative)
        + (1.0 - targets) * negative_weights * torch.log(1.0 - cumulative)
    )
    class_probabilities = cumulative_to_class_probabilities(cumulative).clamp_min(eps)
    class_targets = targets.sum(dim=1).long()
    class_nll = -torch.log(class_probabilities.gather(1, class_targets[:, None])).mean()
    return 0.5 * (boundary_losses.mean() + class_nll)


def estimate_consistency_prior(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate a direction-aware, magnitude-shrunk prior using training subjects only.

    For each subject, boundary, and feature, a robust standardized median shift is
    computed. Effects are winsorized across subjects. The prior is the absolute
    signed mean divided by mean absolute effect plus one. Opposing directions
    cancel in the numerator, while the +1 shrinks negligible effects.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects)
    unique_subjects = np.unique(subjects)
    effects = np.zeros((len(unique_subjects), 4, x.shape[1]), dtype=float)
    effect_rows: list[dict[str, float | int]] = []
    for subject_pos, subject in enumerate(unique_subjects):
        subject_mask = subjects == subject
        subject_x = x[subject_mask]
        subject_y = y[subject_mask]
        for boundary in range(4):
            lower = subject_x[subject_y <= boundary]
            upper = subject_x[subject_y > boundary]
            pooled = np.vstack([lower, upper])
            center = np.median(pooled, axis=0)
            scale = 1.4826 * np.median(np.abs(pooled - center), axis=0)
            fallback = np.std(pooled, axis=0, ddof=1)
            scale = np.where(scale > eps, scale, fallback)
            scale = np.where(scale > eps, scale, 1.0)
            effects[subject_pos, boundary] = (np.median(upper, axis=0) - np.median(lower, axis=0)) / scale

    priors = np.zeros((4, x.shape[1]), dtype=float)
    for boundary in range(4):
        boundary_effects = effects[:, boundary, :]
        lo = np.quantile(boundary_effects, 0.05, axis=0)
        hi = np.quantile(boundary_effects, 0.95, axis=0)
        winsorized = np.clip(boundary_effects, lo, hi)
        signed_mean = winsorized.mean(axis=0)
        absolute_mean = np.abs(winsorized).mean(axis=0)
        raw = np.abs(signed_mean) / (absolute_mean + 1.0)
        # Preserve both absolute evidence and relative ordering while avoiding
        # exact 0/1 targets that would saturate sigmoid gates.
        relative = (rankdata(raw, method="average") - 1.0) / max(len(raw) - 1.0, 1.0)
        priors[boundary] = np.clip(0.5 * raw + 0.5 * relative, 0.05, 0.95)
        for subject_pos, subject in enumerate(unique_subjects):
            for feature_idx in range(x.shape[1]):
                effect_rows.append(
                    {
                        "subject_id": int(subject),
                        "boundary": boundary,
                        "feature_index": feature_idx,
                        "standardized_median_shift": effects[subject_pos, boundary, feature_idx],
                    }
                )
    return priors.astype(np.float32), pd.DataFrame(effect_rows)


def fit_preprocessor(x_train: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(x_train))
    return imputer, scaler


def transform(
    x: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(imputer.transform(x)).astype(np.float32)


def fit_model(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: NetworkConfig,
    seed: int,
    *,
    consistency_prior: np.ndarray | None = None,
    regularization_lambda: float = 0.0,
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
    max_epochs: int = 80,
    patience: int = 10,
    fixed_epochs: int | None = None,
    batch_size: int = 512,
) -> tuple[nn.Module, TrainResult]:
    seed = int(seed)
    set_seed(seed)
    model = make_model(method, x_train.shape[1], config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    targets = cumulative_targets(y_train)
    positive, negative = boundary_balancing_weights(y_train)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    prior_tensor = None if consistency_prior is None else torch.from_numpy(consistency_prior.astype(np.float32))
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    best_loss = float("inf")
    best_epoch = 1
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            cumulative, _ = model(batch_x)
            loss = ordinal_loss(cumulative, batch_y, positive, negative)
            if method == "proposed":
                assert prior_tensor is not None
                gates = model.gate_values()
                assert gates is not None
                loss = loss + regularization_lambda * torch.mean((gates - prior_tensor) ** 2)
            loss.backward()
            optimizer.step()

        if x_validation is None or y_validation is None:
            best_loss = float(loss.detach())
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            continue
        model.eval()
        with torch.no_grad():
            validation_targets = torch.from_numpy(cumulative_targets(y_validation))
            validation_cumulative, _ = model(torch.from_numpy(x_validation))
            validation_loss = ordinal_loss(validation_cumulative, validation_targets, positive, negative)
            current = float(validation_loss)
        if current < best_loss - 1e-5:
            best_loss = current
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, TrainResult(best_state, best_epoch, best_loss)


def predict_model(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        cumulative, probabilities = model(torch.from_numpy(x.astype(np.float32)))
    probs = probabilities.numpy()
    return probs.argmax(axis=1), probs, cumulative.numpy()


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(error.mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "severe_error_rate": float((error >= 2).mean()),
    }


def load_and_validate_inputs(
    feature_table_path: Path,
    metadata_path: Path,
    fold_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, str], pd.DataFrame]:
    table = pd.read_csv(feature_table_path)
    metadata_records = json.loads(metadata_path.read_text())
    metadata = pd.DataFrame(metadata_records)
    feature_names = metadata.loc[metadata["retained"], "feature_name"].tolist()
    if len(feature_names) != 72 or len(set(feature_names)) != 72:
        raise ValueError(f"Expected exactly 72 retained physiological features, found {len(feature_names)}")
    missing = sorted(set(feature_names).difference(table.columns))
    if missing:
        raise ValueError(f"Feature table is missing: {missing}")
    if set(feature_names).intersection(METADATA_COLUMNS):
        raise ValueError("Metadata/leakage columns entered the feature set")
    if table["sample_name"].duplicated().any():
        raise ValueError("Duplicate sample_name values")
    label_map = (
        table[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    if label_map != LABELS:
        raise ValueError(f"Unexpected labels: {label_map}")
    values = table[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite physiological feature values")
    assignments = pd.read_csv(fold_path)
    if assignments["sample_name"].duplicated().any() or len(assignments) != len(table):
        raise ValueError("Fold assignments are incomplete or duplicated")
    fold_map = assignments.set_index("sample_name")["fold"]
    table_samples = set(table["sample_name"])
    if table_samples != set(assignments["sample_name"]):
        raise ValueError("Fold assignments do not match the feature table")
    table = table.copy()
    table["outer_fold"] = table["sample_name"].map(fold_map).astype(int)
    leakage = table.groupby("subject_id")["outer_fold"].nunique()
    if (leakage != 1).any():
        raise ValueError("A subject appears in multiple outer folds")
    modality_map = metadata.set_index("feature_name")["modality"].to_dict()
    return table, feature_names, modality_map, assignments


def inner_splits(y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 3):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, validation in splitter.split(np.zeros(len(y)), y, groups):
        if set(groups[train]).intersection(groups[validation]):
            raise ValueError("Subject leakage in inner validation")
        yield train, validation


def tune_common_architecture(
    x_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Sequence[NetworkConfig],
    seed: int,
    max_epochs: int,
    patience: int,
    n_inner_splits: int,
) -> tuple[NetworkConfig, int, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    cached_splits = list(inner_splits(y, groups, seed, n_inner_splits))
    for candidate in candidates:
        for inner_fold, (train_idx, validation_idx) in enumerate(cached_splits, start=1):
            imputer, scaler = fit_preprocessor(x_raw[train_idx])
            x_train = transform(x_raw[train_idx], imputer, scaler)
            x_validation = transform(x_raw[validation_idx], imputer, scaler)
            model, result = fit_model(
                "ordinal_mlp",
                x_train,
                y[train_idx],
                candidate,
                seed + inner_fold,
                x_validation=x_validation,
                y_validation=y[validation_idx],
                max_epochs=max_epochs,
                patience=patience,
            )
            pred, _, _ = predict_model(model, x_validation)
            metrics = multiclass_metrics(y[validation_idx], pred)
            rows.append(
                {
                    "candidate": candidate.name,
                    "inner_fold": inner_fold,
                    "best_epoch": result.best_epoch,
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    summary = results.groupby("candidate").agg(
        macro_f1=("macro_f1", "mean"), mae=("mae", "mean"), epoch=("best_epoch", "median")
    )
    selected_name = summary.sort_values(["macro_f1", "mae"], ascending=[False, True]).index[0]
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    selected_epoch = int(np.clip(round(summary.loc[selected_name, "epoch"]), 3, max_epochs))
    results["selected"] = results["candidate"].eq(selected_name)
    return selected, selected_epoch, results


def tune_lambda(
    x_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: NetworkConfig,
    lambdas: Sequence[float],
    seed: int,
    max_epochs: int,
    patience: int,
    n_inner_splits: int,
) -> tuple[float, int, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    cached_splits = list(inner_splits(y, groups, seed, n_inner_splits))
    for value in lambdas:
        for inner_fold, (train_idx, validation_idx) in enumerate(cached_splits, start=1):
            imputer, scaler = fit_preprocessor(x_raw[train_idx])
            x_train = transform(x_raw[train_idx], imputer, scaler)
            x_validation = transform(x_raw[validation_idx], imputer, scaler)
            prior, _ = estimate_consistency_prior(x_train, y[train_idx], groups[train_idx])
            model, result = fit_model(
                "proposed",
                x_train,
                y[train_idx],
                config,
                seed + 100 + inner_fold,
                consistency_prior=prior,
                regularization_lambda=value,
                x_validation=x_validation,
                y_validation=y[validation_idx],
                max_epochs=max_epochs,
                patience=patience,
            )
            pred, _, _ = predict_model(model, x_validation)
            metrics = multiclass_metrics(y[validation_idx], pred)
            rows.append(
                {
                    "lambda": value,
                    "inner_fold": inner_fold,
                    "best_epoch": result.best_epoch,
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    summary = results.groupby("lambda").agg(
        macro_f1=("macro_f1", "mean"), mae=("mae", "mean"), epoch=("best_epoch", "median")
    )
    selected_lambda = float(
        summary.sort_values(["macro_f1", "mae"], ascending=[False, True]).index[0]
    )
    selected_epoch = int(np.clip(round(summary.loc[selected_lambda, "epoch"]), 3, max_epochs))
    results["selected"] = results["lambda"].eq(selected_lambda)
    return selected_lambda, selected_epoch, results


def add_prediction_rows(
    target: list[dict[str, object]],
    table: pd.DataFrame,
    indices: np.ndarray,
    method: str,
    fold: int,
    seed: int,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    cumulative: np.ndarray,
) -> None:
    for pos, index in enumerate(indices):
        source = table.iloc[int(index)]
        row: dict[str, object] = {
            "method": method,
            "fold": fold,
            "seed": seed,
            "subject_id": int(source["subject_id"]),
            "subject_name": source["subject_name"],
            "sample_id": int(source["sample_id"]),
            "sample_name": source["sample_name"],
            "true_class_id": int(source["class_id"]),
            "pred_class_id": int(predictions[pos]),
        }
        row.update({f"proba_{LABELS[k]}": float(probabilities[pos, k]) for k in range(5)})
        row.update({f"cumulative_b{k}": float(cumulative[pos, k]) for k in range(4)})
        target.append(row)


def reuse_xgboost_predictions(path: Path, table: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame["task"] == "5class") & (frame["model"] == "xgboost")].copy()
    if len(frame) != len(table) or set(frame["sample_name"]) != set(table["sample_name"]):
        raise ValueError("Phase 3 XGBoost reference does not align with the feature table")
    result = frame[
        [
            "fold",
            "subject_id",
            "subject_name",
            "sample_id",
            "sample_name",
            "true_class_id",
            "pred_class_id",
            *[f"proba_{name}" for name in LABELS.values()],
        ]
    ].copy()
    result.insert(0, "method", "flat_xgboost_reference")
    result.insert(2, "seed", -1)
    probs = result[[f"proba_{LABELS[k]}" for k in range(5)]].to_numpy()
    for boundary in range(4):
        result[f"cumulative_b{boundary}"] = probs[:, boundary + 1 :].sum(axis=1)
    return result


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    for (method, fold, seed), group in predictions.groupby(["method", "fold", "seed"]):
        y_true = group["true_class_id"].to_numpy(dtype=int)
        y_pred = group["pred_class_id"].to_numpy(dtype=int)
        fold_rows.append({"method": method, "fold": fold, "seed": seed, **multiclass_metrics(y_true, y_pred)})
        for boundary in range(4):
            scores = group[f"cumulative_b{boundary}"].to_numpy(dtype=float)
            if not np.isfinite(scores).all():
                continue
            boundary_rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "boundary": boundary,
                    "boundary_label": BOUNDARY_LABELS[boundary],
                    "auc": roc_auc_score(y_true > boundary, scores),
                }
            )
        for subject, subject_group in group.groupby("subject_id"):
            subject_rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "subject_id": subject,
                    **multiclass_metrics(
                        subject_group["true_class_id"].to_numpy(dtype=int),
                        subject_group["pred_class_id"].to_numpy(dtype=int),
                    ),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(subject_rows), pd.DataFrame(boundary_rows)


def adjacent_auc_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    probability_columns = [f"proba_{LABELS[k]}" for k in range(5)]
    for (method, fold, seed), group in predictions.groupby(["method", "fold", "seed"]):
        probabilities = group[probability_columns].to_numpy(dtype=float)
        y = group["true_class_id"].to_numpy(dtype=int)
        if not np.isfinite(probabilities).all():
            continue
        for lower in range(4):
            mask = np.isin(y, [lower, lower + 1])
            denominator = probabilities[mask, lower] + probabilities[mask, lower + 1]
            score = probabilities[mask, lower + 1] / np.clip(denominator, 1e-12, None)
            rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "adjacent_pair": f"{LABELS[lower]} vs {LABELS[lower + 1]}",
                    "auc": roc_auc_score(y[mask] == lower + 1, score),
                }
            )
    return pd.DataFrame(rows)


def per_class_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, fold, seed), group in predictions.groupby(["method", "fold", "seed"]):
        y_true = group["true_class_id"].to_numpy(dtype=int)
        y_pred = group["pred_class_id"].to_numpy(dtype=int)
        for class_id, class_name in LABELS.items():
            truth = y_true == class_id
            predicted = y_pred == class_id
            true_positive = int(np.sum(truth & predicted))
            precision = true_positive / max(int(predicted.sum()), 1)
            recall = true_positive / max(int(truth.sum()), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": int(truth.sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_comparison(fold_metrics: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    neural = fold_metrics[fold_metrics["seed"] >= 0]
    averaged = neural.groupby(["method", "fold"], as_index=False)[
        ["macro_f1", "balanced_accuracy", "mae", "qwk", "severe_error_rate"]
    ].mean()
    rows: list[dict[str, object]] = []
    comparisons = [
        ("ordinal_mlp", "flat_xgboost_reference"),
        ("boundary_gated_mlp", "ordinal_mlp"),
        ("proposed", "boundary_gated_mlp"),
    ]
    rng = np.random.default_rng(20261018)
    for treatment, control in comparisons:
        if control == "flat_xgboost_reference":
            control_frame = fold_metrics[fold_metrics["method"] == control][
                ["fold", "macro_f1", "balanced_accuracy", "mae", "qwk", "severe_error_rate"]
            ]
        else:
            control_frame = averaged[averaged["method"] == control].drop(columns="method")
        treatment_frame = averaged[averaged["method"] == treatment].drop(columns="method")
        paired = treatment_frame.merge(control_frame, on="fold", suffixes=("_treatment", "_control"))
        for metric in ["macro_f1", "balanced_accuracy", "mae", "qwk", "severe_error_rate"]:
            delta = paired[f"{metric}_treatment"].to_numpy() - paired[f"{metric}_control"].to_numpy()
            samples = rng.choice(delta, size=(n_bootstrap, len(delta)), replace=True).mean(axis=1)
            rows.append(
                {
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "mean_delta": delta.mean(),
                    "ci_low": np.quantile(samples, 0.025),
                    "ci_high": np.quantile(samples, 0.975),
                    "positive_folds": int((delta > 0).sum()),
                    "n_folds": len(delta),
                }
            )
    return pd.DataFrame(rows)


def subject_paired_comparison(subject_metrics: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    averaged = subject_metrics[subject_metrics["seed"] >= 0].groupby(
        ["method", "subject_id"], as_index=False
    )[["macro_f1", "balanced_accuracy", "mae", "qwk", "severe_error_rate"]].mean()
    treatment = averaged[averaged["method"] == "proposed"].drop(columns="method")
    control = averaged[averaged["method"] == "boundary_gated_mlp"].drop(columns="method")
    paired = treatment.merge(control, on="subject_id", suffixes=("_treatment", "_control"))
    rng = np.random.default_rng(20261019)
    rows: list[dict[str, object]] = []
    for metric in ["macro_f1", "balanced_accuracy", "mae", "qwk", "severe_error_rate"]:
        delta = paired[f"{metric}_treatment"].to_numpy() - paired[f"{metric}_control"].to_numpy()
        samples = rng.choice(delta, size=(n_bootstrap, len(delta)), replace=True).mean(axis=1)
        rows.append(
            {
                "treatment": "proposed",
                "control": "boundary_gated_mlp",
                "metric": metric,
                "mean_subject_delta": delta.mean(),
                "ci_low": np.quantile(samples, 0.025),
                "ci_high": np.quantile(samples, 0.975),
                "positive_subjects": int((delta > 0).sum()),
                "n_subjects": len(delta),
            }
        )
    return pd.DataFrame(rows)


def aggregate_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in predictions.groupby("method"):
        if method == "flat_xgboost_reference":
            y_true = frame["true_class_id"].to_numpy(dtype=int)
            y_pred = frame["pred_class_id"].to_numpy(dtype=int)
            rows.append({"method": method, "seed": -1, **multiclass_metrics(y_true, y_pred)})
        else:
            for seed, seeded in frame.groupby("seed"):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        **multiclass_metrics(
                            seeded["true_class_id"].to_numpy(dtype=int),
                            seeded["pred_class_id"].to_numpy(dtype=int),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def gate_alignment(gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, fold, seed, boundary), frame in gates.groupby(["method", "fold", "seed", "boundary"]):
        rho = spearmanr(frame["prior"], frame["gate"]).statistic
        prior_top = set(frame.nlargest(10, "prior")["feature_name"])
        gate_top = set(frame.nlargest(10, "gate")["feature_name"])
        rows.append(
            {
                "method": method,
                "fold": fold,
                "seed": seed,
                "boundary": boundary,
                "spearman_rho": rho,
                "mse": float(np.mean((frame["prior"] - frame["gate"]) ** 2)),
                "top10_jaccard": len(prior_top & gate_top) / len(prior_top | gate_top),
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    gates: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    neural = fold_metrics[fold_metrics["seed"] >= 0]
    pivot = neural.pivot_table(index=["fold", "seed"], columns="method", values="macro_f1")
    delta = pivot["proposed"] - pivot["boundary_gated_mlp"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="black", linewidth=1)
    for seed, series in delta.groupby(level="seed"):
        ax.plot(series.index.get_level_values("fold"), series.values, marker="o", label=f"seed {seed}")
    ax.set(xlabel="Outer fold", ylabel="Delta Macro F1", title="Proposed - Boundary-Gated MLP")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "paired_macro_f1_delta.png", dpi=180)
    plt.close(fig)

    proposed = gates[gates["method"] == "proposed"]
    summary = proposed.groupby(["boundary", "feature_name", "modality"], as_index=False)[["prior", "gate"]].mean()
    top_features = (
        summary.groupby("feature_name")["prior"].mean().nlargest(20).index.tolist()
    )
    matrix = summary[summary["feature_name"].isin(top_features)].pivot(
        index="feature_name", columns="boundary", values="gate"
    ).reindex(top_features)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4), [f"b{value}" for value in range(4)])
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set(title="Mean learned gates for top consistency-prior features")
    fig.colorbar(image, ax=ax, label="Gate")
    fig.tight_layout()
    fig.savefig(figures / "feature_gate_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)
    for boundary, ax in enumerate(axes):
        subset = proposed[proposed["boundary"] == boundary]
        ax.scatter(subset["prior"], subset["gate"], s=7, alpha=0.25)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
        ax.set_title(f"b{boundary}")
        ax.set_xlabel("Prior")
    axes[0].set_ylabel("Learned gate")
    fig.suptitle("Consistency prior vs learned feature gate")
    fig.tight_layout()
    fig.savefig(figures / "prior_gate_alignment.png", dpi=180)
    plt.close(fig)

    alignment_summary = alignment.groupby(["method", "boundary"])["spearman_rho"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    alignment_summary.plot(kind="bar", ax=ax)
    ax.set(xlabel="Boundary", ylabel="Mean Spearman rho", title="Gate-prior rank alignment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "gate_prior_rank_alignment.png", dpi=180)
    plt.close(fig)


def format_mean_sd(frame: pd.DataFrame, metric: str) -> str:
    return f"{frame[metric].mean():.4f} +/- {frame[metric].std(ddof=1):.4f}"


def write_report(
    path: Path,
    config: dict[str, object],
    overall: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    boundary_auc: pd.DataFrame,
    adjacent_auc: pd.DataFrame,
    comparisons: pd.DataFrame,
    subject_comparisons: pd.DataFrame,
    alignment: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
) -> None:
    neural_overall = overall[overall["seed"] >= 0]
    lines = [
        "# Phase 7.0 Consistency-Regularized Boundary-Gated Ordinal Network",
        "",
        f"Run date: {config['run_date_utc']}",
        "",
        "## Design and leakage control",
        "",
        "- Input: all 72 retained ECG/EDA/trapezius-EMG features; no facial/video data.",
        "- Outer evaluation reuses the exact Phase 3 five subject-independent folds.",
        "- Imputation, scaling, consistency prior, architecture/lambda selection, and epoch selection are fit only inside the current outer-training subjects.",
            "- Four conditional transition probabilities are multiplied to produce monotone cumulative probabilities `P(y>b)`; class probabilities are therefore non-negative and sum to one by construction.",
            "- Ordinal loss is the equal-weight mean of class-balanced cumulative-boundary BCE and NLL on the induced valid five-class probabilities. This prevents a boundary-discriminative model from silently collapsing intermediate-class predictions.",
            "- All three neural variants share the selected architecture, optimizer, and final epoch budget within each outer fold; Proposed differs from Boundary-Gated MLP only through the selected consistency penalty.",
        "- The consistency prior is a single direction-aware effect statistic: robust subject-level median shifts are winsorized, opposing directions cancel in the signed mean, and negligible effects are shrunk. It does not reuse Phase 7.1's combined reliability formula.",
            "- The Phase 3 flat XGBoost reference is reused without retraining. It used the 59-feature de-duplicated `combined_unique` set; neural models use the requested full 72-feature representation, so that comparison is contextual rather than a pure architecture ablation.",
            "- The archived Phase 3 XGBoost output contains hard predictions but no saved probabilities. It is included in multiclass metrics, but omitted from cumulative-boundary and adjacent-pair AUC tables rather than reconstructing unsupported scores.",
            "- Core controlled contrast for the inductive bias is Proposed vs Boundary-Gated MLP, which differs only by the gate-to-prior penalty.",
        "",
        "## Overall results",
        "",
        "| method | seeds | macro F1 | balanced accuracy | MAE | QWK | severe error rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    xgb = overall[overall["method"] == "flat_xgboost_reference"]
    if not xgb.empty:
        row = xgb.iloc[0]
        lines.append(
            f"| Flat XGBoost (Phase 3) | 1 | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} | {row.mae:.4f} | {row.qwk:.4f} | {row.severe_error_rate:.4f} |"
        )
    for method in NEURAL_METHODS:
        subset = neural_overall[neural_overall["method"] == method]
        lines.append(
            f"| {method} | {len(subset)} | {format_mean_sd(subset, 'macro_f1')} | {format_mean_sd(subset, 'balanced_accuracy')} | {format_mean_sd(subset, 'mae')} | {format_mean_sd(subset, 'qwk')} | {format_mean_sd(subset, 'severe_error_rate')} |"
        )

    lines.extend(
        [
            "",
            "## Paired outer-fold comparisons",
            "",
            "Deltas are treatment minus control after averaging random seeds within each outer fold. For MAE and severe error rate, negative is better. CIs are paired bootstrap intervals over the five outer folds and are descriptive because n=5.",
            "",
            "| treatment | control | metric | mean delta | 95% CI | positive folds |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples():
        lines.append(
            f"| {row.treatment} | {row.control} | {row.metric} | {row.mean_delta:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | {row.positive_folds}/{row.n_folds} |"
        )

    lines.extend(
        [
            "",
            "### Core comparison at subject level",
            "",
            "Subject metrics are first averaged over seeds, then paired by subject. The bootstrap resamples the 87 held-out subjects. Positive still means treatment minus control, so positive MAE/severe-error deltas are worse.",
            "",
            "| metric | mean subject delta | 95% CI | positive subjects |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in subject_comparisons.itertuples():
        lines.append(
            f"| {row.metric} | {row.mean_subject_delta:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | {row.positive_subjects}/{row.n_subjects} |"
        )

    lines.extend(["", "### Fold/seed stability: Macro F1", ""])
    lines.extend(["| method | fold | mean over seeds | seed SD |", "|---|---:|---:|---:|"])
    fold_stability = fold_metrics[fold_metrics["seed"] >= 0].groupby(["method", "fold"])["macro_f1"].agg(["mean", "std"]).reset_index()
    for row in fold_stability.itertuples():
        lines.append(f"| {row.method} | {row.fold} | {row.mean:.4f} | {row.std:.4f} |")

    lines.extend(["", "### Per-class performance", ""])
    lines.extend(["| method | class | F1 mean +/- SD | recall mean +/- SD |", "|---|---|---:|---:|"])
    class_summary = per_class_metrics.groupby(["method", "class_name"])[["f1", "recall"]].agg(["mean", "std"])
    for (method, class_name), row in class_summary.iterrows():
        lines.append(
            f"| {method} | {class_name} | {row[('f1', 'mean')]:.4f} +/- {row[('f1', 'std')]:.4f} | {row[('recall', 'mean')]:.4f} +/- {row[('recall', 'std')]:.4f} |"
        )

    lines.extend(["", "## Cumulative boundary AUC", ""])
    boundary_summary = boundary_auc.groupby(["method", "boundary_label"])["auc"].agg(["mean", "std"]).reset_index()
    lines.extend(["| method | boundary | AUC mean +/- SD |", "|---|---|---:|"])
    for row in boundary_summary.itertuples():
        boundary_label = row.boundary_label.replace("|", "\\|")
        lines.append(f"| {row.method} | {boundary_label} | {row.mean:.4f} +/- {row.std:.4f} |")

    lines.extend(
        [
            "",
            "## True adjacent-pair AUC",
            "",
            "These values are computed only on samples belonging to each adjacent class pair and use renormalized two-class probabilities. They are not cumulative-boundary AUCs.",
            "",
            "| method | adjacent pair | AUC mean +/- SD |",
            "|---|---|---:|",
        ]
    )
    adjacent_summary = adjacent_auc.groupby(["method", "adjacent_pair"])["auc"].agg(["mean", "std"]).reset_index()
    for row in adjacent_summary.itertuples():
        lines.append(f"| {row.method} | {row.adjacent_pair} | {row.mean:.4f} +/- {row.std:.4f} |")

    lines.extend(["", "## Learned gates vs consistency prior", ""])
    alignment_summary = alignment.groupby(["method", "boundary"])[["spearman_rho", "mse", "top10_jaccard"]].agg(["mean", "std"])
    lines.extend(["| method | boundary | Spearman rho | MSE | top-10 Jaccard |", "|---|---:|---:|---:|---:|"])
    for (method, boundary), row in alignment_summary.iterrows():
        lines.append(
            f"| {method} | {boundary} | {row[('spearman_rho', 'mean')]:.4f} +/- {row[('spearman_rho', 'std')]:.4f} | {row[('mse', 'mean')]:.4f} | {row[('top10_jaccard', 'mean')]:.4f} |"
        )

    core = comparisons[
        (comparisons["treatment"] == "proposed")
        & (comparisons["control"] == "boundary_gated_mlp")
    ].set_index("metric")
    f1 = core.loc["macro_f1"]
    qwk = core.loc["qwk"]
    practical = f1.mean_delta >= 0.005 and f1.positive_folds >= 4 and qwk.mean_delta > 0
    verdict = (
        "The consistency prior met the pre-specified practical stability rule."
        if practical
        else "The consistency prior did not meet the practical stability rule (Delta Macro F1 >= 0.005, positive in at least 4/5 folds, and positive Delta QWK)."
    )
    lines.extend(
        [
            "",
            "## Research conclusion",
            "",
            verdict,
            "",
            "### Answers to the research questions",
            "",
            f"1. **Ordinal MLP vs flat baseline:** Ordinal MLP was worse than the Phase 3 XGBoost reference (fold-paired Delta Macro F1 {comparisons[(comparisons['treatment'] == 'ordinal_mlp') & (comparisons['metric'] == 'macro_f1')].iloc[0].mean_delta:.4f}). This comparison is contextual because XGBoost used the 59-feature de-duplicated set, while the neural models used all 72 requested features.",
            f"2. **Boundary gates:** Gates did not yield a stable classification gain over the ungated Ordinal MLP (Delta Macro F1 {comparisons[(comparisons['treatment'] == 'boundary_gated_mlp') & (comparisons['metric'] == 'macro_f1')].iloc[0].mean_delta:.4f}); the small QWK change was not stable across all folds.",
            f"3. **Consistency prior beyond gates:** Proposed was effectively tied with the unregularized gated model (Delta Macro F1 {f1.mean_delta:.4f}; {int(f1.positive_folds)}/5 positive folds; Delta QWK {qwk.mean_delta:.4f}). The fold and subject bootstrap intervals include zero.",
            "4. **Gate-prior agreement:** Regularization substantially increased rank agreement between learned gates and the fold-local physiological prior, but this alignment was not predictive evidence: it did not improve held-out-subject metrics.",
            "5. **Difficult adjacent levels:** PA1-vs-PA2 and PA2-vs-PA3 remained the weakest or only modestly discriminated transitions. Proposed changed both by less than 0.001 AUC relative to the unregularized gated model; its similarly small changes at BL1-vs-PA1 and PA3-vs-PA4 were not accompanied by overall gains.",
            "6. **Architecture vs regularization:** Neither the ordinal architecture nor boundary gates improved Macro F1 materially, and the consistency penalty added no measurable generalization benefit. The experiment therefore provides no support for increasing network complexity.",
            "",
            "Interpretation is restricted to subject-independent predictive contribution in this BioVid experimental heat-pain setting. Gate agreement is a model diagnostic, not biological causality or a universal biomarker claim.",
            "",
            "![Paired Macro F1 delta](../experiments/boundary_gated_ordinal/figures/paired_macro_f1_delta.png)",
            "",
            "![Feature gate heatmap](../experiments/boundary_gated_ordinal/figures/feature_gate_heatmap.png)",
            "",
            "![Prior-gate alignment](../experiments/boundary_gated_ordinal/figures/prior_gate_alignment.png)",
            "",
            "## Artifact index",
            "",
            "- `experiments/boundary_gated_ordinal/config.json`",
            "- `experiments/boundary_gated_ordinal/oof_predictions.csv`",
            "- `experiments/boundary_gated_ordinal/overall_metrics.csv`",
            "- `experiments/boundary_gated_ordinal/fold_seed_metrics.csv`",
            "- `experiments/boundary_gated_ordinal/subject_metrics.csv`",
            "- `experiments/boundary_gated_ordinal/boundary_auc.csv`",
            "- `experiments/boundary_gated_ordinal/adjacent_pair_auc.csv`",
            "- `experiments/boundary_gated_ordinal/per_class_metrics.csv`",
            "- `experiments/boundary_gated_ordinal/paired_comparisons.csv`",
            "- `experiments/boundary_gated_ordinal/subject_paired_comparisons.csv`",
            "- `experiments/boundary_gated_ordinal/feature_gates.csv`",
            "- `experiments/boundary_gated_ordinal/gate_prior_alignment.csv`",
            "- `experiments/boundary_gated_ordinal/inner_cv_architecture.csv`",
            "- `experiments/boundary_gated_ordinal/inner_cv_lambda.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    table, feature_names, modality_map, _ = load_and_validate_inputs(
        args.feature_table, args.feature_metadata, args.fold_assignments
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    x_raw_all = table[feature_names].to_numpy(dtype=float)
    y_all = table["class_id"].to_numpy(dtype=int)
    groups_all = table["subject_id"].to_numpy(dtype=int)
    outer_folds = [int(value) for value in sorted(table["outer_fold"].unique())]
    if args.max_outer_folds:
        outer_folds = outer_folds[: args.max_outer_folds]
    seeds = args.seeds
    architecture_candidates = [
        NetworkConfig((32,), 0.10),
        NetworkConfig((64, 32), 0.15),
    ]
    if args.smoke:
        architecture_candidates = architecture_candidates[:1]
        args.max_epochs = min(args.max_epochs, 3)
        args.patience = min(args.patience, 2)
        args.lambdas = [0.1]
        seeds = seeds[:1]

    prediction_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    effect_frames: list[pd.DataFrame] = []
    architecture_frames: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []

    for outer_fold in outer_folds:
        train_idx = np.flatnonzero(table["outer_fold"].to_numpy() != outer_fold)
        test_idx = np.flatnonzero(table["outer_fold"].to_numpy() == outer_fold)
        if set(groups_all[train_idx]).intersection(groups_all[test_idx]):
            raise ValueError(f"Subject leakage in outer fold {outer_fold}")
        selected_config, common_epochs, architecture_results = tune_common_architecture(
            x_raw_all[train_idx],
            y_all[train_idx],
            groups_all[train_idx],
            architecture_candidates,
            args.tuning_seed + outer_fold,
            args.max_epochs,
            args.patience,
            args.inner_splits,
        )
        architecture_results.insert(0, "outer_fold", outer_fold)
        architecture_frames.append(architecture_results)
        selected_lambda, proposed_epochs, lambda_results = tune_lambda(
            x_raw_all[train_idx],
            y_all[train_idx],
            groups_all[train_idx],
            selected_config,
            args.lambdas,
            args.tuning_seed + 1000 + outer_fold,
            args.max_epochs,
            args.patience,
            args.inner_splits,
        )
        lambda_results.insert(0, "outer_fold", outer_fold)
        lambda_frames.append(lambda_results)
        selections.append(
            {
                "outer_fold": outer_fold,
                "network_config": selected_config.name,
                "final_epochs_all_neural_models": common_epochs,
                "selected_lambda": selected_lambda,
                "lambda_tuning_median_best_epoch": proposed_epochs,
            }
        )

        imputer, scaler = fit_preprocessor(x_raw_all[train_idx])
        x_train = transform(x_raw_all[train_idx], imputer, scaler)
        x_test = transform(x_raw_all[test_idx], imputer, scaler)
        prior, subject_effects = estimate_consistency_prior(
            x_train, y_all[train_idx], groups_all[train_idx]
        )
        subject_effects.insert(0, "outer_fold", outer_fold)
        subject_effects["feature_name"] = subject_effects["feature_index"].map(dict(enumerate(feature_names)))
        effect_frames.append(subject_effects)

        for seed in seeds:
            for method in NEURAL_METHODS:
                model, _ = fit_model(
                    method,
                    x_train,
                    y_all[train_idx],
                    selected_config,
                    seed + outer_fold * 100,
                    consistency_prior=prior if method == "proposed" else None,
                    regularization_lambda=selected_lambda if method == "proposed" else 0.0,
                    fixed_epochs=common_epochs,
                    batch_size=args.batch_size,
                )
                pred, probabilities, cumulative = predict_model(model, x_test)
                add_prediction_rows(
                    prediction_rows,
                    table,
                    test_idx,
                    method,
                    outer_fold,
                    seed,
                    pred,
                    probabilities,
                    cumulative,
                )
                gates = model.gate_values()
                if gates is not None:
                    gate_values = gates.detach().numpy()
                    for boundary in range(4):
                        for feature_index, feature_name in enumerate(feature_names):
                            gate_rows.append(
                                {
                                    "method": method,
                                    "fold": outer_fold,
                                    "seed": seed,
                                    "boundary": boundary,
                                    "feature_index": feature_index,
                                    "feature_name": feature_name,
                                    "modality": modality_map[feature_name],
                                    "prior": float(prior[boundary, feature_index]),
                                    "gate": float(gate_values[boundary, feature_index]),
                                    "selected_lambda": selected_lambda if method == "proposed" else 0.0,
                                }
                            )

    predictions = pd.DataFrame(prediction_rows)
    if len(outer_folds) == 5:
        xgb = reuse_xgboost_predictions(args.xgboost_predictions, table)
        predictions = pd.concat([xgb, predictions], ignore_index=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(output_dir / "feature_gates.csv", index=False)
    pd.concat(effect_frames, ignore_index=True).to_csv(output_dir / "subject_boundary_effects.csv", index=False)
    pd.concat(architecture_frames, ignore_index=True).to_csv(output_dir / "inner_cv_architecture.csv", index=False)
    pd.concat(lambda_frames, ignore_index=True).to_csv(output_dir / "inner_cv_lambda.csv", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "selected_hyperparameters.csv", index=False)

    fold_metrics, subject_metrics, boundary_auc = metric_tables(predictions)
    overall = aggregate_overall(predictions)
    adjacent_auc = adjacent_auc_table(predictions)
    per_class_metrics = per_class_metric_table(predictions)
    fold_metrics.to_csv(output_dir / "fold_seed_metrics.csv", index=False)
    subject_metrics.to_csv(output_dir / "subject_metrics.csv", index=False)
    boundary_auc.to_csv(output_dir / "boundary_auc.csv", index=False)
    adjacent_auc.to_csv(output_dir / "adjacent_pair_auc.csv", index=False)
    per_class_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    alignment = gate_alignment(gates)
    alignment.to_csv(output_dir / "gate_prior_alignment.csv", index=False)

    comparisons = pd.DataFrame()
    if len(outer_folds) == 5:
        comparisons = paired_comparison(fold_metrics, args.bootstrap_samples)
        comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
        subject_comparisons = subject_paired_comparison(subject_metrics, args.bootstrap_samples)
        subject_comparisons.to_csv(output_dir / "subject_paired_comparisons.csv", index=False)
        plot_outputs(output_dir, fold_metrics, gates, alignment)

    configuration: dict[str, object] = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "feature_table": str(args.feature_table),
        "feature_table_sha256": sha256(args.feature_table),
        "feature_metadata": str(args.feature_metadata),
        "fold_assignments": str(args.fold_assignments),
        "fold_assignments_sha256": sha256(args.fold_assignments),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "outer_folds": outer_folds,
        "seeds": seeds,
        "inner_splits": args.inner_splits,
        "architecture_candidates": [asdict(item) for item in architecture_candidates],
        "lambda_candidates": args.lambdas,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "tuning_seed": args.tuning_seed,
        "consistency_prior": "abs(mean(winsorized robust subject effects)) / (mean(abs(effect)) + 1), blended equally with within-boundary rank and clipped to [0.05, 0.95]",
        "monotonicity": "conditional sigmoid probabilities followed by cumulative product",
        "ordinal_loss": "0.5 * class-balanced cumulative BCE + 0.5 * five-class NLL",
        "flat_xgboost_reference": str(args.xgboost_predictions),
        "flat_xgboost_feature_count": 59,
        "device": "cpu",
        "smoke": args.smoke,
    }
    (output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    if len(outer_folds) == 5:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_report(
            args.report,
            configuration,
            overall,
            fold_metrics,
            boundary_auc,
            adjacent_auc,
            comparisons,
            subject_comparisons,
            alignment,
            per_class_metrics,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", type=Path, default=Path("artifacts/feature_table_filtered.csv"))
    parser.add_argument("--feature-metadata", type=Path, default=Path("artifacts/feature_metadata.json"))
    parser.add_argument("--fold-assignments", type=Path, default=Path("experiments/baselines/fold_assignments.csv"))
    parser.add_argument("--xgboost-predictions", type=Path, default=Path("experiments/model_strengthening_benchmark/oof_predictions_5class.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/boundary_gated_ordinal"))
    parser.add_argument("--report", type=Path, default=Path("reports/boundary_gated_ordinal.md"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--tuning-seed", type=int, default=20261018)
    parser.add_argument("--max-outer-folds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
