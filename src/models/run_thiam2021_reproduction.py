from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from src.data.load_samples import EXPECTED_LABEL_MAP, load_samples
from src.data.thiam2021_signals import build_signal_cache, fit_standardization, load_signal_cache
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_prediction_reliability import (
    apply_temperature,
    conformal_quantile,
    ece_score,
    fit_temperature,
    split_model_and_calibration_subjects,
    standard_conformal_sets,
)
from src.models.thiam2021_ddcae import ThiamSupervisedDDCAE, joint_loss


CLASS_IDS = np.arange(5, dtype=int)
CLASS_NAMES = [EXPECTED_LABEL_MAP[int(value)] for value in CLASS_IDS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SignalDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        labels: np.ndarray,
        indices: np.ndarray,
        stats: dict[str, dict[str, float]],
    ) -> None:
        self.arrays = arrays
        self.labels = labels
        self.indices = np.asarray(indices, dtype=int)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        idx = int(self.indices[position])
        inputs = {
            modality: torch.from_numpy(
                ((np.asarray(array[idx], dtype=np.float32) - self.stats[modality]["mean"]) / self.stats[modality]["std"])
                .astype(np.float32, copy=False)[None, :]
            )
            for modality, array in self.arrays.items()
        }
        return inputs, torch.tensor(int(self.labels[idx]), dtype=torch.long), idx


def choose_device(preference: str) -> torch.device:
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=False,
    )


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: values.to(device) for name, values in inputs.items()}


def run_epoch(
    model: ThiamSupervisedDDCAE,
    batches: DataLoader,
    config: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    count = 0
    for inputs, targets, _ in batches:
        inputs = move_inputs(inputs, device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(inputs, corrupt=training)
            loss, pieces = joint_loss(model, outputs, inputs, targets, config)
            if training:
                loss.backward()
                optimizer.step()
        batch_n = len(targets)
        count += batch_n
        for name, value in pieces.items():
            sums[name] = sums.get(name, 0.0) + float(value.cpu()) * batch_n
    return {name: value / count for name, value in sums.items()}


@torch.no_grad()
def predict(
    model: ThiamSupervisedDDCAE,
    batches: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    indices = []
    for inputs, _, batch_indices in batches:
        outputs = model(move_inputs(inputs, device), corrupt=False)
        probabilities.append(torch.softmax(outputs["logits"], dim=1).cpu().numpy())
        indices.append(batch_indices.numpy())
    return np.concatenate(indices).astype(int), np.concatenate(probabilities)


def performance_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mae": float(np.mean(np.abs(y_true - prediction))),
        "qwk": float(cohen_kappa_score(y_true, prediction, weights="quadratic")),
        "per_class_recall": {
            CLASS_NAMES[i]: float(value)
            for i, value in enumerate(recall_score(y_true, prediction, labels=CLASS_IDS, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=CLASS_IDS).astype(int).tolist(),
    }


def adjacent_pair_auc(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {}
    for lower in range(4):
        upper = lower + 1
        mask = np.isin(y_true, [lower, upper])
        score = probabilities[mask, upper] / np.maximum(
            probabilities[mask, lower] + probabilities[mask, upper], 1e-12
        )
        result[f"{CLASS_NAMES[lower]}_vs_{CLASS_NAMES[upper]}"] = float(
            roc_auc_score((y_true[mask] == upper).astype(int), score)
        )
    return result


def bootstrap_subject_ci(
    predictions: pd.DataFrame, probabilities: np.ndarray, repetitions: int, seed: int
) -> dict[str, dict[str, float]]:
    subjects = np.asarray(sorted(predictions["subject_id"].unique()), dtype=int)
    positions = {
        subject: np.flatnonzero(predictions["subject_id"].to_numpy(dtype=int) == subject) for subject in subjects
    }
    y = predictions["true_class_id"].to_numpy(dtype=int)
    pred = probabilities.argmax(axis=1)

    def values(pos: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y[pos], pred[pos])),
            "macro_f1": float(f1_score(y[pos], pred[pos], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y[pos], pred[pos])),
            "mae": float(np.mean(np.abs(y[pos] - pred[pos]))),
            "qwk": float(cohen_kappa_score(y[pos], pred[pos], weights="quadratic")),
        }

    estimate = values(np.arange(len(predictions)))
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in estimate}
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        pos = np.concatenate([positions[int(subject)] for subject in sampled])
        for name, value in values(pos).items():
            draws[name].append(value)
    return {
        name: {
            "estimate": estimate[name],
            "ci_low": float(np.quantile(draws[name], 0.025)),
            "ci_high": float(np.quantile(draws[name], 0.975)),
        }
        for name in estimate
    }


def reliability_metrics(
    calibration_y: np.ndarray,
    calibration_probabilities: np.ndarray,
    test_y: np.ndarray,
    test_probabilities: np.ndarray,
    config: dict,
) -> tuple[dict, np.ndarray, float]:
    temperature = fit_temperature(calibration_probabilities, calibration_y, (0.05, 10.0))
    calibrated_calibration = apply_temperature(calibration_probabilities, temperature)
    calibrated_test = apply_temperature(test_probabilities, temperature)
    prediction = calibrated_test.argmax(axis=1)
    error = (prediction != test_y).astype(int)
    uncertainty = 1.0 - calibrated_test.max(axis=1)
    error_auc = float(roc_auc_score(error, uncertainty)) if len(np.unique(error)) == 2 else float("nan")
    keep = max(1, int(math.ceil(0.5 * len(test_y))))
    accepted = np.argsort(uncertainty, kind="mergesort")[:keep]
    alpha = float(config["evaluation"]["conformal_alpha"])
    scores = 1.0 - calibrated_calibration[np.arange(len(calibration_y)), calibration_y]
    threshold = conformal_quantile(scores, alpha)
    sets = standard_conformal_sets(calibrated_test, threshold)
    reliability = {
        "temperature": temperature,
        "temperature_scaled_ece": ece_score(
            test_y, calibrated_test, int(config["evaluation"]["ece_bins"])
        ),
        "error_detection_auroc": error_auc,
        "selective_accuracy_at_50_percent_coverage": float(accuracy_score(test_y[accepted], prediction[accepted])),
        "conformal_target_coverage": 1.0 - alpha,
        "conformal_empirical_coverage": float(sets[np.arange(len(test_y)), test_y].mean()),
        "conformal_mean_set_width": float(sets.sum(axis=1).mean()),
        "conformal_threshold": float(threshold),
    }
    return reliability, calibrated_test, temperature


def subject_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        y = group["true_class_id"].to_numpy(dtype=int)
        pred = group["pred_class_id"].to_numpy(dtype=int)
        recalls = recall_score(y, pred, labels=CLASS_IDS, average=None, zero_division=0)
        row = {
            "subject_id": int(subject),
            "subject_name": group["subject_name"].iloc[0],
            "n": len(group),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "mae": float(np.mean(np.abs(y - pred))),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        }
        row.update({f"recall_{CLASS_NAMES[i]}": float(value) for i, value in enumerate(recalls)})
        rows.append(row)
    return pd.DataFrame(rows)


def package_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = performance_metrics(y_true, probabilities)
    metrics["adjacent_pair_auc"] = adjacent_pair_auc(y_true, probabilities)
    return metrics


def xgboost_comparison(root: Path, config: dict, fold: int, test_names: set[str]) -> dict:
    path = root / config["evaluation"]["xgboost_predictions"]
    frame = pd.read_csv(path)
    frame = frame[
        (frame["model"] == "xgboost")
        & (frame["feature_set"] == "combined_unique")
        & (frame["fold"] == fold)
    ].copy()
    if set(frame["sample_name"]) != test_names:
        raise ValueError("XGBoost comparison does not contain the exact outer-test trials")
    y_true = frame["true_class_id"].to_numpy(dtype=int)
    probability_columns = [f"proba_{name}" for name in CLASS_NAMES]
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    if np.isfinite(probabilities).all():
        result = package_metrics(y_true, probabilities)
    else:
        # The Phase 3-strengthening artifact stores XGBoost hard predictions but leaves
        # probability columns empty. Do not manufacture probability-based AUC values.
        prediction = frame["pred_class_id"].to_numpy(dtype=int)
        result = {
            "accuracy": float(accuracy_score(y_true, prediction)),
            "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
            "mae": float(np.mean(np.abs(y_true - prediction))),
            "qwk": float(cohen_kappa_score(y_true, prediction, weights="quadratic")),
            "per_class_recall": {
                CLASS_NAMES[i]: float(value)
                for i, value in enumerate(
                    recall_score(y_true, prediction, labels=CLASS_IDS, average=None, zero_division=0)
                )
            },
            "confusion_matrix": confusion_matrix(y_true, prediction, labels=CLASS_IDS).astype(int).tolist(),
            "adjacent_pair_auc": None,
            "probability_note": "Unavailable: stored XGBoost probability columns are all missing.",
        }
    result["source"] = str(path.relative_to(root))
    result["n_test"] = len(frame)
    return result


def phase10_reference(root: Path) -> dict[str, float]:
    calibration = pd.read_csv(root / "experiments/prediction_reliability/calibration_metrics.csv")
    calibration = calibration[
        (calibration["probability_type"] == "temperature_scaled")
        & (calibration["group_type"] == "overall")
    ].iloc[0]
    ranking = pd.read_csv(root / "experiments/prediction_reliability/selective_summary.csv")
    ranking = ranking[ranking["method"] == "max_probability"].iloc[0]
    selective = pd.read_csv(root / "experiments/prediction_reliability/selective_prediction_metrics.csv")
    selective = selective[
        (selective["method"] == "max_probability") & np.isclose(selective["coverage_target"], 0.5)
    ]
    conformal = pd.read_csv(root / "experiments/prediction_reliability/conformal_summary.csv")
    conformal = conformal[
        (conformal["method"] == "standard_classification")
        & np.isclose(conformal["alpha"], 0.1)
        & (conformal["group_type"] == "overall")
    ].iloc[0]
    return {
        "temperature_scaled_ece": float(calibration["ece"]),
        "error_detection_auroc": float(ranking["error_detection_auroc"]),
        "selective_accuracy_at_50_percent_coverage": float(selective["accuracy"].mean()),
        "conformal_empirical_coverage": float(conformal["empirical_coverage"]),
        "conformal_mean_set_width": float(conformal["mean_width"]),
    }


def memory_snapshot(device: torch.device) -> dict[str, float]:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        rss_mb = rss / (1024.0 * 1024.0)
    else:
        rss_mb = rss / 1024.0
    result = {"process_peak_rss_mb": rss_mb}
    if device.type == "mps":
        result["mps_current_allocated_mb"] = float(torch.mps.current_allocated_memory()) / (1024.0**2)
        result["mps_driver_allocated_mb"] = float(torch.mps.driver_allocated_memory()) / (1024.0**2)
    return result


def write_report(path: Path, result: dict) -> None:
    metrics = result["ddcae_test_metrics"]
    xgb = result["xgboost_same_trials"]
    rel = result["reliability"]
    phase10 = result["phase10_xgboost_reference"]
    history = result["history"]
    lines = [
        "# Thiam et al. (2021) supervised DDCAE smoke test",
        "",
        f"Run date: {result['run_timestamp_utc']}",
        "",
        "## Status and interpretation",
        "",
        "This is a one-outer-fold engineering smoke test with a small number of epochs. It is not a paper reproduction result and must not be compared directly with the paper's 87-fold LOSO mean. The implementation preserves the reported three modalities, 1,152-sample inputs, six-stage DDCAE encoders/decoders, attention, feature-wise multimodal gate, joint reconstruction/classification objective, learning rate, and batch size. Protocol and paper ambiguities are documented in `reports/thiam2021_feasibility_review.md`.",
        "",
        "## Execution",
        "",
        f"- Device: `{result['device']}` (MPS available: `{result['mps_available']}`).",
        f"- Outer fold: {result['outer_fold']}; fit/validation/calibration/test subjects: {result['split_counts']['fit_subjects']}/{result['split_counts']['validation_subjects']}/{result['split_counts']['calibration_subjects']}/{result['split_counts']['outer_test_subjects']}.",
        f"- Epochs run: {len(history)} of 100 paper epochs; batch size: {result['batch_size']}; learning rate: {result['learning_rate']}.",
        f"- Epoch wall time: {result['runtime']['epoch_seconds_mean']:.2f} s mean ({result['runtime']['epoch_seconds_min']:.2f}-{result['runtime']['epoch_seconds_max']:.2f} s).",
        f"- Warm-up-aware steady-state projection: {result['runtime']['projected_single_fold_hours_100_epochs_steady']:.2f} h per fold, {result['runtime']['projected_5fold_1seed_hours_steady']:.2f} h for five folds x one seed, and {result['runtime']['projected_5fold_3seed_hours_steady']:.2f} h for five folds x three seeds. A conservative all-epoch-mean projection is {result['runtime']['projected_5fold_1seed_hours']:.2f}/{result['runtime']['projected_5fold_3seed_hours']:.2f} h. Projections exclude cache construction and final reliability evaluation.",
        f"- Peak process RSS: {result['memory']['process_peak_rss_mb']:.1f} MB; MPS current/driver allocation: {result['memory'].get('mps_current_allocated_mb', float('nan')):.1f}/{result['memory'].get('mps_driver_allocated_mb', float('nan')):.1f} MB.",
        "",
        "## Loss sanity check",
        "",
        "| epoch | train total | validation total | train CE | validation CE | seconds |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_total']:.4f} | {row['validation_total']:.4f} | {row['train_classification']:.4f} | {row['validation_classification']:.4f} | {row['seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Best validation checkpoint: epoch {result['best_epoch']}. With only {len(history)} epochs at the paper's small learning rate, loss direction is a pipeline diagnostic, not evidence of convergence.",
            "",
            "## Outer-fold smoke performance and same-trial XGBoost reference",
            "",
            "| model | accuracy | macro F1 | balanced accuracy | MAE | QWK | PA2 recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| DDCAE smoke | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['mae']:.4f} | {metrics['qwk']:.4f} | {metrics['per_class_recall']['PA2']:.4f} |",
            f"| Phase 3-strengthening XGBoost | {xgb['accuracy']:.4f} | {xgb['macro_f1']:.4f} | {xgb['balanced_accuracy']:.4f} | {xgb['mae']:.4f} | {xgb['qwk']:.4f} | {xgb['per_class_recall']['PA2']:.4f} |",
            "",
            "The rows contain the exact same outer-fold test trials. The DDCAE row is intentionally under-trained and is not a fair model comparison.",
            "",
            "### DDCAE per-class recall",
            "",
            "| class | recall |",
            "|---|---:|",
        ]
    )
    for name in CLASS_NAMES:
        lines.append(f"| {name} | {metrics['per_class_recall'][name]:.4f} |")
    lines.extend(["", "### DDCAE adjacent-level AUC", "", "| pair | AUROC |", "|---|---:|"])
    for name, value in metrics["adjacent_pair_auc"].items():
        lines.append(f"| {name.replace('_vs_', ' vs ')} | {value:.4f} |")
    lines.extend(
        [
            "",
            "DDCAE confusion matrix, rows=true and columns=predicted, class order `BL1 PA1 PA2 PA3 PA4`:",
            "",
            "```text",
            *[" ".join(f"{value:4d}" for value in row) for row in metrics["confusion_matrix"]],
            "```",
            "",
            "## Reliability smoke diagnostics",
            "",
            f"- Temperature-scaled ECE: {rel['temperature_scaled_ece']:.4f} (temperature {rel['temperature']:.4f}).",
            f"- Error-detection AUROC: {rel['error_detection_auroc']:.4f}.",
            f"- Accuracy at 50% selective coverage: {rel['selective_accuracy_at_50_percent_coverage']:.4f}.",
            f"- 90% split-conformal target: empirical coverage {rel['conformal_empirical_coverage']:.4f}, mean set width {rel['conformal_mean_set_width']:.3f} of five labels.",
            "",
            "For context, the completed five-fold Phase 10 XGBoost run had temperature-scaled ECE "
            f"{phase10['temperature_scaled_ece']:.4f}, error-detection AUROC {phase10['error_detection_auroc']:.4f}, "
            f"50%-coverage accuracy {phase10['selective_accuracy_at_50_percent_coverage']:.4f}, and 90% conformal coverage/width "
            f"{phase10['conformal_empirical_coverage']:.4f}/{phase10['conformal_mean_set_width']:.3f}. The two-epoch DDCAE is worse in risk ranking and resolution; its very low ECE is caused by nearly uniform probabilities and temperature hitting the upper bound, not useful calibration.",
            "",
            "These values verify that the Phase 10-style, subject-disjoint calibration path works with DDCAE probabilities. They cannot yet answer whether the reliability-resolution trade-off persists after a published deep model is trained: a two-epoch checkpoint collapsed to PA2 and has not learned the intended classifier. The formal answer is therefore **undetermined after smoke testing**; it is deferred until an approved full-scale run.",
            "",
            "## Reproducibility",
            "",
            f"- Training command: `{result['command']}`",
            f"- Finalization command: `{result['finalization_command']}`",
            f"- Seed: `{result['seed']}`",
            f"- Model checkpoint SHA-256: `{result['checkpoint_sha256']}`",
            f"- Raw-signal collection SHA-256: `{result['input_hashes']['raw_signal_collection_sha256']}`",
            f"- Samples metadata SHA-256: `{result['input_hashes']['samples_sha256']}`",
            f"- Fold assignment SHA-256: `{result['input_hashes']['fold_assignments_sha256']}`",
            f"- Paper PDF SHA-256: `{result['input_hashes']['paper_pdf_sha256']}`",
            "- Full dependency versions, split roles, standardization values, configuration, history, predictions, and subject-level metrics are stored in `experiments/thiam2021_smoke/`.",
            "",
            "## Stop decision",
            "",
            "The requested stop condition is met. No other outer fold and no additional seed was started.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/thiam2021_smoke.yaml"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Load the already trained checkpoint/history and only regenerate evaluation outputs.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text())
    fold = int(args.fold or config["training"]["outer_fold"])
    epochs = int(args.epochs or config["training"]["smoke_epochs"])
    seed = int(config["training"]["seed"])
    set_seeds(seed)

    samples = load_samples(root / config["data"]["samples"])
    cache_manifest = build_signal_cache(root, samples, config, force=args.rebuild_cache)
    arrays, cache_manifest = load_signal_cache(root, config)
    labels = samples["class_id"].to_numpy(dtype=int)
    folds = make_fold_indices(samples, load_phase3_folds(root / config["data"]["fold_assignments"]))
    split = next(item for item in folds if int(item["fold"]) == fold)

    development_subjects, calibration_subjects = split_model_and_calibration_subjects(
        split["train_subjects"], float(config["training"]["calibration_subject_fraction"]), seed + fold
    )
    fit_subjects, validation_subjects = split_model_and_calibration_subjects(
        development_subjects,
        float(config["training"]["validation_subject_fraction_of_model_development"]),
        seed + 1000 + fold,
    )
    roles = {
        "fit": fit_subjects,
        "validation": validation_subjects,
        "calibration": calibration_subjects,
        "outer_test": split["test_subjects"],
    }
    for left, left_subjects in roles.items():
        for right, right_subjects in roles.items():
            if left < right and set(left_subjects) & set(right_subjects):
                raise ValueError(f"Subject leakage between {left} and {right}")
    indices = {
        role: np.flatnonzero(samples["subject_id"].isin(subjects).to_numpy()) for role, subjects in roles.items()
    }
    if not np.array_equal(np.sort(indices["outer_test"]), np.sort(split["test_index"])):
        raise ValueError("Derived outer-test indices do not match the Phase 3 fold")
    stats = fit_standardization(arrays, indices["fit"])

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    datasets = {
        role: SignalDataset(arrays, labels, role_indices, stats) for role, role_indices in indices.items()
    }
    loaders = {
        role: loader(
            dataset,
            batch_size,
            shuffle=(role == "fit"),
            seed=seed + offset,
            num_workers=num_workers,
        )
        for offset, (role, dataset) in enumerate(datasets.items())
    }

    device = choose_device(str(config["training"]["device_preference"]))
    model = ThiamSupervisedDDCAE(config).to(device)
    output_dir = root / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.csv"
    if args.finalize_existing:
        if not checkpoint_path.exists() or not history_path.exists():
            raise FileNotFoundError("Cannot finalize: checkpoint or training history is missing")
        history = pd.read_csv(history_path).to_dict(orient="records")
        best_epoch = int(min(history, key=lambda row: row["validation_total"])["epoch"])
        best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best_state)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
        history = []
        best_epoch = 0
        best_loss = float("inf")
        best_state = None
        patience = 0
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_metrics = run_epoch(model, loaders["fit"], config, device, optimizer)
            validation_metrics = run_epoch(model, loaders["validation"], config, device, None)
            seconds = time.perf_counter() - started
            row = {"epoch": epoch, "seconds": seconds}
            row.update({f"train_{name}": value for name, value in train_metrics.items()})
            row.update({f"validation_{name}": value for name, value in validation_metrics.items()})
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation_metrics["total"] < best_loss:
                best_loss = validation_metrics["total"]
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= int(config["training"]["early_stopping_patience"]):
                    break
        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")
        model.load_state_dict(best_state)

    calibration_indices, calibration_raw = predict(model, loaders["calibration"], device)
    test_indices, test_raw = predict(model, loaders["outer_test"], device)
    calibration_y = labels[calibration_indices]
    test_y = labels[test_indices]
    reliability, test_calibrated, _ = reliability_metrics(
        calibration_y, calibration_raw, test_y, test_raw, config
    )
    metrics = package_metrics(test_y, test_calibrated)

    if not args.finalize_existing:
        torch.save(best_state, checkpoint_path)
    predictions = samples.iloc[test_indices][
        ["subject_id", "subject_name", "sample_id", "sample_name", "class_id", "class_name"]
    ].copy()
    predictions = predictions.rename(columns={"class_id": "true_class_id", "class_name": "true_class_name"})
    predictions["fold"] = fold
    predictions["pred_class_id"] = test_calibrated.argmax(axis=1)
    predictions["pred_class_name"] = predictions["pred_class_id"].map(EXPECTED_LABEL_MAP)
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions[f"raw_{class_name}"] = test_raw[:, class_id]
        predictions[f"calibrated_{class_name}"] = test_calibrated[:, class_id]
    predictions.to_csv(output_dir / "outer_test_predictions.csv", index=False)
    subject_frame = subject_metrics(predictions)
    subject_frame.to_csv(output_dir / "subject_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    pd.DataFrame(
        [
            {"subject_id": int(subject), "role": role, "fold": fold}
            for role, subjects in roles.items()
            for subject in subjects
        ]
    ).to_csv(output_dir / "nested_subject_split_manifest.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    epoch_seconds = [row["seconds"] for row in history]
    mean_epoch = float(np.mean(epoch_seconds))
    steady_epoch = float(epoch_seconds[-1])
    xgb = xgboost_comparison(root, config, fold, set(predictions["sample_name"]))
    ci = bootstrap_subject_ci(
        predictions,
        test_calibrated,
        int(config["evaluation"]["bootstrap_repetitions_smoke"]),
        seed + 2000,
    )
    dependencies = {}
    for package in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "PyYAML"]:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    fold_path = root / config["data"]["fold_assignments"]
    paper_path = root / config["paper"]["local_pdf"]
    training_command = " ".join(
        shlex.quote(value)
        for value in [
            sys.executable,
            "-m",
            "src.models.run_thiam2021_reproduction",
            "--config",
            str(args.config),
            "--fold",
            str(fold),
            "--epochs",
            str(epochs),
        ]
    )
    result = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "classification": config["classification"],
        "command": training_command,
        "finalization_command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "outer_fold": fold,
        "seed": seed,
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "batch_size": batch_size,
        "learning_rate": float(config["training"]["learning_rate"]),
        "best_epoch": best_epoch,
        "split_counts": {
            f"{role}_subjects": len(subjects) for role, subjects in roles.items()
        }
        | {f"{role}_trials": int(len(indices[role])) for role in roles},
        "standardization_fit_subjects_only": stats,
        "history": history,
        "ddcae_test_metrics": metrics,
        "ddcae_subject_clustered_95_ci": ci,
        "reliability": reliability,
        "phase10_xgboost_reference": phase10_reference(root),
        "xgboost_same_trials": xgb,
        "runtime": {
            "cache_preprocessing_seconds": float(cache_manifest["preprocessing_seconds"]),
            "epoch_seconds_mean": mean_epoch,
            "epoch_seconds_min": float(np.min(epoch_seconds)),
            "epoch_seconds_max": float(np.max(epoch_seconds)),
            "projected_single_fold_hours_100_epochs": mean_epoch * 100 / 3600,
            "projected_5fold_1seed_hours": mean_epoch * 100 * 5 / 3600,
            "projected_5fold_3seed_hours": mean_epoch * 100 * 5 * 3 / 3600,
            "steady_epoch_seconds": steady_epoch,
            "projected_single_fold_hours_100_epochs_steady": (epoch_seconds[0] + steady_epoch * 99) / 3600,
            "projected_5fold_1seed_hours_steady": (epoch_seconds[0] + steady_epoch * 99) * 5 / 3600,
            "projected_5fold_3seed_hours_steady": (epoch_seconds[0] + steady_epoch * 99) * 5 * 3 / 3600,
        },
        "memory": memory_snapshot(device),
        "input_hashes": {
            "raw_signal_collection_sha256": cache_manifest["raw_signal_collection_sha256"],
            "samples_sha256": cache_manifest["samples_sha256"],
            "fold_assignments_sha256": sha256_file(fold_path),
            "paper_pdf_sha256": sha256_file(paper_path),
            "xgboost_predictions_sha256": sha256_file(root / config["evaluation"]["xgboost_predictions"]),
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dependencies": dependencies,
        "platform": platform.platform(),
        "python": sys.version,
        "paper_fidelity": {
            "faithful_components": [
                "three ECG/EDA/EMG inputs",
                "1152 samples at 256 Hz",
                "reported six-stage DDCAE widths and modality kernels",
                "attention after each encoder convolution",
                "feature-wise latent gate and 128-unit classifier",
                "joint weighted reconstruction/classification loss",
                "Gaussian noise, Adam learning rate, and batch size",
            ],
            "adaptations": [
                "fixed Phase 3 five outer folds instead of 87-fold LOSO",
                "nested validation/calibration subject splits",
                "canonical window only in smoke test instead of nine-window augmentation",
                "per-clip filtering and assumed one-second crop due lack of continuous data/event trigger",
                "fit-subject-only per-modality scalar standardization because paper omits amplitude normalization",
                "PyTorch independent implementation because no official code was found",
            ],
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(root / config["outputs"]["report"], result)
    print(json.dumps({"status": "complete", "output": str(output_dir), "best_epoch": best_epoch}), flush=True)


if __name__ == "__main__":
    main()
