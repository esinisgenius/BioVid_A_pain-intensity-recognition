from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.data.load_samples import EXPECTED_LABEL_MAP, load_samples
from src.data.thiam2021_signals import build_signal_cache, fit_standardization, load_signal_cache
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_prediction_reliability import split_model_and_calibration_subjects
from src.models.run_thiam2021_reproduction import (
    CLASS_IDS,
    CLASS_NAMES,
    SignalDataset,
    choose_device,
    loader,
    memory_snapshot,
    move_inputs,
    package_metrics,
    set_seeds,
    sha256_file,
)
from src.models.thiam2021_ddcae import ThiamSupervisedDDCAE, joint_loss


def parameter_counts(model: ThiamSupervisedDDCAE) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "autoencoders": sum(parameter.numel() for parameter in model.autoencoders.parameters()),
        "gate": sum(parameter.numel() for parameter in model.gate.parameters()),
        "classifier": sum(parameter.numel() for parameter in model.classifier.parameters()),
    }


def gradient_norm(parameters) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    found = False
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum().cpu().double()
            found = True
    return float(torch.sqrt(squared)) if found else 0.0


def train_epoch(
    model: ThiamSupervisedDDCAE,
    batches,
    config: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    classifier_gradients = []
    total_gradients = []
    count = 0
    for inputs, targets, _ in batches:
        inputs = move_inputs(inputs, device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, corrupt=True)
        loss, pieces = joint_loss(model, outputs, inputs, targets, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        classifier_norm = gradient_norm(model.classifier.parameters())
        total_norm = gradient_norm(model.parameters())
        if not math.isfinite(classifier_norm) or not math.isfinite(total_norm):
            raise FloatingPointError("Non-finite gradient norm")
        classifier_gradients.append(classifier_norm)
        total_gradients.append(total_norm)
        optimizer.step()
        batch_n = len(targets)
        count += batch_n
        for name, value in pieces.items():
            sums[name] = sums.get(name, 0.0) + float(value.cpu()) * batch_n
    metrics = {name: value / count for name, value in sums.items()}
    metrics.update(
        {
            "classifier_gradient_norm_mean": float(np.mean(classifier_gradients)),
            "classifier_gradient_norm_max": float(np.max(classifier_gradients)),
            "total_gradient_norm_mean": float(np.mean(total_gradients)),
            "total_gradient_norm_max": float(np.max(total_gradients)),
        }
    )
    return metrics


@torch.no_grad()
def evaluate(
    model: ThiamSupervisedDDCAE,
    batches,
    labels: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    all_indices = []
    all_probabilities = []
    for inputs, targets, indices in batches:
        inputs = move_inputs(inputs, device)
        targets = targets.to(device)
        outputs = model(inputs, corrupt=False)
        loss, pieces = joint_loss(model, outputs, inputs, targets, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        batch_n = len(targets)
        count += batch_n
        for name, value in pieces.items():
            sums[name] = sums.get(name, 0.0) + float(value.cpu()) * batch_n
        all_indices.append(indices.numpy())
        all_probabilities.append(torch.softmax(outputs["logits"], dim=1).cpu().numpy())
    indices = np.concatenate(all_indices).astype(int)
    probabilities = np.concatenate(all_probabilities)
    y_true = labels[indices]
    prediction = probabilities.argmax(axis=1)
    recalls = recall_score(y_true, prediction, labels=CLASS_IDS, average=None, zero_division=0)
    losses = {name: value / count for name, value in sums.items()}
    losses["reconstruction_sum"] = sum(losses[f"reconstruction_{name}"] for name in config["data"]["modalities"])
    return {
        "indices": indices,
        "probabilities": probabilities,
        "prediction": prediction,
        "losses": losses,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "per_class_recall": {CLASS_NAMES[i]: float(value) for i, value in enumerate(recalls)},
        "prediction_counts": {
            CLASS_NAMES[i]: int(np.sum(prediction == i)) for i in range(len(CLASS_NAMES))
        },
        "distinct_predicted_classes": int(len(np.unique(prediction))),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=CLASS_IDS).astype(int).tolist(),
    }


def history_row(epoch: int, seconds: float, train_opt: dict, train_eval: dict, validation: dict) -> dict:
    row = {"epoch": epoch, "seconds": seconds}
    for name, value in train_opt.items():
        row[f"train_optimization_{name}"] = value
    for prefix, result in [("train", train_eval), ("validation", validation)]:
        for name, value in result["losses"].items():
            row[f"{prefix}_{name}"] = value
        row[f"{prefix}_accuracy"] = result["accuracy"]
        row[f"{prefix}_macro_f1"] = result["macro_f1"]
        row[f"{prefix}_distinct_predicted_classes"] = result["distinct_predicted_classes"]
        for name, value in result["per_class_recall"].items():
            row[f"{prefix}_recall_{name}"] = value
        for name, value in result["prediction_counts"].items():
            row[f"{prefix}_pred_count_{name}"] = value
    return row


def convergence_gate(train: dict, validation: dict, minimum_classes: int) -> tuple[bool, str]:
    for name, result in [("train", train), ("validation", validation)]:
        numeric = [result["accuracy"], result["macro_f1"], *result["losses"].values()]
        if not np.isfinite(numeric).all():
            return False, f"non-finite {name} metric"
        if result["distinct_predicted_classes"] < minimum_classes:
            return False, f"{name} predicts only {result['distinct_predicted_classes']} class(es)"
    return True, "train and subject-disjoint validation both predict multiple classes with finite metrics"


def save_predictions(path: Path, samples: pd.DataFrame, result: dict, role: str, fold: int) -> None:
    frame = samples.iloc[result["indices"]][
        ["subject_id", "subject_name", "sample_id", "sample_name", "class_id", "class_name"]
    ].copy()
    frame = frame.rename(columns={"class_id": "true_class_id", "class_name": "true_class_name"})
    frame["role"] = role
    frame["fold"] = fold
    frame["pred_class_id"] = result["prediction"]
    frame["pred_class_name"] = frame["pred_class_id"].map(EXPECTED_LABEL_MAP)
    for class_id, class_name in enumerate(CLASS_NAMES):
        frame[f"proba_{class_name}"] = result["probabilities"][:, class_id]
    frame.to_csv(path, index=False)


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    epoch = history["epoch"]
    axes[0, 0].plot(epoch, history["train_total"], label="train")
    axes[0, 0].plot(epoch, history["validation_total"], label="validation")
    axes[0, 0].set_title("Joint loss (clean evaluation)")
    axes[0, 1].plot(epoch, history["train_classification"], label="train")
    axes[0, 1].plot(epoch, history["validation_classification"], label="validation")
    axes[0, 1].axhline(math.log(5), color="black", linestyle="--", linewidth=1, label="log(5)")
    axes[0, 1].set_title("Classification loss")
    axes[0, 2].plot(epoch, history["train_reconstruction_sum"], label="train")
    axes[0, 2].plot(epoch, history["validation_reconstruction_sum"], label="validation")
    axes[0, 2].set_title("Sum of modality reconstruction MSE")
    axes[1, 0].plot(epoch, history["train_accuracy"], label="train accuracy")
    axes[1, 0].plot(epoch, history["validation_accuracy"], label="validation accuracy")
    axes[1, 0].plot(epoch, history["train_macro_f1"], linestyle="--", label="train macro F1")
    axes[1, 0].plot(epoch, history["validation_macro_f1"], linestyle="--", label="validation macro F1")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Classification performance")
    for class_name in CLASS_NAMES:
        axes[1, 1].plot(epoch, history[f"validation_recall_{class_name}"], label=class_name)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Validation per-class recall")
    axes[1, 2].plot(
        epoch,
        history["train_optimization_classifier_gradient_norm_mean"],
        label="classifier gradient norm",
    )
    axes[1, 2].plot(
        epoch,
        history["validation_distinct_predicted_classes"],
        label="validation predicted classes",
    )
    axes[1, 2].set_title("Optimization and collapse diagnostics")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/thiam2021_fold1_diagnostic.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text())
    fold = int(config["training"]["outer_fold"])
    if fold != 1:
        raise ValueError("This diagnostic is intentionally restricted to outer fold 1")
    seed = int(config["training"]["seed"])
    set_seeds(seed)

    samples = load_samples(root / config["data"]["samples"])
    cache_manifest = build_signal_cache(root, samples, config, force=False)
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
        "calibration_held_out_unused": calibration_subjects,
        "outer_test": split["test_subjects"],
    }
    role_sets = {name: set(values) for name, values in roles.items()}
    for left_index, left in enumerate(roles):
        for right in list(roles)[left_index + 1 :]:
            if role_sets[left] & role_sets[right]:
                raise ValueError(f"Subject leakage between {left} and {right}")
    indices = {
        role: np.flatnonzero(samples["subject_id"].isin(subjects).to_numpy()) for role, subjects in roles.items()
    }
    if not np.array_equal(np.sort(indices["outer_test"]), np.sort(split["test_index"])):
        raise ValueError("Outer-test indices do not match Phase 3 Fold 1")
    stats = fit_standardization(arrays, indices["fit"])

    datasets = {
        role: SignalDataset(arrays, labels, role_indices, stats)
        for role, role_indices in indices.items()
        if role in {"fit", "validation", "outer_test"}
    }
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = loader(datasets["fit"], batch_size, True, seed, num_workers)
    eval_loaders = {
        role: loader(dataset, batch_size, False, seed + offset + 1, num_workers)
        for offset, (role, dataset) in enumerate(datasets.items())
    }
    device = choose_device(str(config["training"]["device_preference"]))
    model = ThiamSupervisedDDCAE(config).to(device)
    counts = parameter_counts(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    output_dir = root / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.csv"
    log_path = output_dir / "training_log.jsonl"
    if checkpoint_path.exists() or history_path.exists() or log_path.exists():
        raise FileExistsError("Diagnostic outputs already exist; refusing to overwrite a completed or partial run")

    initial_train = evaluate(model, eval_loaders["fit"], labels, config, device)
    initial_validation = evaluate(model, eval_loaders["validation"], labels, config, device)
    started_all = time.perf_counter()
    history = []
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    consecutive_collapse = 0
    termination_reason = "max_epochs"
    max_epochs = int(config["training"]["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        started_epoch = time.perf_counter()
        train_optimization = train_epoch(model, train_loader, config, device, optimizer)
        train_evaluation = evaluate(model, eval_loaders["fit"], labels, config, device)
        validation = evaluate(model, eval_loaders["validation"], labels, config, device)
        row = history_row(
            epoch, time.perf_counter() - started_epoch, train_optimization, train_evaluation, validation
        )
        if not np.isfinite([value for value in row.values() if isinstance(value, (int, float))]).all():
            termination_reason = "non_finite_metric"
            break
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

        current = float(validation["losses"]["total"])
        min_delta = float(config["training"]["early_stopping_min_delta"])
        if current < best_loss - min_delta:
            best_loss = current
            best_epoch = epoch
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        collapsed = (
            train_evaluation["distinct_predicted_classes"] == 1
            and validation["distinct_predicted_classes"] == 1
        )
        consecutive_collapse = consecutive_collapse + 1 if collapsed else 0
        if (
            epoch >= int(config["training"]["collapse_check_start_epoch"])
            and consecutive_collapse >= int(config["training"]["collapse_patience"])
        ):
            termination_reason = "persistent_single_class_collapse"
            break
        if epochs_without_improvement >= int(config["training"]["early_stopping_patience"]):
            termination_reason = "early_stopping"
            break

    if best_epoch == 0 or not checkpoint_path.exists():
        raise RuntimeError("No finite best checkpoint was produced")
    best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_state)
    best_train = evaluate(model, eval_loaders["fit"], labels, config, device)
    best_validation = evaluate(model, eval_loaders["validation"], labels, config, device)
    gate_passed, gate_reason = convergence_gate(
        best_train,
        best_validation,
        int(config["evaluation"]["minimum_distinct_classes_for_test"]),
    )
    test_result = None
    if gate_passed:
        test_result = evaluate(model, eval_loaders["outer_test"], labels, config, device)
        save_predictions(output_dir / "outer_test_predictions.csv", samples, test_result, "outer_test", fold)

    save_predictions(output_dir / "fit_predictions.csv", samples, best_train, "fit", fold)
    save_predictions(output_dir / "validation_predictions.csv", samples, best_validation, "validation", fold)
    pd.DataFrame(best_train["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "fit_confusion_matrix.csv"
    )
    pd.DataFrame(best_validation["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "validation_confusion_matrix.csv"
    )
    if test_result is not None:
        pd.DataFrame(test_result["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            output_dir / "outer_test_confusion_matrix.csv"
        )
    history_frame = pd.DataFrame(history)
    plot_history(history_frame, output_dir / "learning_curves.png")
    pd.DataFrame(
        [
            {"fold": fold, "role": role, "subject_id": int(subject)}
            for role, subjects in roles.items()
            for subject in subjects
        ]
    ).to_csv(output_dir / "subject_split_manifest.csv", index=False)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    dependencies = {}
    for package in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML"]:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "outer_fold": fold,
        "seed": seed,
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "parameter_counts": counts,
        "split_counts": {
            f"{role}_subjects": len(subjects) for role, subjects in roles.items()
        }
        | {f"{role}_trials": int(len(indices[role])) for role in roles},
        "standardization_fit_subjects_only": stats,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "termination_reason": termination_reason,
        "early_stopping": {
            "metric": config["training"]["checkpoint_metric"],
            "patience": config["training"]["early_stopping_patience"],
            "min_delta": config["training"]["early_stopping_min_delta"],
            "epochs_without_improvement_at_stop": epochs_without_improvement,
        },
        "collapse_rule": {
            "start_epoch": config["training"]["collapse_check_start_epoch"],
            "patience": config["training"]["collapse_patience"],
            "consecutive_collapse_at_stop": consecutive_collapse,
        },
        "convergence_gate_passed": gate_passed,
        "convergence_gate_reason": gate_reason,
        "outer_test_evaluated": test_result is not None,
        "initial": {
            "train": {key: value for key, value in initial_train.items() if key not in {"indices", "probabilities", "prediction"}},
            "validation": {key: value for key, value in initial_validation.items() if key not in {"indices", "probabilities", "prediction"}},
        },
        "best_checkpoint": {
            "train": {key: value for key, value in best_train.items() if key not in {"indices", "probabilities", "prediction"}},
            "validation": {key: value for key, value in best_validation.items() if key not in {"indices", "probabilities", "prediction"}},
            "outer_test": None
            if test_result is None
            else {key: value for key, value in test_result.items() if key not in {"indices", "probabilities", "prediction"}},
        },
        "runtime": {
            "total_training_and_epoch_evaluation_seconds": time.perf_counter() - started_all,
            "epoch_seconds_mean": float(history_frame["seconds"].mean()),
            "epoch_seconds_min": float(history_frame["seconds"].min()),
            "epoch_seconds_max": float(history_frame["seconds"].max()),
        },
        "memory": memory_snapshot(device),
        "input_hashes": {
            "raw_signal_collection_sha256": cache_manifest["raw_signal_collection_sha256"],
            "samples_sha256": cache_manifest["samples_sha256"],
            "fold_assignments_sha256": sha256_file(root / config["data"]["fold_assignments"]),
            "paper_pdf_sha256": sha256_file(root / config["paper"]["local_pdf"]),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dependencies": dependencies,
        "platform": platform.platform(),
        "python": sys.version,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "termination_reason": termination_reason,
                "convergence_gate_passed": gate_passed,
                "outer_test_evaluated": test_result is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
