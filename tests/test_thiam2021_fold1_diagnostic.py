from __future__ import annotations

import torch

from src.models.run_thiam2021_fold1_diagnostic import convergence_gate, gradient_norm


def _result(distinct: int, finite: bool = True) -> dict:
    value = 1.0 if finite else float("nan")
    return {
        "accuracy": value,
        "macro_f1": value,
        "losses": {"total": value, "classification": value, "reconstruction_sum": value},
        "distinct_predicted_classes": distinct,
    }


def test_convergence_gate_requires_finite_multiclass_train_and_validation_predictions() -> None:
    passed, _ = convergence_gate(_result(5), _result(4), minimum_classes=2)
    assert passed
    passed, reason = convergence_gate(_result(1), _result(4), minimum_classes=2)
    assert not passed and "train predicts only" in reason
    passed, reason = convergence_gate(_result(5), _result(4, finite=False), minimum_classes=2)
    assert not passed and "non-finite validation" in reason


def test_gradient_norm_is_positive_after_classifier_backward() -> None:
    layer = torch.nn.Linear(3, 2)
    loss = layer(torch.ones(4, 3)).square().mean()
    loss.backward()
    assert gradient_norm(layer.parameters()) > 0
