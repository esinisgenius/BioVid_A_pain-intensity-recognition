from __future__ import annotations

import numpy as np
import torch

from src.models.run_boundary_gated_ordinal import (
    BoundaryGatedOrdinalMLP,
    NetworkConfig,
    OrdinalMLP,
    cumulative_to_class_probabilities,
    estimate_consistency_prior,
    inner_splits,
    load_and_validate_inputs,
    metric_tables,
    ordinal_loss,
)


def test_models_produce_monotone_cumulative_and_valid_class_probabilities() -> None:
    x = torch.randn(13, 72)
    config = NetworkConfig((16,), 0.0)
    for model in [OrdinalMLP(72, config), BoundaryGatedOrdinalMLP(72, config)]:
        cumulative, probabilities = model(x)
        assert cumulative.shape == (13, 4)
        assert probabilities.shape == (13, 5)
        assert torch.all(cumulative[:, :-1] >= cumulative[:, 1:])
        assert torch.all(probabilities >= -1e-7)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(13), atol=1e-6)


def test_cumulative_conversion_matches_expected_distribution() -> None:
    cumulative = torch.tensor([[0.8, 0.6, 0.3, 0.1]])
    result = cumulative_to_class_probabilities(cumulative)
    assert torch.allclose(result, torch.tensor([[0.2, 0.2, 0.3, 0.2, 0.1]]), atol=1e-6)


def test_ordinal_loss_is_finite_for_valid_probabilities() -> None:
    cumulative = torch.tensor([[0.8, 0.6, 0.3, 0.1], [0.7, 0.5, 0.2, 0.05]])
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    loss = ordinal_loss(cumulative, targets, torch.ones(4), torch.ones(4))
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_consistency_prior_rewards_directionally_consistent_effects() -> None:
    rng = np.random.default_rng(7)
    subjects = np.repeat(np.arange(12), 10)
    y = np.tile(np.repeat(np.arange(5), 2), 12)
    x = rng.normal(scale=0.2, size=(len(y), 3))
    x[:, 0] += y * 0.8
    for subject in range(12):
        sign = 1 if subject % 2 == 0 else -1
        x[subjects == subject, 1] += sign * y[subjects == subject] * 0.8
    prior, effects = estimate_consistency_prior(x, y, subjects)
    assert prior.shape == (4, 3)
    assert np.all((prior >= 0.05) & (prior <= 0.95))
    assert np.all(prior[:, 0] > prior[:, 1])
    assert len(effects) == 12 * 4 * 3


def test_repository_inputs_have_72_features_and_subject_clean_folds() -> None:
    table, features, _, _ = load_and_validate_inputs(
        feature_table_path=__import__("pathlib").Path("artifacts/feature_table_filtered.csv"),
        metadata_path=__import__("pathlib").Path("artifacts/feature_metadata.json"),
        fold_path=__import__("pathlib").Path("experiments/baselines/fold_assignments.csv"),
    )
    assert len(features) == 72
    assert table.groupby("subject_id")["outer_fold"].nunique().max() == 1


def test_inner_splits_keep_subjects_disjoint() -> None:
    groups = np.repeat(np.arange(12), 10)
    y = np.tile(np.repeat(np.arange(5), 2), 12)
    for train, validation in inner_splits(y, groups, seed=13, n_splits=3):
        assert not set(groups[train]).intersection(groups[validation])


def test_missing_reference_probabilities_are_skipped_for_auc() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "method": ["reference"] * 5,
            "fold": [1] * 5,
            "seed": [-1] * 5,
            "subject_id": [1] * 5,
            "true_class_id": np.arange(5),
            "pred_class_id": np.arange(5),
            **{f"cumulative_b{boundary}": [np.nan] * 5 for boundary in range(4)},
        }
    )
    fold_metrics, subject_metrics, boundary_auc = metric_tables(frame)
    assert len(fold_metrics) == 1
    assert len(subject_metrics) == 1
    assert boundary_auc.empty
