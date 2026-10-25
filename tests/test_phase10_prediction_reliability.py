from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import src.models.run_prediction_reliability as phase10


def test_temperature_scaling_preserves_argmax_and_reduces_calibration_nll() -> None:
    y_true = np.asarray([0, 1, 2, 3, 4] * 20)
    probabilities = np.full((len(y_true), 5), 0.0125)
    wrong = (y_true + 1) % 5
    probabilities[np.arange(len(y_true)), y_true] = 0.35
    probabilities[np.arange(len(y_true)), wrong] = 0.60
    probabilities = phase10.normalize_probabilities(probabilities)

    temperature = phase10.fit_temperature(probabilities, y_true, (0.05, 10.0))
    calibrated = phase10.apply_temperature(probabilities, temperature)

    assert temperature > 1.0
    np.testing.assert_array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1))
    assert log_loss(y_true, calibrated, labels=phase10.CLASS_IDS) < log_loss(
        y_true, probabilities, labels=phase10.CLASS_IDS
    )
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)


def test_nested_subject_split_is_disjoint_deterministic_and_nonempty() -> None:
    subjects = list(range(1, 21))
    model_a, calibration_a = phase10.split_model_and_calibration_subjects(subjects, 0.25, 123)
    model_b, calibration_b = phase10.split_model_and_calibration_subjects(subjects, 0.25, 123)

    assert model_a == model_b
    assert calibration_a == calibration_b
    assert len(calibration_a) == 5
    assert model_a
    assert not set(model_a).intersection(calibration_a)
    assert set(model_a).union(calibration_a) == set(subjects)


def test_finite_sample_conformal_quantile_uses_higher_order_statistic() -> None:
    scores = np.arange(1, 11, dtype=float) / 10.0
    # ceil((10 + 1) * 0.8) = 9, hence the ninth order statistic.
    assert phase10.conformal_quantile(scores, alpha=0.2) == 0.9
    # The finite-sample correction clips an unattainable rank to n.
    assert phase10.conformal_quantile(scores, alpha=0.05) == 1.0


def test_standard_sets_and_ordinal_intervals_have_expected_structure() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.03, 0.02],
            [0.05, 0.15, 0.55, 0.20, 0.05],
        ]
    )
    standard = phase10.standard_conformal_sets(probabilities, threshold=0.80)
    np.testing.assert_array_equal(standard[0], [True, True, False, False, False])

    lower, upper = phase10.ordinal_conformal_intervals(probabilities, radius=1.1)
    membership = phase10.interval_membership(lower, upper)
    assert np.all(lower <= upper)
    for row in membership:
        included = np.flatnonzero(row)
        np.testing.assert_array_equal(included, np.arange(included.min(), included.max() + 1))


def test_selection_metrics_are_computed_only_on_accepted_samples() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    prediction = np.asarray([0, 4, 2, 0])
    accepted = np.asarray([True, False, True, False])
    metrics = phase10.selection_metrics(y_true, prediction, accepted)

    assert metrics["n_accepted"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["severe_error_rate"] == 0.0


def test_build_fold_predictions_never_uses_outer_test_subjects_for_calibration(monkeypatch) -> None:
    rows = []
    for subject_id in range(1, 11):
        for class_id in range(5):
            rows.append(
                {
                    "subject_id": subject_id,
                    "subject_name": f"S{subject_id:02d}",
                    "sample_id": class_id,
                    "sample_name": f"S{subject_id:02d}-{class_id}",
                    "class_id": class_id,
                    "class_name": phase10.EXPECTED_LABEL_MAP[class_id],
                    "feature": float(class_id),
                }
            )
    table = pd.DataFrame(rows)
    test_index = table.index[table["subject_id"].isin([9, 10])].to_numpy()
    train_index = table.index[~table["subject_id"].isin([9, 10])].to_numpy()
    folds = [
        {
            "fold": 1,
            "train_index": train_index,
            "test_index": test_index,
            "train_subjects": list(range(1, 9)),
            "test_subjects": [9, 10],
        }
    ]

    class FakeModel:
        def fit(self, x: np.ndarray, y: np.ndarray):
            self.classes_ = np.arange(5)
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            distance = np.abs(x[:, [0]] - np.arange(5)[None, :])
            scores = np.exp(-distance) + 0.05
            return scores / scores.sum(axis=1, keepdims=True)

    monkeypatch.setattr(phase10, "make_baseline_model", lambda *_args, **_kwargs: FakeModel())
    predictions, conformal, manifest, thresholds = phase10.build_fold_predictions(
        table=table,
        columns=["feature"],
        folds=folds,
        calibration_fraction=0.25,
        seed=7,
        temperature_bounds=(0.05, 10.0),
        alpha_levels=[0.1],
    )

    roles = {role: set(group["subject_id"]) for role, group in manifest.groupby("role")}
    assert not roles["model_training"] & roles["calibration"]
    assert not roles["model_training"] & roles["outer_test"]
    assert not roles["calibration"] & roles["outer_test"]
    assert set(predictions["subject_id"]) == {9, 10}
    assert set(conformal["subject_id"]) == {9, 10}
    assert thresholds.loc[0, "n_calibration"] == 10


def test_subject_cluster_bootstrap_keeps_whole_subject_clusters() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat([1, 2, 3], 4),
            "value": np.repeat([0.0, 1.0, 2.0], 4),
        }
    )
    result = phase10.subject_cluster_bootstrap(
        frame,
        lambda data: float(data["value"].mean()),
        repetitions=100,
        seed=11,
        confidence_level=0.95,
    )
    assert result["estimate"] == 1.0
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
