from __future__ import annotations

import numpy as np
import pandas as pd

import src.models.run_subject_adaptation as phase9


FEATURES = [
    "eda_scl_delta",
    "eda_scl_slope_per_s",
    "eda_scr_amplitude_sum",
    "eda_scr_count",
    "emg_trapezius__tsfresh_linear_trend_slope",
]


def synthetic_table() -> pd.DataFrame:
    rows = []
    sample_id = 0
    for subject_id in range(1, 5):
        for class_id, class_name in ((1, "PA1"), (2, "PA2")):
            for trial in range(4):
                sample_id += 1
                row = {
                    "subject_id": subject_id,
                    "subject_name": f"subject_{subject_id}",
                    "sample_id": sample_id,
                    "sample_name": f"subject_{subject_id}-{class_name}-{trial:03d}",
                    "class_id": class_id,
                    "class_name": class_name,
                }
                shift = 0.3 * class_id + 0.1 * subject_id
                for feature_index, feature in enumerate(FEATURES):
                    row[feature] = shift + 0.01 * feature_index + 0.02 * trial
                rows.append(row)
    return pd.DataFrame(rows)


def synthetic_folds(table: pd.DataFrame) -> list[dict[str, object]]:
    folds = []
    for fold, held_out in ((1, {1, 2}), (2, {3, 4})):
        test_index = table.index[table["subject_id"].isin(held_out)].to_numpy(dtype=int)
        train_index = table.index[~table["subject_id"].isin(held_out)].to_numpy(dtype=int)
        folds.append({"fold": fold, "train_index": train_index, "test_index": test_index})
    return folds


def synthetic_config() -> dict[str, object]:
    return {
        "random_state": 7,
        "features": FEATURES,
        "pain_pairs": [["PA1", "PA2"]],
        "sample_efficiency": {"k_per_class": [0, 1], "repeats": 2},
        "models": {
            "population": {"C": 1.0, "max_iter": 1000},
            "individual": {"C": 0.1, "max_iter": 1000},
            "adaptation": {"l2_penalty": 10.0, "max_iter": 200},
        },
        "bootstrap": {"n_resamples": 50},
        "inference": {"practical_auc_gain": 0.02},
    }


def test_adaptation_split_is_deterministic_balanced_and_disjoint() -> None:
    subject = phase9.pair_table(synthetic_table(), 1, 2)
    subject = subject[subject["subject_id"] == 1].reset_index(drop=True)
    adaptation_a, evaluation_a = phase9.make_adaptation_split(subject, k=2, seed=123)
    adaptation_b, evaluation_b = phase9.make_adaptation_split(subject, k=2, seed=123)

    np.testing.assert_array_equal(adaptation_a, adaptation_b)
    np.testing.assert_array_equal(evaluation_a, evaluation_b)
    assert not set(adaptation_a).intersection(evaluation_a)
    assert set(adaptation_a).union(evaluation_a) == set(range(len(subject)))
    assert subject.iloc[adaptation_a].groupby("binary_label").size().to_dict() == {0: 2, 1: 2}


def test_outer_fold_validation_rejects_subject_overlap() -> None:
    pair = phase9.pair_table(synthetic_table(), 1, 2)
    train_idx = pair.index[pair["subject_id"].isin([1, 2])].to_numpy(dtype=int)
    test_idx = pair.index[pair["subject_id"].isin([2, 3])].to_numpy(dtype=int)
    try:
        phase9.validate_outer_fold(pair, train_idx, test_idx)
    except AssertionError as error:
        assert "subject leakage" in str(error)
    else:
        raise AssertionError("Expected subject-overlap safeguard to fail")


def test_methods_share_evaluation_trials_and_k_zero_is_population_only() -> None:
    table = synthetic_table()
    predictions, metrics, splits = phase9.run_sample_efficiency(
        table,
        synthetic_folds(table),
        synthetic_config(),
    )

    assert set(predictions.loc[predictions["k_per_class"] == 0, "method"]) == {"population_only"}
    nonzero = predictions[predictions["k_per_class"] == 1]
    group_keys = ["pair", "subject_id", "k_per_class", "repeat"]
    for _, group in nonzero.groupby(group_keys):
        samples_by_method = {
            method: set(method_rows["sample_name"])
            for method, method_rows in group.groupby("method")
        }
        assert set(samples_by_method) == set(phase9.METHODS)
        assert len({frozenset(samples) for samples in samples_by_method.values()}) == 1
    assert (splits.loc[splits["k_per_class"] == 1, "n_adaptation"] == 2).all()
    assert (splits.loc[splits["k_per_class"] == 1, "n_evaluation"] == 6).all()
    gains = phase9.paired_gain_table(metrics)
    assert len(gains) == 4 * 2
    assert gains[["adaptation_gain_auc", "individual_minus_population_auc"]].notna().all().all()


def test_residual_zero_coefficients_reproduce_population_scores() -> None:
    base_scores = np.array([0.1, 0.4, 0.8])
    features = np.arange(15, dtype=float).reshape(3, 5)
    corrected = phase9.apply_residual_correction(base_scores, features, np.zeros(6))
    np.testing.assert_allclose(corrected, base_scores)


def test_subject_cluster_bootstrap_averages_repeats_before_resampling() -> None:
    rows = []
    for subject_id, value in ((1, 0.2), (2, 0.6)):
        for repeat in range(4):
            rows.append({"pair": "PA1_vs_PA2", "k": 1, "subject_id": subject_id, "gain": value})
    frame = pd.DataFrame(rows)
    summary = phase9.subject_clustered_summary(frame, ["pair", "k"], "gain", 100, 5)
    assert summary.loc[0, "n_subjects"] == 2
    assert np.isclose(summary.loc[0, "subject_mean"], 0.4)

