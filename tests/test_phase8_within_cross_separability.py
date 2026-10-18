from __future__ import annotations

import numpy as np
import pandas as pd

import src.models.run_within_cross_separability as phase8


def synthetic_table(n_subjects: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for subject_id in range(1, n_subjects + 1):
        subject_name = f"subject_{subject_id:02d}"
        sample_id = (subject_id - 1) * 100
        for class_id, class_name in phase8.EXPECTED_LABEL_MAP.items():
            trial_start = 81 if class_id == 0 else 1 + (class_id - 1) * 20
            for offset in range(20):
                trial = trial_start + offset
                sample_id += 1
                row = {
                    "subject_id": subject_id,
                    "subject_name": subject_name,
                    "sample_id": sample_id,
                    "sample_name": f"{subject_name}-{class_name}-{trial:03d}",
                    "class_id": class_id,
                    "class_name": class_name,
                    "trial_number": trial,
                }
                for feature_pos, feature in enumerate(phase8.FEATURES):
                    row[feature] = class_id * (0.15 + feature_pos * 0.02) + subject_id * 0.1 + rng.normal(0, 1)
                rows.append(row)
    return pd.DataFrame(rows)


def synthetic_folds(table: pd.DataFrame) -> list[dict[str, object]]:
    folds = []
    all_indices = set(range(len(table)))
    for fold, subject_id in enumerate(sorted(table["subject_id"].unique()), start=1):
        test_index = table.index[table["subject_id"] == subject_id].to_numpy(dtype=int)
        train_index = np.asarray(sorted(all_indices.difference(test_index)), dtype=int)
        folds.append(
            {
                "fold": fold,
                "train_index": train_index,
                "test_index": test_index,
                "train_subjects": sorted(table.iloc[train_index]["subject_id"].unique()),
                "test_subjects": [subject_id],
            }
        )
    return folds


def test_trial_number_and_structure_validation() -> None:
    table = synthetic_table(2).drop(columns="trial_number")
    parsed = phase8.add_trial_number(table)
    phase8.validate_trial_structure(parsed)
    assert parsed.loc[parsed["class_id"] == 0, "trial_number"].min() == 81
    assert set(parsed.loc[(parsed["subject_id"] == 1) & (parsed["class_id"] > 0), "trial_number"]) == set(range(1, 81))


def test_effect_size_and_overlap_have_expected_limits() -> None:
    low = np.arange(20, dtype=float)
    high = low + 100.0
    assert phase8.hedges_g(low, high) > 10
    assert phase8.empirical_overlap(low, high) == 0.0
    assert phase8.hedges_g(low, low) == 0.0
    assert phase8.empirical_overlap(low, low) == 1.0


def test_within_subject_predictions_are_oof_and_deterministic(monkeypatch) -> None:
    table = synthetic_table(2)
    monkeypatch.setattr(phase8, "N_REPEATS", 2)
    monkeypatch.setattr(phase8, "N_BOOTSTRAPS", 20)

    metrics_a, predictions_a = phase8.run_within_subject(table)
    metrics_b, predictions_b = phase8.run_within_subject(table)

    assert len(metrics_a) == 2 * len(phase8.PAIRS)
    assert len(predictions_a) == 2 * len(phase8.PAIRS) * 40
    assert (predictions_a["n_oof_predictions"] == 2).all()
    assert not predictions_a[["pair", "sample_name"]].duplicated().any()
    np.testing.assert_allclose(predictions_a["oof_score"], predictions_b["oof_score"])
    np.testing.assert_allclose(metrics_a["roc_auc"], metrics_b["roc_auc"])


def test_cross_subject_predictions_reuse_grouped_folds_without_overlap(monkeypatch) -> None:
    table = synthetic_table(4)
    folds = synthetic_folds(table)
    monkeypatch.setattr(phase8, "N_BOOTSTRAPS", 20)

    metrics, predictions = phase8.run_cross_subject(table, folds)

    assert len(metrics) == 4 * len(phase8.PAIRS)
    assert len(predictions) == 4 * len(phase8.PAIRS) * 40
    assert not predictions[["pair", "sample_name"]].duplicated().any()
    assert predictions.groupby(["pair", "subject_id"])["fold"].nunique().eq(1).all()
    for split in folds:
        train_subjects = set(table.iloc[split["train_index"]]["subject_id"])
        test_subjects = set(table.iloc[split["test_index"]]["subject_id"])
        assert train_subjects.isdisjoint(test_subjects)


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    raw = pd.Series([0.04, 0.001, 0.02, 0.8])
    adjusted = phase8.benjamini_hochberg(raw)
    ordered = pd.DataFrame({"raw": raw, "adjusted": adjusted}).sort_values("raw")
    assert np.all(np.diff(ordered["adjusted"]) >= -1e-12)
    assert ((adjusted >= raw) & (adjusted <= 1)).all()

