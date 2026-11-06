from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.run_xgboost_diagnostic import CANDIDATES, make_model, validate_protocol


def test_diagnostic_grid_covers_capacity_learning_rate_and_regularization() -> None:
    assert {int(row["max_depth"]) for row in CANDIDATES}.issuperset({2, 3, 5})
    assert min(float(row["learning_rate"]) for row in CANDIDATES) <= 0.02
    assert max(float(row["learning_rate"]) for row in CANDIDATES) >= 0.10
    assert min(int(row["n_estimators"]) for row in CANDIDATES) <= 150
    assert max(int(row["n_estimators"]) for row in CANDIDATES) >= 600
    assert min(float(row["reg_lambda"]) for row in CANDIDATES) <= 1.0
    assert max(float(row["reg_lambda"]) for row in CANDIDATES) >= 10.0
    assert min(float(row["reg_alpha"]) for row in CANDIDATES) == 0.0
    assert max(float(row["reg_alpha"]) for row in CANDIDATES) >= 1.0


def test_xgboost_imputer_is_fit_on_training_rows_only() -> None:
    x_train = np.asarray([[1.0, np.nan], [3.0, 20.0], [5.0, 40.0], [7.0, 60.0], [9.0, 80.0]])
    y_train = np.asarray([0, 1, 2, 3, 4])
    model = make_model(CANDIDATES[0], random_state=7)
    model.fit(x_train, y_train)
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_, np.asarray([5.0, 50.0]))


def test_protocol_audit_accepts_valid_synthetic_subject_split() -> None:
    rows = []
    for subject_id in range(1, 6):
        for class_id, class_name in enumerate(["BL1", "PA1", "PA2", "PA3", "PA4"]):
            rows.append(
                {
                    "subject_id": subject_id,
                    "subject_name": f"S{subject_id}",
                    "sample_id": len(rows),
                    "sample_name": f"S{subject_id}_{class_name}",
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": float(subject_id + class_id),
                }
            )
    table = pd.DataFrame(rows)
    folds = []
    for fold, subject_id in enumerate(range(1, 6), start=1):
        test_index = table.index[table["subject_id"] == subject_id].to_numpy()
        train_index = table.index[table["subject_id"] != subject_id].to_numpy()
        folds.append(
            {
                "fold": fold,
                "train_index": train_index,
                "test_index": test_index,
                "train_subjects": sorted(table.iloc[train_index]["subject_id"].unique()),
                "test_subjects": [subject_id],
            }
        )
    checks = validate_protocol(table, folds, ["feature"])
    assert checks["outer_subject_overlap"] == []
    assert checks["feature_metadata_columns_in_model"] == []
    assert checks["label_map_matches_expected"]
