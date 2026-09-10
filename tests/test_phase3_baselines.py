from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.splits import make_subject_grouped_splits
from src.models.baselines import make_baseline_model


def test_subject_grouped_splits_have_no_subject_overlap() -> None:
    rows = []
    for subject_id in range(1, 11):
        for class_id in range(5):
            for trial in range(2):
                rows.append({"subject_id": subject_id, "class_id": class_id, "trial": trial})
    table = pd.DataFrame(rows)
    folds, protocol, _ = make_subject_grouped_splits(
        y=table["class_id"].to_numpy(),
        groups=table["subject_id"].to_numpy(),
        n_splits=5,
        random_state=20260909,
    )
    assert protocol == "StratifiedGroupKFold"
    for split in folds:
        assert set(split.train_subjects).isdisjoint(split.test_subjects)
        assert len(split.test_subjects) == 2


def test_preprocessing_is_fit_only_on_training_fold() -> None:
    x_train = np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    y_train = np.asarray([0, 1, 2])
    x_test = np.asarray([[1001.0, 10010.0], [1003.0, 10030.0]])

    model = make_baseline_model("logistic_regression", random_state=20260909)
    model.fit(x_train, y_train)

    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(imputer.statistics_, np.asarray([3.0, 30.0]))
    np.testing.assert_allclose(scaler.mean_, np.asarray([3.0, 30.0]))
    assert not np.allclose(scaler.mean_, np.vstack([x_train, x_test]).mean(axis=0))
