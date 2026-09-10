from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]


def make_subject_grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
    prefer_stratified: bool = True,
) -> tuple[list[FoldSplit], str, str]:
    """Create subject-independent folds and verify that train/test subjects never overlap."""
    if prefer_stratified:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        protocol = "StratifiedGroupKFold"
        reason = "Used because every subject has all five classes with identical class counts."
        raw_splits = splitter.split(np.zeros_like(y), y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        protocol = "GroupKFold"
        reason = "Used because stratified grouped splitting was disabled."
        raw_splits = splitter.split(np.zeros_like(y), y, groups)

    folds: list[FoldSplit] = []
    for fold, (train_index, test_index) in enumerate(raw_splits, start=1):
        train_subjects = tuple(sorted(pd.unique(groups[train_index]).astype(int).tolist()))
        test_subjects = tuple(sorted(pd.unique(groups[test_index]).astype(int).tolist()))
        overlap = set(train_subjects).intersection(test_subjects)
        if overlap:
            raise ValueError(f"Subject leakage in fold {fold}: {sorted(overlap)}")
        folds.append(
            FoldSplit(
                fold=fold,
                train_index=np.asarray(train_index),
                test_index=np.asarray(test_index),
                train_subjects=train_subjects,
                test_subjects=test_subjects,
            )
        )

    return folds, protocol, reason


def fold_assignment_table(table: pd.DataFrame, folds: list[FoldSplit]) -> pd.DataFrame:
    rows = []
    for split in folds:
        for idx in split.test_index:
            source = table.iloc[int(idx)]
            rows.append(
                {
                    "fold": split.fold,
                    "subject_id": int(source["subject_id"]),
                    "subject_name": source["subject_name"],
                    "sample_id": int(source["sample_id"]),
                    "sample_name": source["sample_name"],
                    "class_id": int(source["class_id"]),
                    "class_name": source["class_name"],
                }
            )
    return pd.DataFrame(rows).sort_values(["fold", "subject_id", "sample_id"]).reset_index(drop=True)
