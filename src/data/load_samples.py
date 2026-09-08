from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_LABEL_MAP = {
    0: "BL1",
    1: "PA1",
    2: "PA2",
    3: "PA3",
    4: "PA4",
}


def load_samples(path: Path) -> pd.DataFrame:
    samples = pd.read_csv(path, sep="\t")
    required = {
        "subject_id",
        "subject_name",
        "class_id",
        "class_name",
        "sample_id",
        "sample_name",
    }
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing sample columns: {sorted(missing)}")

    observed = (
        samples[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    if observed != EXPECTED_LABEL_MAP:
        raise ValueError(f"Unexpected label mapping: {observed}")

    if samples["sample_name"].duplicated().any():
        raise ValueError("Duplicate sample_name values found")
    if samples["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found")

    return samples.sort_values("sample_id").reset_index(drop=True)


def signal_path(source_dir: Path, subject_name: str, sample_name: str) -> Path:
    return source_dir / subject_name / f"{sample_name}_bio.csv"

