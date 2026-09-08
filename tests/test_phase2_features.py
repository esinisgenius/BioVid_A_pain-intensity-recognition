from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_samples import EXPECTED_LABEL_MAP, load_samples
from src.features.extract_trial_features import MODEL_LEAKAGE_COLUMNS, feature_columns


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def table() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "feature_table_filtered.parquet")


def metadata() -> pd.DataFrame:
    return pd.read_json(ARTIFACTS / "feature_metadata.json")


def test_label_mapping_and_row_count() -> None:
    samples = load_samples(ROOT / "starting_point" / "samples.csv")
    features = table()
    assert len(features) == 8700
    assert len(samples) == 8700
    observed = (
        samples[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    assert observed == EXPECTED_LABEL_MAP


def test_subject_sample_mapping_matches_samples_csv() -> None:
    samples = load_samples(ROOT / "starting_point" / "samples.csv")
    features = table()
    assert not features["sample_name"].duplicated().any()
    merged = features[["sample_name", "subject_id", "class_id", "class_name"]].merge(
        samples[["sample_name", "subject_id", "class_id", "class_name"]],
        on="sample_name",
        suffixes=("_features", "_samples"),
    )
    assert len(merged) == 8700
    for col in ["subject_id", "class_id", "class_name"]:
        assert (merged[f"{col}_features"] == merged[f"{col}_samples"]).all()


def test_no_forbidden_channels_or_leakage_features() -> None:
    features = table()
    meta = metadata()
    forbidden_terms = [
        "temperature",
        "emg_corrugator",
        "emg_zygomaticus",
        "trial_number",
        "file_path",
        "source_path",
    ]
    model_features = feature_columns(features)
    for col in model_features:
        assert col not in MODEL_LEAKAGE_COLUMNS
        assert not any(term in col for term in forbidden_terms)
    for name in meta["feature_name"]:
        assert not any(term in name for term in forbidden_terms)


def test_all_feature_values_are_finite_and_metadata_aligned() -> None:
    features = table()
    meta = metadata()
    model_features = feature_columns(features)
    values = features[model_features].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert not meta["feature_name"].duplicated().any()
    retained_metadata = set(meta.loc[meta["retained"], "feature_name"])
    assert set(model_features) == retained_metadata


def test_reproducibility_hash_exists_and_matches() -> None:
    summary = json.loads((ARTIFACTS / "feature_pipeline_summary.json").read_text())
    path = ARTIFACTS / "feature_table_filtered.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["feature_table_csv_sha256"] == digest

