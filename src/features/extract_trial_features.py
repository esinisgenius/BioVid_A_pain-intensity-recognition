from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
from tsfresh.feature_extraction import feature_calculators as fc

from src.data.load_samples import load_samples, signal_path
from src.features.feature_definitions import feature_metadata_records


SAMPLING_RATE_HZ = 512
EXPECTED_ROWS = 2816
MODEL_LEAKAGE_COLUMNS = {
    "subject_id",
    "subject_name",
    "sample_id",
    "sample_name",
    "class_id",
    "class_name",
    "trial_number",
    "file_path",
    "source_path",
}


def finite_or_zero(value: float) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def slope_over_seconds(y: np.ndarray, sampling_rate_hz: int = SAMPLING_RATE_HZ) -> float:
    if len(y) < 2:
        return 0.0
    t = np.arange(len(y), dtype=float) / sampling_rate_hz
    return float(np.polyfit(t, y.astype(float), 1)[0])


def ecg_bandpass_5_20(ecg: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [5.0, 20.0], btype="bandpass", fs=SAMPLING_RATE_HZ, output="sos")
    return signal.sosfiltfilt(sos, ecg.astype(float) - np.median(ecg.astype(float)))


def rate_from_rpeaks(rpeaks: np.ndarray, n_samples: int) -> np.ndarray:
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 3:
        return np.zeros(n_samples, dtype=float)
    rr_s = np.diff(rpeaks) / SAMPLING_RATE_HZ
    valid = rr_s > 0
    if valid.sum() < 2:
        return np.zeros(n_samples, dtype=float)
    hr = 60.0 / rr_s[valid]
    midpoints = ((rpeaks[:-1] + rpeaks[1:]) / 2.0)[valid]
    sample_index = np.arange(n_samples, dtype=float)
    return np.interp(sample_index, midpoints, hr, left=hr[0], right=hr[-1])


def skewness(x: np.ndarray) -> float:
    if np.unique(x).size <= 1:
        return 0.0
    return finite_or_zero(stats.skew(x, bias=False))


def kurtosis(x: np.ndarray) -> float:
    if np.unique(x).size <= 1:
        return 0.0
    return finite_or_zero(stats.kurtosis(x, fisher=True, bias=False))


def extract_ecg_features(ecg: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    diagnostics: dict[str, object] = {
        "ecg_processing_success": True,
        "ecg_error": "",
        "ecg_beat_pipeline": "median_center_plus_5_20hz_bandpass_plus_neurokit_ecg_peaks",
    }
    try:
        clean = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE_HZ, method="neurokit")
    except Exception:
        clean = ecg.astype(float)

    try:
        beat_signal = ecg_bandpass_5_20(ecg)
        _, info = nk.ecg_peaks(
            beat_signal,
            sampling_rate=SAMPLING_RATE_HZ,
            method="neurokit",
            correct_artifacts=False,
        )
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        rate = rate_from_rpeaks(rpeaks, len(ecg))
    except Exception as exc:
        diagnostics["ecg_processing_success"] = False
        diagnostics["ecg_error"] = f"{type(exc).__name__}: {exc}"
        rate = np.zeros_like(clean)
        rpeaks = np.asarray([], dtype=int)

    try:
        quality = nk.ecg_quality(clean, rpeaks=rpeaks, sampling_rate=SAMPLING_RATE_HZ)
        quality = np.asarray(quality, dtype=float)
    except Exception as exc:
        quality = np.zeros_like(clean)
        diagnostics["ecg_quality_success"] = False
        diagnostics["ecg_quality_error"] = f"{type(exc).__name__}: {exc}"
    else:
        diagnostics["ecg_quality_success"] = True
        diagnostics["ecg_quality_error"] = ""

    rr_ms = np.diff(rpeaks) / SAMPLING_RATE_HZ * 1000.0 if len(rpeaks) >= 2 else np.asarray([], dtype=float)
    enough_rpeaks = len(rpeaks) >= 3
    diagnostics["ecg_rpeak_count"] = int(len(rpeaks))
    diagnostics["ecg_rpeak_detection_success"] = bool(enough_rpeaks)

    features = {
        "ecg_rpeak_count": float(len(rpeaks)),
        "ecg_rate_mean_bpm": finite_or_zero(np.nanmean(rate)) if enough_rpeaks else 0.0,
        "ecg_rate_median_bpm": finite_or_zero(np.nanmedian(rate)) if enough_rpeaks else 0.0,
        "ecg_rate_min_bpm": finite_or_zero(np.nanmin(rate)) if enough_rpeaks else 0.0,
        "ecg_rate_max_bpm": finite_or_zero(np.nanmax(rate)) if enough_rpeaks else 0.0,
        "ecg_rate_range_bpm": finite_or_zero(np.nanmax(rate) - np.nanmin(rate)) if enough_rpeaks else 0.0,
        "ecg_rate_slope_bpm_per_s": finite_or_zero(slope_over_seconds(np.nan_to_num(rate))) if enough_rpeaks else 0.0,
        "ecg_rr_mean_ms": finite_or_zero(np.mean(rr_ms)) if rr_ms.size >= 2 else 0.0,
        "ecg_rr_median_ms": finite_or_zero(np.median(rr_ms)) if rr_ms.size >= 2 else 0.0,
        "ecg_quality_mean": finite_or_zero(np.nanmean(quality)),
        "ecg_clean_std": finite_or_zero(np.nanstd(clean, ddof=1)),
        "ecg_clean_rms": finite_or_zero(rms(np.nan_to_num(clean))),
        "ecg_clean_peak_to_peak": finite_or_zero(np.nanmax(clean) - np.nanmin(clean)),
    }
    return features, diagnostics


def extract_eda_features(eda: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    diagnostics: dict[str, object] = {"eda_processing_success": True, "eda_error": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            signals, info = nk.eda_process(eda, sampling_rate=SAMPLING_RATE_HZ)
        amplitudes = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
        amplitudes = amplitudes[np.isfinite(amplitudes)]
        amplitude_threshold = max(0.01, 0.01 * float(np.max(eda) - np.min(eda)))
        amplitudes = amplitudes[amplitudes >= amplitude_threshold]
        scr_count = int(amplitudes.size)
    except Exception as exc:
        diagnostics["eda_processing_success"] = False
        diagnostics["eda_error"] = f"{type(exc).__name__}: {exc}"
        amplitudes = np.asarray([], dtype=float)
        scr_count = 0
        amplitude_threshold = max(0.01, 0.01 * float(np.max(eda) - np.min(eda)))

    diagnostics["eda_scr_count"] = scr_count
    diagnostics["eda_scr_detection_success"] = bool(diagnostics["eda_processing_success"])
    diagnostics["eda_scr_nonzero"] = bool(scr_count > 0)
    diagnostics["eda_scr_amplitude_threshold"] = amplitude_threshold

    t = np.arange(len(eda), dtype=float) / SAMPLING_RATE_HZ
    features = {
        "eda_scl_mean": finite_or_zero(np.mean(eda)),
        "eda_scl_median": finite_or_zero(np.median(eda)),
        "eda_scl_std": finite_or_zero(np.std(eda, ddof=1)),
        "eda_scl_min": finite_or_zero(np.min(eda)),
        "eda_scl_max": finite_or_zero(np.max(eda)),
        "eda_scl_range": finite_or_zero(np.max(eda) - np.min(eda)),
        "eda_scl_auc": finite_or_zero(np.trapezoid(eda, t)),
        "eda_scl_slope_per_s": finite_or_zero(slope_over_seconds(eda)),
        "eda_scl_delta": finite_or_zero(eda[-1] - eda[0]),
        "eda_scr_count": float(scr_count),
        "eda_scr_amplitude_max": finite_or_zero(np.max(amplitudes)) if amplitudes.size else 0.0,
        "eda_scr_amplitude_sum": finite_or_zero(np.sum(amplitudes)) if amplitudes.size else 0.0,
    }
    return features, diagnostics


def emg_activation_features(emg: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    rectified = np.abs(emg.astype(float))
    window = max(1, int(round(0.05 * SAMPLING_RATE_HZ)))
    kernel = np.ones(window, dtype=float) / window
    envelope = np.convolve(rectified, kernel, mode="same")

    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    threshold = median + 3.0 * mad
    min_duration_samples = int(round(0.05 * SAMPLING_RATE_HZ))
    active = envelope > threshold

    segments: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_duration_samples:
                segments.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_duration_samples:
        segments.append((start, len(active)))

    active_clean = np.zeros_like(active, dtype=bool)
    for start, end in segments:
        active_clean[start:end] = True
    durations = np.asarray([(end - start) / SAMPLING_RATE_HZ for start, end in segments], dtype=float)

    features = {
        "emg_trapezius_activation_fraction": finite_or_zero(np.mean(active_clean)),
        "emg_trapezius_activation_count": float(len(segments)),
        "emg_trapezius_activation_mean_duration_s": finite_or_zero(np.mean(durations)) if durations.size else 0.0,
    }
    diagnostics = {
        "emg_activation_threshold_method": "per-window median(envelope)+3*MAD(envelope); no labels or dataset-wide statistics",
        "emg_activation_threshold": threshold,
        "emg_activation_segment_count": len(segments),
    }
    return features, diagnostics


def extract_emg_features(emg: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    abs_emg = np.abs(emg)
    deadband = max(float(np.median(abs_emg) + 3.0 * np.median(np.abs(abs_emg - np.median(abs_emg)))), 1e-12)
    signs = np.sign(np.where(abs_emg <= deadband, 0.0, emg))
    nonzero = signs != 0
    zero_crossings = 0
    if np.any(nonzero):
        compact = signs[nonzero]
        zero_crossings = int(np.sum(compact[1:] * compact[:-1] < 0))

    features = {
        "emg_trapezius_mav": finite_or_zero(np.mean(abs_emg)),
        "emg_trapezius_rms": finite_or_zero(rms(emg)),
        "emg_trapezius_std": finite_or_zero(np.std(emg, ddof=1)),
        "emg_trapezius_iemg": finite_or_zero(np.sum(abs_emg) / SAMPLING_RATE_HZ),
        "emg_trapezius_energy": finite_or_zero(np.sum(np.square(emg))),
        "emg_trapezius_peak_to_peak": finite_or_zero(np.max(emg) - np.min(emg)),
        "emg_trapezius_waveform_length": finite_or_zero(np.sum(np.abs(np.diff(emg)))),
        "emg_trapezius_zero_crossings": float(zero_crossings),
    }
    activation, diagnostics = emg_activation_features(emg)
    features.update(activation)
    diagnostics["emg_zero_crossing_deadband"] = deadband
    return features, diagnostics


def tsfresh_features(prefix: str, x: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}__tsfresh_mean": finite_or_zero(fc.mean(x)),
        f"{prefix}__tsfresh_median": finite_or_zero(fc.median(x)),
        f"{prefix}__tsfresh_standard_deviation": finite_or_zero(fc.standard_deviation(x)),
        f"{prefix}__tsfresh_minimum": finite_or_zero(fc.minimum(x)),
        f"{prefix}__tsfresh_maximum": finite_or_zero(fc.maximum(x)),
        f"{prefix}__tsfresh_root_mean_square": finite_or_zero(fc.root_mean_square(x)),
        f"{prefix}__tsfresh_abs_energy": finite_or_zero(fc.abs_energy(x)),
        f"{prefix}__tsfresh_absolute_sum_of_changes": finite_or_zero(fc.absolute_sum_of_changes(x)),
        f"{prefix}__tsfresh_mean_change": finite_or_zero(fc.mean_change(x)),
        f"{prefix}__tsfresh_linear_trend_slope": finite_or_zero(dict(fc.linear_trend(x, [{"attr": "slope"}]))["attr_\"slope\""]),
        f"{prefix}__tsfresh_skewness": skewness(x),
        f"{prefix}__tsfresh_kurtosis": kurtosis(x),
    }


def extract_one(row: pd.Series, source_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    path = signal_path(source_dir, row["subject_name"], row["sample_name"])
    df = pd.read_csv(path, sep="\t")
    expected_columns = ["time", "gsr", "ecg", "emg_trapezius", "emg_corrugator", "emg_zygomaticus"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected columns for {path}: {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count for {path}: {len(df)}")

    record: dict[str, object] = {
        "subject_id": int(row["subject_id"]),
        "subject_name": row["subject_name"],
        "sample_id": int(row["sample_id"]),
        "sample_name": row["sample_name"],
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "source_dir": source_dir.name,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "n_samples": len(df),
        "duration_s": float((df["time"].iloc[-1] - df["time"].iloc[0]) / 1_000_000),
    }
    diagnostics: dict[str, object] = {
        "sample_name": row["sample_name"],
        "subject_id": int(row["subject_id"]),
        "class_id": int(row["class_id"]),
    }

    ecg = df["ecg"].to_numpy(dtype=float)
    eda = df["gsr"].to_numpy(dtype=float)
    emg = df["emg_trapezius"].to_numpy(dtype=float)

    ecg_features, ecg_diagnostics = extract_ecg_features(ecg)
    eda_features, eda_diagnostics = extract_eda_features(eda)
    emg_features, emg_diagnostics = extract_emg_features(emg)
    generic = {}
    generic.update(tsfresh_features("ecg", ecg))
    generic.update(tsfresh_features("eda", eda))
    generic.update(tsfresh_features("emg_trapezius", emg))

    record.update(ecg_features)
    record.update(eda_features)
    record.update(emg_features)
    record.update(generic)
    diagnostics.update(ecg_diagnostics)
    diagnostics.update(eda_diagnostics)
    diagnostics.update(emg_diagnostics)
    return record, diagnostics


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in MODEL_LEAKAGE_COLUMNS and c not in {"source_dir", "sampling_rate_hz", "n_samples", "duration_s"}]


def compute_feature_stats(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    n = len(table)
    for name in features:
        values = pd.to_numeric(table[name], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        finite_values = values[finite]
        missing_prop = float(1.0 - finite.sum() / n)
        std = float(finite_values.std(ddof=1)) if len(finite_values) > 1 else 0.0
        mean = float(finite_values.mean()) if len(finite_values) else 0.0
        min_value = float(finite_values.min()) if len(finite_values) else 0.0
        max_value = float(finite_values.max()) if len(finite_values) else 0.0
        zero_prop = float((finite_values == 0).sum() / len(finite_values)) if len(finite_values) else 1.0
        unique_count = int(finite_values.nunique(dropna=True))
        near_constant = bool(unique_count <= 1 or std <= 1e-12 or zero_prop >= 0.995)
        rows.append(
            {
                "feature_name": name,
                "mean": mean,
                "std": std,
                "min": min_value,
                "max": max_value,
                "zero_proportion": zero_prop,
                "missing_proportion": missing_prop,
                "unique_value_count": unique_count,
                "near_constant_warning": near_constant,
            }
        )
    return pd.DataFrame(rows)


def apply_degeneracy_policy(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drop_reasons = []
    drop_features = set()
    for row in stats_df.itertuples(index=False):
        if row.missing_proportion > 0:
            drop_features.add(row.feature_name)
            drop_reasons.append((row.feature_name, "non-finite or missing values present"))
        elif row.unique_value_count <= 1:
            drop_features.add(row.feature_name)
            drop_reasons.append((row.feature_name, "constant feature"))
        elif row.zero_proportion >= 0.995:
            drop_features.add(row.feature_name)
            drop_reasons.append((row.feature_name, "zero proportion >= 0.995"))
        elif row.std <= 1e-12:
            drop_features.add(row.feature_name)
            drop_reasons.append((row.feature_name, "near-zero standard deviation"))

    metadata = metadata.copy()
    metadata["retained"] = ~metadata["feature_name"].isin(drop_features)
    reason_map = {name: reason for name, reason in drop_reasons}
    metadata["exclusion_reason"] = metadata["feature_name"].map(reason_map).fillna("")

    metadata_features = set(metadata["feature_name"])
    metadata = metadata[metadata["feature_name"].isin(table.columns)].reset_index(drop=True)
    final_columns = [c for c in table.columns if c not in drop_features]
    dropped = pd.DataFrame(drop_reasons, columns=["feature_name", "reason"])
    missing_metadata = sorted(set(feature_columns(table)).difference(metadata_features))
    if missing_metadata:
        raise ValueError(f"Feature columns missing metadata: {missing_metadata}")
    return table[final_columns].copy(), metadata, dropped


def validate_outputs(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    samples: pd.DataFrame,
    expected_rows: int = 8700,
) -> None:
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(table)}")
    if table["sample_name"].duplicated().any():
        raise ValueError("Duplicate sample_name rows in feature table")
    merged = table[["sample_name", "subject_id", "class_id", "class_name"]].merge(
        samples[["sample_name", "subject_id", "class_id", "class_name"]],
        on="sample_name",
        suffixes=("_features", "_samples"),
    )
    if len(merged) != len(samples):
        raise ValueError("Feature table does not match all samples")
    for col in ["subject_id", "class_id", "class_name"]:
        if not (merged[f"{col}_features"] == merged[f"{col}_samples"]).all():
            raise ValueError(f"Mismatch in {col}")

    forbidden_substrings = ["temperature", "emg_corrugator", "emg_zygomaticus", "trial_number", "file_path"]
    for name in metadata["feature_name"]:
        if any(forbidden in name for forbidden in forbidden_substrings):
            raise ValueError(f"Forbidden feature found: {name}")

    for col in feature_columns(table):
        if col in MODEL_LEAKAGE_COLUMNS:
            raise ValueError(f"Leakage column included as feature: {col}")
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in feature column {col}")

    if metadata["feature_name"].duplicated().any():
        raise ValueError("Duplicate feature metadata rows")
    feature_set = set(feature_columns(table))
    metadata_set = set(metadata.loc[metadata["retained"], "feature_name"])
    if feature_set != metadata_set:
        raise ValueError(f"Retained metadata/features mismatch: {sorted(feature_set ^ metadata_set)[:10]}")


def run_extraction(root: Path, limit: int | None = None) -> None:
    source_dir = root / "biosignals_filtered"
    artifacts = root / "artifacts"
    reports = root / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    samples = load_samples(root / "starting_point" / "samples.csv")
    if limit is not None:
        samples = samples.head(limit).copy()

    records = []
    diagnostics = []
    for idx, row in samples.iterrows():
        record, diagnostic = extract_one(row, source_dir)
        records.append(record)
        diagnostics.append(diagnostic)
        if (idx + 1) % 500 == 0:
            print(f"extracted {idx + 1} samples", flush=True)

    all_table = pd.DataFrame(records)
    metadata = pd.DataFrame(feature_metadata_records())
    stats_all = compute_feature_stats(all_table, feature_columns(all_table))
    final_table, final_metadata, dropped = apply_degeneracy_policy(all_table, metadata, stats_all)
    stats_final = compute_feature_stats(final_table, feature_columns(final_table))

    expected_rows = 8700 if limit is None else len(samples)
    validate_outputs(
        final_table,
        final_metadata,
        load_samples(root / "starting_point" / "samples.csv") if limit is None else samples,
        expected_rows=expected_rows,
    )

    all_table.to_parquet(artifacts / "feature_table_filtered_all_candidates.parquet", index=False)
    final_table.to_parquet(artifacts / "feature_table_filtered.parquet", index=False)
    final_table.to_csv(artifacts / "feature_table_filtered.csv", index=False)
    final_metadata.to_json(artifacts / "feature_metadata.json", orient="records", indent=2)
    stats_all.to_csv(artifacts / "feature_stats_all_candidates.csv", index=False)
    stats_final.to_csv(artifacts / "feature_stats_filtered.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(artifacts / "feature_extraction_diagnostics.csv", index=False)
    dropped.to_csv(artifacts / "feature_exclusions.csv", index=False)

    summary = build_summary(final_table, final_metadata, stats_all, dropped, pd.DataFrame(diagnostics))
    summary["feature_table_csv_sha256"] = hashlib.sha256(
        (artifacts / "feature_table_filtered.csv").read_bytes()
    ).hexdigest()
    (artifacts / "feature_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_summary(
    final_table: pd.DataFrame,
    metadata: pd.DataFrame,
    stats_all: pd.DataFrame,
    dropped: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, object]:
    retained = metadata[metadata["retained"]].copy()
    scr_cols = ["eda_scr_count", "eda_scr_amplitude_max", "eda_scr_amplitude_sum"]
    scr_stats = stats_all[stats_all["feature_name"].isin(scr_cols)].to_dict(orient="records")
    summary = {
        "n_rows": int(len(final_table)),
        "n_retained_features": int(len(feature_columns(final_table))),
        "retained_by_modality": retained.groupby("modality")["feature_name"].count().to_dict(),
        "retained_by_feature_set": retained.groupby("feature_set")["feature_name"].count().to_dict(),
        "retained_by_provenance": retained.groupby("provenance")["feature_name"].count().to_dict(),
        "dropped_features": dropped.to_dict(orient="records"),
        "eda_scr_detection_success_rate": float(diagnostics["eda_scr_detection_success"].mean()),
        "eda_scr_nonzero_trial_rate": float(diagnostics["eda_scr_nonzero"].mean()),
        "eda_scr_stats": scr_stats,
        "ecg_processing_success_rate": float(diagnostics["ecg_processing_success"].mean()),
        "ecg_rpeak_detection_success_rate": float(diagnostics["ecg_rpeak_detection_success"].mean()),
        "ecg_quality_success_rate": float(diagnostics["ecg_quality_success"].mean()) if "ecg_quality_success" in diagnostics else None,
        "ecg_rpeak_count_min": int(diagnostics["ecg_rpeak_count"].min()),
        "ecg_rpeak_count_max": int(diagnostics["ecg_rpeak_count"].max()),
        "emg_activation_threshold": "per-window smoothed rectified amplitude > median(envelope)+3*MAD(envelope), with 50 ms smoothing and minimum 50 ms segment duration; no labels or dataset-wide statistics",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_extraction(args.root.resolve(), args.limit)


if __name__ == "__main__":
    main()
