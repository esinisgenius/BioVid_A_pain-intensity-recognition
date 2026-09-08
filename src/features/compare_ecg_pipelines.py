from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal

from src.data.load_samples import load_samples, signal_path
from src.features.extract_trial_features import SAMPLING_RATE_HZ, slope_over_seconds


BEAT_FEATURES = [
    "rpeak_count",
    "rate_mean_bpm",
    "rate_median_bpm",
    "rate_min_bpm",
    "rate_max_bpm",
    "rate_range_bpm",
    "rate_slope_bpm_per_s",
    "rr_mean_ms",
    "rr_median_ms",
]


def finite_or_zero(value: float) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def bandpass_5_20(ecg: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [5.0, 20.0], btype="bandpass", fs=SAMPLING_RATE_HZ, output="sos")
    return signal.sosfiltfilt(sos, ecg - np.median(ecg))


def features_from_peaks(peaks: np.ndarray, n_samples: int) -> dict[str, float]:
    peaks = np.asarray(peaks, dtype=int)
    enough = len(peaks) >= 3
    if enough:
        rr_s = np.diff(peaks) / SAMPLING_RATE_HZ
        valid = rr_s > 0
        if valid.sum() >= 2:
            hr = 60.0 / rr_s[valid]
            midpoints = ((peaks[:-1] + peaks[1:]) / 2.0)[valid]
            rate = np.interp(np.arange(n_samples, dtype=float), midpoints, hr, left=hr[0], right=hr[-1])
        else:
            rate = np.zeros(n_samples, dtype=float)
    else:
        rate = np.zeros(n_samples, dtype=float)
    rr_ms = np.diff(peaks) / SAMPLING_RATE_HZ * 1000.0 if len(peaks) >= 2 else np.asarray([], dtype=float)
    return {
        "rpeak_count": float(len(peaks)),
        "rate_mean_bpm": finite_or_zero(np.nanmean(rate)) if enough else 0.0,
        "rate_median_bpm": finite_or_zero(np.nanmedian(rate)) if enough else 0.0,
        "rate_min_bpm": finite_or_zero(np.nanmin(rate)) if enough else 0.0,
        "rate_max_bpm": finite_or_zero(np.nanmax(rate)) if enough else 0.0,
        "rate_range_bpm": finite_or_zero(np.nanmax(rate) - np.nanmin(rate)) if enough else 0.0,
        "rate_slope_bpm_per_s": finite_or_zero(slope_over_seconds(np.nan_to_num(rate))) if enough else 0.0,
        "rr_mean_ms": finite_or_zero(np.mean(rr_ms)) if rr_ms.size >= 2 else 0.0,
        "rr_median_ms": finite_or_zero(np.median(rr_ms)) if rr_ms.size >= 2 else 0.0,
    }


def plausibility_flags(features: dict[str, float]) -> dict[str, bool]:
    count = features["rpeak_count"]
    hr_mean = features["rate_mean_bpm"]
    hr_min = features["rate_min_bpm"]
    hr_max = features["rate_max_bpm"]
    rr_mean = features["rr_mean_ms"]
    rr_median = features["rr_median_ms"]
    return {
        "success": bool(count >= 3),
        "hr_mean_plausible": bool(count >= 3 and 35.0 <= hr_mean <= 160.0),
        "hr_min_plausible": bool(count >= 3 and 25.0 <= hr_min <= 180.0),
        "hr_max_plausible": bool(count >= 3 and 35.0 <= hr_max <= 220.0),
        "rr_mean_plausible": bool(count >= 3 and 350.0 <= rr_mean <= 2000.0),
        "rr_median_plausible": bool(count >= 3 and 350.0 <= rr_median <= 2000.0),
    }


def run_a(ecg: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals, info = nk.ecg_process(ecg, sampling_rate=SAMPLING_RATE_HZ)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        features = features_from_peaks(peaks, len(ecg))
        quality_mean = finite_or_zero(np.nanmean(signals["ECG_Quality"].to_numpy(dtype=float)))
        error = ""
    except Exception as exc:
        features = features_from_peaks(np.asarray([], dtype=int), len(ecg))
        quality_mean = 0.0
        error = f"{type(exc).__name__}: {exc}"
    flags = plausibility_flags(features)
    return features, {"error": error, "quality_mean": quality_mean, **flags}


def run_b(ecg: np.ndarray) -> tuple[dict[str, float], dict[str, object], np.ndarray]:
    filtered = bandpass_5_20(ecg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_peaks(filtered, sampling_rate=SAMPLING_RATE_HZ, method="neurokit", correct_artifacts=False)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        features = features_from_peaks(peaks, len(ecg))
        error = ""
    except Exception as exc:
        features = features_from_peaks(np.asarray([], dtype=int), len(ecg))
        error = f"{type(exc).__name__}: {exc}"
    flags = plausibility_flags(features)
    return features, {"error": error, "quality_mean": 0.0, **flags}, filtered


def waveform_summary(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_std": finite_or_zero(np.std(x, ddof=1)),
        f"{prefix}_rms": finite_or_zero(np.sqrt(np.mean(np.square(x)))),
        f"{prefix}_peak_to_peak": finite_or_zero(np.max(x) - np.min(x)),
    }


def corr_records(df: pd.DataFrame, prefix_a: str, prefix_b: str, features: list[str]) -> list[dict[str, object]]:
    rows = []
    both = df["a_success"] & df["b_success"]
    for name in features:
        a = f"{prefix_a}_{name}"
        b = f"{prefix_b}_{name}"
        if both.sum() >= 3:
            corr = df.loc[both, [a, b]].corr().iloc[0, 1]
        else:
            corr = np.nan
        rows.append({"feature": name, "pearson_r_success_both": finite_or_zero(corr)})
    return rows


def compare(root: Path) -> None:
    samples = load_samples(root / "starting_point" / "samples.csv")
    rows = []
    for idx, row in samples.iterrows():
        path = signal_path(root / "biosignals_filtered", row["subject_name"], row["sample_name"])
        df = pd.read_csv(path, sep="\t")
        ecg = df["ecg"].to_numpy(dtype=float)
        a_features, a_diag = run_a(ecg)
        b_features, b_diag, b_signal = run_b(ecg)
        record: dict[str, object] = {
            "subject_id": int(row["subject_id"]),
            "subject_name": row["subject_name"],
            "sample_id": int(row["sample_id"]),
            "sample_name": row["sample_name"],
            "class_id": int(row["class_id"]),
            "class_name": row["class_name"],
        }
        for name, value in a_features.items():
            record[f"a_{name}"] = value
        for name, value in b_features.items():
            record[f"b_{name}"] = value
        record.update(
            {
                "a_success": a_diag["success"],
                "b_success": b_diag["success"],
                "a_error": a_diag["error"],
                "b_error": b_diag["error"],
                "a_quality_mean": a_diag["quality_mean"],
                "a_hr_mean_plausible": a_diag["hr_mean_plausible"],
                "b_hr_mean_plausible": b_diag["hr_mean_plausible"],
                "a_hr_min_plausible": a_diag["hr_min_plausible"],
                "b_hr_min_plausible": b_diag["hr_min_plausible"],
                "a_hr_max_plausible": a_diag["hr_max_plausible"],
                "b_hr_max_plausible": b_diag["hr_max_plausible"],
                "a_rr_mean_plausible": a_diag["rr_mean_plausible"],
                "b_rr_mean_plausible": b_diag["rr_mean_plausible"],
                "a_rr_median_plausible": a_diag["rr_median_plausible"],
                "b_rr_median_plausible": b_diag["rr_median_plausible"],
            }
        )
        record.update(waveform_summary(ecg, "raw_filtered_ecg"))
        record.update(waveform_summary(b_signal, "b_bandpass_ecg"))
        rows.append(record)
        if (idx + 1) % 500 == 0:
            print(f"compared {idx + 1} samples", flush=True)

    out = pd.DataFrame(rows)
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    out.to_csv(artifacts / "ecg_pipeline_comparison.csv", index=False)

    both_success = out["a_success"] & out["b_success"]
    count_diff = out.loc[both_success, "b_rpeak_count"] - out.loc[both_success, "a_rpeak_count"]
    summary = {
        "n_trials": int(len(out)),
        "a_success_rate": float(out["a_success"].mean()),
        "b_success_rate": float(out["b_success"].mean()),
        "a_fail_b_success": int((~out["a_success"] & out["b_success"]).sum()),
        "a_success_b_fail": int((out["a_success"] & ~out["b_success"]).sum()),
        "both_success": int(both_success.sum()),
        "rpeak_count_exact_agreement_both_success": float((count_diff == 0).mean()),
        "rpeak_count_within_1_both_success": float((count_diff.abs() <= 1).mean()),
        "rpeak_count_diff_mean_b_minus_a": float(count_diff.mean()),
        "rpeak_count_diff_abs_mean": float(count_diff.abs().mean()),
        "a_rpeak_count_distribution": out["a_rpeak_count"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict(),
        "b_rpeak_count_distribution": out["b_rpeak_count"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict(),
        "correlations_both_success": corr_records(out, "a", "b", BEAT_FEATURES),
        "a_abnormal_proportions": {
            "failure": float((~out["a_success"]).mean()),
            "hr_mean_not_plausible": float((~out["a_hr_mean_plausible"]).mean()),
            "hr_min_not_plausible": float((~out["a_hr_min_plausible"]).mean()),
            "hr_max_not_plausible": float((~out["a_hr_max_plausible"]).mean()),
            "rr_mean_not_plausible": float((~out["a_rr_mean_plausible"]).mean()),
            "rr_median_not_plausible": float((~out["a_rr_median_plausible"]).mean()),
        },
        "b_abnormal_proportions": {
            "failure": float((~out["b_success"]).mean()),
            "hr_mean_not_plausible": float((~out["b_hr_mean_plausible"]).mean()),
            "hr_min_not_plausible": float((~out["b_hr_min_plausible"]).mean()),
            "hr_max_not_plausible": float((~out["b_hr_max_plausible"]).mean()),
            "rr_mean_not_plausible": float((~out["b_rr_mean_plausible"]).mean()),
            "rr_median_not_plausible": float((~out["b_rr_median_plausible"]).mean()),
        },
        "subject_79_recovered_failures": int(((out["subject_id"] == 79) & ~out["a_success"] & out["b_success"]).sum()),
        "waveform_summary_correlations": corr_records(
            out,
            "raw_filtered_ecg",
            "b_bandpass_ecg",
            ["std", "rms", "peak_to_peak"],
        ),
        "waveform_summary_ratios_b_over_a": {
            "std_median_ratio": float(np.median(out["b_bandpass_ecg_std"] / out["raw_filtered_ecg_std"])),
            "rms_median_ratio": float(np.median(out["b_bandpass_ecg_rms"] / out["raw_filtered_ecg_rms"])),
            "peak_to_peak_median_ratio": float(np.median(out["b_bandpass_ecg_peak_to_peak"] / out["raw_filtered_ecg_peak_to_peak"])),
        },
    }
    (artifacts / "ecg_pipeline_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    subject = (
        out.groupby(["subject_id", "subject_name"])
        .agg(
            n_trials=("sample_name", "count"),
            a_failures=("a_success", lambda x: int((~x).sum())),
            b_failures=("b_success", lambda x: int((~x).sum())),
            a_mean_rpeaks=("a_rpeak_count", "mean"),
            b_mean_rpeaks=("b_rpeak_count", "mean"),
            a_hr_mean_abnormal=("a_hr_mean_plausible", lambda x: int((~x).sum())),
            b_hr_mean_abnormal=("b_hr_mean_plausible", lambda x: int((~x).sum())),
        )
        .reset_index()
    )
    subject.to_csv(artifacts / "ecg_pipeline_comparison_by_subject.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    compare(args.root.resolve())


if __name__ == "__main__":
    main()
