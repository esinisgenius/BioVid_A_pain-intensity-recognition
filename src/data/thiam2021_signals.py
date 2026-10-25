from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


MODALITY_COLUMNS = {"eda": "gsr", "ecg": "ecg", "emg": "emg_trapezius"}


def _sos_filter(signal: np.ndarray, order: int, cutoff, btype: str, sample_rate: float) -> np.ndarray:
    sos = butter(order, cutoff, btype=btype, fs=sample_rate, output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))


def preprocess_trial(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    """Apply the paper filters at 512 Hz, decimate, then take the adapted 4.5 s crop."""
    source_rate = int(config["data"]["source_rate_hz"])
    target_rate = int(config["data"]["model_rate_hz"])
    if source_rate != 2 * target_rate:
        raise ValueError("This audited implementation requires exact factor-2 downsampling")
    pre = config["preprocessing"]
    eda = _sos_filter(
        frame[MODALITY_COLUMNS["eda"]].to_numpy(),
        int(pre["eda_filter_order"]),
        float(pre["eda_lowpass_hz"]),
        "lowpass",
        source_rate,
    )
    ecg = _sos_filter(
        frame[MODALITY_COLUMNS["ecg"]].to_numpy(),
        int(pre["ecg_filter_order"]),
        [float(x) for x in pre["ecg_bandpass_hz"]],
        "bandpass",
        source_rate,
    )
    x = np.linspace(-1.0, 1.0, len(ecg), dtype=np.float64)
    coeff = np.polyfit(x, ecg, int(pre["ecg_polynomial_detrend_degree"]))
    ecg = ecg - np.polyval(coeff, x)
    emg = _sos_filter(
        frame[MODALITY_COLUMNS["emg"]].to_numpy(),
        int(pre["emg_filter_order"]),
        [float(x) for x in pre["emg_bandpass_hz"]],
        "bandpass",
        source_rate,
    )

    start = int(round(float(config["data"]["local_crop_start_seconds"]) * target_rate))
    length = int(round(float(config["data"]["paper_window_seconds"]) * target_rate))
    result = {}
    for modality, values in {"eda": eda, "ecg": ecg, "emg": emg}.items():
        downsampled = values[::2]
        cropped = downsampled[start : start + length]
        if cropped.shape != (length,):
            raise ValueError(f"Unexpected {modality} shape {cropped.shape}; expected {(length,)}")
        if not np.isfinite(cropped).all():
            raise ValueError(f"Non-finite values after preprocessing {modality}")
        result[modality] = cropped.astype(np.float32)
    return result


def cache_paths(directory: Path) -> dict[str, Path]:
    return {
        "eda": directory / "eda.npy",
        "ecg": directory / "ecg.npy",
        "emg": directory / "emg.npy",
        "manifest": directory / "manifest.json",
    }


def build_signal_cache(root: Path, samples: pd.DataFrame, config: dict, force: bool = False) -> dict:
    directory = root / config["data"]["cache_directory"]
    paths = cache_paths(directory)
    if not force and all(path.exists() for path in paths.values()):
        manifest = json.loads(paths["manifest"].read_text())
        if manifest["sample_names"] == samples["sample_name"].tolist():
            return manifest
        raise ValueError("Existing Thiam signal cache sample order does not match samples.csv")

    directory.mkdir(parents=True, exist_ok=True)
    n_samples = len(samples)
    length = int(round(float(config["data"]["paper_window_seconds"]) * int(config["data"]["model_rate_hz"])))
    arrays = {
        modality: np.lib.format.open_memmap(paths[modality], mode="w+", dtype=np.float32, shape=(n_samples, length))
        for modality in ("eda", "ecg", "emg")
    }
    source_digest = hashlib.sha256()
    source_dir = root / config["data"]["raw_signals"]
    started = time.perf_counter()
    for idx, row in samples.iterrows():
        relative = Path(row["subject_name"]) / f"{row['sample_name']}_bio.csv"
        path = source_dir / relative
        raw = path.read_bytes()
        source_digest.update(str(relative).encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(raw)
        frame = pd.read_csv(io.BytesIO(raw), sep="\t", usecols=list(MODALITY_COLUMNS.values()))
        processed = preprocess_trial(frame, config)
        for modality in arrays:
            arrays[modality][idx] = processed[modality]
        if (idx + 1) % 500 == 0:
            print(f"cached {idx + 1}/{n_samples} trials", flush=True)
    for array in arrays.values():
        array.flush()

    samples_path = root / config["data"]["samples"]
    manifest = {
        "n_samples": n_samples,
        "signal_length": length,
        "modalities": ["eda", "ecg", "emg"],
        "sample_names": samples["sample_name"].tolist(),
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "raw_signal_collection_sha256": source_digest.hexdigest(),
        "preprocessing_seconds": time.perf_counter() - started,
        "preprocessing": config["preprocessing"],
        "canonical_window_assumption": {
            "local_clip_start_relative_to_onset": "likely about 3 s; exact trigger unknown",
            "crop_start_seconds_within_local_clip": config["data"]["local_crop_start_seconds"],
            "output_samples": length,
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_signal_cache(root: Path, config: dict) -> tuple[dict[str, np.ndarray], dict]:
    directory = root / config["data"]["cache_directory"]
    paths = cache_paths(directory)
    manifest = json.loads(paths["manifest"].read_text())
    arrays = {modality: np.load(paths[modality], mmap_mode="r") for modality in ("eda", "ecg", "emg")}
    expected = (int(manifest["n_samples"]), int(manifest["signal_length"]))
    for modality, array in arrays.items():
        if array.shape != expected:
            raise ValueError(f"Cached {modality} shape {array.shape} != {expected}")
    return arrays, manifest


def fit_standardization(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, dict[str, float]]:
    stats = {}
    for modality, array in arrays.items():
        selected = np.asarray(array[indices], dtype=np.float64)
        mean = float(selected.mean())
        std = float(selected.std())
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid standardization statistics for {modality}: mean={mean}, std={std}")
        stats[modality] = {"mean": mean, "std": std}
    return stats
