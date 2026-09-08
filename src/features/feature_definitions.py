from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    modality: str
    source_channel: str
    feature_set: str
    provenance: str
    interpretation_type: str
    method: str
    library: str
    definition: str
    units: str
    window_suitability: str
    limitations: str
    enters_main_analysis: bool
    quality_gated: bool
    missing_policy: str
    shared_with: str = ""


def _domain(
    name: str,
    modality: str,
    source: str,
    interpretation: str,
    definition: str,
    units: str = "",
    method: str = "deterministic per-window summary",
    library: str = "numpy/neurokit2",
    suitability: str = "Suitable for 5.5 second event-related windows",
    limitations: str = "",
    quality_gated: bool = False,
    provenance: str = "domain_informed",
    shared_with: str = "",
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_name=name,
        modality=modality,
        source_channel=source,
        feature_set="domain_informed",
        provenance=provenance,
        interpretation_type=interpretation,
        method=method,
        library=library,
        definition=definition,
        units=units,
        window_suitability=suitability,
        limitations=limitations,
        enters_main_analysis=True,
        quality_gated=quality_gated,
        missing_policy="fill deterministic no-event values with 0; record failures in extraction diagnostics",
        shared_with=shared_with,
    )


def _generic(
    name: str,
    modality: str,
    source: str,
    definition: str,
    units: str = "",
    shared_with: str = "",
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_name=name,
        modality=modality,
        source_channel=source,
        feature_set="generic_automated",
        provenance="shared" if shared_with else "generic_automated",
        interpretation_type="generic signal/statistical summary",
        method="explicit tsfresh calculator",
        library="tsfresh",
        definition=definition,
        units=units,
        window_suitability="Suitable as a short-window descriptive statistic",
        limitations="Generic signal summary; do not interpret as a physiological marker",
        enters_main_analysis=False,
        quality_gated=False,
        missing_policy="feature must be finite for every sample or be excluded with an audit record",
        shared_with=shared_with,
    )


def domain_feature_definitions() -> list[FeatureDefinition]:
    ecg_signal = "generic signal/statistical summary"
    rate = "physiological-derived feature"
    eda_level = "physiological-derived feature"
    emg_activity = "physiological-derived feature"
    emg_signal = "generic signal/statistical summary"

    return [
        _domain("ecg_rpeak_count", "ECG", "ecg", rate, "Number of R-peaks detected after median-centering and fixed 5-20 Hz bandpass", "count", "5-20 Hz bandpass plus NeuroKit2 ecg_peaks(method='neurokit')", "scipy/neurokit2", quality_gated=True),
        _domain("ecg_rate_mean_bpm", "ECG", "ecg", rate, "Mean interpolated heart rate from bandpass-detected R-peaks", "bpm", "R-peak interval-derived heart rate", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_rate_median_bpm", "ECG", "ecg", rate, "Median interpolated heart rate from bandpass-detected R-peaks", "bpm", "R-peak interval-derived heart rate", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_rate_min_bpm", "ECG", "ecg", rate, "Minimum interpolated heart rate from bandpass-detected R-peaks", "bpm", "R-peak interval-derived heart rate", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_rate_max_bpm", "ECG", "ecg", rate, "Maximum interpolated heart rate from bandpass-detected R-peaks", "bpm", "R-peak interval-derived heart rate", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_rate_range_bpm", "ECG", "ecg", rate, "Maximum minus minimum interpolated heart rate from bandpass-detected R-peaks", "bpm", "R-peak interval-derived heart rate", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_rate_slope_bpm_per_s", "ECG", "ecg", rate, "Least-squares slope of interpolated heart rate over seconds", "bpm/s", "linear regression over per-sample time", "numpy", quality_gated=True),
        _domain("ecg_rr_mean_ms", "ECG", "ecg", rate, "Mean interval between adjacent bandpass-detected R-peaks", "ms", "R-peak intervals", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_rr_median_ms", "ECG", "ecg", rate, "Median interval between adjacent bandpass-detected R-peaks", "ms", "R-peak intervals", "scipy/neurokit2/numpy", quality_gated=True),
        _domain("ecg_quality_mean", "ECG", "ecg", ecg_signal, "Mean NeuroKit2 ECG quality score", "unitless", "NeuroKit2 ecg_quality", "neurokit2", limitations="Signal-quality covariate, not a physiological marker", quality_gated=False),
        _domain("ecg_clean_std", "ECG", "ecg", ecg_signal, "Standard deviation of cleaned ECG waveform", "signal units", "cleaned waveform summary", "neurokit2/numpy", limitations="Waveform summary; electrode scale and preprocessing can dominate", provenance="shared", shared_with="ecg__tsfresh_standard_deviation"),
        _domain("ecg_clean_rms", "ECG", "ecg", ecg_signal, "Root mean square of cleaned ECG waveform", "signal units", "cleaned waveform summary", "neurokit2/numpy", limitations="Waveform summary, not a direct physiological marker", provenance="shared", shared_with="ecg__tsfresh_root_mean_square"),
        _domain("ecg_clean_peak_to_peak", "ECG", "ecg", ecg_signal, "Maximum minus minimum cleaned ECG waveform value", "signal units", "cleaned waveform summary", "neurokit2/numpy", limitations="Waveform summary, outlier-sensitive"),
        _domain("eda_scl_mean", "EDA/GSR", "gsr", eda_level, "Mean skin conductance level", "signal units", "filtered GSR summary", "numpy", provenance="shared", shared_with="eda__tsfresh_mean"),
        _domain("eda_scl_median", "EDA/GSR", "gsr", eda_level, "Median skin conductance level", "signal units", "filtered GSR summary", "numpy", provenance="shared", shared_with="eda__tsfresh_median"),
        _domain("eda_scl_std", "EDA/GSR", "gsr", "generic signal/statistical summary", "Standard deviation of filtered GSR", "signal units", "filtered GSR summary", "numpy", limitations="Generic variability summary; not a standalone physiological marker", provenance="shared", shared_with="eda__tsfresh_standard_deviation"),
        _domain("eda_scl_min", "EDA/GSR", "gsr", "generic signal/statistical summary", "Minimum filtered GSR value", "signal units", "filtered GSR summary", "numpy", provenance="shared", shared_with="eda__tsfresh_minimum"),
        _domain("eda_scl_max", "EDA/GSR", "gsr", "generic signal/statistical summary", "Maximum filtered GSR value", "signal units", "filtered GSR summary", "numpy", provenance="shared", shared_with="eda__tsfresh_maximum"),
        _domain("eda_scl_range", "EDA/GSR", "gsr", "generic signal/statistical summary", "Maximum minus minimum filtered GSR value", "signal units", "filtered GSR summary", "numpy"),
        _domain("eda_scl_auc", "EDA/GSR", "gsr", eda_level, "Trapezoidal integral of GSR over seconds", "signal units*s", "time integral", "numpy"),
        _domain("eda_scl_slope_per_s", "EDA/GSR", "gsr", eda_level, "Least-squares slope of GSR over seconds", "signal units/s", "linear trend over per-sample time", "numpy", provenance="shared", shared_with="eda__tsfresh_linear_trend_slope"),
        _domain("eda_scl_delta", "EDA/GSR", "gsr", eda_level, "Last minus first GSR value", "signal units", "edge-to-edge change", "numpy", provenance="shared", shared_with="eda__tsfresh_mean_change"),
        _domain("eda_scr_count", "EDA/GSR", "gsr", eda_level, "Number of NeuroKit2-detected SCR peaks after per-window amplitude filtering at max(0.01, 1% of window range)", "count", "NeuroKit2 eda_process/eda_peaks plus deterministic amplitude threshold", "neurokit2/numpy", quality_gated=True),
        _domain("eda_scr_amplitude_max", "EDA/GSR", "gsr", eda_level, "Maximum detected SCR amplitude after per-window amplitude filtering; 0 when no SCR is retained", "signal units", "NeuroKit2 SCR amplitude plus deterministic threshold", "neurokit2/numpy", quality_gated=True),
        _domain("eda_scr_amplitude_sum", "EDA/GSR", "gsr", eda_level, "Sum of detected SCR amplitudes after per-window amplitude filtering; 0 when no SCR is retained", "signal units", "NeuroKit2 SCR amplitude plus deterministic threshold", "neurokit2/numpy", quality_gated=True),
        _domain("emg_trapezius_mav", "Trapezius EMG", "emg_trapezius", emg_activity, "Mean absolute value of filtered trapezius EMG", "signal units", "EMG time-domain amplitude", "numpy"),
        _domain("emg_trapezius_rms", "Trapezius EMG", "emg_trapezius", emg_signal, "Root mean square of filtered trapezius EMG", "signal units", "EMG waveform summary", "numpy", limitations="Generic waveform magnitude; do not interpret directly as physiological marker", provenance="shared", shared_with="emg_trapezius__tsfresh_root_mean_square"),
        _domain("emg_trapezius_std", "Trapezius EMG", "emg_trapezius", emg_signal, "Standard deviation of filtered trapezius EMG", "signal units", "EMG waveform summary", "numpy", provenance="shared", shared_with="emg_trapezius__tsfresh_standard_deviation"),
        _domain("emg_trapezius_iemg", "Trapezius EMG", "emg_trapezius", emg_activity, "Sum of absolute trapezius EMG values divided by sampling rate", "signal units*s", "integrated EMG", "numpy"),
        _domain("emg_trapezius_energy", "Trapezius EMG", "emg_trapezius", emg_signal, "Sum of squared trapezius EMG values", "signal units^2", "waveform energy", "numpy", provenance="shared", shared_with="emg_trapezius__tsfresh_abs_energy"),
        _domain("emg_trapezius_peak_to_peak", "Trapezius EMG", "emg_trapezius", emg_signal, "Maximum minus minimum trapezius EMG value", "signal units", "waveform amplitude span", "numpy"),
        _domain("emg_trapezius_waveform_length", "Trapezius EMG", "emg_trapezius", emg_signal, "Sum of absolute first differences", "signal units", "waveform roughness/activity", "numpy", provenance="shared", shared_with="emg_trapezius__tsfresh_absolute_sum_of_changes"),
        _domain("emg_trapezius_zero_crossings", "Trapezius EMG", "emg_trapezius", emg_signal, "Count of sign changes after a fixed deadband threshold", "count", "thresholded zero-crossing count", "numpy"),
        _domain("emg_trapezius_activation_fraction", "Trapezius EMG", "emg_trapezius", emg_activity, "Fraction of samples whose smoothed rectified amplitude exceeds a per-window deterministic threshold", "fraction", "fixed-rule EMG activation", "numpy", quality_gated=True),
        _domain("emg_trapezius_activation_count", "Trapezius EMG", "emg_trapezius", emg_activity, "Number of activation segments from fixed-rule thresholding", "count", "fixed-rule EMG activation", "numpy", quality_gated=True),
        _domain("emg_trapezius_activation_mean_duration_s", "Trapezius EMG", "emg_trapezius", emg_activity, "Mean duration of activation segments; 0 when no activation segment is detected", "s", "fixed-rule EMG activation", "numpy", quality_gated=True),
    ]


GENERIC_CALCULATORS = [
    ("mean", "Arithmetic mean of signal values"),
    ("median", "Median of signal values"),
    ("standard_deviation", "Standard deviation of signal values"),
    ("minimum", "Minimum signal value"),
    ("maximum", "Maximum signal value"),
    ("root_mean_square", "Root mean square of signal values"),
    ("abs_energy", "Sum of squared signal values"),
    ("absolute_sum_of_changes", "Sum of absolute consecutive differences"),
    ("mean_change", "Mean consecutive difference"),
    ("linear_trend_slope", "Least-squares slope over sample index"),
    ("skewness", "Adjusted Fisher-Pearson skewness"),
    ("kurtosis", "Fisher kurtosis"),
]


GENERIC_MODALITIES = [
    ("ecg", "ECG", "ecg"),
    ("eda", "EDA/GSR", "gsr"),
    ("emg_trapezius", "Trapezius EMG", "emg_trapezius"),
]


SHARED_GENERIC = {
    "ecg__tsfresh_standard_deviation": "ecg_clean_std",
    "ecg__tsfresh_root_mean_square": "ecg_clean_rms",
    "eda__tsfresh_mean": "eda_scl_mean",
    "eda__tsfresh_median": "eda_scl_median",
    "eda__tsfresh_standard_deviation": "eda_scl_std",
    "eda__tsfresh_minimum": "eda_scl_min",
    "eda__tsfresh_maximum": "eda_scl_max",
    "eda__tsfresh_linear_trend_slope": "eda_scl_slope_per_s",
    "eda__tsfresh_mean_change": "eda_scl_delta",
    "emg_trapezius__tsfresh_standard_deviation": "emg_trapezius_std",
    "emg_trapezius__tsfresh_root_mean_square": "emg_trapezius_rms",
    "emg_trapezius__tsfresh_abs_energy": "emg_trapezius_energy",
    "emg_trapezius__tsfresh_absolute_sum_of_changes": "emg_trapezius_waveform_length",
}


def generic_feature_definitions() -> list[FeatureDefinition]:
    features: list[FeatureDefinition] = []
    for prefix, modality, source in GENERIC_MODALITIES:
        for calculator, definition in GENERIC_CALCULATORS:
            name = f"{prefix}__tsfresh_{calculator}"
            features.append(
                _generic(
                    name=name,
                    modality=modality,
                    source=source,
                    definition=definition,
                    shared_with=SHARED_GENERIC.get(name, ""),
                )
            )
    return features


def all_feature_definitions() -> list[FeatureDefinition]:
    return domain_feature_definitions() + generic_feature_definitions()


def feature_metadata_records() -> list[dict[str, object]]:
    return [asdict(feature) for feature in all_feature_definitions()]
