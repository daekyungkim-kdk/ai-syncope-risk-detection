from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import kurtosis, skew

from .loaders import SubjectSignals


def _segment(values: np.ndarray, rate: float, start: float, end: float) -> np.ndarray:
    left = max(0, int(round(start * rate)))
    right = min(len(values), int(round(end * rate)))
    return np.asarray(values[left:right], dtype=float)


def _safe_stat(function, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(function(finite))


def _linear_slope(values: np.ndarray, rate: float) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return float("nan")
    time = np.arange(len(values), dtype=float)[finite] / rate
    return float(np.polyfit(time, values[finite], 1)[0])


def _ppg_features(ppg: np.ndarray, rate: float) -> dict[str, float]:
    result = {
        "ppg_mean": float("nan"), "ppg_std": float("nan"),
        "ppg_skew": float("nan"), "ppg_kurtosis": float("nan"),
        "hr_bpm": float("nan"), "ibi_mean_ms": float("nan"),
        "sdnn_ms": float("nan"), "rmssd_ms": float("nan"),
        "pnn50": float("nan"), "pulse_amplitude": float("nan"),
        "ppg_sqi": 0.0, "peak_count": 0.0,
    }
    ppg = np.asarray(ppg, dtype=float)
    finite_ratio = float(np.isfinite(ppg).mean()) if len(ppg) else 0.0
    if len(ppg) < max(20, int(rate * 8)) or finite_ratio < 0.9:
        return result

    if not np.all(np.isfinite(ppg)):
        ppg = pd.Series(ppg).interpolate(limit_direction="both").to_numpy()
    centered = ppg - np.median(ppg)
    scale = np.median(np.abs(centered)) * 1.4826
    if scale <= 1e-12:
        return result

    sos = butter(3, [0.5, min(4.0, rate * 0.45)], btype="bandpass", fs=rate, output="sos")
    filtered = sosfiltfilt(sos, centered)
    prominence = max(np.std(filtered) * 0.25, 1e-9)
    peaks, properties = find_peaks(
        filtered,
        distance=max(1, int(rate * 0.30)),
        prominence=prominence,
    )
    ibis = np.diff(peaks) / rate
    plausible = ibis[(ibis >= 0.33) & (ibis <= 1.50)]
    plausible_ratio = len(plausible) / max(1, len(ibis))
    expected_min = max(1.0, len(ppg) / rate * 0.5)
    count_score = min(1.0, len(peaks) / expected_min)
    sqi = finite_ratio * plausible_ratio * count_score

    result.update({
        "ppg_mean": float(np.mean(ppg)),
        "ppg_std": float(np.std(ppg)),
        "ppg_skew": _safe_stat(skew, ppg),
        "ppg_kurtosis": _safe_stat(kurtosis, ppg),
        "pulse_amplitude": float(np.median(properties.get("prominences", [np.nan]))),
        "ppg_sqi": float(np.clip(sqi, 0, 1)),
        "peak_count": float(len(peaks)),
    })
    if len(plausible) >= 3:
        ibi_ms = plausible * 1000.0
        diff_ms = np.diff(ibi_ms)
        result.update({
            "hr_bpm": float(60.0 / np.mean(plausible)),
            "ibi_mean_ms": float(np.mean(ibi_ms)),
            "sdnn_ms": float(np.std(ibi_ms, ddof=1)),
            "rmssd_ms": float(np.sqrt(np.mean(diff_ms ** 2))) if len(diff_ms) else float("nan"),
            "pnn50": float(np.mean(np.abs(diff_ms) > 50.0)) if len(diff_ms) else float("nan"),
        })
    return result


def _acc_features(acc: np.ndarray, rate: float) -> dict[str, float]:
    result = {
        "acc_mag_mean": float("nan"), "acc_mag_std": float("nan"),
        "acc_rms": float("nan"), "jerk_rms": float("nan"),
        "tilt_mean_deg": float("nan"),
    }
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[1] < 3 or len(acc) < 3:
        return result
    mag = np.linalg.norm(acc[:, :3], axis=1)
    jerk = np.diff(acc[:, :3], axis=0) * rate
    horizontal = np.linalg.norm(acc[:, :2], axis=1)
    tilt = np.degrees(np.arctan2(horizontal, np.abs(acc[:, 2]) + 1e-12))
    result.update({
        "acc_mag_mean": float(np.nanmean(mag)),
        "acc_mag_std": float(np.nanstd(mag)),
        "acc_rms": float(np.sqrt(np.nanmean(mag ** 2))),
        "jerk_rms": float(np.sqrt(np.nanmean(jerk ** 2))),
        "tilt_mean_deg": float(np.nanmean(tilt)),
    })
    return result


def _slow_features(prefix: str, values: np.ndarray, rate: float) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.nanmean(values)) if len(values) else float("nan"),
        f"{prefix}_std": float(np.nanstd(values)) if len(values) else float("nan"),
        f"{prefix}_slope": _linear_slope(values, rate),
    }


def _window_label(subject: SubjectSignals, start: float, end: float) -> float:
    if subject.labels is None or subject.label_rate is None:
        return float("nan")
    values = _segment(subject.labels, subject.label_rate, start, end)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan")
    if subject.dataset == "wesad":
        labels, counts = np.unique(finite.astype(int), return_counts=True)
        return float(labels[np.argmax(counts)])
    return float(np.mean(finite))


def extract_window_features(
    subject: SubjectSignals,
    window_seconds: float = 60.0,
    step_seconds: float = 15.0,
) -> pd.DataFrame:
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive")
    duration = subject.duration_seconds
    rows: list[dict[str, float | str | bool]] = []
    start = 0.0
    while start + window_seconds <= duration + 1e-9:
        end = start + window_seconds
        bvp = _segment(subject.signals["bvp"], subject.sampling_rates["bvp"], start, end)
        row: dict[str, float | str | bool] = {
            "dataset": subject.dataset,
            "subject_id": subject.subject_id,
            "window_start_s": start,
            "window_end_s": end,
            "source_label": _window_label(subject, start, end),
        }
        row.update(_ppg_features(bvp, subject.sampling_rates["bvp"]))
        if "acc" in subject.signals:
            row.update(_acc_features(
                _segment(subject.signals["acc"], subject.sampling_rates["acc"], start, end),
                subject.sampling_rates["acc"],
            ))
        for name in ("eda", "temp"):
            if name in subject.signals:
                row.update(_slow_features(
                    name,
                    _segment(subject.signals[name], subject.sampling_rates[name], start, end),
                    subject.sampling_rates[name],
                ))
        row["low_signal_quality"] = bool(float(row["ppg_sqi"]) < 0.45)
        rows.append(row)
        start += step_seconds
    return pd.DataFrame(rows)

