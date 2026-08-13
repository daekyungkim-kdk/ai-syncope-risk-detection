from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pickle

import numpy as np


@dataclass
class SubjectSignals:
    subject_id: str
    dataset: str
    signals: dict[str, np.ndarray]
    sampling_rates: dict[str, float]
    labels: np.ndarray | None = None
    label_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        durations = [
            len(values) / self.sampling_rates[name]
            for name, values in self.signals.items()
            if len(values) and self.sampling_rates.get(name, 0) > 0
        ]
        if not durations:
            return 0.0
        return float(min(durations))


def _flat(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        try:
            return pickle.load(stream, encoding="latin1")
        except TypeError:
            stream.seek(0)
            return pickle.load(stream)


def _subject_name(payload: dict[str, Any], path: Path) -> str:
    value = payload.get("subject", path.stem)
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value)


def load_ppg_dalia(path: str | Path) -> SubjectSignals:
    path = Path(path)
    payload = _read_pickle(path)
    wrist = payload["signal"]["wrist"]
    chest = payload["signal"].get("chest", {})

    signals: dict[str, np.ndarray] = {
        "bvp": _flat(wrist["BVP"]),
        "acc": _flat(wrist["ACC"]),
        "eda": _flat(wrist["EDA"]),
        "temp": _flat(wrist["TEMP"]),
    }
    rates = {"bvp": 64.0, "acc": 32.0, "eda": 4.0, "temp": 4.0}
    if "ECG" in chest:
        signals["ecg"] = _flat(chest["ECG"])
        rates["ecg"] = 700.0

    labels = _flat(payload["label"]) if "label" in payload else None
    # PPG-DaLiA heart-rate reference labels are supplied at 2 Hz.
    label_rate = 2.0 if labels is not None else None
    return SubjectSignals(
        subject_id=_subject_name(payload, path),
        dataset="ppg-dalia",
        signals=signals,
        sampling_rates=rates,
        labels=labels,
        label_rate=label_rate,
        metadata={"source_file": str(path)},
    )


def load_wesad(path: str | Path) -> SubjectSignals:
    path = Path(path)
    payload = _read_pickle(path)
    wrist = payload["signal"]["wrist"]
    chest = payload["signal"].get("chest", {})

    signals: dict[str, np.ndarray] = {
        "bvp": _flat(wrist["BVP"]),
        "acc": _flat(wrist["ACC"]),
        "eda": _flat(wrist["EDA"]),
        "temp": _flat(wrist["TEMP"]),
    }
    rates = {"bvp": 64.0, "acc": 32.0, "eda": 4.0, "temp": 4.0}
    if "ECG" in chest:
        signals["ecg"] = _flat(chest["ECG"])
        rates["ecg"] = 700.0

    labels = _flat(payload["label"]) if "label" in payload else None
    return SubjectSignals(
        subject_id=_subject_name(payload, path),
        dataset="wesad",
        signals=signals,
        sampling_rates=rates,
        labels=labels,
        label_rate=700.0 if labels is not None else None,
        metadata={"source_file": str(path)},
    )


def load_subject(path: str | Path, dataset: str) -> SubjectSignals:
    dataset = dataset.lower()
    if dataset == "ppg-dalia":
        return load_ppg_dalia(path)
    if dataset == "wesad":
        return load_wesad(path)
    raise ValueError(f"Unsupported dataset: {dataset}")

