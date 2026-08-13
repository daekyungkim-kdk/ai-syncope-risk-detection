from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


DEFAULT_FEATURES = [
    "hr_bpm", "ibi_mean_ms", "sdnn_ms", "rmssd_ms", "pnn50",
    "pulse_amplitude", "ppg_std", "acc_mag_std", "jerk_rms",
    "tilt_mean_deg", "eda_mean", "eda_slope", "temp_mean", "temp_slope",
]


@dataclass
class PersonalizedAnomalyModel:
    feature_names: list[str]
    contamination: float = 0.05
    caution_threshold: float = 0.45
    risk_threshold: float = 0.70
    random_state: int = 42

    def __post_init__(self) -> None:
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", RobustScaler()),
            ("model", IsolationForest(
                n_estimators=250,
                contamination=self.contamination,
                random_state=self.random_state,
            )),
        ])
        self._raw_low: float | None = None
        self._raw_high: float | None = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **kwargs) -> "PersonalizedAnomalyModel":
        names = [name for name in DEFAULT_FEATURES if name in frame.columns]
        if not names:
            raise ValueError("No supported feature columns were found")
        return cls(feature_names=names, **kwargs)

    def fit(self, baseline: pd.DataFrame) -> "PersonalizedAnomalyModel":
        if len(baseline) < 10:
            raise ValueError("At least 10 baseline windows are required")
        values = baseline[self.feature_names].replace([np.inf, -np.inf], np.nan)
        self.pipeline.fit(values)
        raw = -self.pipeline.decision_function(values)
        self._raw_low = float(np.quantile(raw, 0.05))
        self._raw_high = float(np.quantile(raw, 0.99))
        if self._raw_high <= self._raw_low:
            self._raw_high = self._raw_low + 1e-6
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._raw_low is None or self._raw_high is None:
            raise RuntimeError("Model must be fitted before scoring")
        values = frame[self.feature_names].replace([np.inf, -np.inf], np.nan)
        raw = -self.pipeline.decision_function(values)
        score = np.clip((raw - self._raw_low) / (self._raw_high - self._raw_low), 0, 1)
        stages = np.where(
            score >= self.risk_threshold,
            "risk",
            np.where(score >= self.caution_threshold, "caution", "normal"),
        )
        output = frame.copy()
        output["anomaly_score"] = score
        output["risk_stage"] = stages
        # Poor PPG quality should not generate a physiological risk assertion.
        if "low_signal_quality" in output:
            poor = output["low_signal_quality"].astype(bool)
            output.loc[poor, "risk_stage"] = "check_sensor"
        return output

