from __future__ import annotations

import pickle
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from syncope_proto import PersonalizedAnomalyModel, extract_window_features, load_subject


def test_end_to_end(tmp_path: Path) -> None:
    duration = 180
    rng = np.random.default_rng(7)
    t_bvp = np.arange(duration * 64) / 64
    t_acc = np.arange(duration * 32) / 32
    t_slow = np.arange(duration * 4) / 4
    payload = {
        "subject": "T01",
        "signal": {
            "wrist": {
                "BVP": (np.sin(2 * np.pi * 1.1 * t_bvp) + rng.normal(0, .03, len(t_bvp)))[:, None],
                "ACC": np.column_stack([np.zeros(len(t_acc)), np.zeros(len(t_acc)), np.ones(len(t_acc))]),
                "EDA": np.ones((len(t_slow), 1)),
                "TEMP": np.full((len(t_slow), 1), 33.0),
            },
            "chest": {},
        },
        "label": np.full(duration * 2, 66.0),
    }
    path = tmp_path / "T01.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)

    subject = load_subject(path, "ppg-dalia")
    frame = extract_window_features(subject, window_seconds=30, step_seconds=10)
    assert len(frame) == 16
    assert frame["hr_bpm"].median() > 55
    assert frame["hr_bpm"].median() < 80
    model = PersonalizedAnomalyModel.from_frame(frame).fit(frame.head(10))
    scored = model.score(frame)
    assert scored["anomaly_score"].between(0, 1).all()
    assert set(scored["risk_stage"]).issubset({"normal", "caution", "risk", "check_sensor"})

