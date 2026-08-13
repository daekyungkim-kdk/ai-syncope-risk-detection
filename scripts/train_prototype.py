from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from syncope_proto import PersonalizedAnomalyModel


def select_baseline(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    candidates = frame.loc[~frame["low_signal_quality"].astype(bool)].copy()
    if frame["dataset"].eq("wesad").all() and "source_label" in frame:
        neutral = candidates[candidates["source_label"] == 1]
        if len(neutral) >= 10:
            return neutral
    count = max(10, int(len(candidates) * fraction))
    return candidates.sort_values("window_start_s").head(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-fraction", type=float, default=0.40)
    args = parser.parse_args()
    if not 0 < args.baseline_fraction <= 1:
        raise SystemExit("--baseline-fraction must be in (0, 1]")

    frame = pd.read_csv(args.features)
    all_outputs = []
    models = {}
    for subject_id, subject_frame in frame.groupby("subject_id", sort=False):
        baseline = select_baseline(subject_frame, args.baseline_fraction)
        model = PersonalizedAnomalyModel.from_frame(subject_frame).fit(baseline)
        all_outputs.append(model.score(subject_frame))
        models[str(subject_id)] = model

    scored = pd.concat(all_outputs, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "scored_windows.csv", index=False)
    joblib.dump(models, args.output_dir / "model.joblib")
    summary = {
        "interpretation": "baseline deviation prototype; not syncope diagnosis",
        "subjects": int(scored["subject_id"].nunique()),
        "windows": int(len(scored)),
        "stage_counts": {str(k): int(v) for k, v in scored["risk_stage"].value_counts().items()},
        "caution_threshold": 0.45,
        "risk_threshold": 0.70,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

