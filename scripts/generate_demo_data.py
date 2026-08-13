from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=8.0)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    duration = args.minutes * 60.0

    bvp_t = np.arange(int(duration * 64)) / 64.0
    # 72 bpm pulse with a late, intentionally unusual segment for pipeline testing.
    phase = 2 * np.pi * 1.2 * bvp_t
    bvp = np.sin(phase) + 0.25 * np.sin(2 * phase) + rng.normal(0, 0.08, len(bvp_t))
    bvp[bvp_t > duration * 0.75] += 0.8 * rng.normal(size=(bvp_t > duration * 0.75).sum())

    acc_t = np.arange(int(duration * 32)) / 32.0
    acc = np.column_stack([
        0.03 * rng.normal(size=len(acc_t)),
        0.03 * rng.normal(size=len(acc_t)),
        1.0 + 0.03 * rng.normal(size=len(acc_t)),
    ])
    moving = acc_t > duration * 0.75
    acc[moving, 0] += 0.4 * np.sin(2 * np.pi * 1.8 * acc_t[moving])

    slow_t = np.arange(int(duration * 4)) / 4.0
    eda = 1.2 + 0.02 * rng.normal(size=len(slow_t))
    temp = 33.0 + 0.02 * rng.normal(size=len(slow_t))
    eda[slow_t > duration * 0.75] += 0.5
    temp[slow_t > duration * 0.75] -= 0.4

    label_t = np.arange(int(duration * 2)) / 2.0
    labels = np.full_like(label_t, 72.0)
    payload = {
        "subject": "DEMO01",
        "signal": {
            "wrist": {
                "BVP": bvp[:, None], "ACC": acc,
                "EDA": eda[:, None], "TEMP": temp[:, None],
            },
            "chest": {},
        },
        "label": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream)
    print(f"Saved demo data to {args.output}")


if __name__ == "__main__":
    main()

