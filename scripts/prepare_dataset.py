from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from syncope_proto import extract_window_features, load_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ppg-dalia", "wesad"], required=True)
    parser.add_argument("--input", required=True, help="pickle path or glob pattern")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument("--step", type=float, default=15.0)
    args = parser.parse_args()

    paths = [Path(item) for item in sorted(glob.glob(args.input))]
    if not paths:
        raise SystemExit(f"No files matched: {args.input}")

    frames = []
    for path in paths:
        print(f"Processing {path}")
        subject = load_subject(path, args.dataset)
        frame = extract_window_features(subject, args.window, args.step)
        if len(frame):
            frames.append(frame)
    if not frames:
        raise SystemExit("No complete windows were generated")

    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} windows to {args.output}")


if __name__ == "__main__":
    main()

