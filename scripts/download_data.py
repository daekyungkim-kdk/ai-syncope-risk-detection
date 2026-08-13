from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


SOURCES = {
    "ppg-dalia": {
        "url": "https://archive.ics.uci.edu/static/public/495/ppg%2Bdalia.zip",
        "filename": "ppg-dalia.zip",
    },
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "syncope-prototype/1.0"})
    with urlopen(request) as response, destination.open("wb") as stream:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            copied += len(block)
            if total:
                print(f"\r{copied / total:6.1%}", end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    source = SOURCES[args.dataset]
    archive = args.output_dir / source["filename"]
    if not archive.exists():
        print(f"Downloading {args.dataset} to {archive}")
        download(source["url"], archive)
    else:
        print(f"Using existing archive: {archive}")

    if not args.no_extract:
        target = args.output_dir / args.dataset
        target.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {target}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)


if __name__ == "__main__":
    main()

