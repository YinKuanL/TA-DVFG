"""Download and unpack MovieLens 1M for the TA-DVFG MovieLens experiment."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/movielens"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.data_dir / "ml-1m.zip"
    target = args.data_dir / "ml-1m"
    if target.exists() and not args.force:
        print(f"MovieLens 1M already exists at {target}")
        return 0
    if not zip_path.exists() or args.force:
        print(f"Downloading {URL} -> {zip_path}")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"Unpacking {zip_path} -> {args.data_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(args.data_dir)
    required = [target / "ratings.dat", target / "users.dat", target / "movies.dat"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Unpack completed but required files are missing: " + ", ".join(missing))
    print(f"Prepared MovieLens 1M at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

