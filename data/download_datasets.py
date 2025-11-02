"""Simple helper to download the SIFT1M dataset."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path


SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


def download_sift1m(root: str | Path = "data", overwrite: bool = False) -> None:
    """Download and extract the SIFT1M dataset into ``root``."""
    root_path = Path(root)
    raw_dir = root_path / "raw" / "sift1m"
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_path = raw_dir / "sift.tar.gz"
    if archive_path.exists() and not overwrite:
        print(f"{archive_path} already exists. Skipping download.")
    else:
        tmp_path = archive_path.with_suffix(".tmp")
        print(f"Downloading SIFT1M from {SIFT1M_URL}...")
        urllib.request.urlretrieve(SIFT1M_URL, tmp_path)
        tmp_path.rename(archive_path)
        print(f"Saved archive to {archive_path}")

    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(path=raw_dir)
    print("Extraction complete.")


if __name__ == "__main__":
    download_sift1m()
