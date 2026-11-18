"""
════════════════════════════════════════════════════════════════════════════════
download_places365.py — Downloads ResNet18 Places365 weights and labels
════════════════════════════════════════════════════════════════════════════════

Run this script once before starting the surveillance system:
    python download_places365.py

Downloads:
    1. resnet18_places365.pth.tar  (~44 MB)  — model weights
    2. categories_places365.txt               — 365 scene label names
"""

import urllib.request
import os
import sys

WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
LABELS_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"

WEIGHTS_FILE = "resnet18_places365.pth.tar"
LABELS_FILE = "categories_places365.txt"


def download(url: str, filename: str) -> None:
    """Download a file with progress reporting."""
    if os.path.isfile(filename):
        print(f"  [SKIP] {filename} already exists.")
        return

    print(f"  Downloading {filename} ...")
    print(f"  URL: {url}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [{pct:3d}%] {mb:.1f} / {total_mb:.1f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
        print(f"\n  [OK] Saved → {filename}")
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"  Please download manually from:\n    {url}")


def main():
    print("=" * 60)
    print("Places365 ResNet18 — Model Downloader")
    print("=" * 60)
    print()

    print("[1/2] Scene classifier weights (~44 MB):")
    download(WEIGHTS_URL, WEIGHTS_FILE)
    print()

    print("[2/2] Scene category labels:")
    download(LABELS_URL, LABELS_FILE)
    print()

    print("=" * 60)
    if os.path.isfile(WEIGHTS_FILE) and os.path.isfile(LABELS_FILE):
        print("All files ready! You can now run: python main.py")
    else:
        print("Some files are missing. See errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
