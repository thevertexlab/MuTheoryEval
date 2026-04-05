#!/usr/bin/env python3
"""
Download WildScore image assets to data/wildscore/.

The HuggingFace dataset (GM77/WildScore) stores images as filename references
rather than inline bytes. This script downloads the full dataset repository
(including the images/ folder) into data/wildscore/.

Usage:
    python scripts/download_wildscore.py

Output:
    data/wildscore/images/   ← score images referenced by the dataset
    data/wildscore/           ← also contains dataset parquet files (cached by HF)

Requirements:
    pip install huggingface_hub datasets

The data/ directory is gitignored — do not commit these files.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "wildscore"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GM77/WildScore to {DATA_DIR} ...")
    print("(This downloads score images — expect ~50-200 MB)")

    local_dir = snapshot_download(
        repo_id="GM77/WildScore",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    print(f"\nDone. Dataset saved to: {local_dir}")

    images_dir = DATA_DIR / "images"
    if images_dir.exists():
        n_images = len(list(images_dir.glob("*")))
        print(f"Found {n_images} images in {images_dir}")
    else:
        print(f"Warning: images/ directory not found in {DATA_DIR}")
        print("The dataset may use a different layout — check the HF repo structure.")


if __name__ == "__main__":
    main()
