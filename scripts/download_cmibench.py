#!/usr/bin/env python3
"""
Download CMI-Bench audio assets to data/cmibench/.

CMI-Bench audio is distributed as split zip files on HuggingFace
(repo: nicolaus625/CMI-bench). This script downloads and extracts them.

Usage:
    python scripts/download_cmibench.py

Output:
    data/cmibench/   ← extracted audio files (mp3/wav) + any metadata

The data/ directory is gitignored — do not commit these files.

Requirements:
    pip install huggingface_hub datasets
"""

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "cmibench"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading nicolaus625/CMI-bench to {DATA_DIR} ...")
    print("(Audio zips can be large — expect several GB)")

    local_dir = snapshot_download(
        repo_id="nicolaus625/CMI-bench",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    print(f"\nDownload complete. Raw files in: {local_dir}")

    # Extract any zip files found
    zip_files = sorted(DATA_DIR.rglob("*.zip"))
    if zip_files:
        print(f"\nExtracting {len(zip_files)} zip file(s) via system unzip (multi-part safe)...")
        import shutil, subprocess
        if not shutil.which("unzip"):
            print("ERROR: `unzip` not found. Install it (e.g. `brew install unzip`) and re-run.")
            sys.exit(1)
        for zf in zip_files:
            print(f"  unzip {zf.name} ...")
            result = subprocess.run(
                ["unzip", "-o", "-q", str(zf), "-d", str(DATA_DIR)],
                capture_output=True, text=True,
            )
            if result.returncode not in (0, 1):  # 1 = warnings, still ok
                print(f"  Warning: unzip returned {result.returncode}: {result.stderr[:200]}")
        print("Extraction complete.")
    else:
        print("No zip files found — audio may already be extracted or in a different format.")

    # Summary
    audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    audio_files = [f for f in DATA_DIR.rglob("*") if f.suffix.lower() in audio_exts]
    print(f"\nTotal audio files found: {len(audio_files)}")
    if audio_files:
        print(f"Sample paths:")
        for f in audio_files[:3]:
            print(f"  {f.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
