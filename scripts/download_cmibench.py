#!/usr/bin/env python3
"""
Selective download of CMI-Bench audio assets.

CMI-Bench audio is stored in 551 split-zip shards on HuggingFace
(nicolaus625/CMI-bench).  Full dataset = ~55 GB.  This script downloads
only the ~4 GB subset needed for the 100-item lite evaluation.

Steps
-----
1. Clone metadata JSONLs from GitHub (no audio, <1 MB).
2. Read the ZIP central directory from test_data.zip (already present, 62 MB).
3. Sample 100 items (40 key_detection, 30 singing_technique, 20 pitch, 10 GTZAN).
4. Map each item's audio file → disk number → shard filename.
5. Download only the required shards via hf_hub_download (~41 shards ≈ 4.1 GB).
6. Extract each target file by parsing the local file header + zlib decompress.
7. Delete the downloaded shards to save disk space.

Usage
-----
    python scripts/download_cmibench.py              # full selective download
    python scripts/download_cmibench.py --meta-only  # clone metadata only
    python scripts/download_cmibench.py --no-cleanup # keep shards after extract

Output
------
    data/cmibench/meta/      ← JSONL metadata (GitHub clone)
    data/cmibench/audio/     ← extracted audio files
    data/cmibench/test_data.zip  ← ZIP central directory (already present)

Requirements
------------
    pip install huggingface_hub gitpython
"""

import argparse
import json
import os
import random
import struct
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data" / "cmibench"
META_DIR  = DATA_DIR / "meta"
AUDIO_DIR = DATA_DIR / "audio"
CD_CACHE  = DATA_DIR / "cd_index.json"    # cached central-directory mapping

HF_REPO   = "nicolaus625/CMI-bench"
GITHUB_META = "https://github.com/nicolaus625/CMI-Bench.git"

# Sampling plan for the 100-item lite set.
# Each entry: (task_key, jsonl_relative_path, n_items)
# All four tasks have small fixed label pools suitable for MCQ.
_SAMPLE_PLAN = [
    ("GTZAN",    "GTZAN/CMI_GTZAN.jsonl",                   25),
    ("GS-key",   "GS-key/CMI_GS_key.jsonl",                 25),
    ("NSynth",   "NSynth/CMI_Nsynth_instrument.jsonl",       25),
    ("VocalSet", "VocalSet/CMI_VocalSet_tech.jsonl",         25),
]
_SAMPLE_SEED = 42

# ── ZIP central directory parsing ─────────────────────────────────────────────

_EOCD_SIG   = b"PK\x05\x06"
_EOCD64_SIG = b"PK\x06\x06"
_CD_SIG     = b"PK\x01\x02"
_LFH_SIG    = b"PK\x03\x04"


def _read_eocd64(data: bytes) -> tuple[int, int]:
    """Return (cd_offset, cd_size) from EOCD64 record."""
    pos = data.rfind(_EOCD64_SIG)
    if pos < 0:
        raise ValueError("No EOCD64 found")
    # EOCD64: sig(4) + size(8) + ver_made(2) + ver_need(2) + disk(4) + disk_cd(4)
    #         + entries_disk(8) + entries_total(8) + cd_size(8) + cd_offset(8)
    cd_size   = struct.unpack_from("<Q", data, pos + 40)[0]
    cd_offset = struct.unpack_from("<Q", data, pos + 48)[0]
    return cd_offset, cd_size


def parse_central_directory(zip_path: Path) -> dict[str, dict]:
    """Parse the ZIP central directory and return a mapping of
    zip_path → {"disk": int, "offset": int}.

    zip_path is the *last* shard (test_data.zip) which contains the CD.
    """
    data = zip_path.read_bytes()

    # Try ZIP64 first (this archive spans 550+ disks → needs ZIP64)
    try:
        cd_offset, cd_size = _read_eocd64(data)
    except ValueError:
        # Fall back to standard EOCD
        pos = data.rfind(_EOCD_SIG)
        if pos < 0:
            raise ValueError("No EOCD found in ZIP")
        cd_offset = struct.unpack_from("<I", data, pos + 16)[0]
        cd_size   = struct.unpack_from("<I", data, pos + 12)[0]

    mapping: dict[str, dict] = {}
    pos = cd_offset
    end = cd_offset + cd_size

    while pos < end and pos + 46 <= len(data):
        if data[pos:pos+4] != _CD_SIG:
            break
        # Central directory file header layout:
        # sig(4) ver_made(2) ver_need(2) flags(2) method(2) mod_time(2) mod_date(2)
        # crc(4) comp_size(4) uncomp_size(4) fname_len(2) extra_len(2) comment_len(2)
        # disk_start(2) int_attr(2) ext_attr(4) local_hdr_offset(4)
        fname_len   = struct.unpack_from("<H", data, pos + 28)[0]
        extra_len   = struct.unpack_from("<H", data, pos + 30)[0]
        comment_len = struct.unpack_from("<H", data, pos + 32)[0]
        disk_start  = struct.unpack_from("<H", data, pos + 34)[0]
        local_offset= struct.unpack_from("<I", data, pos + 42)[0]
        fname       = data[pos+46: pos+46+fname_len].decode("utf-8", errors="replace")

        # ZIP64 extra field may override disk_start and local_offset
        extra_data = data[pos+46+fname_len: pos+46+fname_len+extra_len]
        ep = 0
        while ep + 4 <= len(extra_data):
            hdr_id = struct.unpack_from("<H", extra_data, ep)[0]
            hdr_sz = struct.unpack_from("<H", extra_data, ep+2)[0]
            if hdr_id == 0x0001:  # ZIP64 extended information
                # Fields present depend on which values in the base header are 0xFFFF/0xFFFFFFFF
                uncomp_sz_b = struct.unpack_from("<I", data, pos + 24)[0]
                comp_sz_b   = struct.unpack_from("<I", data, pos + 20)[0]
                off64 = ep + 4
                if uncomp_sz_b == 0xFFFFFFFF and off64 + 8 <= len(extra_data):
                    off64 += 8
                if comp_sz_b == 0xFFFFFFFF and off64 + 8 <= len(extra_data):
                    off64 += 8
                if local_offset == 0xFFFFFFFF and off64 + 8 <= len(extra_data):
                    local_offset = struct.unpack_from("<Q", extra_data, off64)[0]
                    off64 += 8
                if disk_start == 0xFFFF and off64 + 4 <= len(extra_data):
                    disk_start = struct.unpack_from("<I", extra_data, off64)[0]
            ep += 4 + hdr_sz

        if fname and not fname.endswith("/"):
            mapping[fname] = {"disk": disk_start, "offset": local_offset}

        pos += 46 + fname_len + extra_len + comment_len

    return mapping


def disk_to_shard(disk: int) -> str:
    """Convert 0-based disk number to HF shard filename.

    Disk 0   → test_data.z01
    Disk 98  → test_data.z99
    Disk 99  → test_data.z100
    Disk 550 → test_data.zip  (central directory — already present)
    """
    n = disk + 1  # shard numbers are 1-based
    if disk == 550:
        return "test_data.zip"
    if n <= 99:
        return f"test_data.z{n:02d}"
    return f"test_data.z{n}"


# ── Local file header extraction ──────────────────────────────────────────────

def extract_file(shard_path: Path, local_offset: int) -> bytes:
    """Extract a single file from a shard by parsing its local file header."""
    data = shard_path.read_bytes()
    pos  = local_offset

    if data[pos:pos+4] != _LFH_SIG:
        raise ValueError(f"No local file header at offset {local_offset} in {shard_path.name}")

    # Local file header layout (30 bytes fixed):
    # sig(4) ver(2) flags(2) method(2) mod_time(2) mod_date(2) crc(4)
    # comp_size(4) uncomp_size(4) fname_len(2) extra_len(2)
    method      = struct.unpack_from("<H", data, pos + 8)[0]
    comp_size   = struct.unpack_from("<I", data, pos + 18)[0]
    uncomp_size = struct.unpack_from("<I", data, pos + 22)[0]
    fname_len   = struct.unpack_from("<H", data, pos + 26)[0]
    extra_len   = struct.unpack_from("<H", data, pos + 28)[0]

    # Handle ZIP64 in local extra field
    extra_start = pos + 30 + fname_len
    extra_data  = data[extra_start: extra_start + extra_len]
    ep = 0
    while ep + 4 <= len(extra_data):
        hdr_id = struct.unpack_from("<H", extra_data, ep)[0]
        hdr_sz = struct.unpack_from("<H", extra_data, ep+2)[0]
        if hdr_id == 0x0001:
            off64 = ep + 4
            if uncomp_size == 0xFFFFFFFF and off64 + 8 <= len(extra_data):
                uncomp_size = struct.unpack_from("<Q", extra_data, off64)[0]
                off64 += 8
            if comp_size == 0xFFFFFFFF and off64 + 8 <= len(extra_data):
                comp_size = struct.unpack_from("<Q", extra_data, off64)[0]
        ep += 4 + hdr_sz

    data_start = pos + 30 + fname_len + extra_len
    compressed = data[data_start: data_start + comp_size]

    if method == 0:   # stored
        return compressed
    if method == 8:   # deflate
        return zlib.decompress(compressed, -15)
    raise ValueError(f"Unsupported compression method {method}")


# ── Step 1: clone metadata ────────────────────────────────────────────────────

def clone_metadata():
    """Git-clone the CMI-Bench GitHub repo (metadata only, no audio)."""
    if META_DIR.exists() and any(META_DIR.rglob("*.jsonl")):
        print(f"  Metadata already present at {META_DIR}")
        return
    META_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning metadata from {GITHUB_META} → {META_DIR} ...")
    import subprocess
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none",
         GITHUB_META, str(META_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: git clone failed:\n{result.stderr}")
        sys.exit(1)
    print("  Metadata clone done.")


# ── Step 2: ensure CD is present ──────────────────────────────────────────────

def ensure_cd_zip() -> Path:
    cd_zip = DATA_DIR / "test_data.zip"
    if cd_zip.exists():
        print(f"  Central directory zip present: {cd_zip} ({cd_zip.stat().st_size//1024//1024} MB)")
        return cd_zip
    print("  Downloading test_data.zip (central directory, ~62 MB) ...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename="test_data.zip",
        local_dir=str(DATA_DIR),
    )
    return Path(path)


# ── Step 3: sample items ──────────────────────────────────────────────────────

def sample_items() -> list[dict]:
    """Return 100 sampled items with task and audio_path set."""
    rng = random.Random(_SAMPLE_SEED)
    sampled: list[dict] = []
    for task, jsonl_rel, n in _SAMPLE_PLAN:
        jsonl = META_DIR / "data" / jsonl_rel
        if not jsonl.exists():
            print(f"  WARN: no JSONL for task {task} at {jsonl}, skipping.")
            continue
        items = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Only use test-split items
                if "test" in d.get("split", []):
                    items.append(d)
        rng.shuffle(items)
        sampled.extend({"_task": task, **it} for it in items[:n])
    print(f"  Sampled {len(sampled)} items across {len(_SAMPLE_PLAN)} tasks.")
    return sampled


def audio_zip_path(item: dict) -> str:
    """Convert item audio_path (list or str) to the path inside the zip.

    JSONL audio_path: ["data/GTZAN/Data/..."]
    ZIP path:          testdata/GTZAN/Data/...
    """
    ap = item.get("audio_path") or item.get("audio") or ""
    rel = ap[0] if isinstance(ap, list) else ap
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return f"testdata/{rel}"


# ── Step 4-7: download shards → extract → cleanup ─────────────────────────────

def build_cd_index(cd_zip: Path) -> dict[str, dict]:
    """Load or build the central-directory index."""
    if CD_CACHE.exists():
        print(f"  Loading cached CD index from {CD_CACHE} ...")
        return json.loads(CD_CACHE.read_text())
    print("  Parsing ZIP central directory (this may take a moment) ...")
    mapping = parse_central_directory(cd_zip)
    CD_CACHE.write_text(json.dumps(mapping))
    print(f"  CD index built: {len(mapping)} entries → cached at {CD_CACHE}")
    return mapping


def download_and_extract(items: list[dict], cd_index: dict, cleanup: bool = True):
    """Download required shards, extract target files, optionally delete shards."""
    from huggingface_hub import hf_hub_download

    # Determine which disks are needed
    disk_to_files: dict[int, list[dict]] = {}
    missing_in_cd = []
    for item in items:
        zip_path = audio_zip_path(item)
        entry = cd_index.get(zip_path)
        if entry is None:
            missing_in_cd.append(zip_path)
            continue
        disk = entry["disk"]
        disk_to_files.setdefault(disk, []).append({
            "item":       item,
            "zip_path":   zip_path,
            "local_offset": entry["offset"],
        })

    if missing_in_cd:
        print(f"  WARN: {len(missing_in_cd)} items not found in CD index — they will be skipped.")

    shards_needed = sorted(disk_to_files.keys())
    print(f"  Need {len(shards_needed)} shards for {len(items)} items.")

    for disk in shards_needed:
        shard_name = disk_to_shard(disk)
        if shard_name == "test_data.zip":
            shard_path = DATA_DIR / "test_data.zip"
        else:
            shard_path = DATA_DIR / shard_name

        if not shard_path.exists():
            print(f"  Downloading {shard_name} ...", end="", flush=True)
            try:
                hf_hub_download(
                    repo_id=HF_REPO, repo_type="dataset",
                    filename=shard_name,
                    local_dir=str(DATA_DIR),
                )
                print(" done")
            except Exception as e:
                print(f" FAILED: {e}")
                continue

        # Extract files from this shard
        for file_info in disk_to_files[disk]:
            item       = file_info["item"]
            zip_path   = file_info["zip_path"]
            loc_offset = file_info["local_offset"]

            # Determine output path (mirrors zip path under AUDIO_DIR)
            ap = item.get("audio_path") or item.get("audio") or ""
            rel = ap[0] if isinstance(ap, list) else ap
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            out_path = AUDIO_DIR / "testdata" / rel
            if out_path.exists():
                continue   # already extracted

            try:
                audio_data = extract_file(shard_path, loc_offset)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(audio_data)
            except Exception as e:
                print(f"  WARN: could not extract {zip_path}: {e}")

        if cleanup and shard_name != "test_data.zip" and shard_path.exists():
            shard_path.unlink()

    # Summary
    audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    extracted = [f for f in AUDIO_DIR.rglob("*") if f.suffix.lower() in audio_exts]
    print(f"\n  Extracted {len(extracted)} audio files to {AUDIO_DIR}/")
    if extracted:
        for f in extracted[:3]:
            print(f"    {f.relative_to(DATA_DIR)}")
        if len(extracted) > 3:
            print(f"    … and {len(extracted)-3} more")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download CMI-Bench audio (selective)")
    parser.add_argument("--meta-only", action="store_true",
                        help="Only clone metadata JSONLs, skip audio download")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep shard files after extraction")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Clone metadata from GitHub ...")
    clone_metadata()

    if args.meta_only:
        print("\nDone (--meta-only).")
        return

    print("\n[Step 2] Ensure ZIP central directory ...")
    cd_zip = ensure_cd_zip()

    print("\n[Step 3] Sample 100 items ...")
    items = sample_items()
    if not items:
        print("  No items to download. Check metadata clone.")
        sys.exit(1)

    print("\n[Step 4] Build/load central directory index ...")
    cd_index = build_cd_index(cd_zip)

    print("\n[Steps 5-7] Download shards → extract → cleanup ...")
    download_and_extract(items, cd_index, cleanup=not args.no_cleanup)

    print("\nDone.")


if __name__ == "__main__":
    main()
