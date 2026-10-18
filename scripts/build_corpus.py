#!/usr/bin/env python3
"""Rebuild the Tigrinya training corpus from OPUS parallel data.

This script downloads the Tigrinya (`ti`) side of every available OPUS
parallel corpus (en-ti), extracts the Tigrinya sentences, cleans and
deduplicates them, merges with any local `Corpus.txt` at the repo root,
and writes the result to `data/tigrinya_corpus_full.txt`.

The result is the same corpus used to train the committed `tokenizer.json`.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --out data/tigrinya_corpus_full.txt

Requirements:
    - Python 3.9+
    - `curl` on PATH (used to download the OPUS zips)
    - write access to ./data/opus/

OPUS terms: each corpus ships its own LICENSE inside its zip. Review
`data/opus/<corpus>.LICENSE` before redistributing the derived corpus.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

# (name, url) pairs. These are the en-ti moses-format zips exposed by the
# OPUS API (https://opus.nlpl.eu/opusapi/?source=ti&target=en&preprocessing=moses).
OPUS_CORPORA = [
    ("NLLB",        "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-ti.txt.zip"),
    ("CCAligned",   "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/en-ti.txt.zip"),
    ("tico-19",     "https://object.pouta.csc.fi/OPUS-tico-19/v2020-10-28/moses/en-ti.txt.zip"),
    ("wikimedia",   "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/en-ti.txt.zip"),
    ("translatewiki", "https://object.pouta.csc.fi/OPUS-translatewiki/v2026-07-01/moses/en-ti.txt.zip"),
    ("Tatoeba",     "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/en-ti.txt.zip"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "tigrinya_corpus_full.txt"
OPUS_DIR = REPO_ROOT / "data" / "opus"
LOCAL_CORPUS = REPO_ROOT / "Corpus.txt"

MAX_LINE_LEN = 2000  # drop absurdly long lines (parsing artifacts)


def is_ethiopic(cp: int) -> bool:
    return (
        0x1200 <= cp <= 0x137F
        or 0x1380 <= cp <= 0x139F
        or 0x2D80 <= cp <= 0x2DDF
        or 0xAB00 <= cp <= 0xAB2F
    )


def download(name: str, url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{name}.zip"
    if out.exists() and out.stat().st_size > 0:
        print(f"  [skip] {name}.zip already present")
        return out
    print(f"  [get ] {name} <- {url}")
    # Prefer curl (handles redirects/Large files well); fall back to urllib.
    try:
        subprocess.run(
            ["curl", "-sSL", "-o", str(out), url], check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        urllib.request.urlretrieve(url, out)
    return out


def extract_ti(zip_path: Path, dest_dir: Path) -> list[Path]:
    import zipfile

    ti_files: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(".ti"):
                zf.extract(member, dest_dir)
                ti_files.append(dest_dir / member)
    return ti_files


def clean_iter(paths: list[Path]):
    """Yield clean, non-empty Tigrinya lines from the given .ti files."""
    for p in paths:
        if not p.exists():
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s or len(s) > MAX_LINE_LEN:
                    continue
                if not any(is_ethiopic(ord(c)) for c in s):
                    continue  # drop pure-English / number leakage
                yield s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output corpus path (default: data/tigrinya_corpus_full.txt)",
    )
    ap.add_argument(
        "--keep-zips", action="store_true",
        help="Keep downloaded zips (default: keep; they are gitignored)",
    )
    args = ap.parse_args()

    print("Downloading Tigrinya corpora from OPUS...")
    ti_files: list[Path] = []
    for name, url in OPUS_CORPORA:
        z = download(name, url, OPUS_DIR)
        ti_files.extend(extract_ti(z, OPUS_DIR))

    print(f"\nCleaning + deduplicating {len(ti_files)} Tigrinya files...")
    seen: set[str] = set()
    kept = 0
    dropped = Counter()
    sources: Counter = Counter()

    def process(lines, source_label: str):
        nonlocal kept
        for s in lines:
            if s in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(s)
            kept += 1
            sources[source_label] += 1

    for ti in ti_files:
        process(clean_iter([ti]), ti.name)

    # Merge any local Corpus.txt (kept in the repo as a seed).
    if LOCAL_CORPUS.exists():
        print(f"Merging local {LOCAL_CORPUS.name}...")
        process(clean_iter([LOCAL_CORPUS]), LOCAL_CORPUS.name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in seen:
            f.write(s + "\n")

    # Stats
    eth = 0
    usyll: set[str] = set()
    with open(args.out, encoding="utf-8", errors="replace") as f:
        for line in f:
            for c in line:
                cp = ord(c)
                if is_ethiopic(cp):
                    eth += 1
                    usyll.add(c)

    print(f"\n=== Wrote {args.out} ===")
    print(f"  Unique lines   : {kept:,}")
    print(f"  Bytes          : {args.out.stat().st_size:,} ({args.out.stat().st_size/1e6:.2f} MB)")
    print(f"  Ge'ez chars    : {eth:,}")
    print(f"  Unique syllables: {len(usyll):,}")
    print(f"  Est. subword tokens: {eth//3:,} - {eth//2:,}")
    print("\nPer-source (lines kept):")
    for k, v in sources.most_common():
        print(f"  {v:>9,}  {k}")
    print("\nDropped:")
    for k, v in dropped.most_common():
        print(f"  {v:>9,}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
