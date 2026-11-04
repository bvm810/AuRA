#!/usr/bin/env python3
"""
blacklist_wavcaps.py

Remove blacklisted clips from a WavCaps CSV (as produced by build_wavcaps_csv.py,
with columns filepath, duration, caption, subset).

The blacklist JSON is a dict keyed by source subset, each mapping to a list of
ids, e.g.:

    {
        "AudioSet":  ["YvfY_TJq7n_U.wav", ...],   # ids carry a .wav extension
        "FreeSound": ["156", "201", ...]           # bare numeric ids
    }

Matching
--------
Matching is on the extension-stripped STEM of the filename (a .wav/.flac/.mp3
suffix is removed on both sides), because the blacklist ids and the CSV
`.flac` filepaths use different extensions.

Matching is SUBSET-SCOPED by default: each blacklist key only removes rows in
its corresponding CSV subset. This is important because ids collide across
subsets (FreeSound and SoundBible both use bare numeric ids), so a global match
would wrongly drop unrelated clips. Pass --ignore-subset to match on stem alone
across all subsets.

Blacklist key -> CSV subset name is resolved via KEY_TO_SUBSET below; unknown
keys are used verbatim as a subset name (with a warning).

Usage
-----
    python blacklist_wavcaps.py <blacklist_json> <in_csv> <out_csv> [--ignore-subset]

Example
-------
    python blacklist_wavcaps.py blacklist_exclude_test_ac.json wavcaps.csv wavcaps_clean.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

AUDIO_EXTS = {".wav", ".flac", ".mp3"}

COLUMNS = ["filepath", "duration", "caption", "subset"]

# Map blacklist JSON keys to the subset names used in the CSV `subset` column.
KEY_TO_SUBSET = {
    "AudioSet": "AudioSet_SL",
    "AudioSet_SL": "AudioSet_SL",
    "BBC": "BBC_Sound_Effects",
    "BBC_Sound_Effects": "BBC_Sound_Effects",
    "FreeSound": "FreeSound",
    "SoundBible": "SoundBible",
}


def stem_of(name: str) -> str:
    """Strip a trailing known audio extension; leave everything else intact."""
    p = Path(str(name))
    if p.suffix.lower() in AUDIO_EXTS:
        return p.stem
    return str(name)


def main():
    parser = argparse.ArgumentParser(description="Remove blacklisted clips from a WavCaps CSV.")
    parser.add_argument("blacklist_json", type=Path, help="Path to a WavCaps blacklist JSON")
    parser.add_argument("in_csv", type=Path, help="Input WavCaps CSV")
    parser.add_argument("out_csv", type=Path, help="Output (filtered) CSV")
    parser.add_argument("--ignore-subset", action="store_true", help="Match on stem across all subsets instead of scoping each blacklist key to its subset")
    args = parser.parse_args()

    if not args.blacklist_json.is_file():
        parser.error(f"blacklist_json not found: {args.blacklist_json}")
    if not args.in_csv.is_file():
        parser.error(f"in_csv not found: {args.in_csv}")

    with args.blacklist_json.open(encoding="utf-8") as f:
        blacklist = json.load(f)
    if not isinstance(blacklist, dict):
        parser.error("blacklist JSON is not a dict keyed by subset")

    df = pd.read_csv(args.in_csv)
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        parser.error(f"input CSV is missing expected columns: {missing_cols}")

    n_in = len(df)
    stems = df["filepath"].map(stem_of)

    remove = pd.Series(False, index=df.index)

    print(f"[info] input: {n_in} rows, {df['filepath'].nunique()} unique clips")

    for key, ids in blacklist.items():
        bl_stems = {stem_of(x) for x in ids}

        if args.ignore_subset:
            mask = stems.isin(bl_stems)
            scope = "all subsets"
        else:
            subset = KEY_TO_SUBSET.get(key)
            if subset is None:
                print(f"[warn] blacklist key {key!r} not in KEY_TO_SUBSET; using it verbatim as a subset name")
                subset = key
            mask = (df["subset"] == subset) & stems.isin(bl_stems)
            scope = f"subset={subset}"

        remove |= mask
        n_rows = int(mask.sum())
        n_clips = int(df.loc[mask, "filepath"].nunique())
        print(f"[info] blacklist[{key!r}]: {len(bl_stems)} ids -> matched {n_clips} clips / {n_rows} rows ({scope})")

    kept = df.loc[~remove, COLUMNS]
    n_removed_rows = int(remove.sum())
    n_removed_clips = int(df.loc[remove, "filepath"].nunique())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.out_csv, index=False)

    print(f"[ok] removed {n_removed_clips} clips / {n_removed_rows} rows; kept {len(kept)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()