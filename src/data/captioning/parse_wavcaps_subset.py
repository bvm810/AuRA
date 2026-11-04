#!/usr/bin/env python3
"""
build_wavcaps_csv.py

Read one WavCaps caption JSON file (one of the four subset files, e.g.
sb_final.json / as_final.json / bbc_final.json / fsd_final.json) and write a
clean CSV with the columns:

    filepath, duration, caption, subset

Column definitions
------------------
    filepath : the entry's `id` field with a `.flac` suffix (see note below)
    duration : the entry's `duration` field (float seconds)
    caption  : the entry's `caption` field
    subset   : the subset name passed on the command line
               (AudioSet_SL, BBC_Sound_Effects, FreeSound, SoundBible)

`filepath` is a bare filename (no directory prefix), mirroring the AudioCaps
`audiocaps_filepaths.csv` convention.

Multiple captions per entry
---------------------------
If an entry's `caption` field is a list, one row is emitted per caption (the
filepath/duration/subset are repeated). A scalar caption yields a single row.
WavCaps ships one caption per audio for all four subsets, so in practice this
just yields one row per clip, but the list case is handled for robustness.

Appending
---------
If <out_csv> already exists (and is non-empty), the new rows are appended to the
bottom without rewriting the header. This lets you accumulate all four subsets
into a single CSV by pointing successive runs at the same output file. If the
existing file's header doesn't match the expected columns, a warning is printed
but the append still proceeds.

Note on the `.flac` suffix
--------------------------
By default the script strips a pre-existing audio extension (.wav/.flac/.mp3)
from the id before appending `.flac`. This matters because some WavCaps subsets
store the id WITH an extension (notably AudioSet_SL, e.g. "Yxxxx.wav"), and a
naive `id + '.flac'` would yield "Yxxxx.wav.flac". For subsets whose ids are
already extensionless (SoundBible, FreeSound, BBC) this stripping is a no-op and
behaves exactly like `id + '.flac'`.

Pass --no-strip-ext to get the literal `id + '.flac'` behavior instead.

Usage
-----
    python build_wavcaps_csv.py <json_file> <subset> <out_csv> [--no-strip-ext]

Example
-------
    python build_wavcaps_csv.py sb_final.json SoundBible wavcaps.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# The four canonical WavCaps subset names.
SUBSETS = ["AudioSet_SL", "BBC_Sound_Effects", "FreeSound", "SoundBible"]

# Audio extensions that may be baked into an entry's `id` and should be stripped
# before appending `.flac`.
AUDIO_EXTS = {".wav", ".flac", ".mp3"}

# Columns to emit, in order.
COLUMNS = ["filepath", "duration", "caption", "subset"]


def make_filepath(raw_id: str, strip_ext: bool) -> str:
    """Turn an entry's `id` into a `<stem>.flac` filename."""
    stem = str(raw_id)
    if strip_ext:
        p = Path(stem)
        if p.suffix.lower() in AUDIO_EXTS:
            stem = p.stem
    return f"{stem}.flac"


def normalize_captions(caption_field):
    """Return a list of captions for an entry.

    A list/tuple yields one caption per element; a scalar yields a
    single-element list.
    """
    if isinstance(caption_field, (list, tuple)):
        return list(caption_field)
    return [caption_field]


def build_dataframe(entries, subset: str, strip_ext: bool) -> pd.DataFrame:
    rows = []
    skipped = 0            # entries with no usable id
    empty_caption = 0      # entries whose caption list was empty
    multi = 0              # entries that produced more than one row

    for e in entries:
        raw_id = e.get("id")
        # An entry with no id can't be given a filepath; drop it rather than
        # emit a broken row.
        if raw_id is None or str(raw_id) == "":
            skipped += 1
            continue

        filepath = make_filepath(raw_id, strip_ext)
        duration = e.get("duration")
        captions = normalize_captions(e.get("caption"))

        if len(captions) == 0:
            empty_caption += 1
            continue
        if len(captions) > 1:
            multi += 1

        for cap in captions:
            rows.append(
                {
                    "filepath": filepath,
                    "duration": duration,
                    "caption": cap,
                    "subset": subset,
                }
            )

    if skipped:
        print(f"[warn] skipped {skipped} entr(y/ies) with a missing/empty id")
    if empty_caption:
        print(f"[warn] skipped {empty_caption} entr(y/ies) with an empty caption list")
    if multi:
        print(f"[info] {multi} entr(y/ies) had multiple captions (expanded to one row each)")

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Report (but keep) rows with missing caption/duration so the user knows.
    n_no_caption = int(df["caption"].isna().sum())
    n_no_duration = int(df["duration"].isna().sum())
    if n_no_caption:
        print(f"[warn] {n_no_caption} row(s) have a missing caption")
    if n_no_duration:
        print(f"[warn] {n_no_duration} row(s) have a missing duration")

    return df


def write_csv(df: pd.DataFrame, out_csv: Path, subset: str) -> None:
    """Write fresh (with header) or append (no header) if the file exists."""
    appending = out_csv.is_file() and out_csv.stat().st_size > 0

    if appending:
        existing_header = out_csv.open(encoding="utf-8").readline().rstrip("\r\n")
        expected_header = ",".join(COLUMNS)
        if existing_header != expected_header:
            print(f"[warn] existing header {existing_header!r} does not match expected {expected_header!r}; appending anyway")
        df.to_csv(out_csv, mode="a", header=False, index=False)
        print(f"[ok] {subset}: appended {len(df)} rows -> {out_csv}")
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[ok] {subset}: wrote {len(df)} rows -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Build a clean WavCaps CSV (filepath, duration, caption, subset) from one subset's caption JSON file.")
    parser.add_argument("json_file", type=Path, help="Path to a WavCaps caption JSON (e.g. sb_final.json)")
    parser.add_argument("subset", choices=SUBSETS, help="Subset name for the `subset` column")
    parser.add_argument("out_csv", type=Path, help="Path to write (or append to) the output CSV")
    parser.add_argument("--no-strip-ext", action="store_true", help="Do NOT strip a pre-existing audio extension from the id; use literal `id + '.flac'`")
    args = parser.parse_args()

    if not args.json_file.is_file():
        parser.error(f"json_file not found: {args.json_file}")

    with args.json_file.open(encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("data")
    if entries is None:
        parser.error("JSON has no top-level 'data' list; is this a WavCaps caption file?")

    df = build_dataframe(entries, args.subset, strip_ext=not args.no_strip_ext)
    write_csv(df, args.out_csv, args.subset)


if __name__ == "__main__":
    main()