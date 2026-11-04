"""
merge_audiocaps_wavcaps.py

Merge an AudioCaps filepath CSV and a WavCaps filepath CSV into one dataset CSV
with columns: filepath, duration, caption, source.

Step 1 - overlap check: only WavCaps AudioSet_SL clips can overlap AudioCaps
(both come from AudioSet). WavCaps names those files 'Y' + the YouTube id, so we
strip '.flac' and a leading 'Y' and compare against the AudioCaps youtube_id
column (per-video granularity). Matching WavCaps rows are dropped; AudioCaps is
kept as the canonical copy.

Step 2 - merged output:
    filepath : <root>/wavcaps/<subset>/<filename>  or  <root>/audiocaps/<filename>
    duration : WavCaps -> clip duration; AudioCaps -> 10.0
    caption  : the caption (all caption rows kept)
    source   : 'AudioCaps' or 'WavCaps (<subset>)'

Usage:
    python merge_audiocaps_wavcaps.py audiocaps_filepaths.csv wavcaps_filepaths.csv \
        merged.csv [--root-dir audio] [--overlap-csv dropped.csv]
"""

import argparse
from pathlib import Path

import pandas as pd

AUDIOCAPS_DURATION = 10.0
AUDIOCAPS_SOURCE = "AudioCaps"


def joinp(*parts):
    return "/".join(str(p) for p in parts if p not in ("", None))


def main():
    ap = argparse.ArgumentParser(
        description="Merge AudioCaps and WavCaps filepath CSVs into one dataset CSV.")
    ap.add_argument("audiocaps_csv", type=Path)
    ap.add_argument("wavcaps_csv", type=Path)
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--root-dir", default="audio", help="Root prefix for every filepath (default 'audio'; pass a real path, or '' for pure-relative paths)")
    ap.add_argument("--overlap-csv", type=Path, default=None, help="Optional path to write the dropped overlapping WavCaps rows")
    args = ap.parse_args()

    root = args.root_dir

    if not args.audiocaps_csv.is_file():
        ap.error("audiocaps_csv not found: " + str(args.audiocaps_csv))
    if not args.wavcaps_csv.is_file():
        ap.error("wavcaps_csv not found: " + str(args.wavcaps_csv))

    ac = pd.read_csv(args.audiocaps_csv)
    wc = pd.read_csv(args.wavcaps_csv)

    for col in ("youtube_id", "caption", "filepath"):
        if col not in ac.columns:
            ap.error("audiocaps_csv missing column: " + col)
    for col in ("filepath", "duration", "caption", "subset"):
        if col not in wc.columns:
            ap.error("wavcaps_csv missing column: " + col)

    # ---- Step 1: overlap check ----
    ac_ytids = set(ac["youtube_id"].astype(str))

    wc_ytid = []
    n_as = 0
    n_as_noY = 0
    for fp, sub in zip(wc["filepath"].astype(str), wc["subset"].astype(str)):
        if sub != "AudioSet_SL":
            wc_ytid.append(None)
            continue
        n_as += 1
        stem = fp[:-5] if fp.endswith(".flac") else fp
        if stem.startswith("Y"):
            wc_ytid.append(stem[1:])
        else:
            n_as_noY += 1
            wc_ytid.append(stem)
    wc = wc.assign(_ytid=wc_ytid)

    overlap_mask = wc["_ytid"].notna() & wc["_ytid"].isin(ac_ytids)
    n_overlap = int(overlap_mask.sum())

    print("[info] AudioCaps: {} rows, {} unique videos".format(len(ac), ac["youtube_id"].nunique()))
    print("[info] WavCaps: {} rows (AudioSet_SL: {})".format(len(wc), n_as))
    if n_as_noY:
        print("[warn] {} AudioSet_SL filepath(s) do not start with 'Y'; matched on raw stem".format(n_as_noY))

    if n_overlap:
        vids = sorted(set(wc.loc[overlap_mask, "_ytid"]))
        print("[warn] {} WavCaps row(s) overlap AudioCaps ({} video(s)); dropping from WavCaps".format(n_overlap, len(vids)))
        print("       e.g. {}".format(vids[:10]))
        if args.overlap_csv is not None:
            wc.loc[overlap_mask].drop(columns="_ytid").to_csv(args.overlap_csv, index=False)
            print("[ok] wrote dropped rows -> {}".format(args.overlap_csv))
    else:
        print("[ok] no overlap between WavCaps and AudioCaps")

    wc_clean = wc.loc[~overlap_mask].copy()

    # ---- Step 2: build merged frame ----
    ac_out = pd.DataFrame({
        "filepath": [joinp(root, "audiocaps", fp) for fp in ac["filepath"].astype(str)],
        "duration": AUDIOCAPS_DURATION,
        "caption": ac["caption"].values,
        "source": AUDIOCAPS_SOURCE,
    })

    wc_out = pd.DataFrame({
        "filepath": [
            joinp(root, "wavcaps", sub, fp)
            for sub, fp in zip(wc_clean["subset"].astype(str), wc_clean["filepath"].astype(str))
        ],
        "duration": wc_clean["duration"].values,
        "caption": wc_clean["caption"].values,
        "source": "WavCaps (" + wc_clean["subset"].astype(str) + ")",
    })

    merged = pd.concat([ac_out, wc_out], ignore_index=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    print("[ok] merged {} AudioCaps + {} WavCaps = {} rows -> {}".format(len(ac_out), len(wc_out), len(merged), args.out_csv))
    print("[info] source counts:")
    for src, cnt in merged["source"].value_counts().items():
        print("        {}: {}".format(src, cnt))


if __name__ == "__main__":
    main()