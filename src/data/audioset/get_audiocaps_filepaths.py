"""Merge the AudioCaps split csvs into one table with a split column and filepath.

Example
-------
    $ python get_audiocaps_filepaths.py audiocaps audio-16k audiocaps_filepaths.csv
    train: 48901/49838, val: 2400/2475, test: 4720/4875 -> audiocaps_filepaths.csv

Counts are rows, not clips.  Train has one caption per clip; val and test have
five captions per clip, all sharing the same youtube_id and start_time, so
their rows repeat the same filepath five times (495 and 975 unique clips).
Use drop_duplicates("youtube_id") if you need one row per audio file.

Expects <audiocaps_root>/{train,val,test}.csv with the official AudioCaps
columns (audiocap_id, youtube_id, start_time, caption).  Clips are assumed to
be 10s, so end_time = start_time + 10, and paths follow the AudioSet
convention.  Rows whose .wav is not on disk are dropped; pass --keep-missing
to keep them.  The output looks like:

    audiocap_id,youtube_id,start_seconds,end_seconds,split,caption,filepath
    91139,r1nicOVtvkQ,130,140.000,train,A woman talks nearby ...,audio-16k/r/r1nicOVtvkQ_130.000_140.000.wav
"""

import argparse
from pathlib import Path

import pandas as pd

DURATION = 10.0
SPLITS = ["train", "val", "test"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audiocaps_root", type=Path, help="dir with train/val/test.csv")
    p.add_argument("audio_root", type=Path, help="e.g. audio-16k")
    p.add_argument("out_csv", type=Path)
    p.add_argument("--keep-missing", action="store_true",
                   help="keep rows whose wav file does not exist")
    args = p.parse_args()

    dfs = []
    for split in SPLITS:
        df = pd.read_csv(args.audiocaps_root / f"{split}.csv")
        df["split"] = split
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    df = df.rename(columns={"start_time": "start_seconds"})
    df["end_seconds"] = df.start_seconds + DURATION
    df["filepath"] = [
        str(args.audio_root / y[0] / f"{y}_{s:.3f}_{e:.3f}.wav")
        for y, s, e in zip(df.youtube_id, df.start_seconds, df.end_seconds)
    ]

    on_disk = {str(p) for p in args.audio_root.glob("*/*.wav")}
    exists = df.filepath.isin(on_disk)
    print(", ".join(f"{s}: {exists[df.split == s].sum()}/{(df.split == s).sum()}"
                    for s in SPLITS) + f" -> {args.out_csv}")

    if not args.keep_missing:
        df = df[exists]
    df = df[["audiocap_id", "youtube_id", "start_seconds", "end_seconds",
             "split", "caption", "filepath"]]
    df.to_csv(args.out_csv, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()