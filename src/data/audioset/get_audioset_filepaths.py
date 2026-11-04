"""Read the AudioSet segments csv and write YTID, start, end, filepath.

Example
-------
    $ python get_filepaths.py balanced_train_segments.csv audio-16k filepaths.csv
    found: 21832, missing: 61 -> filepaths.csv

Rows whose .wav is not on disk are dropped; pass --keep-missing to keep them.
The output looks like:

    YTID,start_seconds,end_seconds,filepath
    ---1_cCGK4M,0.000,10.000,audio-16k/-/---1_cCGK4M_0.000_10.000.wav
    ---2_BBVHAA,30.000,40.000,audio-16k/-/---2_BBVHAA_30.000_40.000.wav
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("segments_csv", type=Path)
    p.add_argument("audio_root", type=Path, help="e.g. audio-16k")
    p.add_argument("out_csv", type=Path)
    p.add_argument("--keep-missing", action="store_true",
                   help="keep rows whose wav file does not exist")
    p.add_argument("--missing-csv", type=Path, default=None,
                   help="also write the rows whose wav file does not exist here")
    args = p.parse_args()

    df = pd.read_csv(
        args.segments_csv,
        comment="#",
        header=None,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
        skipinitialspace=True,
        usecols=[0, 1, 2],
    )

    df["filepath"] = [
        str(args.audio_root / y[0] / f"{y}_{s:.3f}_{e:.3f}.wav")
        for y, s, e in zip(df.YTID, df.start_seconds, df.end_seconds)
    ]

    on_disk = {str(p) for p in args.audio_root.glob("*/*.wav")}
    exists = df.filepath.isin(on_disk)
    print(f"found: {exists.sum()}, missing: {(~exists).sum()} -> {args.out_csv}")

    if args.missing_csv:
        df[~exists].to_csv(args.missing_csv, index=False, float_format="%.3f")

    if not args.keep_missing:
        df = df[exists]
    df.to_csv(args.out_csv, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()