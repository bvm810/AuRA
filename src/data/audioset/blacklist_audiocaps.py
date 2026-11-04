"""Remove AudioCaps test clips from the AudioSet filelist to make a training csv.

Example
-------
    $ python blacklist_audiocaps.py filepaths.csv audiocaps_filepaths.csv train_metadata.csv
    audioset: 1934112 rows, blacklisted 975 youtube_ids from [test] -> removed 968, kept 1933144

Takes the outputs of get_filepaths.py and get_audiocaps_filepaths.py.  Matching
is on youtube_id by default, so every segment of a blacklisted video is dropped,
not just the exact 10s window AudioCaps used.  Pass --by-segment to match on
(youtube_id, start_seconds) instead, and --splits test val to hold out the
validation clips too.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audioset_csv", type=Path, help="output of get_filepaths.py")
    p.add_argument("audiocaps_csv", type=Path, help="output of get_audiocaps_filepaths.py")
    p.add_argument("out_csv", type=Path)
    p.add_argument("--splits", nargs="+", default=["test"],
                   help="AudioCaps splits to blacklist (default: test)")
    p.add_argument("--by-segment", action="store_true",
                   help="match on (youtube_id, start_seconds) instead of youtube_id")
    args = p.parse_args()

    audioset = pd.read_csv(args.audioset_csv)
    audiocaps = pd.read_csv(args.audiocaps_csv)
    blacklist = audiocaps[audiocaps.split.isin(args.splits)]

    if args.by_segment:
        keys = set(zip(blacklist.youtube_id, blacklist.start_seconds.round(3)))
        drop = pd.Series(list(zip(audioset.YTID, audioset.start_seconds.round(3))),
                         index=audioset.index).isin(keys)
    else:
        drop = audioset.YTID.isin(set(blacklist.youtube_id))

    kept = audioset[~drop]
    print(f"audioset: {len(audioset)} rows, "
          f"blacklisted {blacklist.youtube_id.nunique()} youtube_ids from {args.splits} "
          f"-> removed {drop.sum()}, kept {len(kept)}")
    kept.to_csv(args.out_csv, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()