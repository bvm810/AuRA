"""Copy AudioCaps audio into a test dir and a train+val dir, with local csvs.

Example
-------
    $ cd data/audioset
    $ python scripts/copy_audiocaps.py info/audiocaps_filepaths.csv \
          /scratch/audiocaps/test /scratch/audiocaps/train_val
    test:      4875 rows,  975 files copied, 0 skipped -> /scratch/audiocaps/test/test.csv
    train_val: 52313 rows, 50333 files copied, 0 skipped -> /scratch/audiocaps/train_val/train_val.csv

Files are copied flat into each destination (names like <YTID>_<start>_<end>.wav
are already unique), and the csv written into each dir has its filepath column
relative to that dir, so the dir is self-contained and movable.

Val and test have five caption rows per clip: each audio file is copied once,
but all caption rows are kept in the csv.  Already-copied files are skipped, so
the script is safe to re-run.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def copy_split(df, dest, root, name, dry_run, out_csv=None):
    dest.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0

    for src in df.filepath.unique():
        src = root / src
        dst = dest / Path(src).name
        if dst.exists():
            skipped += 1
        else:
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1

    out = df.copy()
    out["filepath"] = [Path(p).name for p in out.filepath]
    out_csv = out_csv or dest / f"{name}.csv"
    if not dry_run:
        out.to_csv(out_csv, index=False, float_format="%.3f")

    print(f"{name}: {len(df)} rows, {copied} files copied, {skipped} skipped -> {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audiocaps_csv", type=Path,
                   help="output of get_audiocaps_filepaths.py")
    p.add_argument("test_dir", type=Path, help="destination for the test split")
    p.add_argument("train_val_dir", type=Path,
                   help="destination for the train and val splits")
    p.add_argument("--root", type=Path, default=Path("."),
                   help="base dir the csv filepaths are relative to (default: .)")
    p.add_argument("--test-csv", type=Path, default=None,
                   help="where to write the test csv (default: <test_dir>/test.csv); "
                        "its filepath column still holds the bare file names")
    p.add_argument("--dry-run", action="store_true", help="report without copying")
    args = p.parse_args()

    df = pd.read_csv(args.audiocaps_csv)

    copy_split(df[df.split == "test"], args.test_dir, args.root, "test", args.dry_run, args.test_csv)
    copy_split(df[df.split != "test"], args.train_val_dir, args.root, "train_val", args.dry_run)


if __name__ == "__main__":
    main()