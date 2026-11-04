"""Build the isolated WavCaps eval set: the test split of a manifest, minus its AudioCaps files.

    python -m src.data.captioning.make_isolated_eval long-segments.csv isolated-eval.csv

The output keeps every column of the manifest and adds `basename`, the file name of each test file.
"""
import argparse
from pathlib import Path

import pandas as pd

EXCLUDED_SOURCE = "AudioCaps"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="manifest written by manifest.py (never modified)")
    p.add_argument("out", type=Path, help="output CSV")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    df = df[(df["split"] == "test") & (df["source"] != EXCLUDED_SOURCE)].copy()
    df["basename"] = df["filepath"].map(lambda f: Path(f).name)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} test rows, {df['filepath'].nunique()} files -> {a.out}")


if __name__ == "__main__":
    main()
