#!/usr/bin/env python3
"""Probe audio files listed in a caption CSV: add sample_rate / num_samples,
fix the duration column, drop too-short files, and rebuild the train/val/test
splits. Reads the input CSV and the audio read-only; writes a new CSV.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torchaudio


def probe(path):
    """Header only, no decoding: (sample_rate, num_samples, bytes, channels)."""
    try:
        i = torchaudio.info(str(path))
        return i.sample_rate, i.num_frames, path.stat().st_size, i.num_channels
    except Exception:
        return None, None, None, None


def norm(s):
    """Loose source matching: 'AudioSet_SL', 'audioset sl' and 'audioset-sl' all agree."""
    return "".join(c for c in str(s).casefold() if c.isalnum())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="input CSV (never modified)")
    p.add_argument("out", type=Path, help="output CSV")
    p.add_argument("--root", type=Path, default=None,
                   help="directory the filepath column is relative to "
                        "(default: the input CSV's directory)")
    p.add_argument("--min-duration", type=float, default=1.0,
                   help="drop files shorter than this many seconds")
    p.add_argument("--max-duration", type=float, default=None,
                   help="drop files longer than this many seconds (off by default; "
                        "use it to discard files whose header reports a bogus length)")
    p.add_argument("--max-captions", type=int, default=4,
                   help="max caption rows kept per val/test file")
    p.add_argument("--exclude-from-eval", nargs="+", default=[], metavar="SOURCE",
                   help="sources that must never appear in val/test (e.g. "
                        "'AudioSet_SL'); their files stay in train, they are not "
                        "dropped. Repeatable or comma-separated, matched loosely "
                        "against the source column")
    p.add_argument("--source-col", default="source",
                   help="column --exclude-from-eval matches against")
    p.add_argument("--eval-frac", type=float, default=0.01)
    p.add_argument("--eval-cap", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    df = pd.read_csv(a.csv)
    root = a.root if a.root is not None else a.csv.parent
    log, n_rows_in, n_files_in = [], len(df), df["filepath"].nunique()

    # Keep whatever the file already calls the validation split.
    val_name = "val" if "val" in set(df.get("split", pd.Series(dtype=str))) else "validation"

    # (1)+(2) probe each unique file once, in parallel (torchaudio.info only
    # reads the header, so this is cheap and I/O bound).
    files = df["filepath"].unique()
    with ThreadPoolExecutor(a.workers) as ex:
        meta = list(ex.map(lambda f: probe(root / f), files))
    work = ["sample_rate", "num_samples", "_bytes", "_ch"]
    df = df.join(pd.DataFrame(meta, index=files, columns=work), on="filepath")

    unreadable = df["sample_rate"].isna()
    if unreadable.any():
        print(f"warning: could not read {df.loc[unreadable, 'filepath'].nunique()} files,"
              f" e.g. {df.loc[unreadable, 'filepath'].iloc[0]}", file=sys.stderr)
        log.append(("dropped, unreadable/missing", unreadable.sum(),
                    df.loc[unreadable, "filepath"].nunique()))
        df = df[~unreadable].copy()
    df[work] = df[work].astype("int64")

    # (4) trust the headers, not the duration column (sample rates vary per file)
    true_duration = df["num_samples"] / df["sample_rate"]
    if "duration" in df:
        bad = (df["duration"] - true_duration).abs() > 1e-3
        log.append(("corrected durations", bad.sum(), df.loc[bad, "filepath"].nunique()))
    df["duration"] = true_duration

    # (3) drop files outside the wanted length range
    short = df["duration"] < a.min_duration
    log.append((f"dropped, under {a.min_duration}s", short.sum(),
                df.loc[short, "filepath"].nunique()))
    df = df[~short].copy()
    if a.max_duration is not None:
        long_ = df["duration"] > a.max_duration
        log.append((f"dropped, over {a.max_duration}s", long_.sum(),
                    df.loc[long_, "filepath"].nunique()))
        df = df[~long_].copy()
    if df.empty:
        sys.exit("error: no rows left after filtering")

    # (5) rebuild the splits on the surviving rows
    uniq = df["filepath"].drop_duplicates()
    n = max(1, min(a.eval_cap, int(len(df) * a.eval_frac)))

    # sources held out of eval keep all their rows, they just cannot be sampled.
    # A file is excluded if *any* of its rows names an excluded source.
    pool = uniq
    if a.exclude_from_eval:
        if a.source_col not in df:
            sys.exit(f"error: --exclude-from-eval needs a '{a.source_col}' column; "
                     f"columns are {list(df.columns)}")
        want = {norm(s) for arg in a.exclude_from_eval for s in arg.split(",") if s.strip()}
        col = df[a.source_col].map(norm)
        missing = want - set(col)
        if missing:
            print(f"warning: no rows for source(s) {sorted(missing)}; present: "
                  f"{sorted(df[a.source_col].astype(str).unique())}", file=sys.stderr)
        held = df.loc[col.isin(want), "filepath"]
        pool = uniq[~uniq.isin(set(held))]
        log.append((f"kept in train, excluded from eval", len(held), held.nunique()))

    if 2 * n > len(pool):
        n = len(pool) // 2
        print(f"warning: only {len(pool)} files eligible for eval, shrinking each "
              f"split to {n}", file=sys.stderr)
        if n == 0:
            sys.exit("error: fewer than 2 files are eligible for the eval splits")
    picked = pool.sample(2 * n, random_state=a.seed).tolist()
    assign = {f: (val_name if i < n else "test") for i, f in enumerate(picked)}
    df["split"] = df["filepath"].map(assign).fillna("train")

    # cap the number of captions per eval file, keeping original row order
    is_eval = df["split"] != "train"
    kept = pd.concat([
        df[~is_eval],
        df[is_eval].groupby("filepath", sort=False).head(a.max_captions),
    ]).sort_index()
    log.append(("dropped, extra eval captions", len(df) - len(kept), 0))
    df = kept

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_bytes", "_ch"]).to_csv(a.out, index=False)

    # ---- summary. durations and sizes are summed once per file, not once per caption.
    per_file = df.drop_duplicates("filepath")
    tbl = per_file.groupby("split").agg(files=("filepath", "size"),
                                        hours=("duration", lambda s: s.sum() / 3600),
                                        GiB=("_bytes", lambda s: s.sum() / 2 ** 30))
    tbl.insert(0, "rows", df.groupby("split").size())
    tbl = tbl.reindex([s for s in ("train", val_name, "test") if s in tbl.index])

    print(f"\n{'=' * 52}\n{a.csv} -> {a.out}\n{'=' * 52}")
    print(f"{'input':<30}{n_rows_in:>7} rows{n_files_in:>8} files")
    for label, rows, nfiles in log:
        print(f"{label:<30}{rows:>7} rows" + (f"{nfiles:>8} files" if nfiles else ""))
    print(f"{'output':<30}{len(df):>7} rows{len(per_file):>8} files")
    print(f"\ntotal duration: {per_file['duration'].sum() / 3600:.2f} h    "
          f"total size: {per_file['_bytes'].sum() / 2 ** 30:.2f} GiB    "
          f"sample rates: {sorted(map(int, df['sample_rate'].unique()))}")
    print(f"\n{tbl.round(2).to_string()}")
    if a.source_col in df:
        print(f"\n{pd.crosstab(df[a.source_col], df['split']).to_string()}")

    # ---- header cross-check, free: file size gives an independent duration estimate.
    # For PCM, seconds ~= bytes / (sample_rate * channels * bytes_per_sample). This
    # under-reads compressed formats, so it is a lower bound, not a measurement --
    # but if it disagrees with the header total by orders of magnitude, the headers
    # are lying (num_frames is an estimate for mp3 and some ffmpeg-demuxed formats).
    est_h = (per_file["_bytes"] /
             (per_file["sample_rate"] * per_file["_ch"] * 2)).sum() / 3600
    d = per_file["duration"]
    print(f"\nheader-derived total : {d.sum() / 3600:,.1f} h")
    print(f"size-implied total   : {est_h:,.1f} h  (lower bound, exact only for 16-bit PCM)")
    print(f"per-file seconds     : median {d.median():.1f}  p99 {d.quantile(0.99):.1f}  "
          f"max {d.max():.4g}")
    if d.sum() > 10 * max(est_h * 3600, 1) or d.max() > 86400:
        print("\n!! the header total is far above what these file sizes can hold, so some\n"
              "   num_frames values are junk. worst offenders:", file=sys.stderr)
        print(per_file.nlargest(5, "duration")[
                  ["filepath", "sample_rate", "num_samples", "duration", "_bytes"]
              ].to_string(index=False), file=sys.stderr)
        print("   rerun with --max-duration to exclude them.", file=sys.stderr)


if __name__ == "__main__":
    main()