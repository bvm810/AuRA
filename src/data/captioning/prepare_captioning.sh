#!/usr/bin/env bash
# Build the captioning metadata used to train the generators: WavCaps minus AudioCaps, plus AudioCaps train + val.
#
# Run from the repository root:
#     bash src/data/captioning/prepare_captioning.sh ~/data/captioning
#
# Expects (relative to the captioning root):
#     json_files/                          the WavCaps captions and blacklists (download.sh)
#     audio/wavcaps/<subset>/<id>.flac     the extracted WavCaps audio (extract.sh)
#     audio/audiocaps/                     the AudioCaps train + val clips and train_val.csv (copy_audiocaps.py)
#
# Writes:
#     long-segments.csv    files of 10 s to 1 h, with 1024-file validation / test splits outside AudioSet_SL
#     isolated-eval.csv    the test split without its AudioCaps files
#
# Intermediate csvs are written to a temporary folder and deleted at the end.

set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?usage: prepare_captioning.sh CAPTIONING_ROOT}"

# Everything below runs from the captioning root so the paths written into the csvs
# are relative to it (audio/wavcaps/...), not tied to this machine's HOME.
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PYTHON="${PYTHON:-python3}"

# in the order of the original csv
i=1
for subset in SoundBible:sb AudioSet_SL:as BBC_Sound_Effects:bbc FreeSound:fsd; do
    name="${subset%%:*}"
    echo "[$i/8] WavCaps $name captions"
    "$PYTHON" "$SCRIPTS/parse_wavcaps_subset.py" \
        "json_files/$name/${subset##*:}_final.json" \
        "$name" \
        "$TMP/wavcaps_no_blacklist.csv"
    i=$((i + 1))
done

echo "[5/8] Blacklisting AudioCaps clips from WavCaps"
"$PYTHON" "$SCRIPTS/blacklist_wavcaps.py" \
    "json_files/blacklist/blacklist_exclude_all_audiocaps.json" \
    "$TMP/wavcaps_no_blacklist.csv" \
    "$TMP/wavcaps.csv"

echo "[6/8] Merging AudioCaps train + val"
"$PYTHON" "$SCRIPTS/merge_audiocaps_wavcaps.py" \
    "audio/audiocaps/train_val.csv" \
    "$TMP/wavcaps.csv" \
    "$TMP/metadata.csv"

echo "[7/8] Manifest"
"$PYTHON" "$SCRIPTS/manifest.py" \
    "$TMP/metadata.csv" \
    "long-segments.csv" \
    --root . \
    --min-duration 10 \
    --max-duration 3600 \
    --exclude-from-eval "WavCaps (AudioSet_SL)"

echo "[8/8] Isolated eval set"
"$PYTHON" "$SCRIPTS/make_isolated_eval.py" \
    "long-segments.csv" \
    "isolated-eval.csv"

echo "done -> $ROOT/long-segments.csv, $ROOT/isolated-eval.csv"