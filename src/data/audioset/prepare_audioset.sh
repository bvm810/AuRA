#!/usr/bin/env bash
# Build the final AudioSet training metadata, minus the AudioCaps test clips.
#
# Run from the repository root:
#     bash src/data/audioset/prepare_audioset.sh ~/data/audioset
#
# Expects (relative to the AudioSet root):
#     audio-16k/<first-char>/<YTID>_<start>_<end>.wav
#     info/unbalanced_train_segments.csv
#     info/audiocaps-splits/{train,val,test}.csv
#
# Writes:
#     info/audioset_filepaths.csv    AudioSet clips found on disk
#     info/not_found.csv             clips in the metadata but missing from disk
#     info/audiocaps_filepaths.csv   merged AudioCaps splits
#     metadata.csv                   final training list, with a train / validation split

set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?usage: prepare_audioset.sh AUDIOSET_ROOT}"

# Everything below runs from the AudioSet root so the paths written into the csvs
# are relative to it (audio-16k/z/...), not tied to this machine's HOME.
cd "$ROOT"
AUDIO="audio-16k"
INFO="info"

PYTHON="${PYTHON:-python3}"

for f in "$INFO/unbalanced_train_segments.csv" \
         "$INFO/audiocaps-splits/train.csv" \
         "$INFO/audiocaps-splits/val.csv" \
         "$INFO/audiocaps-splits/test.csv"; do
    [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done
[ -d "$AUDIO" ] || { echo "missing audio dir: $AUDIO" >&2; exit 1; }

echo "[1/4] AudioSet filepaths"
"$PYTHON" "$SCRIPTS/get_audioset_filepaths.py" \
    "$INFO/unbalanced_train_segments.csv" \
    "$AUDIO" \
    "$INFO/audioset_filepaths.csv" \
    --missing-csv "$INFO/not_found.csv"

echo "[2/4] AudioCaps filepaths"
"$PYTHON" "$SCRIPTS/get_audiocaps_filepaths.py" \
    "$INFO/audiocaps-splits" \
    "$AUDIO" \
    "$INFO/audiocaps_filepaths.csv"

echo "[3/4] Blacklisting AudioCaps test clips"
"$PYTHON" "$SCRIPTS/blacklist_audiocaps.py" \
    "$INFO/audioset_filepaths.csv" \
    "$INFO/audiocaps_filepaths.csv" \
    "metadata.csv"

echo "[4/4] Train / validation split"
"$PYTHON" "$SCRIPTS/make_splits.py" "metadata.csv"

echo "done -> $ROOT/metadata.csv"