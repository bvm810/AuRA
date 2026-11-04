#!/usr/bin/env bash
#
# wavcaps.sh — join and extract the WavCaps split zip archives.
#
# Run this on the machine where the audio will live (your HPC cluster), after
# rsync'ing the Zip_files/ tree across. It does, per subset:
#
#   1. verify every part is present and the numbering has no gaps
#   2. check free disk space before touching anything
#   3. join the parts into one archive   (zip -s 0, with -F / -FF fallbacks)
#   4. extract it                        (unzip -n, so re-runs resume)
#   5. count the extracted files against the published totals
#   6. delete the joined archive         (the parts are kept unless you pass -d)
#
# Usage:
#   ./wavcaps.sh [-i INDIR] [-o OUTDIR] [-t TMPDIR] [-j N] [-k] [-d] [-n] [SUBSET ...]
#
#   -i INDIR    directory containing Zip_files/   (default: ./wavcaps)
#   -o OUTDIR   where the .flac files go          (default: ./wavcaps_audio)
#   -t TMPDIR   scratch for the joined archive    (default: OUTDIR/.joining)
#   -j N        process N subsets in parallel     (default: 1 — see note below)
#   -k          keep the joined archive after extraction
#   -d          delete the zip parts after a verified extraction
#   -p          preserve the archive's internal paths (default: flatten them to
#               OUTDIR/<subset>/, dropping the mnt/fast/nobackup/... prefix)
#   -n          dry run: verify parts and report disk needs, extract nothing
#   SUBSET      one or more of: AudioSet_SL BBC_Sound_Effects FreeSound SoundBible
#               (default: all four, smallest first)
#
# Examples:
#   ./wavcaps.sh -n                                  # preflight check
#   ./wavcaps.sh -i /scratch/wavcaps -o /scratch/audio SoundBible
#   ./wavcaps.sh -i /scratch/wavcaps -o /scratch/audio -d
#
# DISK SPACE
#   Joining writes a second full copy of the subset before extraction, so peak
#   usage is roughly 2x the subset size on TMPDIR + the extracted size on OUTDIR.
#   FreeSound alone is ~534 GB of parts. Subsets are processed one at a time and
#   the joined archive is deleted immediately after, so the 2x spike is per
#   subset, not for the whole dataset. Put TMPDIR and OUTDIR on the same
#   filesystem as each other, on scratch, never on your home quota.
#
# PARALLELISM
#   -j > 1 runs whole subsets concurrently and multiplies the peak disk usage by
#   the same factor. On most shared filesystems this is also slower, not faster,
#   because joining is sequential I/O. Leave it at 1 unless you have measured
#   otherwise and have the space.
#
# SLURM
#   This is a long single-node I/O job. Do not run it on a login node.
#
#     #!/bin/bash
#     #SBATCH --job-name=wavcaps-extract
#     #SBATCH --time=24:00:00
#     #SBATCH --cpus-per-task=4
#     #SBATCH --mem=8G
#     ./wavcaps.sh -i /scratch/$USER/wavcaps -o /scratch/$USER/wavcaps_audio
#
#   If your queue has a wall-clock limit shorter than the job needs, submit one
#   subset per job — each is independent, and unzip -n means a timed-out job
#   resumes cleanly on the next attempt.
#
# Requires: zip, unzip, awk, df.

set -uo pipefail

INDIR="./wavcaps"
OUTDIR="./wavcaps_audio"
TMPDIR_OVERRIDE=""
JOBS=1
KEEP_JOINED=0
DELETE_PARTS=0
FLATTEN=1
DRY_RUN=0

# smallest first, so failures surface cheaply
ALL_SUBSETS=(SoundBible BBC_Sound_Effects AudioSet_SL FreeSound)

# published clip counts, used as the post-extraction sanity check
expected_count() {
  case "$1" in
    AudioSet_SL)       echo 108317 ;;
    BBC_Sound_Effects) echo 31201  ;;
    FreeSound)         echo 262300 ;;
    SoundBible)        echo 1232   ;;
    *)                 echo 0      ;;
  esac
}

usage() { sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts ":i:o:t:j:kdpnh" opt; do
  case "$opt" in
    i) INDIR="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    t) TMPDIR_OVERRIDE="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    k) KEEP_JOINED=1 ;;
    d) DELETE_PARTS=1 ;;
    p) FLATTEN=0 ;;
    n) DRY_RUN=1 ;;
    h) usage 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; usage 1 ;;
    :)  echo "option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))

SUBSETS=("$@")
[[ ${#SUBSETS[@]} -eq 0 ]] && SUBSETS=("${ALL_SUBSETS[@]}")

for cmd in zip unzip awk df; do
  command -v "$cmd" >/dev/null || { echo "error: $cmd not found" >&2; exit 1; }
done

ZIPDIR="${INDIR}/Zip_files"
[[ -d "$ZIPDIR" ]] || { echo "error: $ZIPDIR not found — is -i pointing at the download root?" >&2; exit 1; }

human() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1024/1024/1024}'; }
avail_bytes() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {print $4 * 1024}'; }

# ---------------------------------------------------------------------------
# verify_parts <subset> — check the split archive is complete.
# Prints "<part-count> <total-bytes>" on success.
# ---------------------------------------------------------------------------
verify_parts() {
  local subset="$1" dir="${ZIPDIR}/$1"

  [[ -d "$dir" ]] || { echo "  ! directory missing: $dir" >&2; return 1; }
  [[ -f "${dir}/${subset}.zip" ]] || {
    echo "  ! final part ${subset}.zip missing — this holds the central directory" >&2
    return 1
  }

  # Part numbers are zero-padded to 2 but run past 99 (.z100, .z121), so parse
  # the number rather than relying on lexical sort.
  local nums=() n maxn=0
  for f in "${dir}/${subset}".z[0-9]*; do
    [[ -e "$f" ]] || continue
    n="${f##*.z}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    n=$((10#$n))
    nums+=("$n")
    (( n > maxn )) && maxn=$n
  done

  # Not every subset is split. SoundBible is ~1 GB and ships as a single
  # ordinary .zip with no .zNN parts — that is valid, not a missing-parts error.
  if (( maxn > 0 )); then
    # Every number from 1..maxn must be present; .zip is the (maxn+1)-th part.
    local present=() missing=()
    for n in "${nums[@]}"; do present[$n]=1; done
    for ((i = 1; i <= maxn; i++)); do
      [[ -z "${present[$i]:-}" ]] && missing+=("$(printf '%s.z%02d' "$subset" "$i")")
    done

    if (( ${#missing[@]} > 0 )); then
      echo "  ! ${#missing[@]} part(s) missing: ${missing[*]:0:8}$( ((${#missing[@]} > 8)) && echo ' ...')" >&2
      echo "  ! re-run the download script before extracting" >&2
      return 1
    fi
  fi

  # Build an explicit file list rather than re-globbing, so the unsplit case
  # (no .zNN files) doesn't hand a literal glob string to du/stat.
  local files=("${dir}/${subset}.zip")
  for ((i = 1; i <= maxn; i++)); do
    for f in "${dir}/${subset}".z[0-9]*; do
      [[ -e "$f" ]] && files+=("$f")
    done
    break
  done

  # du -b is GNU-only; fall back to stat if it isn't available.
  local total
  total=$(du -bc "${files[@]}" 2>/dev/null | tail -1 | awk '{print $1}')
  if [[ -z "$total" || "$total" == "0" ]]; then
    total=$(stat -c%s "${files[@]}" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  fi
  [[ -z "$total" ]] && total=0

  echo "$((maxn + 1)) $total"
}

# ---------------------------------------------------------------------------
# process_subset <subset>
# ---------------------------------------------------------------------------
process_subset() {
  local subset="$1"
  local dir="${ZIPDIR}/${subset}"
  local tmp="${TMPDIR_OVERRIDE:-${OUTDIR}/.joining}"
  local joined="${tmp}/${subset}_joined.zip"
  local marker="${OUTDIR}/.${subset}.done"

  echo
  echo "=== ${subset} ==============================================="

  if [[ -f "$marker" ]]; then
    echo "  already extracted (marker: $marker) — skipping"
    return 0
  fi

  local info parts bytes
  info=$(verify_parts "$subset") || return 1
  read -r parts bytes <<< "$info"
  echo "  parts:      ${parts} files, $(human "$bytes")"

  # Peek at the layout. The central directory lives in the final part, so unzip
  # can list the contents without joining first.
  local top
  top=$(unzip -l "${dir}/${subset}.zip" 2>/dev/null \
        | awk 'NR>3 && $4 != "" {print $4}' \
        | awk -F/ 'NF>1 {print $1; exit}')
  [[ -n "$top" ]] && echo "  layout:     ${top}/ (inside the archive)"

  # Disk check: joining needs another full copy, extraction roughly one more
  # (flac is already compressed, so the zip barely shrinks anything).
  mkdir -p "$tmp" "$OUTDIR"
  local free_tmp free_out
  free_tmp=$(avail_bytes "$tmp")
  free_out=$(avail_bytes "$OUTDIR")
  echo "  free:       $(human "$free_tmp") on scratch, $(human "$free_out") on target"

  if (( free_tmp < bytes )); then
    echo "  ! not enough space to join: need $(human "$bytes"), have $(human "$free_tmp")" >&2
    return 1
  fi
  if (( free_out < bytes )); then
    echo "  ! not enough space to extract: need ~$(human "$bytes"), have $(human "$free_out")" >&2
    return 1
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    if (( parts > 1 )); then
      echo "  [dry run] would join ${parts} parts -> ${joined}"
    else
      echo "  [dry run] single archive, no join needed"
    fi
    echo "  [dry run] would extract -> ${OUTDIR}/${subset}/"
    return 0
  fi

  # --- join ---------------------------------------------------------------
  # Only split archives need joining. A single .zip (SoundBible) is read
  # directly, which also skips the 2x disk spike.
  local src
  if (( parts > 1 )); then
    if [[ -f "$joined" ]]; then
      echo "  reusing existing joined archive"
    else
      echo "  joining ${parts} parts (this is the slow part — pure sequential I/O)"
      if ! zip -q -s 0 "${dir}/${subset}.zip" --out "$joined"; then
        echo "  join failed, retrying with zip -F (fix)" >&2
        rm -f "$joined"
        if ! zip -q -F "${dir}/${subset}.zip" --out "$joined"; then
          echo "  zip -F failed, retrying with zip -FF (aggressive fix)" >&2
          rm -f "$joined"
          if ! zip -q -FF "${dir}/${subset}.zip" --out "$joined"; then
            echo "  ! could not reconstruct ${subset} — a part is likely corrupt." >&2
            echo "  ! delete that part and re-download it, then re-run." >&2
            rm -f "$joined"
            return 1
          fi
        fi
      fi
    fi
    echo "  joined:     $(human "$(du -b "$joined" | awk '{print $1}')")"
    src="$joined"
  else
    echo "  single archive, no join needed"
    src="${dir}/${subset}.zip"
  fi

  # --- extract ------------------------------------------------------------
  # -n never overwrites, so an interrupted run resumes instead of redoing work.
  # UNZIP_DISABLE_ZIPBOMB_DETECTION handles the "overlapped components" refusal
  # that newer Info-ZIP builds throw on these archives.
  # Each subset extracts into its own staging dir. The archives were built from
  # absolute paths on the authors' machine, so their internal layout is
  # something like mnt/fast/nobackup/.../flac/xxx.flac — staging per subset
  # keeps those trees from colliding before they get flattened below.
  local stage="${OUTDIR}/.stage_${subset}"
  [[ "$FLATTEN" -eq 0 ]] && stage="${OUTDIR}/${subset}"
  mkdir -p "$stage"
  echo "  extracting to ${stage}/"
  local rc=0
  UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -n "$src" -d "$stage" || rc=$?

  # unzip returns 1 for warnings (e.g. skipped existing files), which is fine.
  if (( rc > 1 )); then
    echo "  ! unzip failed with rc=$rc" >&2
    return 1
  fi

  # --- flatten ------------------------------------------------------------
  # Move the directory that actually holds the audio up to OUTDIR/<subset>,
  # discarding the mnt/fast/nobackup/... scaffolding. Same filesystem, so this
  # is a rename, not a copy — instant regardless of subset size.
  local final="${OUTDIR}/${subset}"
  if [[ "$FLATTEN" -eq 1 ]]; then
    local first audio_dir
    first=$(find "$stage" -type f \( -name '*.flac' -o -name '*.wav' \) -print -quit)
    if [[ -n "$first" ]]; then
      audio_dir=$(dirname "$first")
      if [[ -d "$final" ]]; then
        # resuming a partial run: merge rather than clobber
        find "$audio_dir" -mindepth 1 -maxdepth 1 -exec mv -n {} "$final"/ \;
      else
        mkdir -p "$(dirname "$final")"
        mv "$audio_dir" "$final"
      fi
      echo "  flattened:  ${final}/ (dropped ${top}/... prefix)"
    else
      echo "  ! no audio files found under ${stage}" >&2
      return 1
    fi
    rm -rf "$stage"
  else
    final="$stage"
  fi

  # --- verify -------------------------------------------------------------
  local got want
  got=$(find "$final" -type f \( -name '*.flac' -o -name '*.wav' \) | wc -l)
  want=$(expected_count "$subset")
  echo "  extracted:  ${got} audio files (published count: ${want})"

  if (( want > 0 )) && (( got < want )); then
    echo "  ! short by $((want - got)) files — archive may be truncated" >&2
    [[ "$src" == "$joined" ]] && echo "  ! keeping ${joined} for inspection" >&2
    return 1
  fi

  # --- cleanup ------------------------------------------------------------
  if [[ "$KEEP_JOINED" -eq 0 && "$src" == "$joined" ]]; then
    rm -f "$joined"
    echo "  removed joined archive"
  fi

  if [[ "$DELETE_PARTS" -eq 1 ]]; then
    rm -f "${dir}/${subset}".z[0-9]* "${dir}/${subset}.zip"
    echo "  deleted source parts ($(human "$bytes") reclaimed)"
  fi

  touch "$marker"
  echo "  DONE"
  return 0
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
echo ">> WavCaps extraction"
echo "   source:  ${ZIPDIR}"
echo "   target:  ${OUTDIR}"
echo "   scratch: ${TMPDIR_OVERRIDE:-${OUTDIR}/.joining}"
echo "   subsets: ${SUBSETS[*]}"
[[ "$DRY_RUN" -eq 1 ]]      && echo "   MODE:    dry run"
[[ "$DELETE_PARTS" -eq 1 ]] && echo "   NOTE:    parts will be DELETED after each verified extraction"

FAILED=()
START=$SECONDS

if [[ "$JOBS" -gt 1 ]]; then
  export -f process_subset verify_parts expected_count human avail_bytes
  export ZIPDIR OUTDIR TMPDIR_OVERRIDE KEEP_JOINED DELETE_PARTS DRY_RUN FLATTEN
  printf '%s\n' "${SUBSETS[@]}" | xargs -P "$JOBS" -I{} bash -c 'process_subset {}' || FAILED+=("one or more subsets")
else
  for subset in "${SUBSETS[@]}"; do
    process_subset "$subset" || FAILED+=("$subset")
  done
fi

ELAPSED=$((SECONDS - START))
echo
echo "=============================================================="
printf ">> Finished in %02dh%02dm%02ds\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))

if (( ${#FAILED[@]} > 0 )); then
  echo ">> FAILED: ${FAILED[*]}"
  echo "   Re-run the script — completed subsets are skipped via their .done markers,"
  echo "   and partial extractions resume (unzip -n never overwrites)."
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> Dry run clean. Re-run without -n to extract."
  exit 0
fi

echo ">> All requested subsets extracted to ${OUTDIR}/"
echo "   Captions are in the json_files/ tree you rsync'd alongside Zip_files/."
echo "   Remember to filter against json_files/blacklist/ before training."