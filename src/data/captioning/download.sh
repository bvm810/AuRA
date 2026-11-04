#!/usr/bin/bash
#
# download.sh — download the WavCaps dataset from Hugging Face with curl.
#
# Two things live in cvssp/WavCaps and you need BOTH:
#   json_files/  — the captions and metadata (~180 MB), plus eval-set blacklists
#   Zip_files/   — the audio, as multi-part zip archives (~700 GB)
#
# The audio is split across parts (SUBSET.z01, SUBSET.z02, ..., SUBSET.zip).
# Every part of a subset is required before that archive can be extracted.
#
# Files keep their original names and directory layout:
#   <OUTDIR>/json_files/AudioSet_SL/as_final.json
#   <OUTDIR>/json_files/blacklist/blacklist_exclude_all_ac.json
#   <OUTDIR>/Zip_files/AudioSet_SL/AudioSet_SL.z01
#   <OUTDIR>/Zip_files/AudioSet_SL/AudioSet_SL.zip
#
# Usage:
#   ./download.sh [-o OUTDIR] [-j N] [-n] [-c] [-z] [SUBSET ...]
#
#   -o OUTDIR   destination directory (default: ./wavcaps)
#   -j N        number of parallel downloads (default: 4)
#   -n          dry run — list what would be downloaded, then exit
#   -c          captions only — fetch json_files/, skip all audio
#   -z          audio only — skip json_files/ (not recommended)
#   SUBSET      one or more of: AudioSet_SL BBC_Sound_Effects FreeSound SoundBible
#               (default: all four; affects audio only, captions are always
#                fetched in full since the whole tree is under 200 MB)
#
# Examples:
#   ./download.sh -c                        # captions only, ~180 MB
#   ./download.sh -n                        # see the file list and total size
#   ./download.sh SoundBible                # captions + ~1 GB of audio
#   ./download.sh -o /data/wavcaps -j 8     # everything, 8 connections
#
# Notes:
#   - Audio is ~700 GB compressed, ~800 GB extracted. Check your disk first.
#   - Downloads resume (curl -C -), so re-running after an interruption picks up
#     where it left off. Completed files are skipped by size check.
#   - Set HF_TOKEN in your environment if you hit rate limits.
#   - Requires: curl, and either jq or python3 (to parse the HF file listing).

set -uo pipefail

REPO="cvssp/WavCaps"
REVISION="main"
API="https://huggingface.co/api/datasets/${REPO}/tree/${REVISION}"
RESOLVE="https://huggingface.co/datasets/${REPO}/resolve/${REVISION}"

OUTDIR="./wavcaps"
JOBS=4
DRY_RUN=0
WANT_JSON=1
WANT_AUDIO=1
ALL_SUBSETS=(AudioSet_SL BBC_Sound_Effects FreeSound SoundBible)

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts ":o:j:nczh" opt; do
  case "$opt" in
    o) OUTDIR="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    c) WANT_AUDIO=0 ;;
    z) WANT_JSON=0 ;;
    h) usage 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; usage 1 ;;
    :)  echo "option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ "$WANT_JSON" -eq 0 && "$WANT_AUDIO" -eq 0 ]]; then
  echo "error: -c and -z together leave nothing to download" >&2
  exit 1
fi

SUBSETS=("$@")
[[ ${#SUBSETS[@]} -eq 0 ]] && SUBSETS=("${ALL_SUBSETS[@]}")

command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }

# ---------------------------------------------------------------------------
# JSON parsing: prefer jq, fall back to python3.
# Emits "<path>\t<size>" lines for every file in a repo directory.
# ---------------------------------------------------------------------------
if command -v jq >/dev/null; then
  parse_tree() { jq -r '.[] | select(.type=="file") | "\(.path)\t\(.size)"'; }
elif command -v python3 >/dev/null; then
  parse_tree() {
    python3 -c '
import json, sys
for e in json.load(sys.stdin):
    if e.get("type") == "file":
        print("%s\t%d" % (e["path"], e.get("size", 0)))
'
  }
else
  echo "error: need either jq or python3 to parse the Hugging Face file listing" >&2
  exit 1
fi

AUTH=()
[[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")

MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

# list_dir <repo-path> — append every file under it to the manifest
list_dir() {
  local dir="$1"
  echo "   - ${dir}"
  if ! curl -sSfL "${AUTH[@]}" "${API}/${dir}?recursive=true" \
       | parse_tree >> "$MANIFEST"; then
    echo "error: could not list ${dir} — is the name correct?" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# 1. Build the manifest.
#    Captions first: they're tiny, and if something is wrong with your network
#    or token you find out in seconds rather than after a 5 GB transfer.
# ---------------------------------------------------------------------------
echo ">> Listing files in ${REPO} (revision: ${REVISION})"

if [[ "$WANT_JSON" -eq 1 ]]; then
  # The whole json_files tree: one caption file per source, plus the blacklists
  # marking clips that overlap common evaluation sets.
  list_dir "json_files"
fi

if [[ "$WANT_AUDIO" -eq 1 ]]; then
  for subset in "${SUBSETS[@]}"; do
    list_dir "Zip_files/${subset}"
  done
fi

COUNT=$(wc -l < "$MANIFEST")
[[ "$COUNT" -eq 0 ]] && { echo "error: no files found" >&2; exit 1; }
TOTAL=$(awk -F'\t' '{s+=$2} END {printf "%.1f", s/1024/1024/1024}' "$MANIFEST")
echo ">> ${COUNT} files, ${TOTAL} GB total"

if [[ "$DRY_RUN" -eq 1 ]]; then
  awk -F'\t' '{printf "%10.2f MB  %s\n", $2/1024/1024, $1}' "$MANIFEST"
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Download. One curl per file, resumable, skipping anything already complete.
# ---------------------------------------------------------------------------
mkdir -p "$OUTDIR"

fetch() {
  local path="$1" size="$2" dest="${OUTDIR}/$1"
  mkdir -p "$(dirname "$dest")"

  if [[ -f "$dest" ]]; then
    local have
    have=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
    if [[ "$have" -eq "$size" ]]; then
      echo "[skip] $path"
      return 0
    fi
    echo "[resume] $path ($have / $size bytes)"
  else
    echo "[get] $path"
  fi

  # -C - resumes; --retry survives the transient 5xx that HF's CDN throws on
  # long transfers; -L follows the redirect to the Xet/CDN storage backend.
  curl -fL -C - --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 -# \
       "${AUTH[@]}" \
       -o "$dest" \
       "${RESOLVE}/${path}?download=true"

  local rc=$?
  # curl returns 33 when the server refuses a range request on an already
  # complete file; treat that as success if the size checks out.
  if [[ $rc -ne 0 && $rc -ne 33 ]]; then
    echo "[FAIL rc=$rc] $path" >&2
    return $rc
  fi

  local got
  got=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo 0)
  if [[ "$got" -ne "$size" ]]; then
    echo "[SIZE MISMATCH] $path — expected $size, got $got (re-run to resume)" >&2
    return 1
  fi
  echo "[ok] $path"
}
export -f fetch
export OUTDIR RESOLVE
export HF_TOKEN="${HF_TOKEN:-}"

echo ">> Downloading to ${OUTDIR} with ${JOBS} parallel connection(s)"
FAILED=0
if [[ "$JOBS" -gt 1 ]]; then
  # xargs can't see the AUTH array, so rebuild the header inside the subshell.
  awk -F'\t' '{print $1"\t"$2}' "$MANIFEST" \
    | xargs -P "$JOBS" -I{} bash -c '
        IFS=$'"'"'\t'"'"' read -r p s <<< "{}"
        AUTH=(); [[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
        fetch "$p" "$s"' || FAILED=1
else
  while IFS=$'\t' read -r p s; do
    fetch "$p" "$s" || FAILED=1
  done < "$MANIFEST"
fi

echo
if [[ "$FAILED" -ne 0 ]]; then
  echo "!! Some files failed or are incomplete. Re-run the script — it resumes."
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. Sanity check the captions and report.
# ---------------------------------------------------------------------------
if [[ "$WANT_JSON" -eq 1 && -d "${OUTDIR}/json_files" ]]; then
  echo ">> Captions in ${OUTDIR}/json_files/:"
  find "${OUTDIR}/json_files" -name '*.json' | sort | while read -r f; do
    n=""
    if command -v python3 >/dev/null; then
      n=$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list):
                print("%d entries" % len(v)); break
        else:
            print("%d keys" % len(d))
    elif isinstance(d, list):
        print("%d entries" % len(d))
except Exception:
    print("unreadable")
' "$f" 2>/dev/null)
    fi
    printf "   %-58s %s\n" "${f#"${OUTDIR}/"}" "$n"
  done
fi

if [[ "$WANT_AUDIO" -eq 0 ]]; then
  echo
  echo ">> Captions only. Re-run without -c to fetch the audio archives."
  exit 0
fi

echo ">> Audio archives in ${OUTDIR}/Zip_files/"
cat <<'EOF'

Next step — extraction. These are SPLIT zip archives, so you cannot just unzip
the .zip part on its own. Join the parts first:

  cd wavcaps/Zip_files/AudioSet_SL
  zip -s 0 AudioSet_SL.zip --out AudioSet_SL_joined.zip
  unzip AudioSet_SL_joined.zip -d /path/to/audio/AudioSet_SL
  rm AudioSet_SL_joined.zip          # frees the duplicated space

If you hit "invalid zip file with overlapped components (possible zip bomb)",
the dataset card recommends repairing instead of joining:

  zip -F AudioSet_SL.zip --out AS.zip
  unzip AS.zip

Joining temporarily doubles disk usage for that subset, so do one subset at a
time and delete the parts once extraction succeeds.

Before training, filter your caption lists against json_files/blacklist/ so you
don't train on clips that appear in the evaluation sets you plan to report on.
EOF