#!/usr/bin/env bash
# Download and verify the large OpenBiomechanics GitHub Release assets.

set -euo pipefail

OBP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBP_RELEASE_REPOSITORY="${OBP_GITHUB_REPOSITORY:-drivelineresearch/openbiomechanics}"
OBP_CHECKSUMS="$OBP_ROOT/scripts/release_checksums.sha256"

usage() {
  cat <<'USAGE'
Usage: scripts/download_data.sh [options]

By default, downloads pitching and hitting C3D/full-signal data. C3D archives
are extracted; full-signal .zip files are verified and left zipped in each
module's data/full_sig directory.

Options:
  --discipline pitching  Download only pitching data (repeatable)
  --discipline hitting   Download only hitting data (repeatable)
  --with-media           Also restore computer-vision demo media
  --with-mokka           Also download the Mokka C3D viewer installers
  --skip-data            Download only requested media/tools (no dataset)
  -h, --help             Show this help

Requirements: authenticated gh, unzip, Bash, and either sha256sum or shasum.
Run `gh auth login` once, or provide GH_TOKEN in automation.
On Windows, run from Git Bash or WSL.
USAGE
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' was not found" >&2
    return 1
  fi
}

compute_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_asset() {
  local path="$1"
  local name expected actual
  name="$(basename "$path")"
  expected="$(awk -v name="$name" '$2 == name {print $1}' "$OBP_CHECKSUMS")"
  if [ -z "$expected" ]; then
    echo "error: no committed checksum for release asset '$name'" >&2
    return 1
  fi
  actual="$(compute_sha256 "$path")"
  if [ "$actual" != "$expected" ]; then
    echo "error: checksum mismatch for '$name'" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
  echo "    verified $name"
}

require_asset() {
  if [ ! -f "$1" ]; then
    echo "error: release is missing expected asset '$(basename "$1")'" >&2
    return 1
  fi
}

DOWNLOAD_PITCHING=0
DOWNLOAD_HITTING=0
DISCIPLINE_SELECTED=0
WITH_MEDIA=0
WITH_MOKKA=0
SKIP_DATA=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --discipline)
      if [ "$#" -lt 2 ]; then
        echo "error: --discipline requires pitching or hitting" >&2
        exit 2
      fi
      DISCIPLINE_SELECTED=1
      case "$2" in
        pitching) DOWNLOAD_PITCHING=1 ;;
        hitting) DOWNLOAD_HITTING=1 ;;
        *) echo "error: unknown discipline '$2' (expected pitching or hitting)" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --discipline=*)
      DISCIPLINE_SELECTED=1
      case "${1#--discipline=}" in
        pitching) DOWNLOAD_PITCHING=1 ;;
        hitting) DOWNLOAD_HITTING=1 ;;
        *) echo "error: unknown discipline '${1#--discipline=}' (expected pitching or hitting)" >&2; exit 2 ;;
      esac
      shift
      ;;
    --with-media) WITH_MEDIA=1; shift ;;
    --with-mokka) WITH_MOKKA=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$SKIP_DATA" -eq 1 ] && [ "$DISCIPLINE_SELECTED" -eq 1 ]; then
  echo "error: --skip-data cannot be combined with --discipline" >&2
  exit 2
fi
if [ "$SKIP_DATA" -eq 1 ] && [ "$WITH_MEDIA" -eq 0 ] && [ "$WITH_MOKKA" -eq 0 ]; then
  echo "error: --skip-data requires --with-media or --with-mokka" >&2
  exit 2
fi
if [ "$SKIP_DATA" -eq 0 ] && [ "$DISCIPLINE_SELECTED" -eq 0 ]; then
  DOWNLOAD_PITCHING=1
  DOWNLOAD_HITTING=1
fi

require_command gh
if ! gh auth status >/dev/null 2>&1; then
  echo "error: GitHub CLI is not authenticated; run 'gh auth login' or set GH_TOKEN" >&2
  exit 1
fi
if [ "$SKIP_DATA" -eq 0 ] || [ "$WITH_MEDIA" -eq 1 ]; then
  require_command unzip
fi
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  echo "error: required checksum command 'sha256sum' or 'shasum' was not found" >&2
  exit 1
fi
if [ ! -f "$OBP_CHECKSUMS" ]; then
  echo "error: checksum manifest not found: $OBP_CHECKSUMS" >&2
  exit 1
fi

cd "$OBP_ROOT"
OBP_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$OBP_TMP_DIR"' EXIT

PITCHING_ASSETS=(
  pitching_c3d.zip
  pitching_energy_flow.zip
  pitching_forces_moments.zip
  pitching_force_plate.zip
  pitching_joint_angles.zip
  pitching_joint_velos.zip
  pitching_landmarks.zip
)
HITTING_ASSETS=(
  hitting_c3d.zip
  hitting_force_plate.zip
  hitting_joint_angles.zip
  hitting_joint_velos.zip
  hitting_landmarks.zip
)

if [ "$SKIP_DATA" -eq 0 ]; then
  echo "==> Downloading verified dataset-v1 assets ..."
  DATASET_ARGS=(
    release download dataset-v1
    --repo "$OBP_RELEASE_REPOSITORY"
    --dir "$OBP_TMP_DIR"
    --clobber
  )
  EXPECTED_DATASET_ASSETS=()
  if [ "$DOWNLOAD_PITCHING" -eq 1 ]; then
    DATASET_ARGS+=(--pattern 'pitching_*.zip')
    EXPECTED_DATASET_ASSETS+=("${PITCHING_ASSETS[@]}")
  fi
  if [ "$DOWNLOAD_HITTING" -eq 1 ]; then
    DATASET_ARGS+=(--pattern 'hitting_*.zip')
    EXPECTED_DATASET_ASSETS+=("${HITTING_ASSETS[@]}")
  fi
  gh "${DATASET_ARGS[@]}"

  # Verify every expected selected asset before changing destination files.
  for name in "${EXPECTED_DATASET_ASSETS[@]}"; do
    require_asset "$OBP_TMP_DIR/$name"
    verify_asset "$OBP_TMP_DIR/$name"
  done

  if [ "$DOWNLOAD_PITCHING" -eq 1 ]; then
    mkdir -p baseball_pitching/data/full_sig
    for name in "${PITCHING_ASSETS[@]}"; do
      if [ "$name" != "pitching_c3d.zip" ]; then
        mv -f "$OBP_TMP_DIR/$name" "baseball_pitching/data/full_sig/${name#pitching_}"
        echo "    placed baseball_pitching/data/full_sig/${name#pitching_}"
      fi
    done
    echo "==> Extracting pitching C3D files ..."
    unzip -oq "$OBP_TMP_DIR/pitching_c3d.zip" -d baseball_pitching/data
  fi

  if [ "$DOWNLOAD_HITTING" -eq 1 ]; then
    mkdir -p baseball_hitting/data/full_sig
    for name in "${HITTING_ASSETS[@]}"; do
      if [ "$name" != "hitting_c3d.zip" ]; then
        mv -f "$OBP_TMP_DIR/$name" "baseball_hitting/data/full_sig/${name#hitting_}"
        echo "    placed baseball_hitting/data/full_sig/${name#hitting_}"
      fi
    done
    echo "==> Extracting hitting C3D files ..."
    unzip -oq "$OBP_TMP_DIR/hitting_c3d.zip" -d baseball_hitting/data
  fi
fi

if [ "$WITH_MEDIA" -eq 1 ]; then
  echo "==> Downloading verified computer-vision media ..."
  MEDIA_DIR="$OBP_TMP_DIR/media"
  mkdir -p "$MEDIA_DIR"
  gh release download cv-media-v1 \
    --repo "$OBP_RELEASE_REPOSITORY" --dir "$MEDIA_DIR" --clobber
  require_asset "$MEDIA_DIR/cv_media.zip"
  verify_asset "$MEDIA_DIR/cv_media.zip"
  unzip -oq "$MEDIA_DIR/cv_media.zip" -d "$OBP_ROOT"
  echo "    media restored under computer_vision/"
fi

if [ "$WITH_MOKKA" -eq 1 ]; then
  echo "==> Downloading verified Mokka installers ..."
  MOKKA_DIR="$OBP_TMP_DIR/mokka"
  mkdir -p "$MOKKA_DIR" binaries
  gh release download tools-mokka-0.6.2 \
    --repo "$OBP_RELEASE_REPOSITORY" --dir "$MOKKA_DIR" --clobber
  MOKKA_ASSETS=(Mokka-0.6.2-Windows.64.zip Mokka-0.6.2_MacOSX.dmg)
  for name in "${MOKKA_ASSETS[@]}"; do
    require_asset "$MOKKA_DIR/$name"
    verify_asset "$MOKKA_DIR/$name"
  done
  for name in "${MOKKA_ASSETS[@]}"; do
    mv -f "$MOKKA_DIR/$name" binaries/
  done
  echo "    Mokka installers saved under binaries/"
fi

if [ "$SKIP_DATA" -eq 0 ]; then
  echo "Done. Full-signal archives remain zipped; see examples/03_join_fullsig.py."
else
  echo "Done. Requested optional assets were downloaded and verified."
fi
