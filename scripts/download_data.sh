#!/usr/bin/env bash
#
# Fetch the large OpenBiomechanics data artifacts that live in GitHub Releases
# (they are kept out of git to keep the repository lightweight) and unpack them
# into the layout the module READMEs describe.
#
# Requirements: the GitHub CLI `gh` (https://cli.github.com), authenticated,
# or set GH_TOKEN. Usage:
#
#   scripts/download_data.sh                 # pitching + hitting data (dataset-v1)
#   scripts/download_data.sh --with-media    # also fetch computer_vision demo media
#   scripts/download_data.sh --with-mokka    # also fetch the Mokka C3D viewer installers
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MEDIA=0
WITH_MOKKA=0
for arg in "$@"; do
  case "$arg" in
    --with-media) WITH_MEDIA=1 ;;
    --with-mokka) WITH_MOKKA=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Downloading dataset-v1 ..."
gh release download dataset-v1 -D "$TMP" --clobber

mkdir -p baseball_pitching/data/full_sig baseball_hitting/data/full_sig

# Processed full-signal tables ship as .zip and stay zipped in data/full_sig/.
# (asset name -> destination path)
declare -A FULLSIG=(
  [pitching_energy_flow.zip]=baseball_pitching/data/full_sig/energy_flow.zip
  [pitching_force_plate.zip]=baseball_pitching/data/full_sig/force_plate.zip
  [pitching_forces_moments.zip]=baseball_pitching/data/full_sig/forces_moments.zip
  [pitching_joint_angles.zip]=baseball_pitching/data/full_sig/joint_angles.zip
  [pitching_joint_velos.zip]=baseball_pitching/data/full_sig/joint_velos.zip
  [pitching_landmarks.zip]=baseball_pitching/data/full_sig/landmarks.zip
  [hitting_force_plate.zip]=baseball_hitting/data/full_sig/force_plate.zip
  [hitting_joint_angles.zip]=baseball_hitting/data/full_sig/joint_angles.zip
  [hitting_joint_velos.zip]=baseball_hitting/data/full_sig/joint_velos.zip
  [hitting_landmarks.zip]=baseball_hitting/data/full_sig/landmarks.zip
)
for asset in "${!FULLSIG[@]}"; do
  mv -f "$TMP/$asset" "${FULLSIG[$asset]}"
  echo "    placed ${FULLSIG[$asset]}"
done

# Raw C3D archives unpack into each module's data/ folder (creates data/c3d/).
echo "==> Unpacking raw C3D ..."
unzip -oq "$TMP/pitching_c3d.zip" -d baseball_pitching/data
unzip -oq "$TMP/hitting_c3d.zip"  -d baseball_hitting/data
echo "    c3d unpacked for pitching and hitting"

if [ "$WITH_MEDIA" -eq 1 ]; then
  echo "==> Downloading computer_vision demo media (cv-media-v1) ..."
  gh release download cv-media-v1 -D "$TMP" --clobber
  unzip -oq "$TMP/cv_media.zip" -d "$ROOT"
  echo "    cv media restored under computer_vision/"
fi

if [ "$WITH_MOKKA" -eq 1 ]; then
  echo "==> Downloading Mokka installers (tools-mokka-0.6.2) ..."
  mkdir -p binaries
  gh release download tools-mokka-0.6.2 -D binaries --clobber
  echo "    Mokka installers saved under binaries/"
fi

echo "Done."
