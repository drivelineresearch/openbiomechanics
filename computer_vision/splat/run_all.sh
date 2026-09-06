#!/usr/bin/env bash
# Run the experimental skeleton/Theia configuration on a compatible NVIDIA/CUDA environment.
#
#   bash computer_vision/splat/run_all.sh [WORK_DIR] [FIRST_FRAME] [N_FRAMES] [STRIDE]
#
# Default sequence (historical scores predate skeleton camera exclusion): the 62.7 mph baseball throw, frames 820-1120, every second frame,
# camera 19 held out. Everything is written under WORK_DIR (default ~/obp-splat-work); nothing is written into the repo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$HOME/obp-splat-work}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
FIRST="${2:-820}"; NFRAMES="${3:-151}"; STRIDE="${4:-2}"
if ! [[ "$FIRST" =~ ^[0-9]+$ && "$NFRAMES" =~ ^[0-9]+$ && "$STRIDE" =~ ^[0-9]+$ ]] || ((NFRAMES < 2 || STRIDE < 1)); then
  echo "FIRST_FRAME must be nonnegative; N_FRAMES must be >= 2; STRIDE must be >= 1" >&2
  exit 2
fi
HOLDOUT="${HOLDOUT:-cam19}"
POSE_HOLDOUT="${POSE_HOLDOUT-$HOLDOUT}"
THEIA_FRAME_OFFSET="${THEIA_FRAME_OFFSET:-0}"
REF_T=$(printf "t%04d" $((NFRAMES / 2)))
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export OBP_SPLAT_WORK="$WORK"
mkdir -p "$WORK"
run() { echo; echo "=== $*"; python -m obp_splat "$@" --work "$WORK"; }

# 1. video (public Drive folder) and 2. datasets from the published calibration
python -m pip freeze > "$WORK/environment.txt"
run fetch throw theia
run dataset --videos videos --frame "$FIRST" --frames "$NFRAMES" --stride "$STRIDE" --out data/throw

# 3. calibration sanity check, 4. athlete masks, 5. 3D skeleton
run verify 15 410 190 data/throw/t0000
run masks 'data/throw/t0*'
run pose3d --holdout "$POSE_HOLDOUT" 'data/throw/t0*'

# 6. optional: agreement with a Theia3D C3D of the same trial, and lab-frame alignment
if [ -f "$WORK/theia.c3d" ]; then
  run theia --c3d theia.c3d --frames data/throw --first_video_frame "$FIRST" --c3d_frame_offset "$THEIA_FRAME_OFFSET" --out data/throw/theia_joints.npz
  EXTRA=(--extra_joints data/throw/theia_joints.npz)
else
  echo "note: put a Theia3D C3D at $WORK/theia.c3d to add foot/toe/head joints (the best configuration)"; EXTRA=()
fi

# 7. multi-view initialisation, 8. static background, 9. the athlete
run init --data "data/throw/$REF_T" --out "data/throw/$REF_T/points_dust3r.npz" --holdout "$HOLDOUT"
BGF=()
for ((frame_index=0; frame_index<NFRAMES; frame_index+=15)); do
  printf -v frame_dir 'data/throw/t%04d' "$frame_index"
  BGF+=("$frame_dir")
done
run background --frames "${BGF[@]}" --init_points "data/throw/$REF_T/points_dust3r.npz" --out out/background --iters 8000 --holdout "$HOLDOUT"
run body --frames 'data/throw/t0*' --ref "data/throw/$REF_T" --bg out/background/ckpt.pt \
    --init_points "data/throw/$REF_T/points_dust3r.npz" --out out/body --iters 14000 --n_gauss 60000 \
    --holdout "$HOLDOUT" --repeat 1 --orbit_deg 160 "${EXTRA[@]}"

# 10. videos: four novel angles and an appearance-holdout comparison
run quad --ckpt out/body/ckpt.pt --data data/throw --out out/quad.mp4 --repeat 2
run holdout --ckpt out/body/ckpt.pt --data data/throw --cam "$HOLDOUT" --out out/holdout.mp4

echo; echo "done. videos in $WORK/out, metrics in $WORK/out/body/result.json"
