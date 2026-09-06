#!/usr/bin/env bash
# End-to-end reproduction: the 62.7 mph throw, video frames 800-1099 at 360 fps. Camera 19 is held out of the
# appearance losses for the score. Supplied Theia poses/alignment are conditioning inputs.
# Then all eight cameras train the model behind the videos.
# Everything is written under WORK (default ~/obp4d-work); nothing is written into the repository.
#
#   bash computer_vision/4d_reconstruction/run_all.sh [WORK]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$HOME/obp4d-work}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
python -m pip freeze > "$WORK/environment.txt"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
run() { echo; echo "=== $*"; python -m obp4d "$@" --work "$WORK"; }

run fetch
run dataset
run masks
run plates

# appearance holdout: camera 19 excluded from background/body image losses; alignment uses all eight
run background --holdout 19 --out bg_h19.pt
run body --holdout 19 --bg bg_h19.pt --out body_h19
run render holdout --body body_h19 --bg bg_h19.pt --cam 19 --out holdout_cam19.mp4

# all eight cameras: the videos
run background --out bg.pt
run body --bg bg.pt --out body
run render quad --body body --bg bg.pt --out quad.mp4
run render ring --body body --bg bg.pt --out ring.mp4

echo; echo "done: score in $WORK/body_h19/result.json, videos in $WORK"
