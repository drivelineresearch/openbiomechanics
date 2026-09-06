#!/usr/bin/env bash
# Environment: PyTorch 2.6 (CUDA 12.4), gsplat 1.5.3 from its prebuilt wheel index, and the rest from
# requirements.txt. Linux with an NVIDIA GPU (24 GB was used; 16 GB should do with --cap 60000 on the body).
#
#   bash computer_vision/4d_reconstruction/setup_env.sh [VENV_DIR]      # default ~/obp4d-venv
set -euo pipefail
VENV="${1:-$HOME/obp4d-venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt26cu124 --extra-index-url https://pypi.org/simple
pip install -r "$HERE/requirements.txt"
python -c "import torch, gsplat; print('torch', torch.__version__, torch.cuda.is_available(), 'gsplat', gsplat.__version__)"
echo "environment ready:  source $VENV/bin/activate"
