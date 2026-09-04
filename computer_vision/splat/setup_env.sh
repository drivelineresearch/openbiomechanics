#!/usr/bin/env bash
# Build the environment this pipeline needs: CUDA 12.4, PyTorch 2.6 (cu124), gsplat and the CV dependencies.
# Tested on Ubuntu 22.04 (native and WSL2) with an RTX 3080; gsplat has no Windows wheel, so use WSL2 on Windows.
#
#   bash computer_vision/splat/setup_env.sh [VENV_DIR]      # default ~/obp-splat-venv
set -euxo pipefail
VENV="${1:-$HOME/obp-splat-venv}"
export DEBIAN_FRONTEND=noninteractive

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3-venv python3-pip build-essential ninja-build git wget ffmpeg

if [ ! -x /usr/local/cuda-12.4/bin/nvcc ]; then
  # WSL2 uses the wsl-ubuntu repo; on native Ubuntu swap it for ubuntu2204/x86_64
  REPO=wsl-ubuntu; grep -qi microsoft /proc/version || REPO=ubuntu2204/x86_64
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb 2>/dev/null \
    || wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  $SUDO dpkg -i /tmp/cuda-keyring.deb && $SUDO apt-get update -qq && $SUDO apt-get install -y -qq cuda-toolkit-12-4
fi
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH

python3 -m venv "$VENV"; source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

python - <<'PY'
import torch, gsplat
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("gsplat", gsplat.__version__)
PY
set +x
echo
echo "environment ready:  source $VENV/bin/activate"
echo "set TORCH_CUDA_ARCH_LIST to your GPU (8.6 = RTX 30xx) to avoid recompiling gsplat for every arch"
