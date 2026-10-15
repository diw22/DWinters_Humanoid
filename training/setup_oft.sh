#!/usr/bin/env bash
# One-time OpenVLA-OFT environment (Python 3.10 venv) on a Brev GPU machine. Usage: bash training/setup_oft.sh
set -euo pipefail
# OpenVLA-OFT pins Python 3.10 packages (tensorflow 2.15); uv provides 3.10 on any image (e.g. 3.12 ones).
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -x ~/venv-oft/bin/pip ] || uv venv --python 3.10 --seed ~/venv-oft
source ~/venv-oft/bin/activate
pip install -q --upgrade pip
[ -d ~/openvla-oft ] || git clone -q https://github.com/moojink/openvla-oft.git ~/openvla-oft
pip install -q -e ~/openvla-oft
pip install -q "mujoco>=3.3,<4" "osqp>=1.0,<2" "imageio[ffmpeg]" scipy
# tensorflow-metadata >= 1.16 needs protobuf 5; tensorflow 2.15 (pinned by OFT) keeps protobuf 4.
pip install -q --no-deps "tensorflow-metadata==1.15.0"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
