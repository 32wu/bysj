#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="bysj"
ENV_FILE="$REPO_DIR/bysj_env.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda 未安装或不在 PATH 中，请先安装 Miniconda/Anaconda。"
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null 2>&1; then
  echo "检测到环境 ${ENV_NAME} 已存在，开始更新..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "开始创建环境 ${ENV_NAME}..."
  conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

echo "环境安装完成，开始做基础自检..."
python -c "import numpy, gym, cv2, torch, highway_env; print('numpy', numpy.__version__); print('gym', gym.__version__); print('opencv', cv2.__version__); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"

echo
echo "可以继续执行："
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_DIR}/LANE"
echo "  python run_RL_base.py --help"
