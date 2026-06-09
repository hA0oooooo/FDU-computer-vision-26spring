#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/timing.sh"
TWODGS_DIR="${PROJECT_ROOT}/2d-gaussian-splatting"
DATA_ROOT="${PROJECT_ROOT}/dataset/360_v2"

SCENE="${1:-garden}"
GPU=0

SCENE_DIR="${DATA_ROOT}/${SCENE}"
OUT_DIR="${PROJECT_ROOT}/dataset/${SCENE}_2dgs_output"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cvpj1

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.6"
export TCNN_CUDA_ARCHITECTURES=86

if [ ! -d "$SCENE_DIR" ]; then
  echo "Scene directory not found: $SCENE_DIR" >&2
  exit 1
fi

if [ ! -d "${SCENE_DIR}/sparse" ]; then
  echo "COLMAP sparse directory not found: ${SCENE_DIR}/sparse" >&2
  echo "This is not a valid COLMAP / 2DGS scene directory." >&2
  exit 1
fi

if [ -d "${SCENE_DIR}/images" ]; then
  IMAGE_DIR_NAME="images"
elif [ -d "${SCENE_DIR}/images_4" ]; then
  IMAGE_DIR_NAME="images_4"
elif [ -d "${SCENE_DIR}/images_2" ]; then
  IMAGE_DIR_NAME="images_2"
else
  echo "Image directory not found under: $SCENE_DIR" >&2
  echo "Tried: images, images_4, images_2" >&2
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

cd "$TWODGS_DIR"

run_timed "background_${SCENE}" 2dgs_train \
env CUDA_VISIBLE_DEVICES=$GPU python train.py \
  -s "$SCENE_DIR" \
  -m "$OUT_DIR" \
  --images "$IMAGE_DIR_NAME" \
  --depth_ratio 0

# SCENE: selects one scene under dataset/360_v2, for example garden, counter, or bicycle.
# SCENE_DIR: reads the selected Mip-NeRF 360 scene from dataset/360_v2/<scene>.
# OUT_DIR: writes one 2DGS background reconstruction to dataset/<scene>_2dgs_output.
# --images: selects the image folder inside the scene; the script prefers images, then images_4, then images_2.
# --depth_ratio 0: uses mean depth, which is usually more stable for unbounded scenes.

run_timed "background_${SCENE}" 2dgs_render_export \
env CUDA_VISIBLE_DEVICES=$GPU python render.py \
  -s "$SCENE_DIR" \
  -m "$OUT_DIR" \
  --images "$IMAGE_DIR_NAME" \
  --unbounded \
  --mesh_res 1024 \
  --depth_ratio 0

# render.py: renders the trained 2DGS scene and extracts background visualization assets.
# --unbounded: uses the unbounded mode for Mip-NeRF 360 style outdoor/large scenes.
# --mesh_res: controls mesh extraction resolution.
