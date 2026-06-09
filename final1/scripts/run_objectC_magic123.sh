#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: bash scripts/run_objectC_magic123.sh [lambda_2d_3d]" >&2
  echo "Example: bash scripts/run_objectC_magic123.sh 0.5" >&2
  exit 2
fi

LAMBDA_2D_3D="${1:-1.0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/timing.sh"
MAGIC123_DIR="${PROJECT_ROOT}/Magic123"
OBJECT_C_DIR="${PROJECT_ROOT}/dataset/object_C"
IMAGE_DIR="${OBJECT_C_DIR}/images"
INPUT_IMAGE="${IMAGE_DIR}/0001.jpg"
FOREGROUND_IMAGE="${IMAGE_DIR}/0001_foreground.png"
RGBA_IMAGE="${IMAGE_DIR}/rgba.png"
OUT_ROOT="${OBJECT_C_DIR}/magic123_output"
COARSE_WS="${OUT_ROOT}/object_C_coarse"
FINE_WS="${OUT_ROOT}/object_C_fine"
FIXED_COARSE_CKPT="${COARSE_WS}/checkpoints/object_C_coarse_latest.pth"
GPU=2
LAMBDA_MASK=3

COARSE_LAMBDA_2D=$(awk "BEGIN { print 1.0 * ${LAMBDA_2D_3D} }")
FINE_LAMBDA_2D=$(awk "BEGIN { print 0.001 * ${LAMBDA_2D_3D} }")

# TEXT_PROMPT="a high-resolution DSLR image of a single potted orchid plant with purple and white flowers,  \
# broad green leaves, continuous thin dark branching stems visibly connecting every flower cluster to the plant, \
# full object, centered, natural indoor plant texture"

TEXT_PROMPT="a zoomed out DSLR product photo of a single Nongfu Spring mineral water bottle, red plastic bottle cap,  \
transparent PET bottle body, red label, smooth glossy plastic material, full object, centered, clean silhouette,  \
pure white background, uniform soft lighting, high-quality 3D model"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cvpj1

export NUMBA_CACHE_DIR=/tmp/numba_cache
export U2NET_HOME="${PROJECT_ROOT}/.cache/rembg"
mkdir -p "$NUMBA_CACHE_DIR"
mkdir -p "$U2NET_HOME"

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.6"
export TCNN_CUDA_ARCHITECTURES=86

python - "$INPUT_IMAGE" <<'PY'
import sys
from PIL import Image

image_path = sys.argv[1]
try:
    with Image.open(image_path) as image:
        image.verify()
except Exception as exc:
    raise SystemExit(
        f"Input image is not readable by PIL: {image_path}\n"
        "Please export it as a real JPEG or PNG before running Object C. "
        "A HEIC/HEIF file renamed to .jpg will fail here.\n"
        f"Original error: {exc}"
    )
PY

rm -rf "$COARSE_WS" "$FINE_WS"
mkdir -p "$OUT_ROOT"

# LAMBDA_2D_3D: controls the paper's lambda_2D/3D ratio; 1.0 is the recommended balanced point.
# COARSE_WS: removed before training so the coarse stage cannot resume an incompatible earlier asset.
# FINE_WS: removed before training so the fine stage cannot mix Mesh artifacts from an earlier asset.

cd "$PROJECT_ROOT"

run_timed object_C preprocess_foreground "remove background; lambda_2d_3d=${LAMBDA_2D_3D}" \
  python scripts/preprocess_image.py "$INPUT_IMAGE" objectC

# scripts/preprocess_image.py: removes the background from the original 0001.jpg photo and writes 0001_foreground.png as a transparent RGBA foreground image.
# objectC: uses BRIA-RMBG 2.0 and keeps every foreground component so orchid flowers and stems are not discarded as cleanup fragments.

cd "$MAGIC123_DIR"

run_timed object_C magic123_preprocess_rgba_depth "generate rgba.png and depth.png; lambda_2d_3d=${LAMBDA_2D_3D}" \
env CUDA_VISIBLE_DEVICES=$GPU python preprocess_image.py \
  --path "$FOREGROUND_IMAGE"

# CUDA_VISIBLE_DEVICES: exposes one physical GPU to the Magic123 preprocessing stage.
# --path: reads the transparent RGBA foreground, reuses its alpha channel, and overwrites rgba.png and depth.png.

run_timed object_C magic123_coarse "coarse implicit optimization; lambda_2d_3d=${LAMBDA_2D_3D}; lambda_guidance=${COARSE_LAMBDA_2D},40" \
env CUDA_VISIBLE_DEVICES=$GPU python main.py -O \
  --text "$TEXT_PROMPT" \
  --sd_version 1.5 \
  --image "$RGBA_IMAGE" \
  --workspace "$COARSE_WS" \
  --optim adam \
  --iters 5000 \
  --guidance SD zero123 \
  --lambda_guidance "$COARSE_LAMBDA_2D" 40 \
  --guidance_scale 100 5 \
  --lambda_rgb 8 \
  --lambda_mask "$LAMBDA_MASK" \
  --latent_iter_ratio 0 \
  --normal_iter_ratio 0.2 \
  --t_range 0.2 0.6 \
  --bg_radius -1 \
  --save_mesh

# CUDA_VISIBLE_DEVICES: exposes one physical GPU to the coarse Magic123 stage.
# -O: enables the default accelerated training options.
# --text: describes the potted orchid asset for the Stable Diffusion prior.
# --sd_version: selects Stable Diffusion v1.5.
# --image: reads rgba.png generated from the transparent foreground image.
# --workspace: writes coarse checkpoints, logs, validation images, and the initial Mesh.
# --optim: selects the Adam optimizer.
# --iters: runs 5000 coarse optimization iterations.
# --guidance: combines Stable Diffusion 2D guidance with Zero123 3D guidance.
# --lambda_guidance: sets SD and Zero123 loss weights; the first value is scaled by lambda_2d_3d.
# --guidance_scale: sets classifier-free guidance scales for SD and Zero123.
# --lambda_rgb: strengthens the known-view color constraint so thin dark stems from rgba.png are preserved.
# --lambda_mask: strengthens the known-view alpha constraint so flower clusters remain connected to their stems.
# --latent_iter_ratio: disables latent-only warmup iterations.
# --normal_iter_ratio: uses normal shading during part of coarse optimization.
# --t_range: limits diffusion timesteps used by the coarse stage.
# --bg_radius: disables the learned background model.
# --save_mesh: exports the coarse Mesh after optimization.

LATEST_COARSE_CKPT=$(ls -t "${COARSE_WS}"/checkpoints/*.pth | head -n 1)
cp "$LATEST_COARSE_CKPT" "$FIXED_COARSE_CKPT"

# LATEST_COARSE_CKPT: selects the newest checkpoint generated by the completed coarse stage.
# FIXED_COARSE_CKPT: provides a stable checkpoint path for the following fine stage.

run_timed object_C magic123_fine_export "fine DMTet optimization and final mesh export; lambda_2d_3d=${LAMBDA_2D_3D}; lambda_guidance=${FINE_LAMBDA_2D},0.01" \
env CUDA_VISIBLE_DEVICES=$GPU python main.py -O \
  --text "$TEXT_PROMPT" \
  --sd_version 1.5 \
  --image "$RGBA_IMAGE" \
  --workspace "$FINE_WS" \
  --dmtet \
  --init_ckpt "$FIXED_COARSE_CKPT" \
  --iters 5000 \
  --optim adam \
  --known_view_interval 2 \
  --latent_iter_ratio 0 \
  --guidance SD zero123 \
  --lambda_guidance "$FINE_LAMBDA_2D" 0.01 \
  --guidance_scale 100 5 \
  --lambda_rgb 8 \
  --lambda_mask "$LAMBDA_MASK" \
  --bg_radius -1 \
  --save_mesh

# CUDA_VISIBLE_DEVICES: exposes one physical GPU to the fine Magic123 stage.
# -O: enables the default accelerated training options.
# --text: uses the same potted orchid prompt as the coarse stage.
# --sd_version: selects Stable Diffusion v1.5.
# --image: reads the same rgba.png known view.
# --workspace: writes fine checkpoints, logs, validation images, and the final textured Mesh.
# --dmtet: refines an explicit DMTet Mesh representation.
# --init_ckpt: initializes fine optimization from the completed coarse checkpoint.
# --iters: runs 5000 fine optimization iterations.
# --optim: selects the Adam optimizer.
# --known_view_interval: constrains optimization with the input view every two iterations to preserve thin stems.
# --latent_iter_ratio: disables latent-only warmup iterations.
# --guidance: combines Stable Diffusion 2D guidance with Zero123 3D guidance.
# --lambda_guidance: sets fine-stage SD and Zero123 weights; the first value is scaled by lambda_2d_3d.
# --guidance_scale: sets classifier-free guidance scales for SD and Zero123.
# --lambda_rgb: strengthens the known-view color constraint so thin dark stems from rgba.png are preserved.
# --lambda_mask: strengthens the known-view alpha constraint so flower clusters remain connected to their stems.
# --bg_radius: disables the learned background model.
# --save_mesh: exports the final OBJ, MTL, and texture files.
