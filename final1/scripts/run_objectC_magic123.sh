#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/timing.sh"
source "${PROJECT_ROOT}/scripts/wandb.sh"
MAGIC123_DIR="${PROJECT_ROOT}/Magic123"
OBJECT_C_NAME="object_C"
OBJECT_C_DIR="${PROJECT_ROOT}/dataset/${OBJECT_C_NAME}"
IMAGE_DIR="${OBJECT_C_DIR}/images"
INPUT_IMAGE=""
for candidate in "${IMAGE_DIR}/0001.jpg" "${IMAGE_DIR}/0001.png"; do
  if [ -f "$candidate" ]; then
    INPUT_IMAGE="$candidate"
    break
  fi
done
if [ -z "$INPUT_IMAGE" ]; then
  echo "Input image not found under: $IMAGE_DIR" >&2
  echo "Expected one of: 0001.jpg, 0001.png" >&2
  exit 1
fi
INPUT_STEM="$(basename "${INPUT_IMAGE%.*}")"
FOREGROUND_IMAGE="${IMAGE_DIR}/${INPUT_STEM}_foreground.png"
RGBA_IMAGE="${IMAGE_DIR}/rgba.png"
OUT_ROOT="${OBJECT_C_DIR}/magic123_output"
COARSE_WS="${OUT_ROOT}/object_C_coarse"
FINE_WS="${OUT_ROOT}/object_C_fine"
FIXED_COARSE_CKPT="${COARSE_WS}/checkpoints/object_C_coarse_latest.pth"
GPU=3
MAX_STEPS=10000
LAMBDA_2D_3D=1.0
LAMBDA_MASK=3

COARSE_LAMBDA_2D=$(awk "BEGIN { print 1.0 * ${LAMBDA_2D_3D} }")
FINE_LAMBDA_2D=$(awk "BEGIN { print 0.001 * ${LAMBDA_2D_3D} }")

# object C -main
TEXT_PROMPT="a high-resolution DSLR image of a single potted orchid plant with purple and white flowers,  \
broad green leaves, continuous thin dark branching stems visibly connecting every flower cluster to the plant, \
full object, centered, natural indoor plant texture"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cvpj1

export http_proxy=http://127.0.0.1:9999
export https_proxy=http://127.0.0.1:9999
export HTTP_PROXY=http://127.0.0.1:9999
export HTTPS_PROXY=http://127.0.0.1:9999
export HF_HOME=/mnt/data/haoyang/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data/haoyang/.cache/huggingface/transformers
export HF_HUB_CACHE=/mnt/data/haoyang/.cache/huggingface/hub
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DIFFUSERS_OFFLINE

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
        "A HEIC/HEIF file renamed to .jpg or .png will fail here.\n"
        f"Original error: {exc}"
    )
PY

rm -rf "$COARSE_WS" "$FINE_WS"
mkdir -p "$OUT_ROOT"

# LAMBDA_2D_3D: controls the paper's lambda_2D/3D ratio; 1.0 is the recommended balanced point.
# COARSE_WS: removed before full training so the coarse stage cannot resume an incompatible earlier asset.
# FINE_WS: removed before training so the fine stage cannot mix Mesh artifacts from an earlier asset.

cd "$PROJECT_ROOT"

run_timed "$OBJECT_C_NAME" preprocess_foreground \
  python scripts/preprocess_image.py "$INPUT_IMAGE" objectC

# scripts/preprocess_image.py: removes the background from the original 0001.jpg photo and writes 0001_foreground.png as a transparent RGBA foreground image.
# objectC: uses BRIA-RMBG 2.0 and keeps every foreground component so orchid flowers and stems are not discarded as cleanup fragments.

cd "$MAGIC123_DIR"

run_timed "$OBJECT_C_NAME" magic123_preprocess_rgba_depth \
env CUDA_VISIBLE_DEVICES=$GPU python preprocess_image.py \
  --path "$FOREGROUND_IMAGE"

# CUDA_VISIBLE_DEVICES: exposes one physical GPU to the Magic123 preprocessing stage.
# --path: reads the transparent RGBA foreground, reuses its alpha channel, and overwrites rgba.png and depth.png.

run_timed "$OBJECT_C_NAME" magic123_coarse \
env CUDA_VISIBLE_DEVICES=$GPU python main.py -O \
  --text "$TEXT_PROMPT" \
  --sd_version 1.5 \
  --image "$RGBA_IMAGE" \
  --workspace "$COARSE_WS" \
  --optim adam \
  --iters "$MAX_STEPS" \
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
# --iters: runs MAX_STEPS coarse optimization iterations.
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

run_timed "$OBJECT_C_NAME" magic123_fine_export \
env CUDA_VISIBLE_DEVICES=$GPU python main.py -O \
  --text "$TEXT_PROMPT" \
  --sd_version 1.5 \
  --image "$RGBA_IMAGE" \
  --workspace "$FINE_WS" \
  --dmtet \
  --init_ckpt "$FIXED_COARSE_CKPT" \
  --iters "$MAX_STEPS" \
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
# --iters: runs MAX_STEPS fine optimization iterations.
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

log_metrics_to_wandb "${OBJECT_C_NAME}_coarse" --profile magic123 "${COARSE_WS}/run"
log_metrics_to_wandb "${OBJECT_C_NAME}_fine" --profile magic123 "${FINE_WS}/run"

cd "$PROJECT_ROOT"

run_timed "$OBJECT_C_NAME" magic123_plot_ema \
python - <<'PY'
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPORT_DIR = Path("report")
COARSE_RUN = Path("dataset/object_C/magic123_output/object_C_coarse/run")
FINE_RUN = Path("dataset/object_C/magic123_output/object_C_fine/run")

METRICS = {
    "train/loss": ("objectC_loss_total.png", "Object C Magic123 Loss Total", "magic123/loss_total"),
    "train/loss_rgb": ("objectC_loss_rgb.png", "Object C Magic123 Loss RGB", "magic123/loss_rgb"),
    "train/loss_mask": ("objectC_loss_mask.png", "Object C Magic123 Loss Mask", "magic123/loss_mask"),
}


def event_dirs(root):
    if not root.exists():
        return []
    return sorted({event.parent for event in root.rglob("events.out.tfevents*")})


def read_scalars(root, tag):
    values = []
    for event_dir in event_dirs(root):
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        values.extend((scalar.step, scalar.value) for scalar in accumulator.Scalars(tag))
    values.sort(key=lambda item: item[0])
    return values


def ema(values, alpha=0.15):
    smoothed = []
    current = None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


def plot_metric(tag, filename, title, ylabel):
    series = [
        ("object_C_coarse", read_scalars(COARSE_RUN, tag)),
        ("object_C_fine", read_scalars(FINE_RUN, tag)),
    ]
    if not any(points for _, points in series):
        print(f"No TensorBoard scalar found for {tag}; skip {filename}.")
        return

    plt.figure(figsize=(8, 5), dpi=180)
    colors = {"object_C_coarse": "#d62728", "object_C_fine": "#1f77b4"}

    for name, points in series:
        if not points:
            continue
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        color = colors[name]
        plt.plot(steps, values, color=color, alpha=0.22, linewidth=1.0, label=f"{name} raw")
        plt.plot(steps, ema(values), color=color, alpha=0.95, linewidth=2.6, label=f"{name} EMA")

    plt.title(title, fontsize=13)
    plt.xlabel("Step", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    output_path = REPORT_DIR / filename
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote {output_path}")


REPORT_DIR.mkdir(parents=True, exist_ok=True)
for tag, (filename, title, ylabel) in METRICS.items():
    plot_metric(tag, filename, title, ylabel)
PY

# magic123_plot_ema: writes report/objectC_loss_total.png, report/objectC_loss_rgb.png, and report/objectC_loss_mask.png from coarse/fine TensorBoard raw losses.
