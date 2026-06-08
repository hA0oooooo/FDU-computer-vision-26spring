#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-act_ABC}"
STEP="${2:-050000}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PATH="$ROOT/outputs/$RUN_NAME/checkpoints/$STEP/pretrained_model"
DATASET_REPO="calvin_task_ABC_D_lerobot_3_4"
DATASET_ROOT="$ROOT/dataset/calvin_task_ABC_D/$DATASET_REPO"
OUT_PATH="$ROOT/outputs/$RUN_NAME/${RUN_NAME}_${STEP}_eval.json"

mkdir -p "$(dirname "$OUT_PATH")"
cd "$ROOT/lerobot"

HF_HOME="${HF_HOME:-$ROOT/.hf_home}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.hf_datasets}" \
POLICY_PATH="$POLICY_PATH" \
DATASET_REPO="$DATASET_REPO" \
DATASET_ROOT="$DATASET_ROOT" \
OUT_PATH="$OUT_PATH" \
python - <<'PY'
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

policy_path = Path(os.environ["POLICY_PATH"])
dataset_repo = os.environ["DATASET_REPO"]
dataset_root = Path(os.environ["DATASET_ROOT"])
out_path = Path(os.environ["OUT_PATH"])
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = PreTrainedConfig.from_pretrained(policy_path)
cfg.pretrained_path = policy_path
cfg.device = device

meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
dataset = LeRobotDataset(
    dataset_repo,
    root=dataset_root,
    delta_timestamps=resolve_delta_timestamps(cfg, meta),
    return_uint8=True,
    tolerance_s=0.001,
)

policy = make_policy(cfg=cfg, ds_meta=meta).to(device).eval()
preprocessor, _ = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=policy_path,
    preprocessor_overrides={"device_processor": {"device": device}},
)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
)

abs_sum = 0.0
valid_values = 0
samples = 0

with torch.inference_mode():
    for batch in tqdm(loader, desc="eval on D"):
        for key in dataset.meta.camera_keys:
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        batch = preprocessor(batch)
        pred = policy.predict_action_chunk(batch)
        err = F.l1_loss(pred, batch[ACTION], reduction="none")
        mask = ~batch["action_is_pad"].unsqueeze(-1)
        abs_sum += float((err * mask).sum().item())
        valid_values += int((mask.sum() * err.shape[-1]).item())
        samples += batch[ACTION].shape[0]

result = {
    "policy_path": str(policy_path),
    "dataset_root": str(dataset_root),
    "test_env": "D",
    "evaluated_samples": samples,
    "valid_action_values": valid_values,
    "normalized_action_l1": abs_sum / valid_values,
}
out_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
