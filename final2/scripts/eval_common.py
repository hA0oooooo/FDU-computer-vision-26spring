from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from config_utils import load_config, project_root, require, root_path


REQUIRED_KEYS = [
    "gpu_id",
    "run_name",
    "step",
    "dataset_repo",
    "dataset_root",
    "batch_size",
    "num_workers",
    "tolerance_s",
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} CONFIG.yaml")

    root = project_root()
    cfg = load_config(sys.argv[1])
    require(cfg, REQUIRED_KEYS)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu_id"]
    os.environ.setdefault("HF_HOME", str(root / ".hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / ".hf_datasets"))
    sys.path.insert(0, str(root / "lerobot" / "src"))
    os.chdir(root / "lerobot")

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

    run_name = cfg["run_name"]
    step = cfg["step"]
    policy_path = root / "outputs" / run_name / "checkpoints" / step / "pretrained_model"
    dataset_root = root_path(root, cfg["dataset_root"])
    out_path = root / "outputs" / run_name / f"{run_name}_{step}_eval.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device

    meta = LeRobotDatasetMetadata(cfg["dataset_repo"], root=dataset_root)
    dataset = LeRobotDataset(
        cfg["dataset_repo"],
        root=dataset_root,
        delta_timestamps=resolve_delta_timestamps(policy_cfg, meta),
        return_uint8=True,
        tolerance_s=float(cfg["tolerance_s"]),
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=meta).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
    )

    abs_sum = 0.0
    valid_values = 0
    samples = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"eval {run_name} on D"):
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
