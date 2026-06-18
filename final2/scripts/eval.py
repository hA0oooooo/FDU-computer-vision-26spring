from __future__ import annotations

import os
import sys

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


def parse_config_path() -> str:
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        return sys.argv[2]
    raise SystemExit(f"Usage: python {sys.argv[0]} --config CONFIG.yaml")


def main() -> None:
    root = project_root()
    cfg = load_config(parse_config_path())
    require(cfg, REQUIRED_KEYS)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu_id"]
    os.environ.setdefault("HF_HOME", str(root / ".hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / ".hf_datasets"))
    sys.path.insert(0, str(root / "lerobot" / "src"))
    os.chdir(root / "lerobot")

    import torch

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.act_d_eval import (
        compute_action_l1,
        make_d_eval_loader,
        report_context,
        write_full_eval,
    )

    run_name = cfg["run_name"]
    step = cfg["step"]
    policy_path = root / "outputs" / run_name / "checkpoints" / step / "pretrained_model"
    dataset_root = root_path(root, cfg["dataset_root"])
    out_path = root / "outputs" / run_name / f"{run_name}_{step}_eval.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (policy_path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint not found: {policy_path}")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device

    dataset, loader = make_d_eval_loader(
        policy_cfg,
        cfg["dataset_repo"],
        dataset_root,
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        float(cfg["tolerance_s"]),
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    metrics = compute_action_l1(
        policy,
        preprocessor,
        loader,
        dataset.meta.camera_keys,
        desc=f"full eval {run_name} on D",
        include_breakdown=True,
    )
    context = report_context(root, run_name, int(step), step)
    report = write_full_eval(out_path, context, metrics)
    print(out_path)
    print(report["final_eval"])


if __name__ == "__main__":
    main()
