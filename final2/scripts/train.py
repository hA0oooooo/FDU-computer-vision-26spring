from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from config_utils import load_config, project_root, require, root_path


REQUIRED_KEYS = [
    "gpu_id",
    "run_name",
    "dataset_repo",
    "dataset_root",
    "wandb_project",
    "wandb_entity",
    "steps",
    "batch_size",
    "save_freq",
    "log_freq",
    "tolerance_s",
    "disable_artifact",
]


def parse_config_path() -> str:
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        return sys.argv[2]
    raise SystemExit(f"Usage: python {sys.argv[0]} --config CONFIG.yaml")


def main() -> None:
    root = project_root()
    cfg = load_config(str(Path(parse_config_path())))
    require(cfg, REQUIRED_KEYS)

    run_name = cfg["run_name"]
    output_dir = root / "outputs" / run_name
    log_path = root / "logs" / f"{run_name}_train.log"

    shutil.rmtree(output_dir, ignore_errors=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={cfg['dataset_repo']}",
        f"--dataset.root={root_path(root, cfg['dataset_root'])}",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={run_name}",
        "--wandb.enable=true",
        f"--wandb.project={cfg['wandb_project']}",
        f"--wandb.entity={cfg['wandb_entity']}",
        "--wandb.mode=online",
        f"--wandb.disable_artifact={cfg['disable_artifact']}",
        f"--steps={cfg['steps']}",
        f"--batch_size={cfg['batch_size']}",
        f"--save_freq={cfg['save_freq']}",
        f"--log_freq={cfg['log_freq']}",
        f"--tolerance_s={cfg['tolerance_s']}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["gpu_id"]
    env["PYTHONPATH"] = (
        f"{root / 'lerobot' / 'src'}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(root / "lerobot" / "src")
    )

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=root / "lerobot",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


if __name__ == "__main__":
    main()
