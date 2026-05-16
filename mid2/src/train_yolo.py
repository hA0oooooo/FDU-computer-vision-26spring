from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from ultralytics import YOLO

from .utils import load_yaml, resolve_mid2
from .wandb_utils import finish_wandb, init_wandb, log_ultralytics_results


YOLOV8S_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt"


def select_device(value: str) -> str | int:
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return value


def ensure_pretrained_weight(path: str) -> str:
    weight_path = resolve_mid2(path)
    if weight_path.exists():
        return str(weight_path)
    if weight_path.name != "yolov8s.pt":
        return path

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading yolov8s.pt to: {weight_path}")
    urllib.request.urlretrieve(YOLOV8S_URL, weight_path)
    return str(weight_path)


def make_cache_data_yaml(data_yaml: str, cache_dir: Path) -> Path:
    data = load_yaml(data_yaml)
    data["path"] = str(resolve_mid2("../dataset/VisDrone"))
    cache_yaml = cache_dir / "visdrone_data.yaml"
    with open(cache_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return cache_yaml


@contextmanager
def working_directory(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visdrone_yolov8s.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    run = init_wandb(cfg)

    model = YOLO(ensure_pretrained_weight(cfg["model"]["weights"]))
    cache_dir = resolve_mid2(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_data_yaml = make_cache_data_yaml(data_cfg["yaml"], cache_dir)
    with tempfile.TemporaryDirectory(prefix="yolo_train_", dir=cache_dir) as tmp_dir:
        with working_directory(cache_dir):
            result = model.train(
                data=str(train_data_yaml),
                epochs=int(train_cfg["epochs"]),
                imgsz=int(data_cfg["image_size"]),
                batch=int(train_cfg["batch_size"]),
                workers=int(train_cfg["num_workers"]),
                device=select_device(str(cfg["device"])),
                project=tmp_dir,
                name=cfg["experiment_name"],
                seed=int(cfg.get("seed", 42)),
                plots=False,
                amp=bool(train_cfg["amp"]),
            )

        save_dir = Path(getattr(result, "save_dir", model.trainer.save_dir))
        results_csv = save_dir / "results.csv"
        log_ultralytics_results(results_csv)
        best_src = save_dir / "weights" / "best.pt"
        best_dst = resolve_mid2(cfg["model"]["best_weight"])
        best_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, best_dst)
        run.summary["best_weight"] = str(best_dst)
        print(f"saved best weight: {best_dst}")
    finish_wandb()


if __name__ == "__main__":
    main()
