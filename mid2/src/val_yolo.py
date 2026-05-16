from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from ultralytics import YOLO

from .utils import ensure_parent, load_yaml, resolve_model


def select_device(value: str) -> str | int:
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visdrone_yolov8s.yaml")
    parser.add_argument("--model", choices=["yolov8s", "yolov8s_finetuned", "yolov9_finetuned"], default="yolov8s_finetuned")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    if args.weights is not None:
        model_path = args.weights
    elif args.model == "yolov8s":
        model_path = cfg["model"]["weights"]
    elif args.model == "yolov9_finetuned":
        model_path = cfg["val"]["yolov9_weight"]
    else:
        model_path = cfg["model"]["best_weight"]
    if args.output is not None:
        output_csv = args.output
    elif args.model == "yolov8s":
        output_csv = cfg["val"]["yolov8_baseline_csv"]
    elif args.model == "yolov9_finetuned":
        output_csv = cfg["val"]["yolov9_csv"]
    else:
        output_csv = cfg["val"]["yolov8_csv"]
    model = YOLO(resolve_model(model_path))
    metrics = model.val(
        data=data_cfg["yaml"],
        imgsz=int(data_cfg["image_size"]),
        batch=int(train_cfg["batch_size"]),
        workers=int(train_cfg["num_workers"]),
        device=select_device(str(cfg["device"])),
        save=False,
        plots=False,
        verbose=False,
    )
    row = {
        "model": args.model,
        "source": args.source_label or ("coco_pretrained" if args.model == "yolov8s" else "provided_weight" if args.model == "yolov9_finetuned" else "finetuned_by_us"),
        "precision": f"{metrics.box.mp:.6f}",
        "recall": f"{metrics.box.mr:.6f}",
        "mAP50": f"{metrics.box.map50:.6f}",
        "mAP50_95": f"{metrics.box.map:.6f}",
    }
    path = ensure_parent(output_csv)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"saved metrics: {path}")


if __name__ == "__main__":
    main()
