from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import torch
from ultralytics import YOLO

from .utils import display_path, ensure_parent, load_yaml, resolve_mid2, resolve_model, save_yaml


CSV_FIELDS = [
    "frame_id",
    "track_id",
    "class_id",
    "class_name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
]


def tensor_list(value):
    return value.detach().cpu().tolist()


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def resolve_line(axis: str, position: float, width: int, height: int) -> int:
    if position > 1:
        return int(round(position))
    size = width if axis == "x" else height
    return int(round(size * position))


def crossing_direction(axis: str, prev: float, curr: float) -> str:
    if axis == "x":
        return "left_to_right" if curr > prev else "right_to_left"
    return "top_to_bottom" if curr > prev else "bottom_to_top"


def draw_line_count(frame, axis: str, line_pos: int, count: int, cfg: dict) -> None:
    height, width = frame.shape[:2]
    line_color = tuple(int(v) for v in cfg["line_color"])
    text_color = tuple(int(v) for v in cfg["text_color"])
    text_bg_color = tuple(int(v) for v in cfg["text_bg_color"])
    thickness = int(cfg["line_thickness"])

    if axis == "x":
        cv2.line(frame, (line_pos, 0), (line_pos, height), line_color, thickness)
    else:
        cv2.line(frame, (0, line_pos), (width, line_pos), line_color, thickness)

    text = f"count = {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    x = max(8, width - text_w - 18)
    y = 14 + text_h
    cv2.rectangle(
        frame,
        (x - 8, y - text_h - 8),
        (x + text_w + 8, y + baseline + 8),
        text_bg_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)


def select_device(value: str) -> str | int:
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return value


def resolve_outputs(source: Path, model_name: str, model_cfg: dict, tag: str | None):
    name = tag or source.stem
    return (
        ensure_parent(f"../dataset/videos/output/{name}_{model_name}.mp4"),
        ensure_parent(f"{model_cfg['output_dir']}/{name}_{model_name}.csv"),
        ensure_parent(f"{model_cfg['output_dir']}/line_count.yaml"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/track_line_count.yaml")
    parser.add_argument("--model", default="yolov8s_finetuned")
    parser.add_argument("--source", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    track_cfg = cfg["track"]
    line_cfg = cfg.get("line_count", {})
    if args.model not in cfg["models"]:
        raise ValueError(f"unknown model '{args.model}', choose from {list(cfg['models'])}")
    model_cfg = cfg["models"][args.model]
    model_path = resolve_model(model_cfg["weights"])
    source = resolve_mid2(args.source or cfg["data"]["source"])
    output_video, output_csv, line_yaml = resolve_outputs(source, args.model, model_cfg, args.tag)

    if not source.exists():
        raise FileNotFoundError(f"video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    model = YOLO(model_path)
    results = model.track(
        source=str(source),
        tracker=track_cfg["tracker"],
        persist=True,
        stream=True,
        conf=float(track_cfg["conf"]),
        iou=float(track_cfg["iou"]),
        imgsz=int(track_cfg["image_size"]),
        device=select_device(str(track_cfg["device"])),
        half=bool(track_cfg["half"]) and torch.cuda.is_available(),
        save=False,
        verbose=False,
    )

    writer = open_writer(output_video, fps, width, height)
    enabled_models = line_cfg.get("enabled_models", [args.model])
    line_enabled = bool(line_cfg.get("enabled", False)) and args.model in enabled_models
    line_axis = str(line_cfg.get("axis", "x"))
    center_key = "cx" if line_axis == "x" else "cy"
    line_pos = resolve_line(line_axis, float(line_cfg.get("position", 0.5)), width, height)
    prev_centers = {}
    counted_ids = set()
    events = []

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

        for frame_id, result in enumerate(results):
            annotated = result.plot(
                conf=bool(track_cfg["show_conf"]),
                line_width=int(track_cfg["line_width"]),
                font_size=int(track_cfg["font_size"]),
            )
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))

            boxes = result.boxes
            if boxes is None or boxes.id is None:
                if line_enabled:
                    draw_line_count(annotated, line_axis, line_pos, len(events), line_cfg)
                writer.write(annotated)
                continue

            xyxy = tensor_list(boxes.xyxy)
            track_ids = [int(v) for v in tensor_list(boxes.id)]
            cls_ids = [int(v) for v in tensor_list(boxes.cls)]
            confs = tensor_list(boxes.conf)

            for box, track_id, class_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                csv_writer.writerow(
                    {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "conf": f"{float(conf):.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                        "cx": f"{cx:.2f}",
                        "cy": f"{cy:.2f}",
                    }
                )

                if line_enabled and track_id not in counted_ids:
                    curr = cx if center_key == "cx" else cy
                    prev = prev_centers.get(track_id)
                    prev_centers[track_id] = curr
                    if prev is not None and (prev - line_pos) * (curr - line_pos) < 0:
                        counted_ids.add(track_id)
                        events.append(
                            {
                                "track_id": track_id,
                                "class_id": class_id,
                                "class_name": result.names[class_id],
                                "cross_frame": frame_id,
                                "direction": crossing_direction(line_axis, prev, curr),
                                "prev_center": round(prev, 2),
                                "curr_center": round(curr, 2),
                            }
                        )

            if line_enabled:
                draw_line_count(annotated, line_axis, line_pos, len(events), line_cfg)
            writer.write(annotated)

    writer.release()
    if line_enabled:
        line_result = {
            "video": display_path(source),
            "tracked_video": display_path(output_video),
            "tracks_csv": display_path(output_csv),
            "model": args.model,
            "tracker": track_cfg["tracker"],
            "line_type": "vertical" if line_axis == "x" else "horizontal",
            "total_count": len(events),
            "left_to_right": sum(event["direction"] == "left_to_right" for event in events),
            "right_to_left": sum(event["direction"] == "right_to_left" for event in events),
            "top_to_bottom": sum(event["direction"] == "top_to_bottom" for event in events),
            "bottom_to_top": sum(event["direction"] == "bottom_to_top" for event in events),
            "events": events,
        }
        line_result["line_x" if line_axis == "x" else "line_y"] = line_pos
        save_yaml(line_yaml, line_result)
        print(f"saved line count: {line_yaml}")
    print(f"saved video: {output_video}")
    print(f"saved tracks: {output_csv}")


if __name__ == "__main__":
    main()
