from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .utils import load_yaml, resolve_mid2


def default_video_path(config_path: str, model_name: str) -> Path:
    cfg = load_yaml(config_path)
    source = resolve_mid2(cfg["data"]["source"])
    return resolve_mid2(f"../dataset/videos/output/{source.stem}_{model_name}.mp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/track_line_count.yaml")
    parser.add_argument("--model", default="yolov8s_finetuned")
    parser.add_argument("--source", default=None)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()

    video_path = resolve_mid2(args.source) if args.source else default_video_path(args.config, args.model)
    output_dir = resolve_mid2("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    for frame_id in args.frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_id} from {video_path}")
        frame_path = output_dir / f"{frame_id}.png"
        cv2.imwrite(str(frame_path), image)
    capture.release()

    print(f"saved frames: {output_dir}")


if __name__ == "__main__":
    main()
