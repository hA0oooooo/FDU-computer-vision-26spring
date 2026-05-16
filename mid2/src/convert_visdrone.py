from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2

from .utils import VISDRONE_ROOT


SPLITS = {
    "train": VISDRONE_ROOT / "VisDrone2019-DET-train" / "VisDrone2019-DET-train",
    "val": VISDRONE_ROOT / "VisDrone2019-DET-val" / "VisDrone2019-DET-val",
    "test": VISDRONE_ROOT / "VisDrone2019-DET-test-dev",
}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_box(row: list[str], img_w: int, img_h: int) -> str | None:
    x, y, w, h = map(float, row[:4])
    score = int(float(row[4]))
    cls_raw = int(float(row[5]))

    if score == 0 or cls_raw < 1 or cls_raw > 10 or w <= 0 or h <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    cls = cls_raw - 1
    cx = (x1 + box_w / 2) / img_w
    cy = (y1 + box_h / 2) / img_h
    nw = box_w / img_w
    nh = box_h / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_split(split: str, root: Path) -> tuple[int, int]:
    src_images = root / "images"
    src_annotations = root / "annotations"
    dst_images = VISDRONE_ROOT / "images" / split
    dst_labels = VISDRONE_ROOT / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for image_path in sorted(src_images.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        img_h, img_w = image.shape[:2]

        annotation_path = src_annotations / f"{image_path.stem}.txt"
        labels = []
        if annotation_path.exists():
            with open(annotation_path, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 6:
                        continue
                    label = convert_box(row, img_w, img_h)
                    if label is not None:
                        labels.append(label)

        link_or_copy(image_path, dst_images / image_path.name)
        with open(dst_labels / f"{image_path.stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
            if labels:
                f.write("\n")

        image_count += 1
        label_count += len(labels)

    return image_count, label_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    selected = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
    for split, root in selected.items():
        image_count, label_count = convert_split(split, root)
        print(f"{split}: images={image_count}, labels={label_count}")


if __name__ == "__main__":
    main()
