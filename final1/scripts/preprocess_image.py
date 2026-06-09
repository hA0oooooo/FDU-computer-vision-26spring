import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GENERATED_STEMS = {"depth", "rgba"}
GENERATED_SUFFIXES = ("_foreground", "_mask")
ALPHA_THRESHOLD = 10
OBJECT_PRESETS = {
    "objectA": {
        "rembg_model": "bria-rmbg",
        "component_mode": "largest",
        "min_component_area": 0,
        "alpha_mode": "binary",
    },
    "objectC": {
        "rembg_model": "bria-rmbg",
        "component_mode": "all",
        "min_component_area": 0,
        "alpha_mode": "soft",
    },
}


def resolve_input_path(input_path: str) -> Path:
    raw_path = Path(input_path).expanduser()

    if raw_path.is_absolute():
        return raw_path

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    candidates = [
        project_root / raw_path,
        Path.cwd() / raw_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Input path not found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def is_source_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.lower() not in GENERATED_STEMS
        and not path.stem.lower().endswith(GENERATED_SUFFIXES)
    )


def collect_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if not is_source_image(input_path):
            raise ValueError(f"Unsupported input image: {input_path}")
        return [input_path]

    if input_path.is_dir():
        image_paths = sorted(path for path in input_path.iterdir() if is_source_image(path))
        if image_paths:
            return image_paths
        raise FileNotFoundError(f"No source images found in directory: {input_path}")

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def filter_alpha_components(
    alpha: Image.Image,
    threshold: int,
    component_mode: str,
    min_component_area: int,
) -> Image.Image:
    alpha_arr = np.asarray(alpha)
    binary_mask = (alpha_arr > threshold).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if component_count <= 1:
        return alpha

    if component_mode == "all":
        return alpha

    if component_mode == "min-area":
        keep_labels = {
            label
            for label in range(1, component_count)
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area
        }
        if not keep_labels:
            keep_labels = {1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])}
        keep_mask = np.isin(labels, list(keep_labels))
    else:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep_mask = labels == largest_label

    filtered_alpha = np.where(keep_mask, alpha_arr, 0).astype(np.uint8)
    return Image.fromarray(filtered_alpha)


def apply_alpha_mode(alpha: Image.Image, threshold: int, alpha_mode: str) -> Image.Image:
    if alpha_mode == "soft":
        return alpha

    alpha_arr = np.asarray(alpha)
    hard_alpha = np.where(alpha_arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(hard_alpha)


def set_transparent_rgb(rgba: Image.Image, background=(255, 255, 255)) -> Image.Image:
    rgba_arr = np.array(rgba)
    rgba_arr[rgba_arr[:, :, 3] == 0, :3] = background
    return Image.fromarray(rgba_arr, mode="RGBA")


def preprocess_image(
    image_path: Path,
    foreground_path: Path,
    session,
    threshold: int,
    component_mode: str,
    min_component_area: int,
    alpha_mode: str,
) -> Path:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgba = remove(image, session=session).convert("RGBA")

    if component_mode != "all":
        rgba.putalpha(
            filter_alpha_components(
                rgba.getchannel("A"),
                threshold,
                component_mode,
                min_component_area,
            )
        )

    rgba.putalpha(apply_alpha_mode(rgba.getchannel("A"), threshold, alpha_mode))
    rgba = set_transparent_rgb(rgba)
    rgba.save(foreground_path)
    return foreground_path


def get_foreground_path(image_path: Path, input_path: Path) -> Path:
    if input_path.is_dir():
        output_dir = input_path.parent / "images"
        return output_dir / f"{image_path.stem}.png"

    return image_path.with_name(f"{image_path.stem}_foreground.png")


def prepare_directory_output(input_path: Path) -> None:
    output_dir = input_path.parent / "images"
    if output_dir.resolve() == input_path.resolve():
        raise ValueError("Directory input must be a raw image directory, e.g. images_raw")

    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in OBJECT_PRESETS:
        raise SystemExit("Invalid preprocessing arguments")

    input_path = resolve_input_path(sys.argv[1])
    preset = OBJECT_PRESETS[sys.argv[2]]
    image_paths = collect_image_paths(input_path)
    if input_path.is_dir():
        prepare_directory_output(input_path)
    session = new_session(preset["rembg_model"])

    for image_path in image_paths:
        foreground_path = preprocess_image(
            image_path,
            foreground_path=get_foreground_path(image_path, input_path),
            session=session,
            threshold=ALPHA_THRESHOLD,
            component_mode=preset["component_mode"],
            min_component_area=preset["min_component_area"],
            alpha_mode=preset["alpha_mode"],
        )
        print(foreground_path)


if __name__ == "__main__":
    main()
