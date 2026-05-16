from __future__ import annotations

from pathlib import Path

import yaml


MID2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MID2_ROOT.parent
VISDRONE_ROOT = REPO_ROOT / "dataset" / "VisDrone"


def load_yaml(path: str | Path) -> dict:
    with open(resolve_mid2(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path: str | Path, data: dict) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path


def resolve_mid2(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (MID2_ROOT / path).resolve()


def resolve_model(value: str | Path) -> str:
    text = str(value)
    if "/" not in text and "\\" not in text:
        return text
    return str(resolve_mid2(text))


def display_path(path: str | Path) -> str:
    path = resolve_mid2(path)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_parent(path: str | Path) -> Path:
    path = resolve_mid2(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
