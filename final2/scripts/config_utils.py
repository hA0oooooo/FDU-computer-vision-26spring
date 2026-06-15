from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict[str, str]:
    config: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"Invalid config line: {line}")
            config[key.strip()] = value.strip().strip("\"'")
    return config


def require(config: dict[str, str], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config or config[key] == ""]
    if missing:
        raise KeyError(f"Missing config keys: {', '.join(missing)}")


def root_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
