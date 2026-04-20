from __future__ import annotations

import copy
import csv
import itertools
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from src.train import train_model
    from src.utils import load_config
else:
    from .train import train_model
    from .utils import load_config


SEARCH_KEYS = {
    "lr_values": "lr",
    "hidden_dim_values": "hidden_dim",
    "weight_decay_values": "weight_decay",
    "activation_values": "activation",
    "batch_size_values": "batch_size",
}


def _resolve_config(config_path: str | Path, config: dict | None) -> dict:
    if config is not None:
        return copy.deepcopy(config)
    return load_config(config_path)


def _build_search_space(search_config: dict, base_config: dict) -> list[dict]:
    dimensions: list[tuple[str, list]] = []
    for list_key, param_key in SEARCH_KEYS.items():
        values = search_config.get(list_key)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        dimensions.append((param_key, values))

    if not dimensions:
        return [{k: base_config[k] for k in ("lr", "hidden_dim", "weight_decay") if k in base_config}]

    names = [item[0] for item in dimensions]
    grids = [item[1] for item in dimensions]
    combinations = []
    for values in itertools.product(*grids):
        combinations.append(dict(zip(names, values)))
    return combinations


def _save_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_log_line(line: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def run_search(config_path: str | Path = "configs/search.yaml", config: dict | None = None) -> dict:
    search_config = _resolve_config(config_path, config)
    base_config = load_config(search_config.get("base_config_path", "configs/train.yaml"))
    search_space = _build_search_space(search_config, base_config)
    search_dir = Path(search_config.get("search_dir", "artifacts/search"))
    search_dir.mkdir(parents=True, exist_ok=True)
    train_verbose = bool(search_config.get("verbose", False))
    summary_log_path = search_config.get("trial_log_path", str(search_dir / "search.txt"))
    Path(summary_log_path).write_text("", encoding="utf-8")

    rows = []
    best_row = None

    for trial_idx, params in enumerate(search_space, start=1):
        trial_config = copy.deepcopy(base_config)
        trial_config.update(params)
        trial_config["save_best_model"] = False
        trial_config["save_best_meta"] = False
        trial_config["save_history"] = False
        trial_config["save_plots"] = False
        trial_config["verbose"] = train_verbose

        summary = train_model(config=trial_config)
        row = {
            "trial": trial_idx,
            **params,
            "best_val_acc": summary["best_val_acc"],
            "best_epoch": summary["best_epoch"],
            "epochs_ran": summary["epochs_ran"],
            "final_train_acc": summary["final_train_acc"],
            "final_val_acc": summary["final_val_acc"],
            "final_train_loss": summary["final_train_loss"],
            "final_val_loss": summary["final_val_loss"],
        }
        rows.append(row)

        if best_row is None or row["best_val_acc"] > best_row["best_val_acc"]:
            best_row = row

        trial_line = (
            f"[{trial_idx:02d}/{len(search_space):02d}] "
            f"lr={row.get('lr')} | hidden_dim={row.get('hidden_dim')} | "
            f"weight_decay={row.get('weight_decay')} | best_val_acc={row['best_val_acc']:.4f}"
        )
        print(trial_line)
        _append_log_line(trial_line, summary_log_path)

    csv_path = search_dir / "search.csv"
    _save_csv(rows, csv_path)

    best_line = (
        f"best_trial: lr={best_row.get('lr')} | hidden_dim={best_row.get('hidden_dim')} | "
        f"weight_decay={best_row.get('weight_decay')} | best_val_acc={best_row['best_val_acc']:.4f}"
    )
    print(best_line)
    _append_log_line(best_line, summary_log_path)

    return {
        "search_space_size": len(search_space),
        "best_trial": best_row,
        "results_csv": str(csv_path),
        "trial_log_path": str(summary_log_path),
    }


if __name__ == "__main__":
    run_search()
