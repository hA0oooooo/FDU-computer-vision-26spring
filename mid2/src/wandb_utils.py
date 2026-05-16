from __future__ import annotations

import csv

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class _DisabledRun:
    summary = {}


def init_wandb(cfg):
    if wandb is None or not cfg.get("wandb", {}).get("enabled", True):
        return _DisabledRun()

    return wandb.init(
        entity=cfg.get("wandb", {}).get("entity"),
        project=cfg.get("wandb", {}).get("project", "mid-project"),
        name=cfg["experiment_name"],
        config=cfg,
    )


def log_ultralytics_results(results_csv) -> None:
    if wandb is None or wandb.run is None or not results_csv.exists():
        return

    with open(results_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clean = {key.strip(): value for key, value in row.items()}
            epoch = int(float(clean.pop("epoch")))
            log_dict = {}
            for key, value in clean.items():
                if value == "":
                    continue
                log_dict[key] = float(value)
            wandb.log(log_dict, step=epoch)


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()
