import copy
import itertools
import os

import pandas as pd

from .train import train_experiment
from .utils import ensure_dir, load_config


def run_search(config_path="configs/search.yaml"):
    search_cfg = load_config(config_path)
    base_cfg = load_config(search_cfg["base_config"])
    output_dir = search_cfg.get("output_dir", "outputs/search")
    ensure_dir(output_dir)

    search_space = search_cfg["search_space"]
    keys = ["backbone_lr", "head_lr", "batch_size", "epochs"]
    trials = itertools.product(*(search_space[key] for key in keys))

    rows = []
    for trial_id, values in enumerate(trials, start=1):
        backbone_lr, head_lr, batch_size, epochs = values
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment_name"] = f"search_trial_{trial_id:02d}"
        cfg["output_dir"] = os.path.join(output_dir, cfg["experiment_name"])
        cfg["train"]["backbone_lr"] = backbone_lr
        cfg["train"]["head_lr"] = head_lr
        cfg["train"]["batch_size"] = batch_size
        cfg["train"]["epochs"] = epochs

        result = train_experiment(cfg)
        row = {
            "trial": trial_id,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "output_dir": result["output_dir"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "search_summary.csv"), index=False)

    return rows


def main():
    run_search()


if __name__ == "__main__":
    main()
