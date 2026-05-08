import argparse
import os

import pandas as pd
import torch
from torch import nn

from .dataset import build_dataloaders
from .models import build_model
from .train import run_epoch
from .utils import count_parameters, ensure_dir, get_device, load_config


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def evaluate_experiment(cfg, checkpoint_path=None):
    output_dir = cfg["output_dir"]
    root_output_dir = "outputs"
    device = get_device(cfg.get("device", "auto"))
    checkpoint_path = checkpoint_path or os.path.join(output_dir, "checkpoints", "best.pt")

    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        use_amp=False,
        desc="test",
    )

    total_params, trainable_params = count_parameters(model)
    result = {
        "experiment_name": cfg["experiment_name"],
        "checkpoint": checkpoint_path,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    summary_path = os.path.join(root_output_dir, "eval.csv")
    ensure_dir(root_output_dir)
    pd.DataFrame([result]).to_csv(
        summary_path,
        mode="a",
        header=not os.path.exists(summary_path),
        index=False,
    )

    print(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    evaluate_experiment(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
