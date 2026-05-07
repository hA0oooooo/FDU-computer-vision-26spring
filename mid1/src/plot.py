import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_history(history_path: str, save_path: str):
    history = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_acc"], marker="o", label="train_acc")
    axes[0].plot(history["epoch"], history["val_acc"], marker="o", label="val_acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_loss"], marker="o", label="train_loss")
    axes[1].plot(history["epoch"], history["val_loss"], marker="o", label="val_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--save", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_history(args.history, args.save)


if __name__ == "__main__":
    main()
