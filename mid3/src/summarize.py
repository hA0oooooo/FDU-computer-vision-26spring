import argparse
import json
import os

import pandas as pd

from .utils import ensure_dir


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(output_root="outputs", experiments=None):
    experiments = experiments or ["unet_ce", "unet_dice", "unet_ce_dice"]
    eval_path = os.path.join(output_root, "eval.csv")
    eval_df = pd.read_csv(eval_path) if os.path.exists(eval_path) else pd.DataFrame()

    rows = []
    for experiment_name in experiments:
        summary_path = os.path.join(output_root, experiment_name, "summary.json")
        if not os.path.exists(summary_path):
            continue

        summary = load_json(summary_path)
        row = {
            "experiment_name": experiment_name,
            "loss": summary.get("loss"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_mIoU": summary.get("best_val_mIoU"),
            "val_pixel_acc": summary.get("best_val_pixel_acc"),
            "foreground_iou": summary.get("best_val_foreground_iou"),
            "background_iou": summary.get("best_val_background_iou"),
            "boundary_iou": summary.get("best_val_boundary_iou"),
            "total_params": summary.get("total_params"),
        }

        if not eval_df.empty:
            matches = eval_df[(eval_df["experiment_name"] == experiment_name) & (eval_df["split"] == "val")]
            if len(matches) > 0:
                latest = matches.iloc[-1]
                row.update(
                    {
                        "val_pixel_acc": latest["pixel_acc"],
                        "foreground_iou": latest["foreground_iou"],
                        "background_iou": latest["background_iou"],
                        "boundary_iou": latest["boundary_iou"],
                        "eval_val_mIoU": latest["mIoU"],
                    }
                )

        rows.append(row)

    ensure_dir(output_root)
    out_path = os.path.join(output_root, "summary.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    build_summary(output_root=args.output_root)


if __name__ == "__main__":
    main()
