#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TWODGS_METRICS = {
    "train_loss_patches/normal_loss": "twodgs/loss_normal",
    "train_loss_patches/total_loss": "twodgs/loss_total",
    "train/loss_viewpoint - l1_loss": "twodgs/eval_l1_loss",
    "train/loss_viewpoint - psnr": "twodgs/eval_psnr",
}

DREAMFUSION_METRICS = {
    "train/loss_sds": "dreamfusion/loss_sds",
    "train/loss_orient": "dreamfusion/loss_orient",
    "train/loss_sparsity": "dreamfusion/loss_sparsity",
    "train/loss_opaque": "dreamfusion/loss_opaque",
}

MAGIC123_METRICS = {
    "train/loss": "magic123/loss_total",
    "train/loss_rgb": "magic123/loss_rgb",
    "train/loss_mask": "magic123/loss_mask",
}


def metric_map(profile: str):
    if profile == "twodgs":
        return TWODGS_METRICS
    if profile == "dreamfusion":
        return DREAMFUSION_METRICS
    if profile == "magic123":
        return MAGIC123_METRICS
    raise ValueError(f"Unknown metric profile: {profile}")


def wandb_group(profile: str):
    if profile == "magic123":
        return "object_C_magic123"
    return profile


def event_dirs(path: Path):
    if path.is_file() and path.name.startswith("events.out.tfevents"):
        return [path.parent]
    if path.is_dir():
        return sorted({event.parent for event in path.rglob("events.out.tfevents*")})
    return []


def log_tensorboard_scalars(source: Path, profile: str):
    count = 0
    selected_metrics = metric_map(profile)
    metrics_by_step = {}
    dirs = event_dirs(source)
    for event_dir in dirs:
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if tag not in selected_metrics:
                continue
            metric = selected_metrics[tag]
            for scalar in accumulator.Scalars(tag):
                metrics_by_step.setdefault(scalar.step, {})[metric] = scalar.value
                count += 1
    for step in sorted(metrics_by_step):
        wandb.log(metrics_by_step[step], step=step)
    return count


def maybe_float(value):
    if value == "" or value.lower() == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def log_csv_scalars(source: Path, profile: str):
    count = 0
    selected_metrics = metric_map(profile)
    csv_files = []
    if source.is_file() and source.suffix == ".csv":
        csv_files = [source]
    elif source.is_dir():
        csv_files = sorted(source.rglob("metrics.csv"))

    for csv_file in csv_files:
        with csv_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                step_value = maybe_float(row.get("step", ""))
                step = int(step_value) if step_value is not None else row_index
                payload = {}
                for key, value in row.items():
                    if key in {"step", "epoch"}:
                        continue
                    if key not in selected_metrics:
                        continue
                    numeric = maybe_float(value)
                    if numeric is not None:
                        payload[selected_metrics[key]] = numeric
                if payload:
                    count += len(payload)
                    wandb.log(payload, step=step)
    return count


def main():
    parser = argparse.ArgumentParser(description="Upload existing training metrics to WandB.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", required=True, choices=["twodgs", "dreamfusion", "magic123"])
    parser.add_argument("sources", nargs="+", help="TensorBoard log dirs/files or CSV log dirs/files")
    args = parser.parse_args()

    existing_sources = [Path(source) for source in args.sources if Path(source).exists()]
    if not existing_sources:
        print(f"No metric sources found for {args.run_name}; skipping WandB upload.")
        return

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        job_type="metrics-upload",
        group=wandb_group(args.profile),
        tags=[args.profile],
        config={"sources": [str(source) for source in existing_sources]},
        settings=wandb.Settings(silent=True, x_disable_viewer=True),
    )
    total = 0
    try:
        for source in existing_sources:
            total += log_tensorboard_scalars(source, args.profile)
            total += log_csv_scalars(source, args.profile)
    finally:
        run.finish()

    print(f"Uploaded {total} scalar values to WandB run {args.run_name}.")


if __name__ == "__main__":
    main()
