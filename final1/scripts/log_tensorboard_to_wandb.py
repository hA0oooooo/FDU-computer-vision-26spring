#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def event_dirs(logdir: Path):
    return sorted({path.parent for path in logdir.rglob("events.out.tfevents*")})


def main():
    parser = argparse.ArgumentParser(description="Upload TensorBoard scalars to WandB.")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    dirs = event_dirs(logdir)
    if not dirs:
        raise SystemExit(f"No TensorBoard event files found under: {logdir}")

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        job_type="tensorboard-import",
        config={"source_logdir": str(logdir)},
    )

    logged = 0
    try:
        for event_dir in dirs:
            prefix = event_dir.relative_to(logdir).as_posix()
            if prefix == ".":
                prefix = ""
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                metric_name = f"{prefix}/{tag}" if prefix else tag
                for scalar in accumulator.Scalars(tag):
                    wandb.log({metric_name: scalar.value}, step=scalar.step)
                    logged += 1
    finally:
        run.finish()

    print(f"Uploaded {logged} TensorBoard scalar points from {logdir}")


if __name__ == "__main__":
    main()
