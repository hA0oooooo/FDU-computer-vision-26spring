from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGSIZE = (8, 6)
DPI = 100
LEGACY_OUTPUTS = [
    "compare_zero_shot_l1.png",
    "compare_dimension_l1.png",
]
RUNS = {
    "act_B": {
        "path": OUTPUT_DIR / "act_B" / "act_B_050000_eval.json",
        "color": "#d62728",
    },
    "act_ABC": {
        "path": OUTPUT_DIR / "act_ABC" / "act_ABC_050000_eval.json",
        "color": "#1f77b4",
    },
}


def load_results() -> dict:
    results = {}
    for name, spec in RUNS.items():
        if not spec["path"].exists():
            raise FileNotFoundError(spec["path"])
        results[name] = json.loads(spec["path"].read_text())
    return results


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_history(results: dict) -> None:
    plt.figure(figsize=FIGSIZE)
    for name, data in results.items():
        history = data.get("eval_history", [])
        steps = [item["step"] for item in history]
        values = [item["normalized_action_l1"] for item in history]
        plt.plot(steps, values, label=name, color=RUNS[name]["color"], linewidth=1.8)
    plt.title("Sampled D Eval During Training")
    plt.xlabel("Training step")
    plt.ylabel("Normalized action L1")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(OUTPUT_DIR / "compare_eval_history.png")


def plot_action_l1(results: dict) -> None:
    dims = list(next(iter(results.values()))["final_eval"]["per_dimension_l1"])
    labels = ["zero_shot"] + dims
    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=FIGSIZE)
    for i, (name, data) in enumerate(results.items()):
        offset = (i - 0.5) * width
        values = [data["final_eval"]["normalized_action_l1"]]
        values.extend(data["final_eval"]["per_dimension_l1"][dim] for dim in dims)
        plt.bar([v + offset for v in x], values, width=width, label=name, color=RUNS[name]["color"])
    plt.title("Zero-shot D Action L1: Overall and Per-dimension")
    plt.xlabel("Action metric")
    plt.ylabel("Normalized action L1")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(OUTPUT_DIR / "compare_action_l1.png")


def plot_chunk_step(results: dict) -> None:
    plt.figure(figsize=FIGSIZE)
    for name, data in results.items():
        items = data["final_eval"]["chunk_step_l1"]
        steps = [item["chunk_step"] for item in items]
        values = [item["normalized_action_l1"] for item in items]
        plt.plot(steps, values, label=name, color=RUNS[name]["color"], linewidth=1.8)
    plt.title("ACT Chunk-step Action L1 on D")
    plt.xlabel("Chunk step")
    plt.ylabel("Normalized action L1")
    plt.legend()
    plt.grid(alpha=0.25)
    savefig(OUTPUT_DIR / "compare_chunk_l1.png")


def write_chunk_summary(results: dict) -> None:
    with open(OUTPUT_DIR / "compare_chunk_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "normalized_action_l1",
                "front_half_l1",
                "back_half_l1",
                "degradation_ratio",
            ],
        )
        writer.writeheader()
        for name, data in results.items():
            summary = data["final_eval"]["chunk_summary"]
            writer.writerow(
                {
                    "model": name,
                    "normalized_action_l1": data["final_eval"]["normalized_action_l1"],
                    "front_half_l1": summary["front_half_l1"],
                    "back_half_l1": summary["back_half_l1"],
                    "degradation_ratio": summary["degradation_ratio"],
                }
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_OUTPUTS:
        (OUTPUT_DIR / filename).unlink(missing_ok=True)
    results = load_results()
    plot_history(results)
    plot_action_l1(results)
    plot_chunk_step(results)
    write_chunk_summary(results)


if __name__ == "__main__":
    main()
