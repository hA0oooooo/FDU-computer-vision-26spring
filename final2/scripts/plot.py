from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
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
    plt.savefig(path, dpi=220)
    plt.close()


def plot_history(results: dict) -> None:
    plt.figure(figsize=(7, 4))
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


def plot_final_l1(results: dict) -> None:
    names = list(results)
    values = [results[name]["final_eval"]["normalized_action_l1"] for name in names]
    plt.figure(figsize=(5, 4))
    plt.bar(names, values, color=[RUNS[name]["color"] for name in names])
    plt.title("Zero-shot D Action Error")
    plt.xlabel("Model")
    plt.ylabel("Normalized action L1")
    savefig(OUTPUT_DIR / "compare_zero_shot_l1.png")


def plot_per_dimension(results: dict) -> None:
    dims = list(next(iter(results.values()))["final_eval"]["per_dimension_l1"])
    x = list(range(len(dims)))
    width = 0.35
    plt.figure(figsize=(8, 4))
    for i, (name, data) in enumerate(results.items()):
        offset = (i - 0.5) * width
        values = [data["final_eval"]["per_dimension_l1"][dim] for dim in dims]
        plt.bar([v + offset for v in x], values, width=width, label=name, color=RUNS[name]["color"])
    plt.title("Per-dimension Action L1 on D")
    plt.xlabel("Action dimension")
    plt.ylabel("Normalized action L1")
    plt.xticks(x, dims)
    plt.legend()
    savefig(OUTPUT_DIR / "compare_dimension_l1.png")


def plot_chunk_step(results: dict) -> None:
    plt.figure(figsize=(8, 4))
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
    results = load_results()
    plot_history(results)
    plot_final_l1(results)
    plot_per_dimension(results)
    plot_chunk_step(results)
    write_chunk_summary(results)


if __name__ == "__main__":
    main()
