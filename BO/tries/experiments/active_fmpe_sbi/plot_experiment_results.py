#!/usr/bin/env python3
"""Create diagnostic plots for the active Lotka-Volterra SBI experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def as_int(rows: list[dict[str, str]], key: str) -> list[int]:
    return [int(row[key]) for row in rows]


def plot_metric_panel(ax: plt.Axes, rows: list[dict[str, str]], metric: str, ylabel: str) -> None:
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            grouped[int(row["simulations"])].append(float(row[metric]))
        xs = sorted(grouped)
        means = np.array([np.mean(grouped[x]) for x in xs])
        stds = np.array([np.std(grouped[x]) for x in xs])
        ax.plot(
            xs,
            means,
            marker="o",
            linewidth=2,
            label=method,
        )
        if any(len(grouped[x]) > 1 for x in xs):
            ax.fill_between(xs, means - stds, means + stds, alpha=0.18)
    ax.set_xlabel("Simulator calls")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()


def plot_trace_panel(ax: plt.Axes, trace: list[dict[str, str]], key: str, ylabel: str) -> None:
    if not trace:
        ax.text(0.5, 0.5, "No BO trace found", ha="center", va="center")
        ax.set_axis_off()
        return
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in trace:
        grouped[int(row["round"])].append(float(row[key]))
    xs = sorted(grouped)
    means = np.array([np.mean(grouped[x]) for x in xs])
    stds = np.array([np.std(grouped[x]) for x in xs])
    ax.plot(xs, means, marker="o", linewidth=2)
    if any(len(grouped[x]) > 1 for x in xs):
        ax.fill_between(xs, means - stds, means + stds, alpha=0.18)
    ax.set_xlabel("BO round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def create_summary(input_dir: Path, output: Path) -> None:
    metrics = read_csv(input_dir / "metrics.csv")
    trace_path = input_dir / "bo_trace.csv"
    trace = read_csv(trace_path) if trace_path.exists() else []
    metric_keys = set(metrics[0].keys()) if metrics else set()
    trace_keys = set(trace[0].keys()) if trace else set()

    if "validation_log_posterior" in metric_keys:
        quality_metric = "validation_log_posterior"
        quality_label = "Validation log posterior\n(higher is better)"
        quality_title = "Posterior Quality"
    elif "posterior_mean_rmse" in metric_keys:
        quality_metric = "posterior_mean_rmse"
        quality_label = "Posterior mean RMSE\n(lower is better)"
        quality_title = "Posterior Quality"
    else:
        raise ValueError("metrics.csv must contain validation_log_posterior or posterior_mean_rmse.")

    if "reward_delta_log_posterior" in trace_keys:
        reward_metric = "reward_delta_log_posterior"
        reward_label = "BO reward\n(delta validation log posterior)"
    elif "reward_delta_negative_rmse" in trace_keys:
        reward_metric = "reward_delta_negative_rmse"
        reward_label = "BO reward\n(delta negative RMSE)"
    else:
        reward_metric = ""
        reward_label = "BO reward"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_metric_panel(
        axes[0, 0],
        metrics,
        quality_metric,
        quality_label,
    )
    axes[0, 0].set_title(quality_title)

    plot_metric_panel(
        axes[0, 1],
        metrics,
        "coverage_error",
        "Coverage error\n(lower is better)",
    )
    axes[0, 1].set_title("Calibration Proxy")

    if reward_metric:
        plot_trace_panel(axes[1, 0], trace, reward_metric, reward_label)
    else:
        axes[1, 0].text(0.5, 0.5, "No BO reward trace found", ha="center", va="center")
        axes[1, 0].set_axis_off()
    axes[1, 0].set_title("BO Reward Trace")

    if trace:
        for key, label in [
            ("prey0", "initial prey"),
            ("pred0", "initial predator"),
            ("t_start", "observation start"),
            ("t_span", "observation span"),
        ]:
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in trace:
                grouped[int(row["round"])].append(float(row[key]))
            xs = sorted(grouped)
            means = np.array([np.mean(grouped[x]) for x in xs])
            axes[1, 1].plot(xs, means, marker="o", label=label)
        axes[1, 1].set_xlabel("BO round")
        axes[1, 1].set_ylabel("Decoded design value")
        axes[1, 1].set_title("BO-Selected Designs")
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No BO designs found", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.suptitle("Active Lotka-Volterra SBI: Random Design vs BO-Guided Design", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "outputs_check",
        help="Directory containing metrics.csv and bo_trace.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to INPUT_DIR/experiment_summary.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input_dir / "experiment_summary.png"
    create_summary(args.input_dir, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
