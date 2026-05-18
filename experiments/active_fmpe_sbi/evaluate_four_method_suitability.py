#!/usr/bin/env python3
"""Four-method suitability check for the Lotka-Volterra SBI setting.

This diagnostic tests whether the current simulator setting is learnable at
all, and whether simple methods are already stronger than the NN-based
posterior estimators.

The compared methods are:

- prior_mean
- abc_knn
- gaussian_npe
- rectified_fmpe

Rows are written live to diagnostics.csv after each method-budget-replicate
unit, so interrupted runs keep completed results and can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from evaluate_fmpe_quality import (
    PARAM_NAMES,
    evaluate_fmpe,
    evaluate_gaussian,
    evaluate_prior,
    make_metric_row,
    posterior_mean_predictive_rmse,
    sample_coverage_error,
)
from run_lotka_volterra import (
    THETA_BOUNDS,
    sample_psi,
    sample_theta,
    simulate_batch,
)


METHODS = ("prior_mean", "abc_knn", "gaussian_npe", "rectified_fmpe")
BASE_FIELDNAMES = [
    "method",
    "budget",
    "replicate",
    "seed",
    "design_space",
    "abc_k_requested",
    "abc_k_used",
    "validation",
    "n_obs",
    "noise_std",
    "flow_samples_per_pair",
    "posterior_samples",
    "ode_steps",
    "fmpe_max_iter",
    "raw_rmse",
    "range_normalized_rmse",
    "prior_std_normalized_rmse",
    "coverage_error",
    "posterior_mean_predictive_rmse",
    "validation_log_posterior",
]
PER_PARAM_FIELDNAMES = []
for param_name in PARAM_NAMES:
    PER_PARAM_FIELDNAMES.extend(
        [
            f"rmse_{param_name}",
            f"range_normalized_rmse_{param_name}",
            f"prior_std_normalized_rmse_{param_name}",
        ]
    )
FIELDNAMES = BASE_FIELDNAMES + PER_PARAM_FIELDNAMES


def choose_abc_k(n_train: int, requested_k: int | None) -> int:
    if requested_k is not None:
        return max(1, min(requested_k, n_train))
    return max(1, min(n_train, max(10, min(100, int(np.sqrt(n_train))))))


def evaluate_abc_knn(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicate: int,
    budget: int,
    seed: int,
    n_obs: int,
    design_space: str,
    abc_k: int | None,
) -> tuple[dict[str, float | int | str], int]:
    theta_train, psi_train, x_train = train
    theta_val, psi_val, x_val = val
    k = choose_abc_k(len(theta_train), abc_k)

    train_features = np.hstack([x_train, psi_train])
    val_features = np.hstack([x_val, psi_val])
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train_features)
    val_std = scaler.transform(val_features)

    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(train_std)
    indices = neighbors.kneighbors(val_std, return_distance=False)
    samples = theta_train[indices]
    pred = samples.mean(axis=1)
    row = make_metric_row(
        "abc_knn",
        replicate,
        budget,
        theta_val,
        pred,
        sample_coverage_error(samples, theta_val),
        posterior_mean_predictive_rmse(pred, psi_val, x_val, seed + 41, n_obs, design_space),
    )
    return row, k


def add_run_metadata(
    row: dict[str, float | int | str],
    args: argparse.Namespace,
    rep_seed: int,
    abc_k_used: int | None = None,
) -> dict[str, float | int | str]:
    row = dict(row)
    row.update(
        {
            "seed": rep_seed,
            "design_space": args.design_space,
            "abc_k_requested": "" if args.abc_k is None else args.abc_k,
            "abc_k_used": "" if abc_k_used is None else abc_k_used,
            "validation": args.validation,
            "n_obs": args.n_obs,
            "noise_std": args.noise_std,
            "flow_samples_per_pair": args.flow_samples_per_pair,
            "posterior_samples": args.posterior_samples,
            "ode_steps": args.ode_steps,
            "fmpe_max_iter": args.fmpe_max_iter,
        }
    )
    return row


def completed_key(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row["method"]),
        int(row["budget"]),
        int(row["replicate"]),
        int(row["seed"]),
        str(row["design_space"]),
    )


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_live_row(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_paths(output_dir: Path, replicate: int) -> dict[str, Path]:
    rep_dir = output_dir / "simulated_data" / f"replicate_{replicate:03d}"
    return {
        "dir": rep_dir,
        "train_npz": rep_dir / "train_full.npz",
        "validation_npz": rep_dir / "validation.npz",
        "train_csv": rep_dir / "train_full.csv",
        "validation_csv": rep_dir / "validation.csv",
        "metadata": rep_dir / "metadata.json",
        "readme": rep_dir / "README.md",
    }


def replicate_data_exists(output_dir: Path, replicate: int) -> bool:
    paths = dataset_paths(output_dir, replicate)
    return paths["train_npz"].exists() and paths["validation_npz"].exists()


def dataset_to_csv(path: Path, data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    theta, psi, x = data
    fieldnames = (
        ["index"]
        + [f"theta_{name}" for name in PARAM_NAMES]
        + [f"psi_{i}" for i in range(psi.shape[1])]
        + [f"x_{i}" for i in range(x.shape[1])]
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(theta)):
            row: dict[str, float | int] = {"index": i}
            for name, value in zip(PARAM_NAMES, theta[i]):
                row[f"theta_{name}"] = float(value)
            for j, value in enumerate(psi[i]):
                row[f"psi_{j}"] = float(value)
            for j, value in enumerate(x[i]):
                row[f"x_{j}"] = float(value)
            writer.writerow(row)


def save_simulated_data(
    output_dir: Path,
    replicate: int,
    rep_seed: int,
    args: argparse.Namespace,
    budgets: list[int],
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    paths = dataset_paths(output_dir, replicate)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    theta_train, psi_train, x_train = train
    theta_val, psi_val, x_val = val

    np.savez_compressed(
        paths["train_npz"],
        theta=theta_train,
        psi=psi_train,
        x=x_train,
        budgets=np.asarray(budgets),
        seed=np.asarray([rep_seed]),
    )
    np.savez_compressed(
        paths["validation_npz"],
        theta=theta_val,
        psi=psi_val,
        x=x_val,
        seed=np.asarray([rep_seed]),
    )
    dataset_to_csv(paths["train_csv"], train)
    dataset_to_csv(paths["validation_csv"], val)

    metadata = {
        "replicate": replicate,
        "seed": rep_seed,
        "design_space": args.design_space,
        "budgets": budgets,
        "max_budget": max(budgets),
        "validation": args.validation,
        "n_obs": args.n_obs,
        "noise_std": args.noise_std,
        "train_rows": int(len(theta_train)),
        "validation_rows": int(len(theta_val)),
        "notes": "Each training budget uses the first N rows of train_full.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    paths["readme"].write_text(
        "\n".join(
            [
                f"# Simulated Data Replicate {replicate}",
                "",
                "This folder stores the exact random-design synthetic data generated for Approach 1.3.",
                "",
                "`train_full.npz` and `train_full.csv` contain the synthetic training data that feeds `gaussian_npe` and `rectified_fmpe`.",
                "",
                "Budgeted training sets are prefixes:",
                "",
                "```text",
                "train_budget_N = first N rows of train_full",
                "```",
                "",
                "`validation.npz` and `validation.csv` contain the held-out validation observations used for evaluation.",
                "",
            ]
        )
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    numeric_keys = [
        "raw_rmse",
        "range_normalized_rmse",
        "prior_std_normalized_rmse",
        "coverage_error",
        "posterior_mean_predictive_rmse",
        "validation_log_posterior",
        "abc_k_used",
    ]
    numeric_keys += [f"rmse_{name}" for name in PARAM_NAMES]
    numeric_keys += [f"range_normalized_rmse_{name}" for name in PARAM_NAMES]
    numeric_keys += [f"prior_std_normalized_rmse_{name}" for name in PARAM_NAMES]

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row:
            continue
        groups[(str(row["method"]), int(row["budget"]))].append(row)

    summary: list[dict[str, float | int | str]] = []
    method_order = {method: i for i, method in enumerate(METHODS)}
    for (method, budget), group in sorted(groups.items(), key=lambda item: (item[0][1], method_order.get(item[0][0], 99))):
        out: dict[str, float | int | str] = {
            "method": method,
            "budget": budget,
            "replicates": len(group),
        }
        for key in numeric_keys:
            vals = [float(row[key]) for row in group if key in row and row[key] not in {"", None}]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        summary.append(out)
    return summary


def plot_summary(summary_rows: list[dict[str, float | int | str]], output: Path) -> None:
    labels = {
        "prior_mean": "prior mean",
        "abc_knn": "ABC-kNN",
        "gaussian_npe": "Gaussian NPE",
        "rectified_fmpe": "Rectified FMPE",
    }
    panels = [
        ("range_normalized_rmse_mean", "Prior-range normalized RMSE\n(lower is better)"),
        ("prior_std_normalized_rmse_mean", "Prior-std normalized RMSE\n(lower is better)"),
        ("coverage_error_mean", "Coverage error\n(lower is better)"),
        ("posterior_mean_predictive_rmse_mean", "Posterior-mean predictive RMSE\n(lower is better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, ylabel) in zip(axes.ravel(), panels):
        for method in METHODS:
            subset = [row for row in summary_rows if row["method"] == method and metric in row]
            if not subset:
                continue
            xs = np.array([int(row["budget"]) for row in subset])
            ys = np.array([float(row[metric]) for row in subset])
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker="o", linewidth=2, label=labels[method])
        ax.set_xlabel("Random-design training simulations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Approach 1.3: Four-Method Suitability Check", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def describe_final_budget(summary_rows: list[dict[str, float | int | str]]) -> list[str]:
    if not summary_rows:
        return ["No completed rows were available."]
    final_budget = max(int(row["budget"]) for row in summary_rows)
    final_rows = [row for row in summary_rows if int(row["budget"]) == final_budget]
    by_method = {str(row["method"]): row for row in final_rows}
    ordered = sorted(
        final_rows,
        key=lambda row: float(row.get("range_normalized_rmse_mean", float("inf"))),
    )

    lines = [
        f"Final evaluated budget: `{final_budget}` simulator calls.",
        "",
        "## Final-Budget Summary",
        "",
        "| Method | Range-norm RMSE | Prior-std RMSE | Coverage error | Predictive RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = by_method.get(method)
        if row is None:
            continue
        lines.append(
            "| "
            + method
            + " | "
            + f"{float(row['range_normalized_rmse_mean']):.4f} | "
            + f"{float(row['prior_std_normalized_rmse_mean']):.4f} | "
            + f"{float(row['coverage_error_mean']):.4f} | "
            + f"{float(row['posterior_mean_predictive_rmse_mean']):.4f} |"
        )

    lines += [
        "",
        "## Current Interpretation",
        "",
        "For lower-is-better metrics, the ideal error pattern is:",
        "",
        "```text",
        "prior_mean > abc_knn > gaussian_npe > rectified_fmpe",
        "```",
        "",
        "Observed final-budget ordering by range-normalized RMSE:",
        "",
        "```text",
        " < ".join(str(row["method"]) for row in ordered),
        "```",
        "",
    ]
    if ordered and str(ordered[0]["method"]) in {"abc_knn", "prior_mean"}:
        lines += [
            "A simple method is strongest at the final budget. Treat this as useful diagnostic signal: the simulator may be learnable, but the NN-based estimators need calibration, training-budget adjustment, or architecture work.",
            "",
        ]
    elif ordered and str(ordered[0]["method"]) == "rectified_fmpe":
        lines += [
            "Rectified FMPE is strongest on the final-budget point-estimate metric. Check coverage before treating it as a well-calibrated posterior.",
            "",
        ]
    else:
        lines += [
            "The final-budget ordering is mixed. Use per-metric and per-parameter diagnostics before deciding whether the setting is suitable.",
            "",
        ]
    return lines


def write_markdown_summary(
    path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Approach 1.3 Four-Method Suitability Check Results",
        "",
        "This diagnostic evaluates whether the Lotka-Volterra setting is suitable for increasingly complex posterior estimators.",
        "",
        "## Run Configuration",
        "",
        "```text",
        f"design_space = {args.design_space}",
        f"budgets = {' '.join(str(b) for b in sorted(set(args.budgets)))}",
        f"validation = {args.validation}",
        f"repeats = {args.repeats}",
        f"seed = {args.seed}",
        f"abc_k = {'sqrt(N), clipped to [10, 100]' if args.abc_k is None else args.abc_k}",
        f"flow_samples_per_pair = {args.flow_samples_per_pair}",
        f"posterior_samples = {args.posterior_samples}",
        f"ode_steps = {args.ode_steps}",
        f"fmpe_max_iter = {args.fmpe_max_iter}",
        f"save_sim_data = {args.save_sim_data}",
        "```",
        "",
        "The synthetic training and validation data are saved under:",
        "",
        "```text",
        "simulated_data/replicate_*/",
        "```",
        "",
        "## Methods",
        "",
        "```text",
        *METHODS,
        "```",
        "",
    ]
    lines.extend(describe_final_budget(summary_rows))
    path.write_text("\n".join(lines))


def generate_random_design_dataset(
    rng: np.random.Generator,
    target_n: int,
    n_obs: int,
    noise_std: float,
    design_space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total = 0
    while total < target_n:
        need = target_n - total
        batch_n = max(need, 16)
        batch = simulate_batch(
            sample_theta(rng, batch_n),
            sample_psi(rng, batch_n, design_space),
            rng,
            n_obs,
            noise_std,
            design_space,
        )
        parts.append(batch)
        total += len(batch[0])
    theta = np.vstack([part[0] for part in parts])[:target_n]
    psi = np.vstack([part[1] for part in parts])[:target_n]
    x = np.vstack([part[2] for part in parts])[:target_n]
    return theta, psi, x


def evaluate_unit(
    method: str,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    replicate: int,
    budget: int,
    rep_seed: int,
) -> dict[str, float | int | str]:
    if method == "prior_mean":
        row = evaluate_prior(val, replicate, budget, rep_seed + budget, args.n_obs, args.design_space)
        return add_run_metadata(row, args, rep_seed)
    if method == "abc_knn":
        row, k_used = evaluate_abc_knn(
            train,
            val,
            replicate,
            budget,
            rep_seed + 500 + budget,
            args.n_obs,
            args.design_space,
            args.abc_k,
        )
        return add_run_metadata(row, args, rep_seed, abc_k_used=k_used)
    if method == "gaussian_npe":
        row = evaluate_gaussian(
            train,
            val,
            replicate,
            budget,
            rep_seed + 1000 + budget,
            args.n_obs,
            args.design_space,
        )
        return add_run_metadata(row, args, rep_seed)
    if method == "rectified_fmpe":
        row = evaluate_fmpe(
            train,
            val,
            replicate,
            budget,
            rep_seed + 2000 + budget,
            args.n_obs,
            args.design_space,
            args.flow_samples_per_pair,
            args.posterior_samples,
            args.ode_steps,
            args.fmpe_max_iter,
        )
        return add_run_metadata(row, args, rep_seed)
    raise ValueError(f"Unknown method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-space", choices=("full", "window", "hard_window"), default="hard_window")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 250, 500, 1000])
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--abc-k", type=int, default=None)
    parser.add_argument("--flow-samples-per-pair", type=int, default=4)
    parser.add_argument("--posterior-samples", type=int, default=48)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--fmpe-max-iter", type=int, default=350)
    parser.add_argument("--seed", type=int, default=616)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "results" / "approach_1_3_four_method_suitability_check",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-save-sim-data",
        action="store_false",
        dest="save_sim_data",
        help="Do not write generated training/validation datasets to simulated_data/.",
    )
    parser.set_defaults(save_sim_data=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.budgets = [40, 70]
        args.validation = 40
        args.repeats = 1
        args.posterior_samples = 24
        args.ode_steps = 8
        args.fmpe_max_iter = 160

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.output_dir / "diagnostics.csv"
    if args.force:
        for path in [
            diagnostics_path,
            args.output_dir / "summary_by_budget.csv",
            args.output_dir / "four_method_suitability_summary.png",
            args.output_dir / "RESULTS.md",
        ]:
            if path.exists():
                path.unlink()
        sim_data_dir = args.output_dir / "simulated_data"
        if args.save_sim_data and sim_data_dir.exists():
            shutil.rmtree(sim_data_dir)

    existing_rows = load_existing_rows(diagnostics_path)
    completed = {completed_key(row) for row in existing_rows}
    budgets = sorted(set(args.budgets))
    max_budget = max(budgets)

    for rep in range(args.repeats):
        rep_seed = args.seed + rep * 10_000
        expected_rep_keys = {
            (method, budget, rep, rep_seed, args.design_space)
            for budget in budgets
            for method in METHODS
        }
        if expected_rep_keys.issubset(completed) and (
            not args.save_sim_data or replicate_data_exists(args.output_dir, rep)
        ):
            print(f"Skipping completed replicate={rep}")
            continue
        rng = np.random.default_rng(rep_seed)
        val = generate_random_design_dataset(
            rng,
            args.validation,
            args.n_obs,
            args.noise_std,
            args.design_space,
        )
        full_train = generate_random_design_dataset(
            rng,
            max_budget,
            args.n_obs,
            args.noise_std,
            args.design_space,
        )
        if args.save_sim_data:
            save_simulated_data(args.output_dir, rep, rep_seed, args, budgets, full_train, val)
            print(f"Wrote simulated data for replicate={rep}")

        if expected_rep_keys.issubset(completed):
            print(f"Skipping completed replicate={rep}")
            continue

        for budget in budgets:
            expected_budget_keys = {
                (method, budget, rep, rep_seed, args.design_space)
                for method in METHODS
            }
            if expected_budget_keys.issubset(completed):
                print(f"Skipping completed budget={budget} replicate={rep}")
                continue
            train = tuple(part[:budget] for part in full_train)
            for method in METHODS:
                key = (method, budget, rep, rep_seed, args.design_space)
                if key in completed:
                    print(f"Skipping completed {method} budget={budget} replicate={rep}")
                    continue
                row = evaluate_unit(method, train, val, args, rep, budget, rep_seed)
                append_live_row(diagnostics_path, row)
                completed.add(completed_key(row))
                print(f"Wrote {method} budget={budget} replicate={rep}")

    rows = load_existing_rows(diagnostics_path)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "summary_by_budget.csv", summary_rows)
    plot_summary(summary_rows, args.output_dir / "four_method_suitability_summary.png")
    write_markdown_summary(args.output_dir / "RESULTS.md", args, summary_rows)

    print(f"Wrote diagnostics to {diagnostics_path}")
    print(f"Wrote summary to {args.output_dir / 'summary_by_budget.csv'}")
    print(f"Wrote plot to {args.output_dir / 'four_method_suitability_summary.png'}")
    print(f"Wrote report to {args.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
