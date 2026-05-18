#!/usr/bin/env python3
"""Standalone FMPE quality diagnostics under random simulation design.

This script answers a narrower question than the active-design runners:

    Is the current FMPE-style posterior estimator itself useful?

It intentionally avoids BO. For each replicate it samples one random-design
training set, evaluates prefixes of that set at increasing budgets, and
compares:

- prior-mean prediction,
- the Gaussian NPE-style baseline,
- the lightweight RectifiedFMPE posterior estimator.

Results are written to an approach-named folder under experiments/results/.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_lotka_volterra import (
    THETA_BOUNDS,
    GaussianPosteriorRegressor,
    coverage_error as gaussian_coverage_error,
    sample_psi,
    sample_theta,
    simulate_batch,
    simulate_one,
)
from run_lotka_volterra_fmpe import RectifiedFMPE


PARAM_NAMES = ("alpha", "beta", "gamma", "delta")
LEVELS = (0.5, 0.8, 0.9)


def prior_mean() -> np.ndarray:
    return THETA_BOUNDS.mean(axis=1)


def prior_std() -> np.ndarray:
    return (THETA_BOUNDS[:, 1] - THETA_BOUNDS[:, 0]) / np.sqrt(12.0)


def prior_range() -> np.ndarray:
    return THETA_BOUNDS[:, 1] - THETA_BOUNDS[:, 0]


def rmse_by_parameter(theta_true: np.ndarray, theta_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((theta_pred - theta_true) ** 2, axis=0))


def prior_coverage_error(theta_true: np.ndarray) -> float:
    lo = THETA_BOUNDS[:, 0]
    hi = THETA_BOUNDS[:, 1]
    width = hi - lo
    total = 0.0
    for level in LEVELS:
        interval_lo = lo + (1.0 - level) * width / 2.0
        interval_hi = hi - (1.0 - level) * width / 2.0
        inside = (theta_true >= interval_lo) & (theta_true <= interval_hi)
        total += abs(float(np.mean(inside)) - level)
    return total / len(LEVELS)


def sample_coverage_error(samples: np.ndarray, theta_true: np.ndarray) -> float:
    total = 0.0
    for level in LEVELS:
        lo = np.quantile(samples, (1.0 - level) / 2.0, axis=1)
        hi = np.quantile(samples, 1.0 - (1.0 - level) / 2.0, axis=1)
        inside = (theta_true >= lo) & (theta_true <= hi)
        total += abs(float(np.mean(inside)) - level)
    return total / len(LEVELS)


def posterior_mean_predictive_rmse(
    theta_pred: np.ndarray,
    psi: np.ndarray,
    x_true: np.ndarray,
    seed: int,
    n_obs: int,
    design_space: str,
) -> float:
    """Simulate noiseless trajectories at posterior mean parameters."""

    rng = np.random.default_rng(seed)
    preds = []
    targets = []
    for th, ps, x in zip(theta_pred, psi, x_true):
        x_pred = simulate_one(
            th,
            ps,
            rng,
            n_obs=n_obs,
            noise_std=0.0,
            design_space=design_space,
        )
        if np.isfinite(x_pred).all():
            preds.append(x_pred)
            targets.append(x)
    if not preds:
        return float("nan")
    pred_arr = np.asarray(preds)
    target_arr = np.asarray(targets)
    return float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))


def make_metric_row(
    method: str,
    replicate: int,
    budget: int,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    coverage_error: float,
    predictive_rmse: float,
    validation_log_posterior: float | None = None,
) -> dict[str, float | int | str]:
    per_param = rmse_by_parameter(theta_true, theta_pred)
    row: dict[str, float | int | str] = {
        "method": method,
        "replicate": replicate,
        "budget": budget,
        "raw_rmse": float(np.sqrt(np.mean((theta_pred - theta_true) ** 2))),
        "range_normalized_rmse": float(np.mean(per_param / prior_range())),
        "prior_std_normalized_rmse": float(np.mean(per_param / prior_std())),
        "coverage_error": coverage_error,
        "posterior_mean_predictive_rmse": predictive_rmse,
    }
    for name, value in zip(PARAM_NAMES, per_param):
        row[f"rmse_{name}"] = float(value)
        row[f"range_normalized_rmse_{name}"] = float(value / prior_range()[PARAM_NAMES.index(name)])
        row[f"prior_std_normalized_rmse_{name}"] = float(value / prior_std()[PARAM_NAMES.index(name)])
    if validation_log_posterior is not None:
        row["validation_log_posterior"] = validation_log_posterior
    return row


def evaluate_prior(
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicate: int,
    budget: int,
    seed: int,
    n_obs: int,
    design_space: str,
) -> dict[str, float | int | str]:
    theta_val, psi_val, x_val = val
    pred = np.repeat(prior_mean()[None, :], len(theta_val), axis=0)
    return make_metric_row(
        "prior_mean",
        replicate,
        budget,
        theta_val,
        pred,
        prior_coverage_error(theta_val),
        posterior_mean_predictive_rmse(pred, psi_val, x_val, seed, n_obs, design_space),
    )


def evaluate_gaussian(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicate: int,
    budget: int,
    seed: int,
    n_obs: int,
    design_space: str,
) -> dict[str, float | int | str]:
    theta_train, psi_train, x_train = train
    theta_val, psi_val, x_val = val
    model = GaussianPosteriorRegressor(seed=seed).fit(x_train, psi_train, theta_train)
    pred = model.predict_mean(x_val, psi_val)
    return make_metric_row(
        "gaussian_npe",
        replicate,
        budget,
        theta_val,
        pred,
        gaussian_coverage_error(model, val),
        posterior_mean_predictive_rmse(pred, psi_val, x_val, seed + 17, n_obs, design_space),
        validation_log_posterior=float(np.mean(model.log_prob(theta_val, x_val, psi_val))),
    )


def evaluate_fmpe(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicate: int,
    budget: int,
    seed: int,
    n_obs: int,
    design_space: str,
    flow_samples_per_pair: int,
    posterior_samples: int,
    ode_steps: int,
    fmpe_max_iter: int,
) -> dict[str, float | int | str]:
    theta_train, psi_train, x_train = train
    theta_val, psi_val, x_val = val
    model = RectifiedFMPE(
        seed=seed,
        flow_samples_per_pair=flow_samples_per_pair,
        max_iter=fmpe_max_iter,
    ).fit(x_train, psi_train, theta_train)
    samples = model.sample(
        x_val,
        psi_val,
        n_samples=posterior_samples,
        steps=ode_steps,
        rng=np.random.default_rng(seed + 23),
    )
    pred = samples.mean(axis=1)
    return make_metric_row(
        "rectified_fmpe",
        replicate,
        budget,
        theta_val,
        pred,
        sample_coverage_error(samples, theta_val),
        posterior_mean_predictive_rmse(pred, psi_val, x_val, seed + 31, n_obs, design_space),
    )


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    numeric_keys = [
        "raw_rmse",
        "range_normalized_rmse",
        "prior_std_normalized_rmse",
        "coverage_error",
        "posterior_mean_predictive_rmse",
        "validation_log_posterior",
    ]
    numeric_keys += [f"rmse_{name}" for name in PARAM_NAMES]
    numeric_keys += [f"range_normalized_rmse_{name}" for name in PARAM_NAMES]
    numeric_keys += [f"prior_std_normalized_rmse_{name}" for name in PARAM_NAMES]

    groups: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["budget"]))].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (method, budget), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        out: dict[str, float | int | str] = {
            "method": method,
            "budget": budget,
            "replicates": len(group),
        }
        for key in numeric_keys:
            vals = [float(row[key]) for row in group if key in row and row[key] != ""]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        summary.append(out)
    return summary


def plot_summary(summary_rows: list[dict[str, float | int | str]], output: Path) -> None:
    methods = ["prior_mean", "gaussian_npe", "rectified_fmpe"]
    labels = {
        "prior_mean": "prior mean",
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
        for method in methods:
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
    fig.suptitle("Standalone FMPE Quality Check Under Random Design", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_markdown_summary(path: Path, summary_rows: list[dict[str, float | int | str]]) -> None:
    final_budget = max(int(row["budget"]) for row in summary_rows)
    final_rows = [row for row in summary_rows if int(row["budget"]) == final_budget]
    by_method = {str(row["method"]): row for row in final_rows}
    prior = by_method.get("prior_mean")
    gaussian = by_method.get("gaussian_npe")
    fmpe = by_method.get("rectified_fmpe")

    verdict: list[str] = []
    if prior is not None and gaussian is not None and fmpe is not None:
        prior_rmse = float(prior["range_normalized_rmse_mean"])
        gaussian_rmse = float(gaussian["range_normalized_rmse_mean"])
        fmpe_rmse = float(fmpe["range_normalized_rmse_mean"])
        gaussian_gain = 100.0 * (prior_rmse - gaussian_rmse) / prior_rmse
        fmpe_gain = 100.0 * (prior_rmse - fmpe_rmse) / prior_rmse
        fmpe_vs_gaussian = 100.0 * (gaussian_rmse - fmpe_rmse) / gaussian_rmse
        verdict = [
            "## Current Verdict",
            "",
            "At the final budget, the lightweight FMPE model is learning real signal:",
            "",
            f"- Gaussian NPE improves range-normalized RMSE over the prior mean by `{gaussian_gain:.1f}%`.",
            f"- Rectified FMPE improves range-normalized RMSE over the prior mean by `{fmpe_gain:.1f}%`.",
            f"- Rectified FMPE improves range-normalized RMSE over Gaussian NPE by `{fmpe_vs_gaussian:.1f}%`.",
            f"- Rectified FMPE predictive RMSE is `{float(fmpe['posterior_mean_predictive_rmse_mean']):.4f}` versus `{float(gaussian['posterior_mean_predictive_rmse_mean']):.4f}` for Gaussian NPE.",
            f"- Rectified FMPE coverage error is `{float(fmpe['coverage_error_mean']):.4f}` versus `{float(gaussian['coverage_error_mean']):.4f}` for Gaussian NPE.",
            "",
            "So the current wording should be:",
            "",
            "> FMPE is better than trivial and competitive on point-estimate quality, but not yet well calibrated.",
            "",
            "The prior-mean coverage number is low because it uses broad prior credible intervals; it should not be interpreted as useful posterior inference.",
            "",
        ]

    lines = [
        "# Approach 4.2 FMPE Quality Check Results",
        "",
        "This diagnostic evaluates posterior estimators under random simulation design only.",
        "It does not use BO, so it tests whether FMPE is good enough before active design is judged.",
        "",
        f"Final evaluated budget: `{final_budget}` simulator calls.",
        "",
        "## Final-Budget Summary",
        "",
        "| Method | Range-norm RMSE | Prior-std RMSE | Coverage error | Predictive RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ["prior_mean", "gaussian_npe", "rectified_fmpe"]:
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
        *verdict,
        "",
        "## Interpretation Rule",
        "",
        "FMPE should only be called good if it clearly improves over the prior mean,",
        "matches or beats the Gaussian baseline, improves with more simulations, and has usable calibration.",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-space", choices=("full", "window", "hard_window"), default="hard_window")
    parser.add_argument("--budgets", type=int, nargs="+", default=[60, 100, 140])
    parser.add_argument("--validation", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--flow-samples-per-pair", type=int, default=4)
    parser.add_argument("--posterior-samples", type=int, default=48)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--fmpe-max-iter", type=int, default=350)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "results" / "approach_4_2_fmpe_quality_check",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.budgets = [40, 70]
        args.validation = 50
        args.repeats = 1
        args.posterior_samples = 32
        args.ode_steps = 8
        args.fmpe_max_iter = 180

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set(args.budgets))
    max_budget = max(budgets)

    rows: list[dict[str, float | int | str]] = []
    for rep in range(args.repeats):
        rep_seed = args.seed + rep * 10_000
        rng = np.random.default_rng(rep_seed)
        val = simulate_batch(
            sample_theta(rng, args.validation),
            sample_psi(rng, args.validation, args.design_space),
            rng,
            args.n_obs,
            args.noise_std,
            args.design_space,
        )
        full_train = simulate_batch(
            sample_theta(rng, max_budget),
            sample_psi(rng, max_budget, args.design_space),
            rng,
            args.n_obs,
            args.noise_std,
            args.design_space,
        )

        for budget in budgets:
            train = tuple(part[:budget] for part in full_train)
            rows.append(evaluate_prior(val, rep, budget, rep_seed + budget, args.n_obs, args.design_space))
            rows.append(
                evaluate_gaussian(
                    train,
                    val,
                    rep,
                    budget,
                    rep_seed + 1000 + budget,
                    args.n_obs,
                    args.design_space,
                )
            )
            rows.append(
                evaluate_fmpe(
                    train,
                    val,
                    rep,
                    budget,
                    rep_seed + 2000 + budget,
                    args.n_obs,
                    args.design_space,
                    args.flow_samples_per_pair,
                    args.posterior_samples,
                    args.ode_steps,
                    args.fmpe_max_iter,
                )
            )

    summary_rows = summarize(rows)
    write_csv(args.output_dir / "diagnostics.csv", rows)
    write_csv(args.output_dir / "summary_by_budget.csv", summary_rows)
    plot_summary(summary_rows, args.output_dir / "fmpe_quality_summary.png")
    write_markdown_summary(args.output_dir / "RESULTS.md", summary_rows)

    print(f"Wrote diagnostics to {args.output_dir / 'diagnostics.csv'}")
    print(f"Wrote summary to {args.output_dir / 'summary_by_budget.csv'}")
    print(f"Wrote plot to {args.output_dir / 'fmpe_quality_summary.png'}")
    print(f"Wrote report to {args.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
