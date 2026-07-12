#!/usr/bin/env python3
"""Approach 1.4: probe whether categorical psi choices carry reward signal.

This diagnostic freezes one initial training set per replicate, then adds one
candidate batch at each categorical psi. It can also sweep increasingly large
candidate batches. Each augmented dataset trains a fresh posterior estimator,
and several higher-is-better reward definitions are computed as absolute scores
and as deltas from the shared initial baseline.

The goal is not to optimize yet. The goal is to ask whether the reward
landscape over psi is learnable enough for BO to have something real to model.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluate_bo_design_effect import (
    CATEGORICAL_DESIGN_LABELS,
    categorical_design_policies,
    categorical_psi_sampler,
    fixed_psi_sampler,
    make_evaluation_set,
    simulate_until_count,
    uniform_psi_sampler,
)
from posterior_diagnostic_metrics import evaluate_fmpe, evaluate_gaussian
from run_lotka_volterra import append_data, decode_design, design_dim, sample_theta, simulate_batch


PARAM_NAMES = ("alpha", "beta", "gamma", "delta")
METRIC_KEYS = (
    "raw_rmse",
    "range_normalized_rmse",
    "prior_std_normalized_rmse",
    "coverage_error",
    "posterior_mean_predictive_rmse",
    "validation_log_posterior",
    "range_normalized_rmse_alpha",
    "range_normalized_rmse_beta",
    "range_normalized_rmse_gamma",
    "range_normalized_rmse_delta",
)
REWARD_LABELS = {
    "neg_range_rmse": "-range-normalized RMSE",
    "neg_prior_std_rmse": "-prior-std-normalized RMSE",
    "neg_raw_rmse": "-raw RMSE",
    "neg_coverage_error": "-coverage error",
    "neg_predictive_rmse": "-posterior-mean predictive RMSE",
    "rmse_coverage": "-RMSE - coverage penalty",
    "rmse_predictive": "-RMSE - predictive penalty",
    "rmse_coverage_predictive": "-RMSE - coverage - predictive penalties",
    "neg_range_rmse_alpha": "-range-normalized RMSE alpha",
    "neg_range_rmse_beta": "-range-normalized RMSE beta",
    "neg_range_rmse_gamma": "-range-normalized RMSE gamma",
    "neg_range_rmse_delta": "-range-normalized RMSE delta",
    "validation_log_posterior": "validation log posterior",
    "log_posterior_coverage": "validation log posterior - coverage penalty",
}
HEATMAP_REWARDS = (
    "neg_range_rmse",
    "neg_coverage_error",
    "neg_predictive_rmse",
    "rmse_coverage",
    "rmse_predictive",
    "rmse_coverage_predictive",
)
BATCH_SWEEP_REWARDS = (
    "neg_range_rmse",
    "neg_predictive_rmse",
    "rmse_coverage_predictive",
)
POLICY_ROWS = ("short", "medium", "long")
POLICY_COLS = ("early", "middle", "late")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def slice_data(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(part[:n].copy() for part in data)  # type: ignore[return-value]


def finite_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, np.nan)
    if value in {"", None}:
        return float("nan")
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def combine_rewards(*values: float) -> float:
    if not all(np.isfinite(value) for value in values):
        return float("nan")
    return float(sum(values))


def reward_values(row: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    """Return higher-is-better rewards from one metric row."""

    range_rmse = finite_value(row, "range_normalized_rmse")
    prior_std_rmse = finite_value(row, "prior_std_normalized_rmse")
    raw_rmse = finite_value(row, "raw_rmse")
    coverage = finite_value(row, "coverage_error")
    predictive = finite_value(row, "posterior_mean_predictive_rmse")
    logp = finite_value(row, "validation_log_posterior")

    rewards = {
        "neg_range_rmse": -range_rmse,
        "neg_prior_std_rmse": -prior_std_rmse,
        "neg_raw_rmse": -raw_rmse,
        "neg_coverage_error": -coverage,
        "neg_predictive_rmse": -predictive,
        "rmse_coverage": combine_rewards(-range_rmse, -args.coverage_weight * coverage),
        "rmse_predictive": combine_rewards(-range_rmse, -args.predictive_weight * predictive),
        "rmse_coverage_predictive": combine_rewards(
            -range_rmse,
            -args.coverage_weight * coverage,
            -args.predictive_weight * predictive,
        ),
    }
    for name in PARAM_NAMES:
        key = f"range_normalized_rmse_{name}"
        rewards[f"neg_range_rmse_{name}"] = -finite_value(row, key)
    if np.isfinite(logp):
        rewards["validation_log_posterior"] = logp
        rewards["log_posterior_coverage"] = combine_rewards(logp, -args.coverage_weight * coverage)
    return rewards


def add_reward_fields(
    row: dict[str, Any],
    base_rewards: dict[str, float] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rewards = reward_values(row, args)
    for name, value in rewards.items():
        row[f"reward_abs_{name}"] = value
        if base_rewards is not None and name in base_rewards:
            base_value = base_rewards[name]
            row[f"reward_delta_{name}"] = value - base_value if np.isfinite(value) and np.isfinite(base_value) else np.nan
    return row


def score_estimator(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    replicate: int,
    seed: int,
) -> dict[str, Any]:
    budget = len(train[0])
    if args.estimator == "gaussian_npe":
        row = evaluate_gaussian(
            train,
            val,
            replicate,
            budget,
            seed,
            args.n_obs,
            args.design_space,
            predictive_timeout_seconds=args.predictive_timeout_seconds,
        )
    elif args.estimator == "rectified_fmpe":
        row = evaluate_fmpe(
            train,
            val,
            replicate,
            budget,
            seed,
            args.n_obs,
            args.design_space,
            args.flow_samples_per_pair,
            args.posterior_samples,
            args.ode_steps,
            args.fmpe_max_iter,
            predictive_timeout_seconds=args.predictive_timeout_seconds,
        )
    else:
        raise ValueError(f"Unknown estimator: {args.estimator}")
    row["estimator"] = args.estimator
    row["simulations"] = budget
    return row


def initial_psi_sampler(args: argparse.Namespace, policies: list[tuple[str, np.ndarray]]):
    if args.initial_design_mode == "categorical":
        return categorical_psi_sampler(policies)
    return uniform_psi_sampler(args.design_space)


def simulate_paired_category_batches(
    rng: np.random.Generator,
    target_n: int,
    policies: list[tuple[str, np.ndarray]],
    n_obs: int,
    noise_std: float,
    design_space: str,
    max_attempts: int = 200,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Use the same accepted theta values for every category.

    A theta is kept only if all category simulations are finite, which isolates
    psi effects from different theta draws.
    """

    theta_parts: dict[str, list[np.ndarray]] = {name: [] for name, _ in policies}
    psi_parts: dict[str, list[np.ndarray]] = {name: [] for name, _ in policies}
    x_parts: dict[str, list[np.ndarray]] = {name: [] for name, _ in policies}
    accepted = 0
    attempts = 0
    while accepted < target_n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not generate {target_n} paired category simulations after {max_attempts} attempts.")
        request_n = max(target_n - accepted, 16)
        theta_candidates = sample_theta(rng, request_n)
        for theta in theta_candidates:
            candidate_batches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            all_valid = True
            for name, psi in policies:
                try:
                    batch = simulate_batch(
                        theta[None, :],
                        psi[None, :],
                        rng,
                        n_obs,
                        noise_std,
                        design_space,
                    )
                except RuntimeError:
                    all_valid = False
                    break
                if len(batch[0]) != 1:
                    all_valid = False
                    break
                candidate_batches[name] = batch
            if not all_valid:
                continue
            for name, batch in candidate_batches.items():
                theta_parts[name].append(batch[0][0])
                psi_parts[name].append(batch[1][0])
                x_parts[name].append(batch[2][0])
            accepted += 1
            if accepted >= target_n:
                break

    return {
        name: (np.asarray(theta_parts[name]), np.asarray(psi_parts[name]), np.asarray(x_parts[name]))
        for name, _ in policies
    }


def simulate_unpaired_category_batches(
    rng: np.random.Generator,
    target_n: int,
    policies: list[tuple[str, np.ndarray]],
    n_obs: int,
    noise_std: float,
    design_space: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        name: simulate_until_count(
            rng,
            target_n,
            fixed_psi_sampler(psi),
            n_obs,
            noise_std,
            design_space,
        )
        for name, psi in policies
    }


def simulate_category_batches(
    rng: np.random.Generator,
    target_n: int,
    policies: list[tuple[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if args.batch_theta_mode == "paired":
        return simulate_paired_category_batches(
            rng,
            target_n,
            policies,
            args.n_obs,
            args.noise_std,
            args.design_space,
        )
    return simulate_unpaired_category_batches(
        rng,
        target_n,
        policies,
        args.n_obs,
        args.noise_std,
        args.design_space,
    )


def decoded_policy_fields(name: str, psi: np.ndarray, design_space: str) -> dict[str, Any]:
    design = decode_design(psi, design_space)
    row: dict[str, Any] = {
        "policy": name,
        "policy_label": CATEGORICAL_DESIGN_LABELS.get(name, name.replace("_", " ")),
        "prey0": design.prey0,
        "pred0": design.pred0,
        "t_start": design.t_start,
        "t_span": design.t_span,
        "t_end": design.t_end,
    }
    for index, value in enumerate(psi):
        row[f"psi{index}_unit"] = float(value)
    return row


def mean_std_sem(values: list[float]) -> tuple[float, float, float]:
    vals = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0, float("nan")
    std = float(np.std(vals, ddof=1))
    return float(np.mean(vals)), std, float(std / np.sqrt(len(vals)))


def summarize_by_policy(rows: list[dict[str, Any]], reward_names: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["batch_size"]), str(row["policy"]))].append(row)

    out_rows: list[dict[str, Any]] = []
    for (batch_size, policy), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[1][0]["policy_label"]))):
        first = group[0]
        out: dict[str, Any] = {
            "batch": batch_size,
            "batch_size": batch_size,
            "final_simulations": int(first["initial"]) + batch_size,
            "policy": policy,
            "policy_label": first["policy_label"],
            "replicates": len({int(row["replicate"]) for row in group}),
            "n": len(group),
            "prey0": first["prey0"],
            "pred0": first["pred0"],
            "t_start": first["t_start"],
            "t_span": first["t_span"],
            "t_end": first["t_end"],
        }
        for index in range(design_dim(str(first["design_space"]))):
            out[f"psi{index}_unit"] = first.get(f"psi{index}_unit", "")
        for key in METRIC_KEYS:
            values = [finite_value(row, key) for row in group]
            mean, std, sem = mean_std_sem(values)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_sem"] = sem
        for reward_name in reward_names:
            for prefix in ("reward_abs", "reward_delta"):
                key = f"{prefix}_{reward_name}"
                values = [finite_value(row, key) for row in group]
                mean, std, sem = mean_std_sem(values)
                out[f"{key}_mean"] = mean
                out[f"{key}_std"] = std
                out[f"{key}_sem"] = sem
                if prefix == "reward_delta":
                    out[f"{key}_snr"] = abs(mean) / std if np.isfinite(mean) and np.isfinite(std) and std > 0.0 else np.nan
        out_rows.append(out)
    return out_rows


def eta_squared(rows: list[dict[str, Any]], key: str) -> float:
    finite_rows = [row for row in rows if np.isfinite(finite_value(row, key))]
    if len(finite_rows) < 2:
        return float("nan")
    values = np.asarray([finite_value(row, key) for row in finite_rows])
    overall = float(np.mean(values))
    ss_total = float(np.sum((values - overall) ** 2))
    if ss_total <= 0.0:
        return float("nan")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in finite_rows:
        groups[str(row["policy"])].append(finite_value(row, key))
    ss_between = 0.0
    for vals in groups.values():
        arr = np.asarray(vals)
        ss_between += len(arr) * (float(np.mean(arr)) - overall) ** 2
    return float(ss_between / ss_total)


def paired_eta_squared(rows: list[dict[str, Any]], key: str) -> float:
    finite_rows = [row for row in rows if np.isfinite(finite_value(row, key))]
    if len(finite_rows) < 2:
        return float("nan")
    rep_means: dict[int, float] = {}
    by_rep: dict[int, list[float]] = defaultdict(list)
    for row in finite_rows:
        by_rep[int(row["replicate"])].append(finite_value(row, key))
    for replicate, values in by_rep.items():
        rep_means[replicate] = float(np.mean(values))

    residual_rows = []
    for row in finite_rows:
        residual_rows.append((str(row["policy"]), finite_value(row, key) - rep_means[int(row["replicate"])]))
    residuals = np.asarray([value for _, value in residual_rows])
    ss_total = float(np.sum(residuals ** 2))
    if ss_total <= 0.0:
        return float("nan")
    groups: dict[str, list[float]] = defaultdict(list)
    for policy, residual in residual_rows:
        groups[policy].append(float(residual))
    ss_policy = 0.0
    for vals in groups.values():
        arr = np.asarray(vals)
        ss_policy += len(arr) * float(np.mean(arr)) ** 2
    return float(ss_policy / ss_total)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def rank_stability(rows: list[dict[str, Any]], key: str, policies: list[str]) -> float:
    by_rep: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = finite_value(row, key)
        if np.isfinite(value):
            by_rep[int(row["replicate"])][str(row["policy"])] = value
    rank_vectors = []
    for values_by_policy in by_rep.values():
        if not all(policy in values_by_policy for policy in policies):
            continue
        # Higher reward is better, so rank negative values for descending order.
        values = np.asarray([-values_by_policy[policy] for policy in policies])
        rank_vectors.append(rankdata(values))
    if len(rank_vectors) < 2:
        return float("nan")
    cors = []
    for left, right in combinations(rank_vectors, 2):
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            continue
        cors.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(cors)) if cors else float("nan")


def winner_stability(rows: list[dict[str, Any]], key: str, best_policy: str) -> float:
    by_rep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if np.isfinite(finite_value(row, key)):
            by_rep[int(row["replicate"])].append(row)
    winners = []
    for group in by_rep.values():
        if not group:
            continue
        winners.append(str(max(group, key=lambda row: finite_value(row, key))["policy"]))
    if not winners:
        return float("nan")
    return float(np.mean([winner == best_policy for winner in winners]))


def summarize_linkage(rows: list[dict[str, Any]], reward_names: list[str], policies: list[str]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    labels = {str(row["policy"]): str(row["policy_label"]) for row in rows}
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    for batch_size in batch_sizes:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch_size]
        for reward_name in reward_names:
            key = f"reward_delta_{reward_name}"
            finite_rows = [row for row in batch_rows if np.isfinite(finite_value(row, key))]
            if not finite_rows:
                continue
            groups: dict[str, list[float]] = defaultdict(list)
            for row in finite_rows:
                groups[str(row["policy"])].append(finite_value(row, key))
            policy_means = {policy: float(np.mean(vals)) for policy, vals in groups.items() if vals}
            if not policy_means:
                continue
            best_policy = max(policy_means, key=policy_means.get)
            worst_policy = min(policy_means, key=policy_means.get)
            within_stds = []
            for vals in groups.values():
                if len(vals) > 1:
                    within_stds.append(float(np.std(vals, ddof=1)))
            initial = int(finite_rows[0]["initial"])
            values = [finite_value(row, key) for row in finite_rows]
            out_rows.append(
                {
                    "batch": batch_size,
                    "batch_size": batch_size,
                    "final_simulations": initial + batch_size,
                    "reward": reward_name,
                    "reward_label": REWARD_LABELS.get(reward_name, reward_name),
                    "n": len(finite_rows),
                    "replicates": len({int(row["replicate"]) for row in finite_rows}),
                    "policies": len(groups),
                    "raw_eta_squared": eta_squared(finite_rows, key),
                    "paired_eta_squared": paired_eta_squared(finite_rows, key),
                    "rank_stability_spearman": rank_stability(finite_rows, key, policies),
                    "winner_stability": winner_stability(finite_rows, key, best_policy),
                    "best_policy": best_policy,
                    "best_label": labels.get(best_policy, best_policy),
                    "best_mean_delta": policy_means[best_policy],
                    "worst_policy": worst_policy,
                    "worst_label": labels.get(worst_policy, worst_policy),
                    "worst_mean_delta": policy_means[worst_policy],
                    "mean_policy_delta": float(np.mean(list(policy_means.values()))),
                    "range_of_policy_means": policy_means[best_policy] - policy_means[worst_policy],
                    "positive_policy_fraction": float(np.mean([value > 0.0 for value in policy_means.values()])),
                    "positive_observation_fraction": float(np.mean([value > 0.0 for value in values])),
                    "between_policy_std": float(np.std(list(policy_means.values()), ddof=1)) if len(policy_means) > 1 else 0.0,
                    "mean_within_policy_std": float(np.mean(within_stds)) if within_stds else np.nan,
                }
            )
    return sorted(out_rows, key=lambda row: (int(row["batch_size"]), str(row["reward"])))


def policy_grid_value(summary_rows: list[dict[str, Any]], key: str) -> np.ndarray:
    grid = np.full((len(POLICY_ROWS), len(POLICY_COLS)), np.nan)
    for row in summary_rows:
        policy = str(row["policy"])
        parts = policy.split("_")
        if len(parts) != 2:
            continue
        row_name, col_name = parts
        if row_name not in POLICY_ROWS or col_name not in POLICY_COLS:
            continue
        value = finite_value(row, key)
        grid[POLICY_ROWS.index(row_name), POLICY_COLS.index(col_name)] = value
    return grid


def plot_reward_heatmaps(
    summary_rows: list[dict[str, Any]],
    reward_names: tuple[str, ...],
    output: Path,
    statistic: str,
    title: str,
    center_zero: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, reward_name in zip(axes.ravel(), reward_names):
        key = f"reward_delta_{reward_name}_{statistic}"
        grid = policy_grid_value(summary_rows, key)
        finite = grid[np.isfinite(grid)]
        if len(finite) == 0:
            ax.axis("off")
            continue
        if center_zero:
            vmax = max(float(np.max(np.abs(finite))), 1e-9)
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmin == vmax:
                vmin -= 1e-9
                vmax += 1e-9
            cmap = "viridis"
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=8)
        ax.set_title(REWARD_LABELS.get(reward_name, reward_name), fontsize=10)
        ax.set_xticks(range(len(POLICY_COLS)))
        ax.set_xticklabels(POLICY_COLS)
        ax.set_yticks(range(len(POLICY_ROWS)))
        ax.set_yticklabels(POLICY_ROWS)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_linkage_summary(linkage_rows: list[dict[str, Any]], output: Path) -> None:
    if not linkage_rows:
        return
    batch_sizes = sorted({int(row["batch_size"]) for row in linkage_rows})
    if len(batch_sizes) > 1:
        rewards = [reward for reward in BATCH_SWEEP_REWARDS if any(row["reward"] == reward for row in linkage_rows)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        panels = [
            ("paired_eta_squared", "Paired eta squared"),
            ("rank_stability_spearman", "Rank stability"),
            ("best_mean_delta", "Best mean delta"),
            ("positive_policy_fraction", "Fraction of positive category means"),
        ]
        for ax, (key, title) in zip(axes.ravel(), panels):
            for reward in rewards:
                subset = [row for row in linkage_rows if row["reward"] == reward]
                xs = np.asarray([int(row["batch_size"]) for row in subset], dtype=float)
                ys = np.asarray([finite_value(row, key) for row in subset], dtype=float)
                order = np.argsort(xs)
                ax.plot(xs[order], ys[order], marker="o", linewidth=2, label=reward)
            ax.set_xscale("log", base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(value) for value in batch_sizes])
            ax.set_xlabel("Added simulations per category")
            ax.set_title(title)
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Reward-Batch Sweep Diagnostics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return

    rows = sorted(linkage_rows, key=lambda row: finite_value(row, "paired_eta_squared"), reverse=True)
    labels = [str(row["reward"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, title in [
        (axes[0], "paired_eta_squared", "Paired eta squared"),
        (axes[1], "range_of_policy_means", "Range of policy means"),
        (axes[2], "rank_stability_spearman", "Rank stability"),
    ]:
        values = [finite_value(row, key) for row in rows]
        ax.bar(x, values, color="#4C78A8", alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Reward-Psi Linkage Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table_linkage(linkage_rows: list[dict[str, Any]]) -> list[str]:
    if not linkage_rows:
        return ["No finite reward-linkage rows were produced."]
    batch_sizes = sorted({int(row["batch_size"]) for row in linkage_rows})
    rows = sorted(
        linkage_rows,
        key=lambda row: (int(row["batch_size"]), -finite_value(row, "paired_eta_squared")),
    )
    lines = [
        "| Batch | Reward | Paired eta^2 | Rank stability | Best category | Best mean delta | Positive categories | Policy range |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    if len(batch_sizes) == 1:
        rows = sorted(linkage_rows, key=lambda row: finite_value(row, "paired_eta_squared"), reverse=True)
    for row in rows:
        lines.append(
            "| "
            + str(int(row["batch_size"]))
            + " | "
            + str(row["reward"])
            + " | "
            + f"{finite_value(row, 'paired_eta_squared'):.3f} | "
            + f"{finite_value(row, 'rank_stability_spearman'):.3f} | "
            + str(row["best_label"])
            + " | "
            + f"{finite_value(row, 'best_mean_delta'):.4f} | "
            + f"{finite_value(row, 'positive_policy_fraction'):.2f} | "
            + f"{finite_value(row, 'range_of_policy_means'):.4f} |"
        )
    return lines


def parse_batch_sizes(args: argparse.Namespace) -> list[int]:
    values = args.batch_sizes if args.batch_sizes is not None else [args.batch]
    batch_sizes = sorted({int(value) for value in values})
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("Batch sizes must be positive integers.")
    return batch_sizes


def write_report(
    path: Path,
    args: argparse.Namespace,
    reward_names: list[str],
    linkage_rows: list[dict[str, Any]],
) -> None:
    batch_sizes = parse_batch_sizes(args)
    config_lines = [
        f"design_space = {args.design_space}",
        f"initial = {args.initial}",
        f"batch = {args.batch}",
        f"batch_sizes = {' '.join(str(value) for value in batch_sizes)}",
        f"validation = {args.validation}",
        f"repeats = {args.repeats}",
        f"initial_design_mode = {args.initial_design_mode}",
        f"batch_theta_mode = {args.batch_theta_mode}",
        f"target_mode = {args.target_mode}",
        f"estimator = {args.estimator}",
        f"flow_samples_per_pair = {args.flow_samples_per_pair}",
        f"posterior_samples = {args.posterior_samples}",
        f"ode_steps = {args.ode_steps}",
        f"fmpe_max_iter = {args.fmpe_max_iter}",
        f"coverage_weight = {args.coverage_weight}",
        f"predictive_weight = {args.predictive_weight}",
        f"predictive_timeout_seconds = {args.predictive_timeout_seconds}",
        f"seed = {args.seed}",
    ]
    if args.bo_category_policies:
        config_lines.append(f"category_policies = {' '.join(args.bo_category_policies)}")
    else:
        config_lines.append("category_policies = all")

    if linkage_rows:
        best = max(linkage_rows, key=lambda row: finite_value(row, "paired_eta_squared"))
        best_delta = max(linkage_rows, key=lambda row: finite_value(row, "best_mean_delta"))
        best_eta = finite_value(best, "paired_eta_squared")
        max_positive_fraction = max(finite_value(row, "positive_policy_fraction") for row in linkage_rows)
        if args.repeats < 2:
            interpretation = "This run has fewer than two repeats, so variance and rank-stability diagnostics are only a smoke check."
        elif max_positive_fraction <= 0.0:
            interpretation = "No tested batch size produced a positive mean category delta. The reward landscape may still rank designs, but the feedback remains mostly least-harmful rather than improving."
        elif finite_value(best_delta, "best_mean_delta") > 0.0 and best_eta >= 0.2:
            interpretation = "At least one larger batch produces positive reward deltas with detectable category-level structure. This is the first condition BO needs before surrogate tuning is meaningful."
        elif best_eta >= 0.5:
            interpretation = "At least one reward shows strong category-level signal after controlling for replicate difficulty."
        elif best_eta >= 0.2:
            interpretation = "At least one reward shows moderate category-level signal, but stability should be checked before trusting BO."
        else:
            interpretation = "The tested rewards show weak category-level signal; BO may not have a stable landscape to learn."
    else:
        interpretation = "No finite linkage diagnostics were produced."

    lines = [
        "# Approach 1.4 Reward Landscape Results",
        "",
        "This diagnostic tests whether categorical observation-window choices `psi` are visibly coupled to posterior-quality rewards before asking BO to optimize them.",
        "",
        "For each replicate, the same initial training set is scored once. Then one batch is added at each categorical `psi`, one fresh estimator is trained per category, and each reward is computed as both an absolute value and a delta from the shared initial baseline.",
        "",
        "When multiple batch sizes are configured, larger batches are nested prefixes of the same generated category batch. This makes the sweep ask how the same category intervention strengthens as the added simulation budget grows.",
        "",
        "All rewards in this report are higher-is-better. Delta rewards are the clearest BO signal because BO receives feedback from improvement after choosing the next batch.",
        "",
        "## Run Configuration",
        "",
        "```text",
        *config_lines,
        "```",
        "",
        "## Reward Linkage",
        "",
        *markdown_table_linkage(linkage_rows),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "Use `reward_delta_landscape.png` to see the denoised 3x3 category grid at the largest configured batch, and `reward_snr_landscape.png` to see where that mean delta is large relative to repeat noise. Use `reward_linkage_summary.png` to see how linkage and positivity change across batch sizes.",
        "",
        "## Reward Definitions",
        "",
        "| Reward | Meaning |",
        "| --- | --- |",
    ]
    for reward_name in reward_names:
        lines.append(f"| `{reward_name}` | {REWARD_LABELS.get(reward_name, reward_name)} |")
    lines += [
        "",
        "## Output Files",
        "",
        "```text",
        "base_metrics.csv",
        "reward_landscape.csv",
        "reward_policy_summary.csv",
        "reward_linkage_summary.csv",
        "reward_delta_landscape.png",
        "reward_snr_landscape.png",
        "reward_linkage_summary.png",
        "RESULTS.md",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-space", choices=("full", "window", "hard_window", "wide_window"), default="wide_window")
    parser.add_argument("--initial", type=int, default=80)
    parser.add_argument("--batch", type=int, default=14)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional added-simulation batch sizes to sweep. Defaults to --batch.",
    )
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument(
        "--initial-design-mode",
        choices=("categorical", "continuous"),
        default="categorical",
        help="How to sample the shared initial training set before probing category batches.",
    )
    parser.add_argument(
        "--batch-theta-mode",
        choices=("paired", "unpaired"),
        default="paired",
        help="Use the same theta batch for all categories, or sample independent batches per category.",
    )
    parser.add_argument(
        "--category-policies",
        dest="bo_category_policies",
        nargs="+",
        default=None,
        choices=tuple(CATEGORICAL_DESIGN_LABELS),
        help="Optional subset of categorical policies to probe.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("validation_set", "fixed_x0"),
        default="validation_set",
        help="Evaluate rewards on a broad validation set or on one fixed synthetic observation x0.",
    )
    parser.add_argument(
        "--target-theta",
        type=float,
        nargs="+",
        default=None,
        help="Theta for fixed_x0 mode: alpha beta gamma delta. Defaults to the prior midpoint.",
    )
    parser.add_argument(
        "--target-psi",
        type=float,
        nargs="+",
        default=None,
        help="Unit-cube psi for fixed_x0 mode. Defaults to the center of the design space.",
    )
    parser.add_argument("--target-seed", type=int, default=919)
    parser.add_argument(
        "--estimator",
        choices=("gaussian_npe", "rectified_fmpe"),
        default="rectified_fmpe",
    )
    parser.add_argument("--flow-samples-per-pair", type=int, default=3)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--fmpe-max-iter", type=int, default=220)
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--predictive-weight", type=float, default=0.1)
    parser.add_argument("--predictive-timeout-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=717)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "results" / "lotka_volterra" / "approach_1_4_reward_landscape_categorical_psi",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.initial = 32
        args.batch = 4
        args.batch_sizes = [4, 8]
        args.validation = 24
        args.repeats = 1
        args.flow_samples_per_pair = min(args.flow_samples_per_pair, 2)
        args.posterior_samples = min(args.posterior_samples, 12)
        args.ode_steps = min(args.ode_steps, 5)
        args.fmpe_max_iter = min(args.fmpe_max_iter, 70)
        args.predictive_timeout_seconds = min(args.predictive_timeout_seconds, 0.25)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = categorical_design_policies(args)
    policy_names = [name for name, _ in policies]
    batch_sizes = parse_batch_sizes(args)
    max_batch_size = max(batch_sizes)
    if design_dim(args.design_space) != 2:
        raise ValueError("The reward-landscape heatmaps currently expect a two-dimensional psi design space.")

    base_rows: list[dict[str, Any]] = []
    landscape_rows: list[dict[str, Any]] = []

    fixed_target_val = None
    if args.target_mode == "fixed_x0":
        fixed_target_val = make_evaluation_set(args, np.random.default_rng(args.target_seed))

    for replicate in range(args.repeats):
        rep_seed = args.seed + 10_000 * replicate
        setup_rng = np.random.default_rng(rep_seed)
        val = fixed_target_val if fixed_target_val is not None else make_evaluation_set(args, setup_rng)
        initial_train = simulate_until_count(
            setup_rng,
            args.initial,
            initial_psi_sampler(args, policies),
            args.n_obs,
            args.noise_std,
            args.design_space,
        )
        base_row = score_estimator(
            initial_train,
            val,
            args,
            replicate,
            rep_seed + 3000,
        )
        base_row.update(
            {
                "replicate": replicate,
                "seed": rep_seed,
                "role": "base_initial",
                "design_space": args.design_space,
                "initial": args.initial,
                "batch": args.batch,
                "batch_sizes": ",".join(str(value) for value in batch_sizes),
                "validation": args.validation,
                "initial_design_mode": args.initial_design_mode,
                "batch_theta_mode": args.batch_theta_mode,
                "target_mode": args.target_mode,
            }
        )
        add_reward_fields(base_row, None, args)
        base_rewards = reward_values(base_row, args)
        base_rows.append(base_row)

        batch_rng = np.random.default_rng(rep_seed + 4000)
        category_batches = simulate_category_batches(batch_rng, max_batch_size, policies, args)
        for policy_index, (policy, psi) in enumerate(policies):
            for batch_index, batch_size in enumerate(batch_sizes):
                extra = slice_data(category_batches[policy], batch_size)
                train = append_data(initial_train, extra)
                row = score_estimator(
                    train,
                    val,
                    args,
                    replicate,
                    rep_seed + 5000 + policy_index * 100 + batch_index,
                )
                row.update(decoded_policy_fields(policy, psi, args.design_space))
                row.update(
                    {
                        "replicate": replicate,
                        "seed": rep_seed,
                        "role": "one_batch_probe",
                        "design_space": args.design_space,
                        "initial": args.initial,
                        "batch": batch_size,
                        "batch_size": batch_size,
                        "batch_sizes": ",".join(str(value) for value in batch_sizes),
                        "final_simulations": args.initial + batch_size,
                        "validation": args.validation,
                        "initial_design_mode": args.initial_design_mode,
                        "batch_theta_mode": args.batch_theta_mode,
                        "target_mode": args.target_mode,
                    }
                )
                add_reward_fields(row, base_rewards, args)
                landscape_rows.append(row)

    reward_names = sorted(
        {
            key.removeprefix("reward_abs_")
            for row in [*base_rows, *landscape_rows]
            for key in row
            if key.startswith("reward_abs_")
        }
    )
    policy_summary_rows = summarize_by_policy(landscape_rows, reward_names)
    linkage_rows = summarize_linkage(landscape_rows, reward_names, policy_names)
    max_batch_policy_summary_rows = [
        row for row in policy_summary_rows if int(row["batch_size"]) == max_batch_size
    ]

    write_csv(args.output_dir / "base_metrics.csv", base_rows)
    write_csv(args.output_dir / "reward_landscape.csv", landscape_rows)
    write_csv(args.output_dir / "reward_policy_summary.csv", policy_summary_rows)
    write_csv(args.output_dir / "reward_linkage_summary.csv", linkage_rows)
    plot_reward_heatmaps(
        max_batch_policy_summary_rows,
        HEATMAP_REWARDS,
        args.output_dir / "reward_delta_landscape.png",
        "mean",
        f"Mean Delta Reward by Psi Category (+{max_batch_size} simulations)",
        center_zero=True,
    )
    plot_reward_heatmaps(
        max_batch_policy_summary_rows,
        HEATMAP_REWARDS,
        args.output_dir / "reward_snr_landscape.png",
        "snr",
        f"Delta Reward Signal-to-Noise by Psi Category (+{max_batch_size} simulations)",
        center_zero=False,
    )
    plot_linkage_summary(linkage_rows, args.output_dir / "reward_linkage_summary.png")
    write_report(args.output_dir / "RESULTS.md", args, reward_names, linkage_rows)

    print(f"Wrote base metrics to {args.output_dir / 'base_metrics.csv'}")
    print(f"Wrote reward landscape to {args.output_dir / 'reward_landscape.csv'}")
    print(f"Wrote policy summary to {args.output_dir / 'reward_policy_summary.csv'}")
    print(f"Wrote linkage summary to {args.output_dir / 'reward_linkage_summary.csv'}")
    print(f"Wrote report to {args.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
