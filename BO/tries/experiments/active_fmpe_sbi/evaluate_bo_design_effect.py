#!/usr/bin/env python3
"""Approach 1.3: test whether BO design choices improve posterior estimation.

This diagnostic compares four design strategies under the same simulation
budget:

- random: every new batch uses uniformly random design variables psi.
- fixed_dumb: every new batch uses one fixed intentionally naive psi.
- bo: each new batch is centered on an adaptive BO-selected psi.
- bo_marginal_random: after BO finishes, draw random designs from BO's
  empirical marginal psi distribution, without adaptive feedback.

The BO-marginal random control separates two explanations:

1. BO found an informative design region.
2. The adaptive BO sequence itself adds value beyond sampling that region.

The default posterior estimator is the lightweight rectified FMPE model. Since
this estimator samples from the posterior but does not expose exact densities,
the default BO reward is sample-based: lower posterior mean error with a
coverage penalty.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from posterior_diagnostic_metrics import (
    evaluate_fmpe,
    evaluate_gaussian,
    posterior_quality_objective as compute_posterior_quality_objective,
)
from run_lotka_volterra import (
    THETA_BOUNDS,
    append_data,
    decode_design,
    design_dim,
    propose_design,
    sample_psi,
    sample_theta,
    simulate_batch,
)


METHODS = ("random", "fixed_dumb", "bo", "bo_marginal_random")
THETA_TARGET_DEFAULT = THETA_BOUNDS.mean(axis=1)
METHOD_LABELS = {
    "random": "uniform random",
    "fixed_dumb": "fixed dumb",
    "bo": "BO adaptive",
    "bo_marginal_random": "BO-marginal random",
}
STRUCTURED_DESIGN_LABELS = {
    "short_early": "short early",
    "short_late": "short late",
    "long_early": "long early",
    "long_late": "long late",
    "medium_mid": "medium middle",
}
CATEGORICAL_DESIGN_LABELS = {
    "short_early": "short early",
    "short_middle": "short middle",
    "short_late": "short late",
    "medium_early": "medium early",
    "medium_middle": "medium middle",
    "medium_late": "medium late",
    "long_early": "long early",
    "long_middle": "long middle",
    "long_late": "long late",
    "low_all": "low all",
    "center_all": "center all",
    "high_all": "high all",
}
METRIC_KEYS = [
    "raw_rmse",
    "range_normalized_rmse",
    "prior_std_normalized_rmse",
    "coverage_error",
    "posterior_mean_predictive_rmse",
    "posterior_quality_objective",
    "validation_log_posterior",
]
REWARD_MODE_LABELS = {
    "log_posterior": "validation log posterior",
    "log_posterior_coverage": "validation log posterior minus coverage penalty",
    "rmse_coverage": "negative range-normalized RMSE minus coverage penalty",
    "rmse_coverage_predictive": "negative range-normalized RMSE minus coverage and predictive penalties",
}

PsiSampler = Callable[[np.random.Generator, int], np.ndarray]


def clone_data(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(part.copy() for part in data)  # type: ignore[return-value]


def uniform_psi_sampler(design_space: str) -> PsiSampler:
    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return sample_psi(rng, n, design_space)

    return sampler


def parse_fixed_psi(args: argparse.Namespace) -> np.ndarray:
    dim = design_dim(args.design_space)
    values = args.dumb_psi if args.dumb_psi is not None else [0.0] * dim
    if len(values) != dim:
        raise ValueError(f"--dumb-psi needs {dim} values for design_space={args.design_space}.")
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def fixed_psi_sampler(psi_fixed: np.ndarray) -> PsiSampler:
    def sampler(_rng: np.random.Generator, n: int) -> np.ndarray:
        return np.repeat(psi_fixed[None, :], n, axis=0)

    return sampler


def categorical_design_policies(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    """Discrete design choices used by categorical BO and categorical random."""

    dim = design_dim(args.design_space)
    if dim == 2:
        policies = [
            ("short_early", np.array([0.0, 0.0])),
            ("short_middle", np.array([0.0, 0.5])),
            ("short_late", np.array([0.0, 1.0])),
            ("medium_early", np.array([0.5, 0.0])),
            ("medium_middle", np.array([0.5, 0.5])),
            ("medium_late", np.array([0.5, 1.0])),
            ("long_early", np.array([1.0, 0.0])),
            ("long_middle", np.array([1.0, 0.5])),
            ("long_late", np.array([1.0, 1.0])),
        ]
    else:
        policies = [
            ("low_all", np.zeros(dim)),
            ("center_all", np.full(dim, 0.5)),
            ("high_all", np.ones(dim)),
        ]
    if args.bo_category_policies:
        allowed = set(args.bo_category_policies)
        policies = [(name, psi) for name, psi in policies if name in allowed]
    if not policies:
        raise ValueError("No categorical BO policies remain after applying --bo-category-policies.")
    return policies


def categorical_psi_sampler(policies: list[tuple[str, np.ndarray]]) -> PsiSampler:
    psi_values = np.asarray([psi for _, psi in policies])

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        idx = rng.integers(0, len(psi_values), size=n)
        return psi_values[idx]

    return sampler


def random_design_sampler(args: argparse.Namespace) -> PsiSampler:
    if args.bo_design_mode == "categorical":
        return categorical_psi_sampler(categorical_design_policies(args))
    return uniform_psi_sampler(args.design_space)


def parse_unit_vector(values: list[float] | None, dim: int, default: float, name: str) -> np.ndarray:
    raw_values = values if values is not None else [default] * dim
    if len(raw_values) != dim:
        raise ValueError(f"{name} needs {dim} values for the selected design space.")
    return np.clip(np.asarray(raw_values, dtype=float), 0.0, 1.0)


def parse_target_theta(args: argparse.Namespace) -> np.ndarray:
    if args.target_theta is None:
        return THETA_TARGET_DEFAULT.copy()
    if len(args.target_theta) != 4:
        raise ValueError("--target-theta needs 4 values: alpha beta gamma delta.")
    theta = np.asarray(args.target_theta, dtype=float)
    lo = THETA_BOUNDS[:, 0]
    hi = THETA_BOUNDS[:, 1]
    if np.any(theta < lo) or np.any(theta > hi):
        raise ValueError(f"--target-theta must lie inside prior bounds: {THETA_BOUNDS.tolist()}.")
    return theta


def parse_target_psi(args: argparse.Namespace) -> np.ndarray:
    dim = design_dim(args.design_space)
    return parse_unit_vector(args.target_psi, dim, 0.5, "--target-psi")


def make_evaluation_set(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create either a broad validation set or one synthetic target observation."""

    if args.target_mode == "validation_set":
        return simulate_until_count(
            rng,
            args.validation,
            uniform_psi_sampler(args.design_space),
            args.n_obs,
            args.noise_std,
            args.design_space,
        )

    if args.target_mode == "fixed_x0":
        theta = parse_target_theta(args)[None, :]
        psi = parse_target_psi(args)[None, :]
        return simulate_batch(theta, psi, rng, args.n_obs, args.noise_std, args.design_space)

    raise ValueError(f"Unknown target mode: {args.target_mode}")


def structured_design_policies(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    """Fixed window policies used to score simple, interpretable designs."""

    dim = design_dim(args.design_space)
    if dim == 2:
        policies = [
            ("short_early", np.array([0.0, 0.0])),
            ("short_late", np.array([0.0, 1.0])),
            ("long_early", np.array([1.0, 0.0])),
            ("long_late", np.array([1.0, 1.0])),
            ("medium_mid", np.array([0.5, 0.5])),
        ]
    else:
        policies = [
            ("low_all", np.zeros(dim)),
            ("center_all", np.full(dim, 0.5)),
            ("high_all", np.ones(dim)),
        ]
    if args.structured_policies:
        allowed = set(args.structured_policies)
        policies = [(name, psi) for name, psi in policies if name in allowed]
    return policies


def focused_bo_sampler(
    psi_focus: np.ndarray,
    args: argparse.Namespace,
    random_fraction: float,
) -> PsiSampler:
    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        n_random = int(round(n * random_fraction))
        n_focused = max(0, n - n_random)
        parts = []
        if n_focused:
            parts.append(np.repeat(psi_focus[None, :], n_focused, axis=0))
        if n_random:
            parts.append(random_design_sampler(args)(rng, n_random))
        return np.vstack(parts)

    return sampler


def empirical_marginal_sampler(
    psi_values: np.ndarray,
    design_space: str,
    jitter_std: float,
) -> PsiSampler:
    if len(psi_values) == 0:
        return uniform_psi_sampler(design_space)

    dim = design_dim(design_space)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        cols = []
        for j in range(dim):
            idx = rng.integers(0, len(psi_values), size=n)
            cols.append(psi_values[idx, j])
        psi = np.column_stack(cols)
        if jitter_std > 0.0:
            psi = psi + rng.normal(0.0, jitter_std, size=psi.shape)
        return np.clip(psi, 0.0, 1.0)

    return sampler


def empirical_joint_sampler(
    psi_values: np.ndarray,
    design_space: str,
) -> PsiSampler:
    if len(psi_values) == 0:
        return uniform_psi_sampler(design_space)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        idx = rng.integers(0, len(psi_values), size=n)
        return psi_values[idx]

    return sampler


def propose_categorical_design(
    rng: np.random.Generator,
    tried_policy_indices: list[int],
    rewards: list[float],
    policies: list[tuple[str, np.ndarray]],
    exploit_best_prob: float,
    ucb_weight: float,
) -> tuple[str, np.ndarray, str]:
    """Finite-arm UCB over named design categories."""

    n_policies = len(policies)
    if rewards and rng.uniform() < exploit_best_prob:
        best_seen = int(np.argmax(rewards))
        policy_index = tried_policy_indices[best_seen]
        name, psi = policies[policy_index]
        return name, psi, "exploit_best_observed"

    untried = [i for i in range(n_policies) if i not in tried_policy_indices]
    if untried:
        policy_index = int(rng.choice(untried))
        name, psi = policies[policy_index]
        return name, psi, "try_unobserved_category"

    counts = np.zeros(n_policies, dtype=float)
    means = np.zeros(n_policies, dtype=float)
    for policy_index in range(n_policies):
        vals = [reward for idx, reward in zip(tried_policy_indices, rewards) if idx == policy_index]
        counts[policy_index] = len(vals)
        means[policy_index] = float(np.mean(vals))
    bonus = ucb_weight * np.sqrt(np.log(len(rewards) + 1.0) / np.maximum(counts, 1.0))
    policy_index = int(np.argmax(means + bonus))
    name, psi = policies[policy_index]
    return name, psi, "categorical_ucb"


def simulate_until_count(
    rng: np.random.Generator,
    target_n: int,
    psi_sampler: PsiSampler,
    n_obs: int,
    noise_std: float,
    design_space: str,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total = 0
    attempts = 0
    while total < target_n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not generate {target_n} valid simulations after {max_attempts} attempts.")
        request_n = max(target_n - total, 16)
        psi = psi_sampler(rng, request_n)
        theta = sample_theta(rng, request_n)
        try:
            batch = simulate_batch(theta, psi, rng, n_obs, noise_std, design_space)
        except RuntimeError:
            continue
        parts.append(batch)
        total += len(batch[0])

    theta = np.vstack([part[0] for part in parts])[:target_n]
    psi = np.vstack([part[1] for part in parts])[:target_n]
    x = np.vstack([part[2] for part in parts])[:target_n]
    return theta, psi, x


def decoded_design_row(
    method: str,
    replicate: int,
    round_index: int,
    batch_index: int,
    psi: np.ndarray,
    design_space: str,
) -> dict[str, float | int | str]:
    design = decode_design(psi, design_space)
    row: dict[str, float | int | str] = {
        "method": method,
        "replicate": replicate,
        "round": round_index,
        "batch_index": batch_index,
        "prey0": design.prey0,
        "pred0": design.pred0,
        "t_start": design.t_start,
        "t_span": design.t_span,
        "t_end": design.t_end,
    }
    for i, value in enumerate(psi):
        row[f"psi{i}_unit"] = float(value)
    return row


def design_rows(
    method: str,
    replicate: int,
    round_index: int,
    psi_batch: np.ndarray,
    design_space: str,
) -> list[dict[str, float | int | str]]:
    return [
        decoded_design_row(method, replicate, round_index, i, psi, design_space)
        for i, psi in enumerate(psi_batch)
    ]


def score_design_strategy(
    method: str,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    replicate: int,
    round_index: int,
    rep_seed: int,
) -> dict[str, float | int | str]:
    budget = len(train[0])
    seed = rep_seed + 3000 + round_index
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
    row.update(
        {
            "method": method,
            "estimator": args.estimator,
            "round": round_index,
            "simulations": budget,
            "seed": rep_seed,
            "design_space": args.design_space,
            "initial": args.initial,
            "batch": args.batch,
            "rounds": args.rounds,
            "validation": args.validation,
            "n_obs": args.n_obs,
            "noise_std": args.noise_std,
            "target_mode": args.target_mode,
            "target_seed": args.target_seed,
            "target_theta": ",".join(str(float(v)) for v in parse_target_theta(args)),
            "target_psi_unit": ",".join(str(float(v)) for v in parse_target_psi(args)),
            "bo_random_fraction": args.bo_random_fraction,
            "bo_design_mode": args.bo_design_mode,
            "bo_category_policies": ",".join(args.bo_category_policies or []),
            "categorical_ucb_weight": args.categorical_ucb_weight,
            "marginal_jitter_std": args.marginal_jitter_std,
            "reward_mode": args.reward_mode,
            "coverage_weight": args.coverage_weight,
            "predictive_weight": args.predictive_weight,
            "predictive_timeout_seconds": args.predictive_timeout_seconds,
            "final_score_only": args.final_score_only,
            "flow_samples_per_pair": args.flow_samples_per_pair,
            "posterior_samples": args.posterior_samples,
            "ode_steps": args.ode_steps,
            "fmpe_max_iter": args.fmpe_max_iter,
            "dumb_psi_unit": ",".join(str(float(v)) for v in parse_fixed_psi(args)),
        }
    )
    row["posterior_quality_objective"] = posterior_quality_objective(row, args)
    return row


def posterior_quality_objective(row: dict[str, float | int | str], args: argparse.Namespace) -> float:
    return compute_posterior_quality_objective(
        row,
        args.reward_mode,
        args.coverage_weight,
        args.predictive_weight,
    )


def run_uniform_random(
    initial_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    rep_seed: int,
    replicate: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    rng = np.random.default_rng(rep_seed + 101)
    train = clone_data(initial_train)
    rows = []
    if not args.final_score_only:
        rows.append(score_design_strategy("random", train, val, args, replicate, 0, rep_seed))
    trace = design_rows("random", replicate, 0, train[1], args.design_space)
    sampler = random_design_sampler(args)
    for round_index in range(1, args.rounds + 1):
        extra = simulate_until_count(rng, args.batch, sampler, args.n_obs, args.noise_std, args.design_space)
        trace.extend(design_rows("random", replicate, round_index, extra[1], args.design_space))
        train = append_data(train, extra)
        if not args.final_score_only or round_index == args.rounds:
            rows.append(score_design_strategy("random", train, val, args, replicate, round_index, rep_seed))
    return rows, trace


def run_fixed_dumb(
    initial_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    rep_seed: int,
    replicate: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    rng = np.random.default_rng(rep_seed + 151)
    train = clone_data(initial_train)
    rows = []
    if not args.final_score_only:
        rows.append(score_design_strategy("fixed_dumb", train, val, args, replicate, 0, rep_seed))
    trace = design_rows("fixed_dumb", replicate, 0, train[1], args.design_space)
    sampler = fixed_psi_sampler(parse_fixed_psi(args))
    for round_index in range(1, args.rounds + 1):
        extra = simulate_until_count(rng, args.batch, sampler, args.n_obs, args.noise_std, args.design_space)
        trace.extend(design_rows("fixed_dumb", replicate, round_index, extra[1], args.design_space))
        train = append_data(train, extra)
        if not args.final_score_only or round_index == args.rounds:
            rows.append(score_design_strategy("fixed_dumb", train, val, args, replicate, round_index, rep_seed))
    return rows, trace


def run_bo(
    initial_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    rep_seed: int,
    replicate: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    np.ndarray,
]:
    rng = np.random.default_rng(rep_seed + 202)
    train = clone_data(initial_train)
    rows = [score_design_strategy("bo", train, val, args, replicate, 0, rep_seed)]
    trace = design_rows("bo", replicate, 0, train[1], args.design_space)
    bo_trace: list[dict[str, float | int | str]] = []
    tried_psi: list[np.ndarray] = []
    tried_policy_indices: list[int] = []
    rewards: list[float] = []
    old_objective = posterior_quality_objective(rows[-1], args)
    categorical_policies = categorical_design_policies(args) if args.bo_design_mode == "categorical" else []

    for round_index in range(1, args.rounds + 1):
        proposal_reason = "continuous_gp_ucb"
        design_policy = ""
        if args.bo_design_mode == "categorical":
            design_policy, psi_next, proposal_reason = propose_categorical_design(
                rng,
                tried_policy_indices,
                rewards,
                categorical_policies,
                args.exploit_best_prob,
                args.categorical_ucb_weight,
            )
        else:
            psi_next = propose_design(
                rng,
                tried_psi,
                rewards,
                n_candidates=args.bo_candidates,
                design_space=args.design_space,
                exploit_best_prob=args.exploit_best_prob,
            )
        sampler = focused_bo_sampler(psi_next, args, args.bo_random_fraction)
        extra = simulate_until_count(rng, args.batch, sampler, args.n_obs, args.noise_std, args.design_space)
        trace.extend(design_rows("bo", replicate, round_index, extra[1], args.design_space))
        candidate_train = append_data(train, extra)
        new_row = score_design_strategy("bo", candidate_train, val, args, replicate, round_index, rep_seed)
        new_objective = posterior_quality_objective(new_row, args)
        reward = new_objective - old_objective
        tried_psi.append(psi_next)
        if args.bo_design_mode == "categorical":
            tried_policy_indices.append([name for name, _ in categorical_policies].index(design_policy))
        rewards.append(reward)

        design = decode_design(psi_next, args.design_space)
        trace_row: dict[str, float | int | str] = {
            "replicate": replicate,
            "round": round_index,
            "reward_delta_objective": reward,
            "objective": new_objective,
            "reward_mode": args.reward_mode,
            "coverage_weight": args.coverage_weight,
            "predictive_weight": args.predictive_weight,
            "range_normalized_rmse": float(new_row["range_normalized_rmse"]),
            "coverage_error": float(new_row["coverage_error"]),
            "posterior_mean_predictive_rmse": float(new_row["posterior_mean_predictive_rmse"]),
            "validation_log_posterior": new_row.get("validation_log_posterior", ""),
            "simulations": len(candidate_train[0]),
            "bo_design_mode": args.bo_design_mode,
            "design_policy": design_policy,
            "proposal_reason": proposal_reason,
            "prey0": design.prey0,
            "pred0": design.pred0,
            "t_start": design.t_start,
            "t_span": design.t_span,
            "t_end": design.t_end,
        }
        for i, value in enumerate(psi_next):
            trace_row[f"psi{i}_unit"] = float(value)
        bo_trace.append(trace_row)

        train = candidate_train
        old_objective = new_objective
        rows.append(new_row)

    return rows, trace, bo_trace, np.asarray(tried_psi)


def run_bo_marginal_random(
    initial_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    bo_psi: np.ndarray,
    args: argparse.Namespace,
    rep_seed: int,
    replicate: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    rng = np.random.default_rng(rep_seed + 303)
    train = clone_data(initial_train)
    rows = []
    if not args.final_score_only:
        rows.append(score_design_strategy("bo_marginal_random", train, val, args, replicate, 0, rep_seed))
    trace = design_rows("bo_marginal_random", replicate, 0, train[1], args.design_space)
    if args.bo_design_mode == "categorical":
        sampler = empirical_joint_sampler(bo_psi, args.design_space)
    else:
        sampler = empirical_marginal_sampler(bo_psi, args.design_space, args.marginal_jitter_std)
    for round_index in range(1, args.rounds + 1):
        extra = simulate_until_count(rng, args.batch, sampler, args.n_obs, args.noise_std, args.design_space)
        trace.extend(design_rows("bo_marginal_random", replicate, round_index, extra[1], args.design_space))
        train = append_data(train, extra)
        if not args.final_score_only or round_index == args.rounds:
            rows.append(score_design_strategy("bo_marginal_random", train, val, args, replicate, round_index, rep_seed))
    return rows, trace


def run_structured_design_scores(
    initial_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    rep_seed: int,
    replicate: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Score simple fixed design policies at the same final simulation budget."""

    rng = np.random.default_rng(rep_seed + 404)
    extra_n = args.batch * args.rounds
    rows: list[dict[str, float | int | str]] = []
    trace: list[dict[str, float | int | str]] = []
    for policy, psi_fixed in structured_design_policies(args):
        sampler = fixed_psi_sampler(psi_fixed)
        train = clone_data(initial_train)
        if extra_n > 0:
            extra = simulate_until_count(rng, extra_n, sampler, args.n_obs, args.noise_std, args.design_space)
            trace.extend(design_rows(f"structured_{policy}", replicate, args.rounds, extra[1], args.design_space))
            train = append_data(train, extra)
        row = score_design_strategy(
            f"structured_{policy}",
            train,
            val,
            args,
            replicate,
            args.rounds,
            rep_seed,
        )
        design = decode_design(psi_fixed, args.design_space)
        row.update(
            {
                "structured_policy": policy,
                "structured_label": STRUCTURED_DESIGN_LABELS.get(policy, policy.replace("_", " ")),
                "prey0": design.prey0,
                "pred0": design.pred0,
                "t_start": design.t_start,
                "t_span": design.t_span,
                "t_end": design.t_end,
            }
        )
        for i, value in enumerate(psi_fixed):
            row[f"psi{i}_unit"] = float(value)
        rows.append(row)
    return rows, trace


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["round"]))].append(row)

    method_order = {method: i for i, method in enumerate(METHODS)}
    out_rows: list[dict[str, float | int | str]] = []
    for (method, round_index), group in sorted(groups.items(), key=lambda item: (item[0][1], method_order[item[0][0]])):
        out: dict[str, float | int | str] = {
            "method": method,
            "round": round_index,
            "replicates": len(group),
            "simulations_mean": float(np.mean([float(row["simulations"]) for row in group])),
        }
        for key in METRIC_KEYS:
            vals = [float(row[key]) for row in group if key in row and row[key] not in {"", None}]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        out_rows.append(out)
    return out_rows


def summarize_designs(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["round"]))].append(row)

    method_order = {method: i for i, method in enumerate(METHODS)}
    out_rows: list[dict[str, float | int | str]] = []
    for (method, round_index), group in sorted(groups.items(), key=lambda item: (item[0][1], method_order[item[0][0]])):
        out: dict[str, float | int | str] = {
            "method": method,
            "round": round_index,
            "n_designs": len(group),
        }
        for key in ("t_start", "t_span", "t_end", "prey0", "pred0"):
            vals = [float(row[key]) for row in group if key in row and row[key] not in {"", None}]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        out_rows.append(out)
    return out_rows


def summarize_design_choice_scores(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["structured_policy"])].append(row)

    policy_order = [
        "short_early",
        "short_late",
        "long_early",
        "long_late",
        "medium_mid",
        "low_all",
        "center_all",
        "high_all",
    ]
    order = {name: i for i, name in enumerate(policy_order)}
    out_rows: list[dict[str, float | int | str]] = []
    for policy, group in sorted(groups.items(), key=lambda item: order.get(item[0], 999)):
        first = group[0]
        out: dict[str, float | int | str] = {
            "structured_policy": policy,
            "structured_label": str(first.get("structured_label", policy)),
            "replicates": len(group),
            "prey0": float(first.get("prey0", np.nan)),
            "pred0": float(first.get("pred0", np.nan)),
            "t_start": float(first.get("t_start", np.nan)),
            "t_span": float(first.get("t_span", np.nan)),
            "t_end": float(first.get("t_end", np.nan)),
        }
        for key in METRIC_KEYS:
            vals = [float(row[key]) for row in group if key in row and row[key] not in {"", None}]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        out_rows.append(out)
    return out_rows


def plot_metric_summary(summary_rows: list[dict[str, float | int | str]], output: Path) -> None:
    panels = [
        ("range_normalized_rmse_mean", "Prior-range normalized RMSE\n(lower is better)"),
        ("posterior_quality_objective_mean", "BO objective\n(higher is better)"),
        ("coverage_error_mean", "Coverage error\n(lower is better)"),
        ("posterior_mean_predictive_rmse_mean", "Posterior-mean predictive RMSE\n(lower is better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, ylabel) in zip(axes.ravel(), panels):
        for method in METHODS:
            subset = [row for row in summary_rows if row["method"] == method and metric in row]
            if not subset:
                continue
            xs = np.array([float(row["simulations_mean"]) for row in subset])
            ys = np.array([float(row[metric]) for row in subset])
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker="o", linewidth=2, label=METHOD_LABELS[method])
        ax.set_xlabel("Training simulations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Approach 1.3: BO Design Effect Check", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_design_summary(
    design_rows_all: list[dict[str, float | int | str]],
    design_summary_rows: list[dict[str, float | int | str]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for method in METHODS:
        subset = [row for row in design_rows_all if row["method"] == method and int(row["round"]) > 0]
        if not subset:
            continue
        ax.scatter(
            [float(row["t_start"]) for row in subset],
            [float(row["t_span"]) for row in subset],
            s=12,
            alpha=0.35,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("t_start")
    ax.set_ylabel("t_span")
    ax.set_title("Chosen observation windows")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    for ax, metric, title in [
        (axes[1], "t_start_mean", "Mean t_start by round"),
        (axes[2], "t_span_mean", "Mean t_span by round"),
    ]:
        for method in METHODS:
            subset = [row for row in design_summary_rows if row["method"] == method and metric in row]
            xs = np.array([int(row["round"]) for row in subset])
            ys = np.array([float(row[metric]) for row in subset])
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker="o", linewidth=2, label=METHOD_LABELS[method])
        ax.set_xlabel("Round")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1].set_ylabel("t_start")
    axes[2].set_ylabel("t_span")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_design_choice_scores(summary_rows: list[dict[str, float | int | str]], output: Path) -> None:
    if not summary_rows:
        return
    labels = [str(row["structured_label"]) for row in summary_rows]
    x = np.arange(len(labels))
    metrics = [
        ("range_normalized_rmse_mean", "Range-norm RMSE", "lower is better"),
        ("posterior_quality_objective_mean", "BO objective", "higher is better"),
        ("coverage_error_mean", "Coverage error", "lower is better"),
        ("posterior_mean_predictive_rmse_mean", "Predictive RMSE", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (key, title, subtitle) in zip(axes.ravel(), metrics):
        vals = [float(row[key]) for row in summary_rows]
        ax.bar(x, vals, color="#4C78A8", alpha=0.82)
        ax.set_title(f"{title}\n({subtitle})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Structured Fixed Design Scores", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def final_interpretation(summary_rows: list[dict[str, float | int | str]]) -> list[str]:
    final_round = max(int(row["round"]) for row in summary_rows)
    final_rows = [row for row in summary_rows if int(row["round"]) == final_round]
    by_method = {str(row["method"]): row for row in final_rows}
    random_row = by_method.get("random")
    dumb_row = by_method.get("fixed_dumb")
    bo_row = by_method.get("bo")
    marginal_row = by_method.get("bo_marginal_random")
    if not random_row or not dumb_row or not bo_row or not marginal_row:
        return ["Not enough final rows for interpretation."]

    random_rmse = float(random_row["range_normalized_rmse_mean"])
    dumb_rmse = float(dumb_row["range_normalized_rmse_mean"])
    bo_rmse = float(bo_row["range_normalized_rmse_mean"])
    marginal_rmse = float(marginal_row["range_normalized_rmse_mean"])
    random_objective = float(random_row.get("posterior_quality_objective_mean", float("nan")))
    bo_objective = float(bo_row.get("posterior_quality_objective_mean", float("nan")))
    marginal_objective = float(marginal_row.get("posterior_quality_objective_mean", float("nan")))
    random_to_bo = 100.0 * (random_rmse - bo_rmse) / random_rmse
    dumb_to_bo = 100.0 * (dumb_rmse - bo_rmse) / dumb_rmse
    random_to_marginal = 100.0 * (random_rmse - marginal_rmse) / random_rmse
    marginal_to_bo = 100.0 * (marginal_rmse - bo_rmse) / marginal_rmse

    lines = [
        "## Current Interpretation",
        "",
        f"At the final round, BO changes range-normalized RMSE by `{random_to_bo:.1f}%` relative to uniform random design.",
        f"BO changes range-normalized RMSE by `{dumb_to_bo:.1f}%` relative to the fixed dumb design.",
        f"The BO-marginal random control changes range-normalized RMSE by `{random_to_marginal:.1f}%` relative to uniform random design.",
        f"BO changes range-normalized RMSE by `{marginal_to_bo:.1f}%` relative to the BO-marginal random control.",
        "",
    ]
    if bo_rmse < random_rmse and bo_rmse < marginal_rmse:
        if np.isfinite(bo_objective) and np.isfinite(marginal_objective) and bo_objective < marginal_objective:
            lines.append("BO is strongest on final range-normalized RMSE, but not on the combined BO objective. This is a mixed signal: more sequential feedback helps point estimation, while coverage/predictive terms still favor another design strategy.")
        else:
            lines.append("BO is strongest at the final budget. This is initial evidence that adaptive design can help beyond simply sampling the BO-favored design region.")
    elif marginal_rmse <= bo_rmse < random_rmse:
        lines.append("BO and the BO-marginal random control both beat uniform random design. This suggests the selected design region is useful, but the adaptive sequence itself is not yet clearly better than sampling that region.")
    elif bo_rmse < random_rmse and marginal_rmse >= random_rmse:
        lines.append("BO beats uniform random design while the marginal control does not. This points toward an adaptive-design effect, but more repeats are needed because this pattern can be noisy.")
    else:
        lines.append("There is no clear final-budget evidence that BO improves over uniform random design in this run.")
    if np.isfinite(random_objective) and np.isfinite(bo_objective) and bo_objective < random_objective:
        lines.append("The combined objective remains worse for BO than for uniform random design, so the final conclusion should not rely on RMSE alone.")
    lines.append("")
    return lines


def structured_design_interpretation(
    summary_rows: list[dict[str, float | int | str]],
    design_choice_summary_rows: list[dict[str, float | int | str]],
) -> list[str]:
    if not design_choice_summary_rows:
        return []
    final_round = max(int(row["round"]) for row in summary_rows)
    final_rows = [row for row in summary_rows if int(row["round"]) == final_round]
    by_method = {str(row["method"]): row for row in final_rows}
    bo_row = by_method.get("bo")
    if bo_row is None:
        return []

    best_rmse = min(design_choice_summary_rows, key=lambda row: float(row["range_normalized_rmse_mean"]))
    best_objective = max(design_choice_summary_rows, key=lambda row: float(row["posterior_quality_objective_mean"]))
    bo_rmse = float(bo_row["range_normalized_rmse_mean"])
    bo_objective = float(bo_row["posterior_quality_objective_mean"])
    best_rmse_value = float(best_rmse["range_normalized_rmse_mean"])
    best_objective_value = float(best_objective["posterior_quality_objective_mean"])

    lines = [
        "## Structured-Design Interpretation",
        "",
        f"The best fixed structured window by RMSE is `{best_rmse['structured_label']}` with RMSE `{best_rmse_value:.4f}`.",
        f"The best fixed structured window by objective is `{best_objective['structured_label']}` with objective `{best_objective_value:.4f}`.",
        "",
    ]
    if best_rmse_value < bo_rmse or best_objective_value > bo_objective:
        lines.append("At least one simple structured fixed window beats BO on a final diagnostic. This means BO must be compared against structured non-adaptive designs before claiming adaptive-design value.")
    else:
        lines.append("BO beats the structured fixed windows on these final diagnostics, which is stronger evidence for adaptive design than a comparison to random alone.")
    lines.append("")
    return lines


def write_markdown_summary(
    path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, float | int | str]],
    design_choice_summary_rows: list[dict[str, float | int | str]],
) -> None:
    final_round = max(int(row["round"]) for row in summary_rows)
    final_rows = [row for row in summary_rows if int(row["round"]) == final_round]
    by_method = {str(row["method"]): row for row in final_rows}
    dumb_psi = parse_fixed_psi(args)
    dumb_design = decode_design(dumb_psi, args.design_space)
    config_lines = [
        f"design_space = {args.design_space}",
        f"initial = {args.initial}",
        f"batch = {args.batch}",
        f"rounds = {args.rounds}",
        f"final_budget = {args.initial + args.batch * args.rounds}",
        f"validation = {args.validation}",
        f"repeats = {args.repeats}",
        f"n_obs = {args.n_obs}",
        f"seed = {args.seed}",
        f"target_mode = {args.target_mode}",
        f"bo_design_mode = {args.bo_design_mode}",
    ]
    if args.target_mode == "fixed_x0":
        config_lines.extend(
            [
                f"target_seed = {args.target_seed}",
                f"target_theta = {' '.join(str(float(v)) for v in parse_target_theta(args))}",
                f"target_psi_unit = {' '.join(str(float(v)) for v in parse_target_psi(args))}",
            ]
        )
    config_lines.extend(
        [
            f"bo_candidates = {args.bo_candidates}",
            f"bo_random_fraction = {args.bo_random_fraction}",
            f"bo_category_policies = {' '.join(args.bo_category_policies) if args.bo_category_policies else 'all'}",
            f"categorical_ucb_weight = {args.categorical_ucb_weight}",
            f"marginal_jitter_std = {args.marginal_jitter_std}",
            f"estimator = {args.estimator}",
            f"reward_mode = {args.reward_mode}",
            f"coverage_weight = {args.coverage_weight}",
            f"predictive_weight = {args.predictive_weight}",
            f"predictive_timeout_seconds = {args.predictive_timeout_seconds}",
            f"final_score_only = {args.final_score_only}",
            f"structured_policies = {' '.join(args.structured_policies) if args.structured_policies else 'all'}",
            f"flow_samples_per_pair = {args.flow_samples_per_pair}",
            f"posterior_samples = {args.posterior_samples}",
            f"ode_steps = {args.ode_steps}",
            f"fmpe_max_iter = {args.fmpe_max_iter}",
            f"dumb_psi_unit = {' '.join(str(float(v)) for v in dumb_psi)}",
        ]
    )

    lines = [
        "# Approach 1.3 BO Design Effect Check Results",
        "",
        "This diagnostic asks whether adaptive BO-selected simulator designs improve parameter estimation, or whether gains mainly come from adding more data.",
        "",
        "## Run Configuration",
        "",
        "```text",
        *config_lines,
        "```",
        "",
        "## Design Focus",
        "",
        "BO controls design variables `psi`, not the inferred physical parameters `theta`.",
        "",
        "For `wide_window`, `psi = (t_span, t_start)`: initial populations and `n_obs` are fixed, while BO can choose between short dense observation windows and longer trend windows.",
        "",
        "When `bo_design_mode = categorical`, BO and the random baseline choose from the same finite set of named design categories instead of the full continuous `psi` space.",
        "",
        "When `target_mode = fixed_x0`, the reward is evaluated only on one synthetic observed time series. This mimics the real SBI use case more closely than averaging over many validation observations, but it is noisier and should not be interpreted as global posterior quality.",
        "",
        "The BO surrogate is fit to round-to-round changes in the configured posterior-quality objective:",
        "",
        "```text",
        REWARD_MODE_LABELS[args.reward_mode],
        "```",
        "",
        "The fixed dumb baseline uses the same shared initial random data, then adds every later batch at one fixed design:",
        "",
        "```text",
        f"prey0 = {dumb_design.prey0:.3f}",
        f"pred0 = {dumb_design.pred0:.3f}",
        f"t_start = {dumb_design.t_start:.3f}",
        f"t_span = {dumb_design.t_span:.3f}",
        f"t_end = {dumb_design.t_end:.3f}",
        "```",
        "",
        "## Methods",
        "",
        "| Method | Meaning |",
        "| --- | --- |",
        "| random | Uniform random allowed design choice at every round. |",
        "| fixed_dumb | One fixed naive design at every round after the shared initial data. |",
        "| bo | Adaptive BO-selected design with a small random-design fraction. |",
        "| bo_marginal_random | Random designs sampled from BO's empirical selected-design distribution after BO finishes. |",
        "",
        f"Final evaluated round: `{final_round}`.",
        "",
        "## Final-Round Summary",
        "",
        "| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |",
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
            + f"{float(row['posterior_quality_objective_mean']):.4f} | "
            + f"{float(row['coverage_error_mean']):.4f} | "
            + f"{float(row['posterior_mean_predictive_rmse_mean']):.4f} |"
        )
    if design_choice_summary_rows:
        lines += [
            "",
            "## Structured Fixed Design Scores",
            "",
            "These policies add all post-initial simulations at one interpretable fixed window. They are not adaptive, but they test whether simple design structure can explain or beat the BO choices.",
            "",
            "| Design | t_start | t_span | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in design_choice_summary_rows:
            lines.append(
                "| "
                + str(row["structured_label"])
                + " | "
                + f"{float(row['t_start']):.3f} | "
                + f"{float(row['t_span']):.3f} | "
                + f"{float(row['range_normalized_rmse_mean']):.4f} | "
                + f"{float(row['posterior_quality_objective_mean']):.4f} | "
                + f"{float(row['coverage_error_mean']):.4f} | "
                + f"{float(row['posterior_mean_predictive_rmse_mean']):.4f} |"
            )
    lines += [
        "",
        *final_interpretation(summary_rows),
        *structured_design_interpretation(summary_rows, design_choice_summary_rows),
        "## Output Files",
        "",
        "```text",
        "metrics.csv",
        "summary_by_round.csv",
        "design_trace.csv",
        "design_summary_by_round.csv",
        "bo_trace.csv",
        "design_choice_scores.csv",
        "design_choice_summary.csv",
        "design_choice_trace.csv",
        "design_choice_scores.png",
        "bo_design_effect_summary.png",
        "design_structure_summary.png",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-space", choices=("full", "window", "hard_window", "wide_window"), default="wide_window")
    parser.add_argument("--initial", type=int, default=80)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.06)
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
    parser.add_argument(
        "--target-seed",
        type=int,
        default=919,
        help="Seed used to generate the fixed synthetic observation x0.",
    )
    parser.add_argument(
        "--bo-design-mode",
        choices=("continuous", "categorical"),
        default="continuous",
        help="Let BO choose continuous psi values or named categorical design choices.",
    )
    parser.add_argument(
        "--bo-category-policies",
        nargs="+",
        default=None,
        choices=tuple(CATEGORICAL_DESIGN_LABELS),
        help="Optional subset of categorical design policies BO/random may choose.",
    )
    parser.add_argument(
        "--categorical-ucb-weight",
        type=float,
        default=1.0,
        help="Exploration weight for finite-arm categorical BO.",
    )
    parser.add_argument("--bo-candidates", type=int, default=192)
    parser.add_argument("--exploit-best-prob", type=float, default=0.25)
    parser.add_argument("--bo-random-fraction", type=float, default=0.2)
    parser.add_argument("--marginal-jitter-std", type=float, default=0.04)
    parser.add_argument(
        "--reward-mode",
        choices=("log_posterior", "log_posterior_coverage", "rmse_coverage", "rmse_coverage_predictive"),
        default="rmse_coverage_predictive",
        help="Scalar objective BO tries to improve after each added simulation batch.",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.25,
        help="Penalty weight for coverage_error in combined reward modes.",
    )
    parser.add_argument(
        "--predictive-weight",
        type=float,
        default=0.1,
        help="Penalty weight for posterior_mean_predictive_rmse when reward-mode is rmse_coverage_predictive.",
    )
    parser.add_argument(
        "--predictive-timeout-seconds",
        type=float,
        default=0.5,
        help="Per-sample timeout for posterior-predictive simulator calls; non-positive disables the timeout.",
    )
    parser.add_argument(
        "--estimator",
        choices=("gaussian_npe", "rectified_fmpe"),
        default="rectified_fmpe",
        help="Posterior estimator used for the design-effect diagnostic.",
    )
    parser.add_argument("--flow-samples-per-pair", type=int, default=3)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument(
        "--fmpe-max-iter",
        type=int,
        default=220,
        help="Max MLP iterations for the lightweight rectified FMPE estimator.",
    )
    parser.add_argument(
        "--dumb-psi",
        type=float,
        nargs="+",
        default=None,
        help="Unit-cube psi values for the fixed dumb baseline. Defaults to all zeros.",
    )
    parser.add_argument("--seed", type=int, default=717)
    parser.add_argument(
        "--no-score-structured-designs",
        action="store_true",
        help="Skip the extra fixed-window design scores.",
    )
    parser.add_argument(
        "--structured-policies",
        nargs="+",
        default=None,
        choices=(
            "short_early",
            "short_late",
            "long_early",
            "long_late",
            "medium_mid",
            "low_all",
            "center_all",
            "high_all",
        ),
        help="Optional subset of structured fixed-design policies to score.",
    )
    parser.add_argument(
        "--final-score-only",
        action="store_true",
        help="Only score non-BO baselines at the final round; BO is still scored every round for rewards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "results" / "lotka_volterra" / "approach_1_3_bo_design_effect_check",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.initial = 50
        args.batch = 10
        args.rounds = 2
        args.validation = 60
        args.repeats = 1
        args.bo_candidates = 64
        args.flow_samples_per_pair = min(args.flow_samples_per_pair, 2)
        args.posterior_samples = min(args.posterior_samples, 16)
        args.ode_steps = min(args.ode_steps, 6)
        args.fmpe_max_iter = min(args.fmpe_max_iter, 120)

    if args.estimator == "rectified_fmpe" and args.reward_mode.startswith("log_posterior"):
        raise ValueError("rectified_fmpe does not expose exact log posterior; use an RMSE-based reward mode.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | int | str]] = []
    design_trace_rows: list[dict[str, float | int | str]] = []
    bo_trace_rows: list[dict[str, float | int | str]] = []
    design_choice_rows: list[dict[str, float | int | str]] = []
    design_choice_trace_rows: list[dict[str, float | int | str]] = []

    fixed_target_val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if args.target_mode == "fixed_x0":
        fixed_target_val = make_evaluation_set(args, np.random.default_rng(args.target_seed))

    for rep in range(args.repeats):
        rep_seed = args.seed + rep * 10_000
        setup_rng = np.random.default_rng(rep_seed)
        val = fixed_target_val if fixed_target_val is not None else make_evaluation_set(args, setup_rng)
        initial_train = simulate_until_count(
            setup_rng,
            args.initial,
            random_design_sampler(args),
            args.n_obs,
            args.noise_std,
            args.design_space,
        )

        random_rows, random_trace = run_uniform_random(initial_train, val, args, rep_seed, rep)
        dumb_rows, dumb_trace = run_fixed_dumb(initial_train, val, args, rep_seed, rep)
        bo_rows, bo_design_trace, bo_trace, bo_psi = run_bo(initial_train, val, args, rep_seed, rep)
        marginal_rows, marginal_trace = run_bo_marginal_random(initial_train, val, bo_psi, args, rep_seed, rep)
        if not args.no_score_structured_designs:
            structured_rows, structured_trace = run_structured_design_scores(initial_train, val, args, rep_seed, rep)
            design_choice_rows.extend(structured_rows)
            design_choice_trace_rows.extend(structured_trace)

        metric_rows.extend(random_rows)
        metric_rows.extend(dumb_rows)
        metric_rows.extend(bo_rows)
        metric_rows.extend(marginal_rows)
        design_trace_rows.extend(random_trace)
        design_trace_rows.extend(dumb_trace)
        design_trace_rows.extend(bo_design_trace)
        design_trace_rows.extend(marginal_trace)
        bo_trace_rows.extend(bo_trace)

    summary_rows = summarize(metric_rows)
    design_summary_rows = summarize_designs(design_trace_rows)
    design_choice_summary_rows = summarize_design_choice_scores(design_choice_rows)

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "summary_by_round.csv", summary_rows)
    write_csv(args.output_dir / "design_trace.csv", design_trace_rows)
    write_csv(args.output_dir / "design_summary_by_round.csv", design_summary_rows)
    write_csv(args.output_dir / "bo_trace.csv", bo_trace_rows)
    write_csv(args.output_dir / "design_choice_scores.csv", design_choice_rows)
    write_csv(args.output_dir / "design_choice_summary.csv", design_choice_summary_rows)
    write_csv(args.output_dir / "design_choice_trace.csv", design_choice_trace_rows)
    plot_metric_summary(summary_rows, args.output_dir / "bo_design_effect_summary.png")
    plot_design_summary(design_trace_rows, design_summary_rows, args.output_dir / "design_structure_summary.png")
    plot_design_choice_scores(design_choice_summary_rows, args.output_dir / "design_choice_scores.png")
    write_markdown_summary(args.output_dir / "RESULTS.md", args, summary_rows, design_choice_summary_rows)

    print(f"Wrote metrics to {args.output_dir / 'metrics.csv'}")
    print(f"Wrote summary to {args.output_dir / 'summary_by_round.csv'}")
    print(f"Wrote design trace to {args.output_dir / 'design_trace.csv'}")
    print(f"Wrote BO trace to {args.output_dir / 'bo_trace.csv'}")
    print(f"Wrote report to {args.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
