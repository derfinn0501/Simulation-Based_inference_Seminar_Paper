"""Shared posterior diagnostic helpers for the Lotka-Volterra SBI experiments."""

from __future__ import annotations

import contextlib
import signal

import numpy as np

from run_lotka_volterra import (
    THETA_BOUNDS,
    GaussianPosteriorRegressor,
    coverage_error as gaussian_coverage_error,
    simulate_one,
)
from run_lotka_volterra_fmpe import RectifiedFMPE


PARAM_NAMES = ("alpha", "beta", "gamma", "delta")
LEVELS = (0.5, 0.8, 0.9)


class PredictiveSimulationTimeout(RuntimeError):
    """Raised when a posterior-predictive simulator call takes too long."""


@contextlib.contextmanager
def predictive_simulation_time_limit(seconds: float | None):
    if seconds is None or seconds <= 0.0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise PredictiveSimulationTimeout

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


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
    timeout_seconds: float | None = 0.5,
) -> float:
    """Simulate noiseless trajectories at posterior mean parameters."""

    rng = np.random.default_rng(seed)
    preds = []
    targets = []
    for th, ps, x in zip(theta_pred, psi, x_true):
        try:
            with predictive_simulation_time_limit(timeout_seconds):
                x_pred = simulate_one(
                    th,
                    ps,
                    rng,
                    n_obs=n_obs,
                    noise_std=0.0,
                    design_space=design_space,
                )
        except PredictiveSimulationTimeout:
            continue
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
    prior_ranges = prior_range()
    prior_stds = prior_std()
    row: dict[str, float | int | str] = {
        "method": method,
        "replicate": replicate,
        "budget": budget,
        "raw_rmse": float(np.sqrt(np.mean((theta_pred - theta_true) ** 2))),
        "range_normalized_rmse": float(np.mean(per_param / prior_ranges)),
        "prior_std_normalized_rmse": float(np.mean(per_param / prior_stds)),
        "coverage_error": coverage_error,
        "posterior_mean_predictive_rmse": predictive_rmse,
    }
    for i, (name, value) in enumerate(zip(PARAM_NAMES, per_param)):
        row[f"rmse_{name}"] = float(value)
        row[f"range_normalized_rmse_{name}"] = float(value / prior_ranges[i])
        row[f"prior_std_normalized_rmse_{name}"] = float(value / prior_stds[i])
    if validation_log_posterior is not None:
        row["validation_log_posterior"] = validation_log_posterior
    return row


def posterior_quality_objective(
    row: dict[str, float | int | str],
    reward_mode: str,
    coverage_weight: float,
    predictive_weight: float,
) -> float:
    range_rmse = float(row["range_normalized_rmse"])
    coverage_error = float(row["coverage_error"])
    if reward_mode == "log_posterior":
        return float(row["validation_log_posterior"])
    if reward_mode == "log_posterior_coverage":
        return float(row["validation_log_posterior"]) - coverage_weight * coverage_error
    if reward_mode == "rmse_coverage":
        return -range_rmse - coverage_weight * coverage_error
    if reward_mode == "rmse_coverage_predictive":
        predictive_rmse = float(row["posterior_mean_predictive_rmse"])
        return -range_rmse - coverage_weight * coverage_error - predictive_weight * predictive_rmse
    raise ValueError(f"Unknown reward mode: {reward_mode}")


def evaluate_prior(
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicate: int,
    budget: int,
    seed: int,
    n_obs: int,
    design_space: str,
    predictive_timeout_seconds: float | None = 0.5,
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
        posterior_mean_predictive_rmse(
            pred,
            psi_val,
            x_val,
            seed,
            n_obs,
            design_space,
            timeout_seconds=predictive_timeout_seconds,
        ),
    )


def evaluate_gaussian(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicate: int,
    budget: int,
    seed: int,
    n_obs: int,
    design_space: str,
    predictive_timeout_seconds: float | None = 0.5,
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
        posterior_mean_predictive_rmse(
            pred,
            psi_val,
            x_val,
            seed + 17,
            n_obs,
            design_space,
            timeout_seconds=predictive_timeout_seconds,
        ),
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
    predictive_timeout_seconds: float | None = 0.5,
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
        posterior_mean_predictive_rmse(
            pred,
            psi_val,
            x_val,
            seed + 31,
            n_obs,
            design_space,
            timeout_seconds=predictive_timeout_seconds,
        ),
    )
