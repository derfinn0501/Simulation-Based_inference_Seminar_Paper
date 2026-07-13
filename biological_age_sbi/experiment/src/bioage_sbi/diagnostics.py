"""Small, estimator-agnostic diagnostics for the biological-age SBI study.

The functions in this module operate on held-out synthetic ground truths and
conditional mean/quantile estimates.  They deliberately do not train a model
or construct a simulator, which keeps diagnostic calculations comparable
across simulator and network variants.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom


def _as_1d(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).squeeze()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after squeezing; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def validate_quantile_predictions(
    truth: np.ndarray | Sequence[float],
    quantiles: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and standardize arrays used by quantile diagnostics."""

    truth_array = _as_1d(truth, "truth")
    quantile_array = np.asarray(quantiles, dtype=float)
    levels = _as_1d(quantile_levels, "quantile_levels")

    if quantile_array.ndim == 3 and quantile_array.shape[-1] == 1:
        quantile_array = quantile_array[..., 0]
    if quantile_array.ndim != 2:
        raise ValueError(f"quantiles must have shape (n_cases, n_levels); got {quantile_array.shape}.")
    if quantile_array.shape != (truth_array.size, levels.size):
        raise ValueError(
            "quantiles shape does not match truth and quantile_levels: "
            f"{quantile_array.shape} != ({truth_array.size}, {levels.size})."
        )
    if not np.all(np.isfinite(quantile_array)):
        raise ValueError("quantiles contains non-finite values.")
    if np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("quantile_levels must lie strictly between zero and one.")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile_levels must be strictly increasing.")

    return truth_array, quantile_array, levels


def point_accuracy_metrics(
    truth: np.ndarray | Sequence[float],
    point_estimate: np.ndarray | Sequence[float],
    prior_std: float | None = None,
) -> pd.Series:
    """Compute point-recovery metrics on held-out synthetic cases."""

    truth_array = _as_1d(truth, "truth")
    estimate_array = _as_1d(point_estimate, "point_estimate")
    if truth_array.shape != estimate_array.shape:
        raise ValueError(f"truth and point_estimate shapes differ: {truth_array.shape} != {estimate_array.shape}.")

    error = estimate_array - truth_array
    rmse = float(np.sqrt(np.mean(error**2)))
    metrics = {
        "n_cases": int(truth_array.size),
        "correlation_r": float(np.corrcoef(truth_array, estimate_array)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "bias": float(np.mean(error)),
    }
    if prior_std is not None:
        if not np.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("prior_std must be positive and finite.")
        metrics["prior_std_normalized_rmse"] = rmse / float(prior_std)
    return pd.Series(metrics, name="point_accuracy")


def quantile_calibration_table(
    truth: np.ndarray | Sequence[float],
    quantiles: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
    confidence: float = 0.99,
) -> pd.DataFrame:
    """Return empirical quantile frequencies and pointwise binomial bands."""

    truth_array, quantile_array, levels = validate_quantile_predictions(truth, quantiles, quantile_levels)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")

    n_cases = truth_array.size
    empirical = np.mean(truth_array[:, None] <= quantile_array, axis=0)
    alpha = 1.0 - confidence
    lower = binom.ppf(alpha / 2.0, n_cases, levels) / n_cases
    upper = binom.ppf(1.0 - alpha / 2.0, n_cases, levels) / n_cases
    return pd.DataFrame(
        {
            "quantile_level": levels,
            "empirical_frequency": empirical,
            "calibration_error": empirical - levels,
            "absolute_calibration_error": np.abs(empirical - levels),
            "pointwise_lower": lower,
            "pointwise_upper": upper,
            "inside_pointwise_band": (empirical >= lower) & (empirical <= upper),
        }
    )


def interval_diagnostics(
    truth: np.ndarray | Sequence[float],
    quantiles: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
    interval_levels: Iterable[float] = (0.5, 0.8, 0.9),
    prior_samples: np.ndarray | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compute central-interval coverage, width, and optional prior contraction."""

    truth_array, quantile_array, levels = validate_quantile_predictions(truth, quantiles, quantile_levels)
    prior_array = None if prior_samples is None else _as_1d(prior_samples, "prior_samples")
    rows = []

    for interval_level in interval_levels:
        interval_level = float(interval_level)
        if not 0.0 < interval_level < 1.0:
            raise ValueError("Every interval level must lie strictly between zero and one.")
        lower_level = (1.0 - interval_level) / 2.0
        upper_level = 1.0 - lower_level
        lower_matches = np.flatnonzero(np.isclose(levels, lower_level))
        upper_matches = np.flatnonzero(np.isclose(levels, upper_level))
        if lower_matches.size != 1 or upper_matches.size != 1:
            raise ValueError(
                f"Central {interval_level:.0%} interval requires quantile levels "
                f"{lower_level:g} and {upper_level:g}."
            )

        lower = quantile_array[:, lower_matches[0]]
        upper = quantile_array[:, upper_matches[0]]
        width = upper - lower
        covered = (truth_array >= lower) & (truth_array <= upper)
        row = {
            "interval_level": interval_level,
            "lower_quantile": lower_level,
            "upper_quantile": upper_level,
            "empirical_coverage": float(np.mean(covered)),
            "coverage_error": float(np.mean(covered) - interval_level),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)),
        }
        if prior_array is not None:
            prior_lower, prior_upper = np.quantile(prior_array, [lower_level, upper_level])
            prior_width = float(prior_upper - prior_lower)
            if prior_width <= 0:
                raise ValueError("The prior interval width must be positive.")
            row["prior_width"] = prior_width
            row["mean_width_over_prior"] = float(np.mean(width) / prior_width)
            row["mean_quantile_contraction"] = float(1.0 - np.mean(width) / prior_width)
        rows.append(row)

    return pd.DataFrame(rows)


def quantile_crossing_rate(quantiles: np.ndarray) -> float:
    """Return the fraction of cases with at least one decreasing quantile."""

    quantile_array = np.asarray(quantiles, dtype=float)
    if quantile_array.ndim == 3 and quantile_array.shape[-1] == 1:
        quantile_array = quantile_array[..., 0]
    if quantile_array.ndim != 2:
        raise ValueError("quantiles must have shape (n_cases, n_levels).")
    return float(np.mean(np.any(np.diff(quantile_array, axis=1) < 0.0, axis=1)))


def age_binned_diagnostics(
    truth: np.ndarray | Sequence[float],
    point_estimate: np.ndarray | Sequence[float],
    quantiles: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
    bins: np.ndarray | Sequence[float],
    interval_level: float = 0.8,
) -> pd.DataFrame:
    """Summarize local recovery, interval coverage, and width by true-age bin."""

    truth_array, quantile_array, levels = validate_quantile_predictions(truth, quantiles, quantile_levels)
    estimate_array = _as_1d(point_estimate, "point_estimate")
    if estimate_array.shape != truth_array.shape:
        raise ValueError("point_estimate and truth must have the same shape.")

    lower_level = (1.0 - interval_level) / 2.0
    upper_level = 1.0 - lower_level
    lower_index = np.flatnonzero(np.isclose(levels, lower_level))
    upper_index = np.flatnonzero(np.isclose(levels, upper_level))
    if lower_index.size != 1 or upper_index.size != 1:
        raise ValueError(
            f"Central {interval_level:.0%} interval requires quantile levels {lower_level:g} and {upper_level:g}."
        )

    lower = quantile_array[:, lower_index[0]]
    upper = quantile_array[:, upper_index[0]]
    error = estimate_array - truth_array
    frame = pd.DataFrame(
        {
            "truth": truth_array,
            "point_estimate": estimate_array,
            "error": error,
            "absolute_error": np.abs(error),
            "squared_error": error**2,
            "interval_width": upper - lower,
            "covered": (truth_array >= lower) & (truth_array <= upper),
            "age_bin": pd.cut(truth_array, bins=np.asarray(bins, dtype=float), include_lowest=True),
        }
    )
    grouped = frame.groupby("age_bin", observed=True)
    return grouped.agg(
        n_cases=("truth", "size"),
        truth_mean=("truth", "mean"),
        estimate_mean=("point_estimate", "mean"),
        bias=("error", "mean"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        empirical_coverage=("covered", "mean"),
        mean_interval_width=("interval_width", "mean"),
    ).reset_index()


def sample_from_quantile_grid(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray | Sequence[float],
    num_samples: int,
    seed: int = 1234,
) -> np.ndarray:
    """Draw approximate samples by linearly interpolating predicted quantiles.

    Sampling is restricted to the supplied quantile-level range; this function
    does not invent unobserved posterior tails.  It is intended only as an
    explicit bridge for toy posterior-predictive checks of quantile networks.
    """

    quantile_array = np.asarray(quantiles, dtype=float)
    levels = _as_1d(quantile_levels, "quantile_levels")
    if quantile_array.ndim == 3 and quantile_array.shape[-1] == 1:
        quantile_array = quantile_array[..., 0]
    if quantile_array.ndim != 2 or quantile_array.shape[1] != levels.size:
        raise ValueError("quantiles must have shape (n_cases, n_levels).")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile_levels must be strictly increasing.")
    if np.any(np.diff(quantile_array, axis=1) < 0.0):
        raise ValueError("Cannot interpolate crossed quantiles.")

    rng = np.random.default_rng(seed)
    probabilities = rng.uniform(levels[0], levels[-1], size=(quantile_array.shape[0], num_samples))
    samples = np.empty_like(probabilities)
    for case_index in range(quantile_array.shape[0]):
        samples[case_index] = np.interp(probabilities[case_index], levels, quantile_array[case_index])
    return samples


def continuous_predictive_metrics(
    observed: np.ndarray,
    posterior_predictive: np.ndarray,
    feature_names: Sequence[str] | None = None,
    interval_level: float = 0.8,
) -> pd.DataFrame:
    """Evaluate continuous posterior predictives against one observation per case."""

    observed_array = np.asarray(observed, dtype=float)
    predictive_array = np.asarray(posterior_predictive, dtype=float)
    if observed_array.ndim == 1:
        observed_array = observed_array[:, None]
    if predictive_array.ndim == 2:
        predictive_array = predictive_array[:, :, None]
    if observed_array.ndim != 2 or predictive_array.ndim != 3:
        raise ValueError("Expected observed (n_cases, n_features) and predictive (n_cases, n_draws, n_features).")
    if observed_array.shape[0] != predictive_array.shape[0] or observed_array.shape[1] != predictive_array.shape[2]:
        raise ValueError("Observed and posterior-predictive case/feature dimensions differ.")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie strictly between zero and one.")

    n_features = observed_array.shape[1]
    names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length does not match the feature dimension.")

    alpha = (1.0 - interval_level) / 2.0
    predictive_median = np.median(predictive_array, axis=1)
    lower = np.quantile(predictive_array, alpha, axis=1)
    upper = np.quantile(predictive_array, 1.0 - alpha, axis=1)
    rows = []
    for feature_index, feature_name in enumerate(names):
        residual = predictive_median[:, feature_index] - observed_array[:, feature_index]
        observed_std = float(np.std(observed_array[:, feature_index], ddof=1))
        rmse = float(np.sqrt(np.mean(residual**2)))
        rows.append(
            {
                "feature": feature_name,
                "mae": float(np.mean(np.abs(residual))),
                "rmse": rmse,
                "observed_std_normalized_rmse": rmse / observed_std if observed_std > 0 else np.nan,
                "predictive_coverage": float(
                    np.mean(
                        (observed_array[:, feature_index] >= lower[:, feature_index])
                        & (observed_array[:, feature_index] <= upper[:, feature_index])
                    )
                ),
                "mean_predictive_width": float(np.mean(upper[:, feature_index] - lower[:, feature_index])),
            }
        )
    return pd.DataFrame(rows)


def binary_predictive_metrics(
    observed: np.ndarray,
    posterior_predictive: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Brier and log scores for binary posterior predictives."""

    observed_array = np.asarray(observed, dtype=float)
    predictive_array = np.asarray(posterior_predictive, dtype=float)
    if observed_array.ndim == 1:
        observed_array = observed_array[:, None]
    if predictive_array.ndim == 2:
        predictive_array = predictive_array[:, :, None]
    if observed_array.ndim != 2 or predictive_array.ndim != 3:
        raise ValueError("Expected observed (n_cases, n_features) and predictive (n_cases, n_draws, n_features).")
    if observed_array.shape[0] != predictive_array.shape[0] or observed_array.shape[1] != predictive_array.shape[2]:
        raise ValueError("Observed and posterior-predictive case/feature dimensions differ.")

    n_features = observed_array.shape[1]
    names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length does not match the feature dimension.")

    probability = np.clip(np.mean(predictive_array, axis=1), 1e-8, 1.0 - 1e-8)
    rows = []
    for feature_index, feature_name in enumerate(names):
        outcome = observed_array[:, feature_index]
        predicted = probability[:, feature_index]
        rows.append(
            {
                "feature": feature_name,
                "brier_score": float(np.mean((predicted - outcome) ** 2)),
                "log_score": float(-np.mean(outcome * np.log(predicted) + (1.0 - outcome) * np.log(1.0 - predicted))),
                "observed_prevalence": float(np.mean(outcome)),
                "mean_predicted_probability": float(np.mean(predicted)),
            }
        )
    return pd.DataFrame(rows)
