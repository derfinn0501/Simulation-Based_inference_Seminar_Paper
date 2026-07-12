"""Fit and serialize the empirical non-lab biological-age simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from sklearn.linear_model import LogisticRegression

from .config import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    DEFAULT_FEATURE_SET_NAME,
    FEATURE_SETS,
    MODEL_NAME,
    OBSERVED_KEY_BY_COLUMN,
)


def _safe_std(values: pd.Series | np.ndarray, min_std: float = 1e-3) -> float:
    std = float(np.nanstd(values, ddof=1))
    return max(std, min_std)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _clip_unit_interval(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(values, eps, 1.0 - eps)


def _standardization_stats(frame: pd.DataFrame, continuous_columns: list[str]) -> dict[str, float]:
    stats = {
        "age_mean": float(frame["biological_age"].mean()),
        "age_std": _safe_std(frame["biological_age"]),
    }
    for col in continuous_columns:
        stats[f"{col}_mean"] = float(frame[col].mean())
        stats[f"{col}_std"] = _safe_std(frame[col])
    return stats


def _feature_series(frame: pd.DataFrame, feature: str, stats: dict[str, float]) -> np.ndarray:
    if feature == "intercept":
        return np.ones(len(frame), dtype=np.float64)
    if feature == "age_z":
        return ((frame["biological_age"] - stats["age_mean"]) / stats["age_std"]).to_numpy(np.float64)
    if feature == "age_z2":
        age_z = (frame["biological_age"] - stats["age_mean"]) / stats["age_std"]
        return np.square(age_z).to_numpy(np.float64)
    if feature.endswith("_z"):
        col = feature.removesuffix("_z")
        return ((frame[col] - stats[f"{col}_mean"]) / stats[f"{col}_std"]).to_numpy(np.float64)
    if feature in frame:
        return frame[feature].to_numpy(np.float64)
    raise KeyError(f"Unknown feature: {feature}")


def design_matrix(frame: pd.DataFrame, features: list[str], stats: dict[str, float]) -> np.ndarray:
    return np.column_stack([_feature_series(frame, feature, stats) for feature in features])


def feature_vector_from_state(state: dict[str, float], features: list[str], stats: dict[str, float]) -> np.ndarray:
    values = []
    for feature in features:
        if feature == "intercept":
            values.append(1.0)
        elif feature == "age_z":
            values.append((state["biological_age"] - stats["age_mean"]) / stats["age_std"])
        elif feature == "age_z2":
            age_z = (state["biological_age"] - stats["age_mean"]) / stats["age_std"]
            values.append(age_z * age_z)
        elif feature.endswith("_z"):
            col = feature.removesuffix("_z")
            values.append((state[col] - stats[f"{col}_mean"]) / stats[f"{col}_std"])
        elif feature in state:
            values.append(state[feature])
        else:
            raise KeyError(f"Unknown feature: {feature}")
    return np.asarray(values, dtype=np.float64)


def _fit_linear(frame: pd.DataFrame, response: str, features: list[str], stats: dict[str, float]) -> dict[str, Any]:
    x = design_matrix(frame, features, stats)
    y = frame[response].to_numpy(np.float64)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    return {
        "response": response,
        "features": features,
        "coefficients": coefficients.tolist(),
        "residual_std": _safe_std(residuals),
        "clip": np.quantile(y, [0.005, 0.995]).astype(float).tolist(),
    }


def _fit_logistic(frame: pd.DataFrame, response: str, features: list[str], stats: dict[str, float]) -> dict[str, Any]:
    x = design_matrix(frame, features, stats)
    y = frame[response].to_numpy(np.int64)

    if len(np.unique(y)) != 2:
        raise ValueError(f"Binary response {response!r} does not contain both classes.")

    model = LogisticRegression(
        fit_intercept=False,
        C=50.0,
        solver="lbfgs",
        max_iter=2_000,
    )
    model.fit(x, y)
    return {
        "response": response,
        "features": features,
        "coefficients": model.coef_.reshape(-1).astype(float).tolist(),
        "prob_clip": [0.001, 0.999],
    }


def _linear_mean(frame: pd.DataFrame, spec: dict[str, Any], stats: dict[str, float]) -> np.ndarray:
    x = design_matrix(frame, spec["features"], stats)
    coefficients = np.asarray(spec["coefficients"], dtype=np.float64)
    return x @ coefficients


def _binary_probability(frame: pd.DataFrame, spec: dict[str, Any], stats: dict[str, float]) -> np.ndarray:
    x = design_matrix(frame, spec["features"], stats)
    coefficients = np.asarray(spec["coefficients"], dtype=np.float64)
    lower, upper = spec["prob_clip"]
    return np.clip(_sigmoid_array(x @ coefficients), lower, upper)


def _nearest_correlation(matrix: np.ndarray, min_eigenvalue: float = 1e-4) -> tuple[np.ndarray, float]:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    min_before = float(eigenvalues.min())
    clipped = np.clip(eigenvalues, min_eigenvalue, None)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired, min_before


def _fit_gaussian_copula(
    frame: pd.DataFrame,
    continuous_models: dict[str, dict[str, Any]],
    binary_models: dict[str, dict[str, Any]],
    stats: dict[str, float],
    continuous_columns: list[str],
    binary_columns: list[str],
    seed: int = 2026,
) -> dict[str, Any]:
    """Fit a conditional Gaussian copula on marginal PIT residuals.

    Continuous variables use the fitted normal residual CDF. Binary variables
    use randomized PIT intervals so their Bernoulli marginals remain intact.
    """

    rng = np.random.default_rng(seed)
    columns = [*continuous_columns, *binary_columns]
    pit_columns = []

    for col in continuous_columns:
        spec = continuous_models[col]
        mean = _linear_mean(frame, spec, stats)
        residual_z = (frame[col].to_numpy(np.float64) - mean) / float(spec["residual_std"])
        pit_columns.append(_clip_unit_interval(ndtr(residual_z)))

    for col in binary_columns:
        spec = binary_models[col]
        probability = _binary_probability(frame, spec, stats)
        observed = frame[col].to_numpy(np.float64)
        lower = np.where(observed >= 0.5, 1.0 - probability, 0.0)
        upper = np.where(observed >= 0.5, 1.0, 1.0 - probability)
        randomized_pit = lower + rng.uniform(size=len(frame)) * (upper - lower)
        pit_columns.append(_clip_unit_interval(randomized_pit))

    z_values = ndtri(np.column_stack(pit_columns))
    correlation, min_eigenvalue_before = _nearest_correlation(np.corrcoef(z_values, rowvar=False))

    return {
        "enabled": True,
        "type": "conditional_gaussian_copula",
        "columns": columns,
        "correlation": correlation.astype(float).tolist(),
        "pit_seed": seed,
        "min_eigenvalue_before_repair": min_eigenvalue_before,
        "fit_note": (
            "Correlation estimated from PIT-transformed conditional residuals. "
            "Continuous variables use normal residual PIT; binary variables use randomized Bernoulli PIT."
        ),
    }


def _adapter_statistics(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    binary_columns: list[str],
    observed_key_by_column: dict[str, str],
) -> dict[str, list[float]]:
    columns = [*continuous_columns, *binary_columns]
    means = []
    stds = []
    for col in columns:
        means.append(float(frame[col].mean()))
        stds.append(_safe_std(frame[col], min_std=0.05))

    return {
        "condition_keys": [observed_key_by_column[col] for col in columns],
        "mean": means,
        "std": stds,
    }


def _feature_is_available(feature: str, continuous_columns: list[str], binary_columns: list[str]) -> bool:
    if feature in {"intercept", "age_z", "age_z2"}:
        return True
    if feature.endswith("_z"):
        return feature.removesuffix("_z") in continuous_columns
    return feature in binary_columns


def _features_for_response(
    response: str,
    configured_features: dict[str, list[str]],
    continuous_columns: list[str],
    binary_columns: list[str],
) -> list[str]:
    features = configured_features.get(response)
    if features and all(_feature_is_available(feature, continuous_columns, binary_columns) for feature in features):
        return features
    return ["intercept", "age_z", "age_z2"]


def _continuous_observation_noise(frame: pd.DataFrame, continuous_columns: list[str]) -> dict[str, float]:
    noise = {}
    for col in continuous_columns:
        if col == "bmi":
            noise[col] = 0.40
        elif col == "sbp":
            noise[col] = 3.00
        else:
            noise[col] = max(0.05 * _safe_std(frame[col]), 1e-3)
    return noise


def _continuous_calibration(
    frame: pd.DataFrame,
    continuous_models: dict[str, dict[str, Any]],
    stats: dict[str, float],
    continuous_columns: list[str],
) -> dict[str, Any]:
    """Store empirical residual pools for later simulator calibration.

    The calibration is serialized but disabled by default. Notebooks can turn it
    on explicitly when testing non-Gaussian shape and age-bin residual structure.
    """

    age_bin_edges = np.arange(40.0, 101.0, 10.0)
    calibration_by_column = {}

    for col in continuous_columns:
        spec = continuous_models[col]
        mean = _linear_mean(frame, spec, stats)
        residuals = frame[col].to_numpy(np.float64) - mean
        biological_age = frame["biological_age"].to_numpy(np.float64)

        residuals_by_age_bin = []
        bin_counts = []
        bin_means = []
        bin_stds = []

        for lower, upper in zip(age_bin_edges[:-1], age_bin_edges[1:], strict=True):
            if upper == age_bin_edges[-1]:
                mask = (biological_age >= lower) & (biological_age <= upper)
            else:
                mask = (biological_age >= lower) & (biological_age < upper)
            bin_residuals = residuals[mask]
            residuals_by_age_bin.append(bin_residuals.astype(float).tolist())
            bin_counts.append(int(mask.sum()))
            bin_means.append(float(np.mean(bin_residuals)) if len(bin_residuals) else 0.0)
            bin_stds.append(_safe_std(bin_residuals) if len(bin_residuals) > 1 else float(spec["residual_std"]))

        calibration_by_column[col] = {
            "all_residuals": residuals.astype(float).tolist(),
            "residuals_by_age_bin": residuals_by_age_bin,
            "bin_counts": bin_counts,
            "bin_mean_residual": bin_means,
            "bin_residual_std": bin_stds,
        }

    return {
        "enabled": False,
        "empirical_residual_bootstrap_enabled": False,
        "age_bin_edges": age_bin_edges.astype(float).tolist(),
        "min_bin_residuals": 30,
        "columns": calibration_by_column,
        "effect_scales": {
            "sbp_bmi": 1.0,
        },
        "age_bin_mean_adjustment": {
            "enabled": False,
            "columns": {},
        },
        "fit_note": (
            "Continuous residual pools are estimated from fitted conditional models. "
            "When enabled, the simulator samples residuals from the matching biological-age bin "
            "instead of using Gaussian residual noise."
        ),
    }


def fit_empirical_model(
    frame: pd.DataFrame,
    model_name: str = MODEL_NAME,
    feature_set_name: str | None = None,
) -> dict[str, Any]:
    """Fit a sequential biological-age simulator model from a prepared Mendeley frame."""

    if feature_set_name is None:
        feature_set_name = model_name if model_name in FEATURE_SETS else DEFAULT_FEATURE_SET_NAME
    if feature_set_name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set_name!r}")

    feature_set = FEATURE_SETS[feature_set_name]
    continuous_columns = list(feature_set.continuous_columns)
    binary_columns = list(feature_set.binary_columns)
    columns = [*continuous_columns, *binary_columns]
    observed_key_by_column = {col: OBSERVED_KEY_BY_COLUMN[col] for col in columns}

    stats = _standardization_stats(frame, continuous_columns)
    prior_probs = np.linspace(0.0, 1.0, 201)
    prior_values = np.quantile(frame["biological_age"], prior_probs).astype(float)

    continuous_models = {
        response: _fit_linear(
            frame,
            response,
            _features_for_response(response, CONTINUOUS_FEATURES, continuous_columns, binary_columns),
            stats,
        )
        for response in continuous_columns
    }
    binary_models = {
        response: _fit_logistic(
            frame,
            response,
            _features_for_response(response, BINARY_FEATURES, continuous_columns, binary_columns),
            stats,
        )
        for response in binary_columns
    }
    dependence_model = _fit_gaussian_copula(
        frame,
        continuous_models=continuous_models,
        binary_models=binary_models,
        stats=stats,
        continuous_columns=continuous_columns,
        binary_columns=binary_columns,
    )

    return {
        "version": 2,
        "model_name": model_name,
        "feature_set_name": feature_set.name,
        "feature_set_description": feature_set.description,
        "model_family": "sequential_conditionals_with_gaussian_copula",
        "description": (
            f"{feature_set.description} Interdependence is modeled through conditional "
            "regressions, a small age-dependent latent risk layer, and a conditional "
            "Gaussian copula over marginal residuals."
        ),
        "columns": columns,
        "continuous_columns": continuous_columns,
        "binary_columns": binary_columns,
        "observed_key_by_column": observed_key_by_column,
        "true_key_by_column": {col: f"true_{col}" for col in columns},
        "standardization": stats,
        "prior": {
            "type": "empirical_quantile",
            "probs": prior_probs.astype(float).tolist(),
            "values": prior_values.tolist(),
            "min": float(frame["biological_age"].min()),
            "max": float(frame["biological_age"].max()),
        },
        "continuous_models": continuous_models,
        "binary_models": binary_models,
        "dependence_model": dependence_model,
        "calibration": {
            "continuous": _continuous_calibration(frame, continuous_models, stats, continuous_columns),
        },
        "latent_factors": {
            "enabled": True,
            "factors": {
                "metabolic_risk": {
                    "mean_terms": {"age_z": 0.35},
                    "std": 0.80,
                },
                "cardiovascular_risk": {
                    "mean_terms": {"age_z": 0.40, "metabolic_risk": 0.35},
                    "std": 0.75,
                },
                "joint_burden": {
                    "mean_terms": {"age_z": 0.45, "metabolic_risk": 0.20},
                    "std": 0.80,
                },
                "behavior_risk": {
                    "mean_terms": {"age_z": -0.15},
                    "std": 0.85,
                },
            },
            "continuous_loadings": {
                "bmi": {"metabolic_risk": 1.40},
                "sbp": {"cardiovascular_risk": 4.00, "metabolic_risk": 1.20},
            },
            "binary_logit_loadings": {
                "smoke": {"behavior_risk": 0.35},
                "drink": {"behavior_risk": 0.30},
                "hypertension": {"cardiovascular_risk": 0.45},
                "diabetes": {"metabolic_risk": 0.55},
                "cvd": {"cardiovascular_risk": 0.45},
                "arthritis": {"joint_burden": 0.45},
                "koa": {"joint_burden": 0.55},
            },
        },
        "observation_model": {
            "continuous_noise_std": _continuous_observation_noise(frame, continuous_columns),
            "binary_flip_probability": 0.00,
        },
        "adapter": _adapter_statistics(frame, continuous_columns, binary_columns, observed_key_by_column),
        "fit_summary": {
            "n_train": int(len(frame)),
            "biological_age_mean": float(frame["biological_age"].mean()),
            "biological_age_std": _safe_std(frame["biological_age"]),
            "continuous_columns": continuous_columns,
            "binary_columns": binary_columns,
            "binary_prevalence": {
                col: float(frame[col].mean())
                for col in binary_columns
            },
        },
    }


def save_empirical_model(model: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def load_empirical_model(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
