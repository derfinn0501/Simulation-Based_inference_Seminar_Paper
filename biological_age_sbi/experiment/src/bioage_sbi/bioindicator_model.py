"""Bioindicator component for the empirical biological-age simulator."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from .config import TRUE_KEY_BY_COLUMN
from .empirical_model import feature_vector_from_state


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _age_z(state: dict[str, float], stats: dict[str, float]) -> float:
    return (state["biological_age"] - stats["age_mean"]) / stats["age_std"]


def _latent_adjustment(loadings: dict[str, float], state: dict[str, float]) -> float:
    return float(sum(float(weight) * state[name] for name, weight in loadings.items()))


def _scaled_coefficients(response: str, spec: dict, continuous_calibration: dict) -> np.ndarray:
    coefficients = np.asarray(spec["coefficients"], dtype=np.float64).copy()
    if not continuous_calibration.get("enabled", False):
        return coefficients

    effect_scales = continuous_calibration.get("effect_scales", {})
    if response == "sbp" and "bmi_z" in spec["features"]:
        bmi_index = spec["features"].index("bmi_z")
        coefficients[bmi_index] *= float(effect_scales.get("sbp_bmi", 1.0))

    return coefficients


def _sample_copula_uniforms(dependence_model: dict, rng: np.random.Generator) -> dict[str, float]:
    if not dependence_model.get("enabled", False):
        return {}

    columns = dependence_model["columns"]
    correlation = np.asarray(dependence_model["correlation"], dtype=np.float64)
    normal_draw = rng.multivariate_normal(np.zeros(len(columns)), correlation)
    uniforms = np.clip(ndtr(normal_draw), 1e-6, 1.0 - 1e-6)
    return dict(zip(columns, uniforms.astype(float), strict=True))


def _sample_latent_factors(
    latent_config: dict,
    state: dict[str, float],
    stats: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    if not latent_config.get("enabled", False):
        return {}

    latents = {}
    for name, spec in latent_config["factors"].items():
        mean = 0.0
        for term, weight in spec["mean_terms"].items():
            if term == "age_z":
                value = _age_z(state, stats)
            else:
                value = latents[term]
            mean += float(weight) * value
        latents[name] = float(rng.normal(mean, float(spec["std"])))
    return latents


def _age_bin_index(age: float, age_bin_edges: list[float]) -> int | None:
    edges = np.asarray(age_bin_edges, dtype=np.float64)
    if len(edges) < 2 or age < edges[0] or age > edges[-1]:
        return None
    index = int(np.searchsorted(edges, age, side="right") - 1)
    return min(index, len(edges) - 2)


def _empirical_residual(
    response: str,
    biological_age: float,
    continuous_calibration: dict,
    rng: np.random.Generator,
    uniform: float | None = None,
) -> float | None:
    if not continuous_calibration.get("enabled", False):
        return None
    if not continuous_calibration.get("empirical_residual_bootstrap_enabled", False):
        return None

    column_calibration = continuous_calibration.get("columns", {}).get(response)
    if not column_calibration:
        return None

    min_bin_residuals = int(continuous_calibration.get("min_bin_residuals", 30))
    residuals = None
    bin_index = _age_bin_index(biological_age, continuous_calibration.get("age_bin_edges", []))
    if bin_index is not None:
        residuals_by_age_bin = column_calibration.get("residuals_by_age_bin", [])
        if bin_index < len(residuals_by_age_bin):
            bin_residuals = residuals_by_age_bin[bin_index]
            if len(bin_residuals) >= min_bin_residuals:
                residuals = np.asarray(bin_residuals, dtype=np.float64)

    if residuals is None:
        residuals = np.asarray(column_calibration["all_residuals"], dtype=np.float64)

    if len(residuals) == 0:
        return None
    if uniform is None:
        return float(rng.choice(residuals))
    return float(np.quantile(residuals, np.clip(uniform, 1e-6, 1.0 - 1e-6)))


def _age_bin_mean_adjustment(response: str, biological_age: float, continuous_calibration: dict) -> float:
    if not continuous_calibration.get("enabled", False):
        return 0.0

    adjustment_config = continuous_calibration.get("age_bin_mean_adjustment", {})
    if not adjustment_config.get("enabled", False):
        return 0.0

    adjustments = adjustment_config.get("columns", {}).get(response)
    if adjustments is None:
        return 0.0

    bin_index = _age_bin_index(biological_age, continuous_calibration.get("age_bin_edges", []))
    if bin_index is None or bin_index >= len(adjustments):
        return 0.0
    return float(adjustments[bin_index])


def _sample_linear(
    response: str,
    spec: dict,
    state: dict[str, float],
    stats: dict[str, float],
    rng: np.random.Generator,
    latent_loadings: dict[str, float] | None = None,
    uniform: float | None = None,
    continuous_calibration: dict | None = None,
) -> float:
    features = feature_vector_from_state(state, spec["features"], stats)
    continuous_calibration = continuous_calibration or {}
    coefficients = _scaled_coefficients(response, spec, continuous_calibration)
    mean = float(features @ coefficients)
    if latent_loadings:
        mean += _latent_adjustment(latent_loadings, state)
    mean += _age_bin_mean_adjustment(response, state["biological_age"], continuous_calibration)

    residual = _empirical_residual(
        response,
        state["biological_age"],
        continuous_calibration,
        rng,
        uniform=uniform,
    )
    if residual is not None:
        value = mean + residual
    elif uniform is None:
        value = rng.normal(mean, float(spec["residual_std"]))
    else:
        value = mean + float(ndtri(np.clip(uniform, 1e-6, 1.0 - 1e-6))) * float(spec["residual_std"])
    lower, upper = spec["clip"]
    return float(np.clip(value, lower, upper))


def _sample_binary(
    spec: dict,
    state: dict[str, float],
    stats: dict[str, float],
    rng: np.random.Generator,
    latent_loadings: dict[str, float] | None = None,
    uniform: float | None = None,
) -> float:
    features = feature_vector_from_state(state, spec["features"], stats)
    coefficients = np.asarray(spec["coefficients"], dtype=np.float64)
    logit = float(features @ coefficients)
    if latent_loadings:
        logit += _latent_adjustment(latent_loadings, state)
    probability = _sigmoid(logit)
    lower, upper = spec["prob_clip"]
    probability = float(np.clip(probability, lower, upper))
    if uniform is not None:
        return float(float(uniform) >= 1.0 - probability)
    return float(rng.binomial(1, probability))


def make_bioindicator_model(empirical_model: dict, rng: np.random.Generator):
    """Create the sequential conditional bioindicator model.

    The generated order is important: later variables can depend on earlier
    generated variables, which induces interdependence without hidden latent
    factors.
    """

    stats = empirical_model["standardization"]
    continuous_models = empirical_model["continuous_models"]
    binary_models = empirical_model["binary_models"]
    continuous_columns = empirical_model.get("continuous_columns", list(continuous_models))
    binary_columns = empirical_model.get("binary_columns", list(binary_models))
    true_key_by_column = empirical_model.get(
        "true_key_by_column",
        {col: TRUE_KEY_BY_COLUMN[col] for col in [*continuous_columns, *binary_columns]},
    )
    dependence_model = empirical_model.get("dependence_model", {"enabled": False})
    latent_config = empirical_model.get("latent_factors", {"enabled": False})
    continuous_latent_loadings = latent_config.get("continuous_loadings", {})
    binary_latent_loadings = latent_config.get("binary_logit_loadings", {})
    continuous_calibration = empirical_model.get("calibration", {}).get("continuous", {})

    def bioindicator_model(biological_age):
        state = {"biological_age": float(np.asarray(biological_age).squeeze())}
        state.update(_sample_latent_factors(latent_config, state, stats, rng))
        copula_uniforms = _sample_copula_uniforms(dependence_model, rng)

        for col in continuous_columns:
            state[col] = _sample_linear(
                col,
                continuous_models[col],
                state,
                stats,
                rng,
                latent_loadings=continuous_latent_loadings.get(col),
                uniform=copula_uniforms.get(col),
                continuous_calibration=continuous_calibration,
            )

        for col in binary_columns:
            state[col] = _sample_binary(
                binary_models[col],
                state,
                stats,
                rng,
                latent_loadings=binary_latent_loadings.get(col),
                uniform=copula_uniforms.get(col),
            )

        latent_output = {
            f"latent_{name}": np.float32(state[name])
            for name in latent_config.get("factors", {})
            if name in state
        }
        true_output = {
            true_key_by_column[col]: np.float32(state[col])
            for col in [*continuous_columns, *binary_columns]
        }
        return {**latent_output, **true_output}

    return bioindicator_model
