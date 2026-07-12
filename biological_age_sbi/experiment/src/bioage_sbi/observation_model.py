"""Observation component for the empirical biological-age simulator."""

from __future__ import annotations

import numpy as np

from .config import OBSERVED_KEY_BY_COLUMN, TRUE_KEY_BY_COLUMN


def _maybe_flip_binary(value: float, flip_probability: float, rng: np.random.Generator) -> float:
    if flip_probability <= 0:
        return value
    if rng.uniform() < flip_probability:
        return 1.0 - value
    return value


def make_observation_model(empirical_model: dict, rng: np.random.Generator):
    """Create a BayesFlow-compatible observation model."""

    noise_std = empirical_model["observation_model"]["continuous_noise_std"]
    flip_probability = float(empirical_model["observation_model"]["binary_flip_probability"])
    continuous_columns = empirical_model.get("continuous_columns", list(empirical_model["continuous_models"]))
    binary_columns = empirical_model.get("binary_columns", list(empirical_model["binary_models"]))
    observed_key_by_column = empirical_model.get(
        "observed_key_by_column",
        {col: OBSERVED_KEY_BY_COLUMN[col] for col in [*continuous_columns, *binary_columns]},
    )
    true_key_by_column = empirical_model.get(
        "true_key_by_column",
        {col: TRUE_KEY_BY_COLUMN[col] for col in [*continuous_columns, *binary_columns]},
    )
    clips = {
        col: empirical_model["continuous_models"][col]["clip"]
        for col in continuous_columns
    }

    def observation_model(**kwargs):
        true_values = {
            col: float(np.asarray(kwargs[true_key_by_column[col]]).squeeze())
            for col in [*continuous_columns, *binary_columns]
        }

        observed = {}
        for col in continuous_columns:
            lower, upper = clips[col]
            value = rng.normal(true_values[col], float(noise_std[col]))
            observed[observed_key_by_column[col]] = np.float32(np.clip(value, lower, upper))

        for col in binary_columns:
            value = _maybe_flip_binary(true_values[col], flip_probability, rng)
            observed[observed_key_by_column[col]] = np.float32(value)

        return observed

    return observation_model
