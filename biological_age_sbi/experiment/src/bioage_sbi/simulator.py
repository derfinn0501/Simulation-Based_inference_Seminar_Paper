"""Simulator assembly helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bioindicator_model import make_bioindicator_model
from .empirical_model import load_empirical_model
from .observation_model import make_observation_model
from .prior import make_prior


def make_component_functions(empirical_model: dict, seed: int = 1234):
    """Return prior, bioindicator, and observation functions sharing one RNG."""

    rng = np.random.default_rng(seed)
    prior = make_prior(empirical_model, rng)
    bioindicator_model = make_bioindicator_model(empirical_model, rng)
    observation_model = make_observation_model(empirical_model, rng)
    return prior, bioindicator_model, observation_model


def make_bayesflow_simulator(empirical_model: dict, seed: int = 1234):
    """Create a BayesFlow simulator from the fitted empirical model."""

    import bayesflow as bf

    return bf.make_simulator(list(make_component_functions(empirical_model, seed=seed)))


def sample_component_model(empirical_model: dict, num_samples: int, seed: int = 1234) -> pd.DataFrame:
    """Sample the components directly without constructing a BayesFlow workflow."""

    prior, bioindicator_model, observation_model = make_component_functions(empirical_model, seed=seed)
    true_key_by_column = empirical_model.get("true_key_by_column", {})
    true_keys = [true_key_by_column.get(col, f"true_{col}") for col in empirical_model["columns"]]
    rows = []
    for _ in range(num_samples):
        prior_draw = prior()
        true_indicators = bioindicator_model(**prior_draw)
        observation_inputs = {key: true_indicators[key] for key in true_keys}
        observations = observation_model(**observation_inputs)
        rows.append({**prior_draw, **true_indicators, **observations})
    return pd.DataFrame(rows)


def load_model_and_make_simulator(model_path: Path, seed: int = 1234):
    empirical_model = load_empirical_model(model_path)
    return empirical_model, make_bayesflow_simulator(empirical_model, seed=seed)
