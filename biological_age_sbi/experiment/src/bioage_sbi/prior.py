"""Prior component for the empirical biological-age simulator."""

from __future__ import annotations

import numpy as np


def make_prior(empirical_model: dict, rng: np.random.Generator):
    """Create a BayesFlow-compatible prior function."""

    probs = np.asarray(empirical_model["prior"]["probs"], dtype=np.float64)
    values = np.asarray(empirical_model["prior"]["values"], dtype=np.float64)

    def prior():
        biological_age = np.interp(rng.uniform(), probs, values)
        return {"biological_age": np.float32(biological_age)}

    return prior
