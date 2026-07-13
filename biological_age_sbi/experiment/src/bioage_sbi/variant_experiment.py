"""Run config-driven biological-age diagnostic variants.

This module keeps simulator mutation, BayesFlow training, held-out evaluation,
and artifact creation out of the comparison notebook.  Each run uses separate
simulator instances (and therefore separate RNG streams) for training,
validation, calibration, and posterior-predictive simulation.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import (
    age_binned_diagnostics,
    binary_predictive_metrics,
    continuous_predictive_metrics,
    interval_diagnostics,
    point_accuracy_metrics,
    quantile_calibration_table,
    quantile_crossing_rate,
    sample_from_quantile_grid,
)
from .empirical_model import load_empirical_model
from .simulator import make_bayesflow_simulator, make_component_functions


def load_variant_config(path: str | Path) -> dict[str, Any]:
    """Load one diagnostic-variant JSON config."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_variant_config(empirical_model: dict[str, Any], simulator_config: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit structural and noise switches to a fitted empirical model."""

    model = copy.deepcopy(empirical_model)
    bioindicator_config = simulator_config["bioindicator_model"]
    observation_config = simulator_config["observation_model"]

    residual_distribution = bioindicator_config["continuous_residual_distribution"]
    if residual_distribution != "gaussian":
        raise ValueError(
            "The current comparison runner supports gaussian continuous residuals only; "
            f"got {residual_distribution!r}."
        )

    model.setdefault("dependence_model", {})["enabled"] = bool(bioindicator_config["copula_enabled"])
    latent_config = model.setdefault("latent_factors", {})
    latent_config["enabled"] = bool(bioindicator_config["latent_factors_enabled"])
    # The component model reads loading maps independently of the sampling
    # switch, so remove them as well for a genuinely latent-free ablation.
    if not latent_config["enabled"]:
        latent_config["continuous_loadings"] = {}
        latent_config["binary_logit_loadings"] = {}

    continuous_calibration = model.setdefault("calibration", {}).setdefault("continuous", {})
    continuous_calibration["enabled"] = bool(bioindicator_config["continuous_calibration_enabled"])
    continuous_calibration["empirical_residual_bootstrap_enabled"] = False
    if continuous_calibration["enabled"]:
        raise ValueError("Gaussian comparison variants must keep continuous calibration disabled.")

    residual_scale = float(bioindicator_config["continuous_residual_noise_scale"])
    latent_scale = float(bioindicator_config["latent_std_scale"])
    observation_scale = float(observation_config["continuous_noise_scale"])
    if min(residual_scale, latent_scale, observation_scale) < 0.0:
        raise ValueError("Noise scales must be non-negative.")

    for spec in model["continuous_models"].values():
        spec["residual_std"] = float(spec["residual_std"] * residual_scale)
    for spec in model.get("latent_factors", {}).get("factors", {}).values():
        spec["std"] = float(spec["std"] * latent_scale)
    for key, value in model["observation_model"]["continuous_noise_std"].items():
        model["observation_model"]["continuous_noise_std"][key] = float(value * observation_scale)
    model["observation_model"]["binary_flip_probability"] = float(
        observation_config["binary_flip_probability"]
    )

    model["active_diagnostic_variant"] = {
        "name": simulator_config["name"],
        "data_source": simulator_config["data_source"]["name"],
        "copula_enabled": bool(bioindicator_config["copula_enabled"]),
        "latent_factors_enabled": bool(bioindicator_config["latent_factors_enabled"]),
        "continuous_residual_noise_scale": residual_scale,
        "latent_std_scale": latent_scale,
        "continuous_observation_noise_scale": observation_scale,
        "binary_flip_probability": float(observation_config["binary_flip_probability"]),
    }
    return model


def _project_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def _build_adapter(model: dict[str, Any]):
    import bayesflow as bf

    continuous_columns = list(model["continuous_columns"])
    binary_columns = list(model["binary_columns"])
    columns = [*continuous_columns, *binary_columns]
    true_keys = [model["true_key_by_column"][column] for column in columns]
    latent_keys = []
    if model.get("latent_factors", {}).get("enabled", False):
        latent_keys = [f"latent_{name}" for name in model["latent_factors"].get("factors", {})]

    adapter_config = model["adapter"]
    condition_mean = np.asarray(adapter_config["mean"], dtype="float32")
    condition_std = np.asarray(adapter_config["std"], dtype="float32")
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .drop([*latent_keys, *true_keys])
        .concatenate(["biological_age"], into="inference_variables")
        .concatenate(adapter_config["condition_keys"], into="inference_conditions")
        .standardize("inference_conditions", mean=condition_mean, std=condition_std)
    )


def _build_network(network_config: dict[str, Any]):
    import bayesflow as bf

    if network_config["type"] != "ScoringRuleNetwork":
        raise ValueError("The current comparison runner expects a ScoringRuleNetwork.")
    levels = np.asarray(network_config["outputs"]["quantile_levels"], dtype=float)
    network = bf.networks.ScoringRuleNetwork(
        scoring_rules={
            "mean": bf.scoring_rules.MeanScore(),
            "quantiles": bf.scoring_rules.QuantileScore(levels),
        },
        subnet=network_config["subnet"],
        subnet_kwargs={
            "widths": tuple(network_config["widths"]),
            "norm": network_config["norm"],
            "dropout": float(network_config["dropout"]),
            "residual": bool(network_config["residual"]),
        },
    )
    return network, levels


def _extract_estimates(estimates: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    block = estimates.get("biological_age", estimates.get("inference_variables"))
    if block is None:
        raise KeyError(f"Could not find biological-age estimates in keys {sorted(estimates)}.")
    point_estimate = np.asarray(block["mean"], dtype=float).squeeze()
    quantiles = np.asarray(block["quantiles"], dtype=float)
    if quantiles.ndim == 3 and quantiles.shape[-1] == 1:
        quantiles = quantiles[..., 0]
    return point_estimate, quantiles


def _history_frame(history: Any) -> pd.DataFrame:
    values = getattr(history, "history", history)
    if not isinstance(values, dict):
        return pd.DataFrame()
    return pd.DataFrame({key: np.asarray(value) for key, value in values.items()})


def _sample_prior(model: dict[str, Any], num_samples: int, seed: int) -> np.ndarray:
    prior, _, _ = make_component_functions(model, seed=seed)
    return np.asarray([prior()["biological_age"] for _ in range(num_samples)], dtype=float).squeeze()


def _posterior_predictives(
    model: dict[str, Any],
    calibration_data: dict[str, Any],
    posterior_age_draws: np.ndarray,
    num_cases: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resimulate observations for a subset of cases and posterior age draws."""

    _, bioindicator_model, observation_model = make_component_functions(model, seed=seed)
    continuous_columns = list(model["continuous_columns"])
    binary_columns = list(model["binary_columns"])
    all_columns = [*continuous_columns, *binary_columns]
    true_keys = [model["true_key_by_column"][column] for column in all_columns]
    observed_keys = model["observed_key_by_column"]
    num_cases = min(int(num_cases), posterior_age_draws.shape[0])
    case_indices = np.linspace(0, posterior_age_draws.shape[0] - 1, num_cases, dtype=int)
    selected_draws = posterior_age_draws[case_indices]

    observed_continuous = np.column_stack(
        [np.asarray(calibration_data[observed_keys[column]])[case_indices].squeeze() for column in continuous_columns]
    )
    observed_binary = np.column_stack(
        [np.asarray(calibration_data[observed_keys[column]])[case_indices].squeeze() for column in binary_columns]
    )
    ppc_continuous = np.empty((num_cases, selected_draws.shape[1], len(continuous_columns)), dtype=float)
    ppc_binary = np.empty((num_cases, selected_draws.shape[1], len(binary_columns)), dtype=float)

    for case_index in range(num_cases):
        for draw_index, age in enumerate(selected_draws[case_index]):
            true_values = bioindicator_model(biological_age=np.float32(age))
            observed_values = observation_model(**{key: true_values[key] for key in true_keys})
            ppc_continuous[case_index, draw_index] = [
                observed_values[observed_keys[column]] for column in continuous_columns
            ]
            ppc_binary[case_index, draw_index] = [
                observed_values[observed_keys[column]] for column in binary_columns
            ]

    return {
        "ppc_case_indices": case_indices,
        "observed_continuous": observed_continuous,
        "ppc_continuous": ppc_continuous,
        "continuous_feature_names": np.asarray(continuous_columns),
        "observed_binary": observed_binary,
        "ppc_binary": ppc_binary,
        "binary_feature_names": np.asarray(binary_columns),
    }


def evaluate_variant_artifact(config: dict[str, Any], artifact: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute all diagnostic tables from one saved evaluation artifact."""

    truth = np.asarray(artifact["truth"], dtype=float).squeeze()
    point_estimate = np.asarray(artifact["point_estimate"], dtype=float).squeeze()
    quantiles = np.asarray(artifact["quantiles"], dtype=float)
    quantile_levels = np.asarray(artifact["quantile_levels"], dtype=float).squeeze()
    prior_samples = np.asarray(artifact["prior_samples"], dtype=float).squeeze()
    evaluation_config = config["evaluation_config"]

    point_metrics = point_accuracy_metrics(truth, point_estimate, prior_std=float(np.std(prior_samples, ddof=1)))
    calibration = quantile_calibration_table(
        truth,
        quantiles,
        quantile_levels,
        confidence=float(evaluation_config["pointwise_confidence"]),
    )
    intervals = interval_diagnostics(
        truth,
        quantiles,
        quantile_levels,
        interval_levels=evaluation_config["interval_levels"],
        prior_samples=prior_samples,
    )
    age_summary = age_binned_diagnostics(
        truth,
        point_estimate,
        quantiles,
        quantile_levels,
        bins=evaluation_config["age_bins"],
        interval_level=0.8,
    )

    continuous_ppc = None
    if {"observed_continuous", "ppc_continuous", "continuous_feature_names"}.issubset(artifact):
        continuous_ppc = continuous_predictive_metrics(
            artifact["observed_continuous"],
            artifact["ppc_continuous"],
            feature_names=artifact["continuous_feature_names"].astype(str).tolist(),
            interval_level=0.8,
        )
    binary_ppc = None
    if {"observed_binary", "ppc_binary", "binary_feature_names"}.issubset(artifact):
        binary_ppc = binary_predictive_metrics(
            artifact["observed_binary"],
            artifact["ppc_binary"],
            feature_names=artifact["binary_feature_names"].astype(str).tolist(),
        )

    return {
        "config": config,
        "artifact": artifact,
        "truth": truth,
        "point_estimate": point_estimate,
        "quantiles": quantiles,
        "quantile_levels": quantile_levels,
        "prior_samples": prior_samples,
        "point_metrics": point_metrics,
        "calibration": calibration,
        "intervals": intervals,
        "age_summary": age_summary,
        "quantile_crossing_rate": quantile_crossing_rate(quantiles),
        "continuous_ppc": continuous_ppc,
        "binary_ppc": binary_ppc,
    }


def load_variant_results(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Load and evaluate an existing variant artifact."""

    project_root = Path(project_root)
    artifact_path = _project_path(project_root, config["evaluation_config"]["artifact_path"])
    artifact = dict(np.load(artifact_path, allow_pickle=False))
    result = evaluate_variant_artifact(config, artifact)
    history_path = _project_path(project_root, config["evaluation_config"]["result_dir"]) / "training_history.csv"
    result["history"] = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    result["reused_artifact"] = True
    return result


def run_diagnostic_variant(
    config: dict[str, Any],
    project_root: str | Path,
    budget_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and save one config-driven diagnostic variant."""

    import bayesflow as bf
    import keras

    project_root = Path(project_root)
    run_config = copy.deepcopy(config)
    training_config = run_config["training_config"]
    evaluation_config = run_config["evaluation_config"]
    if budget_overrides:
        for key, value in budget_overrides.items():
            if key in training_config:
                training_config[key] = int(value)
            elif key in evaluation_config:
                evaluation_config[key] = int(value)
            else:
                raise KeyError(f"Unknown budget override: {key}")
    if training_config.get("optimizer") != "bayesflow_default":
        raise ValueError("Only optimizer='bayesflow_default' is currently implemented.")

    empirical_model_path = _project_path(project_root, run_config["simulator_config"]["empirical_model_path"])
    empirical_model = load_empirical_model(empirical_model_path)
    active_model = apply_variant_config(empirical_model, run_config["simulator_config"])

    keras.backend.clear_session()
    keras.utils.set_random_seed(int(training_config["network_seed"]))
    adapter = _build_adapter(active_model)
    network, quantile_levels = _build_network(run_config["network_config"])

    training_simulator = make_bayesflow_simulator(active_model, seed=int(training_config["training_seed"]))
    validation_simulator = make_bayesflow_simulator(active_model, seed=int(training_config["validation_seed"]))
    training_data = training_simulator.sample(int(training_config["num_simulations"]))
    validation_data = validation_simulator.sample(int(training_config["num_validation_simulations"]))
    workflow = bf.BasicWorkflow(simulator=training_simulator, adapter=adapter, inference_network=network)
    history = workflow.fit_offline(
        training_data,
        epochs=int(training_config["epochs"]),
        batch_size=int(training_config["batch_size"]),
        validation_data=validation_data,
    )
    history_frame = _history_frame(history)

    checkpoint_path = _project_path(project_root, training_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    keras.saving.save_model(workflow.approximator, checkpoint_path)

    calibration_simulator = make_bayesflow_simulator(active_model, seed=int(evaluation_config["calibration_seed"]))
    calibration_data = calibration_simulator.sample(int(evaluation_config["num_calibration_cases"]))
    estimates = workflow.approximator.estimate(conditions=calibration_data)
    point_estimate, quantiles = _extract_estimates(estimates)
    truth = np.asarray(calibration_data["biological_age"], dtype=float).squeeze()
    prior_samples = _sample_prior(active_model, truth.size, seed=int(evaluation_config["calibration_seed"]) + 10_000)

    crossing_rate = quantile_crossing_rate(quantiles)
    ppc_quantiles = quantiles
    if crossing_rate > 0.0:
        warnings.warn(
            "Predicted quantiles cross for some cases. Raw quantiles are retained for all "
            "diagnostics; only the approximate PPC sampling bridge uses row-wise monotone "
            "rearrangement.",
            stacklevel=2,
        )
        ppc_quantiles = np.sort(quantiles, axis=1)
    posterior_age_draws = sample_from_quantile_grid(
        ppc_quantiles,
        quantile_levels,
        num_samples=int(evaluation_config["num_posterior_predictive_samples"]),
        seed=int(evaluation_config["posterior_predictive_seed"]),
    )
    ppc_arrays = _posterior_predictives(
        active_model,
        calibration_data,
        posterior_age_draws,
        num_cases=int(evaluation_config.get("num_posterior_predictive_cases", 100)),
        seed=int(evaluation_config["posterior_predictive_seed"]),
    )
    artifact = {
        "truth": truth,
        "point_estimate": point_estimate,
        "quantiles": quantiles,
        "quantile_levels": quantile_levels,
        "prior_samples": prior_samples,
        "ppc_used_monotone_rearrangement": np.asarray(crossing_rate > 0.0),
        **ppc_arrays,
    }

    result_dir = _project_path(project_root, evaluation_config["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _project_path(project_root, evaluation_config["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **artifact)

    result = evaluate_variant_artifact(run_config, artifact)
    result["history"] = history_frame
    result["reused_artifact"] = False
    runtime_config = copy.deepcopy(run_config)
    runtime_config["training_config"]["status"] = "completed"
    (result_dir / "run_config.json").write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    history_frame.to_csv(result_dir / "training_history.csv", index=False)
    result["point_metrics"].to_frame().T.to_csv(result_dir / "point_accuracy_metrics.csv", index=False)
    result["calibration"].to_csv(result_dir / "quantile_calibration.csv", index=False)
    result["intervals"].to_csv(result_dir / "interval_diagnostics.csv", index=False)
    result["age_summary"].to_csv(result_dir / "age_binned_diagnostics.csv", index=False)
    if result["continuous_ppc"] is not None:
        result["continuous_ppc"].to_csv(result_dir / "continuous_predictive_metrics.csv", index=False)
    if result["binary_ppc"] is not None:
        result["binary_ppc"].to_csv(result_dir / "binary_predictive_metrics.csv", index=False)
    return result
