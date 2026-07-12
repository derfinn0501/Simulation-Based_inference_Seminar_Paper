# Lotka-Volterra BayesFlow Example

This notebook demonstrates simulation-based inference for the Lotka-Volterra predator-prey model. The goal is to infer hidden ecological parameters from noisy population observations.

## Simulator

The simulator has three parts:

1. **Prior over hidden parameters**
   - Parameters are `theta = (alpha, beta, gamma, delta)`.
   - They are sampled by transforming standard normal draws with a sigmoid and scaling them to roughly `(0.1, 4.0)`.

2. **Ecology model**
   - Solves the Lotka-Volterra ODE:
     - `dx/dt = alpha * x - beta * x * y`
     - `dy/dt = -gamma * y + delta * x * y`
   - Uses initial state `(x0, y0) = (1, 1)`.
   - Produces latent trajectories `x`, `y`, and `t` over 100 time steps.

3. **Observation model**
   - Adds Gaussian noise to the latent prey and predator trajectories.
   - Subsamples every 10th time step.
   - Produces `observed_x`, `observed_y`, and `observed_t`.

The simulator output contains both the hidden parameters, the full latent trajectories, and the noisy observations. For inference, the target is `theta`; the observations or summaries are the conditions.

## Prior Predictive Trajectories

The first trajectory plot shows the prior predictive distribution. Each faint line is one simulated world with its own sampled parameters. The dark prey and predator curves are pointwise medians across all simulated worlds, not necessarily real trajectories from one parameter setting. The shaded bands are pointwise 95% intervals.

## Inference Approaches

### 1. Expert Statistics + Point Estimation

The notebook first computes hand-crafted expert statistics from the observations:

- means
- log variances
- autocorrelations
- cross-correlation
- dominant period

The adapter converts the simulator output to neural-network format:

- drops unused full trajectories and raw observations
- concatenates `alpha`, `beta`, `gamma`, `delta` into `inference_variables`
- concatenates expert statistics into `inference_conditions`

The network is:

```python
bf.networks.ScoringRuleNetwork(...)
```

It predicts posterior point summaries, especially posterior means and quantiles. This is fast and interpretable, but it does not model the full posterior directly.

### 2. Expert Statistics + Flow Matching

The same expert statistics are used as input conditions, but the inference network changes to:

```python
bf.networks.FlowMatching()
```

This learns a full posterior sampler instead of only point estimates. It can represent dependencies and correlations between parameters.

### 3. Learned Summaries + Flow Matching

In the final version, the hand-crafted expert statistics are dropped. The adapter keeps the raw observed time series and concatenates them into `summary_variables`.

The workflow uses:

```python
summary_network = bf.networks.TimeSeriesNetwork()
inference_network = bf.networks.FlowMatching()
```

The `TimeSeriesNetwork` learns an embedding of `observed_x`, `observed_y`, and `observed_t`. This learned embedding replaces the expert statistics and is trained jointly with the flow-matching posterior network.

## Training

Training uses simulated datasets from the model. An epoch is one full pass through the training data. Within each epoch, the data are processed in mini-batches, and the network weights are updated once per batch.

Too few epochs can underfit; too many can overfit to the finite simulated training set. Validation simulations are used to monitor generalization.

## Posterior Evaluation

The notebook evaluates posterior quality with:

- recovery plots comparing estimates to true simulated parameters
- calibration ECDF plots
- posterior pair plots to inspect marginal distributions and correlations
- z-score contraction diagnostics
- posterior predictive checks

For posterior predictive checks, posterior parameter samples are fed back into the ecology model. The resulting simulated trajectories are compared with the observed data. This checks whether inferred parameters generate plausible ecological dynamics, not just plausible posterior plots.

## Main Progression

The notebook moves from a fast baseline to richer inference:

```text
expert statistics + point estimation
-> expert statistics + flow matching
-> learned summaries + flow matching
```

The overall lesson is to start with a simple, inspectable SBI workflow, diagnose it, then replace approximations step by step with more expressive learned components.
