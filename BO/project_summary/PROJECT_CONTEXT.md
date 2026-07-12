# Project Context

## Core Topic

The project studies Simulation-Based Inference, especially Neural Posterior
Estimation and Flow Matching as scalable posterior-learning methods.

SBI is relevant when the likelihood is unavailable, intractable, or too
expensive, but simulation from a model is possible.

The basic SBI workflow is:

1. define a prior over parameters `theta`
2. sample `theta` from the prior
3. simulate data `x` from `p(x | theta)`
4. train an inference network on simulated pairs `(theta, x)`
5. condition on observed data `x_0` to approximate `p(theta | x_0)`

## Main Research Question

Can flow matching be used to learn accurate and scalable posterior samplers for
simulation-based inference?

## Personal Research Angle

The goal is to understand the method deeply and make a small but concrete
contribution.

Useful contribution types could be:

- compare NPE and flow matching on controlled toy simulators
- study posterior evaluation metrics
- analyze when active simulation or Bayesian optimization helps
- build a clean minimal implementation for educational purposes
- identify failure cases of flow matching in SBI

## Current Experimental Context

The current prototype uses Lotka-Volterra predator-prey dynamics.

The inferred physical parameters are:

```text
theta = (alpha, beta, gamma, delta)
```

The current design variables are observation-window variables, and in some
settings initial populations:

```text
psi = design variables chosen by random design or BO
```

The key principle is:

```text
BO chooses psi, not theta.
theta is still sampled from the prior.
```

The current result is:

- the simulator is learnable by Gaussian NPE and lightweight FMPE
- FMPE is strong on point-estimate and predictive metrics
- FMPE calibration remains weaker than simpler methods
- the current BO design loop does not yet beat random design
