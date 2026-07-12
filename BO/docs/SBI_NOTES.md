# SBI Notes

## Core Idea

Simulation-Based Inference estimates posterior distributions when the likelihood
is unavailable or hard to evaluate, but a simulator can generate data.

The basic joint sampling process is:

```text
theta ~ p(theta)
x ~ p_sim(x | theta)
```

The target is:

```text
p(theta | x_0)
```

for an observed datum `x_0`.

## Neural Posterior Estimation

NPE trains a conditional density model:

```text
q_phi(theta | x) approx p(theta | x)
```

After training, inference for a new observation is amortized:

```text
x_0 -> q_phi(theta | x_0)
```

## Current Project Mapping

In the Lotka-Volterra prototype:

```text
theta = (alpha, beta, gamma, delta)
```

where:

- `alpha`: prey growth
- `beta`: predation rate
- `gamma`: predator death
- `delta`: predator reproduction from prey

Design variables are called `psi` and may include observation-window settings.
BO should choose `psi`, not `theta`.

## Evaluation Reminder

Learning a conditional posterior is not enough.
Posterior validity requires calibration, predictive checks, and synthetic
ground-truth recovery tests where possible.
