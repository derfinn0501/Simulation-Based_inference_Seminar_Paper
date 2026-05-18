# 001 Fixed Observation Count With Adaptive Time Window

Status: idea

## Question

Given a fixed number of observations, should BO choose a short dense observation window or a long sparse observation window?

## Motivation

The current Lotka-Volterra setting keeps:

```text
n_obs = fixed
```

This is useful because the neural networks receive a fixed-size input:

```text
x_dim = 2 * n_obs
```

However, BO can still choose where those fixed observations are placed in time by controlling:

```text
psi = (t_start, t_span)
```

The simulator then observes:

```text
times = linspace(t_start, t_start + t_span, n_obs)
```

This creates a clean experimental-design question:

> With a fixed measurement budget, is it better to observe densely over a short period or sparsely over a long period?

## Intuition

A short window gives dense local information:

- local growth or decay,
- slope information,
- fast predator-prey interaction changes,
- short-term phase behavior.

A long window gives broader dynamical information:

- oscillation period,
- amplitude,
- predator-prey phase relation,
- long-term tendencies.

Both may be useful. The best design may depend on the current posterior uncertainty.

## Possible Implementation

Keep the observation dimension fixed:

```text
n_obs = 10
```

Let BO choose:

```text
t_start
t_span
```

and generate uniformly spaced observations within that window.

This is already close to the current `hard_window` setup:

```text
psi = (t_span, t_start)
```

Later extensions could include:

```text
spacing_type
individual observation times
sampling density parameter
```

but those should only be added after the simpler fixed-grid window design is understood.

## Why This Is Attractive

- Fixed-size NN input is preserved.
- BO has a low-dimensional and interpretable design space.
- The design has a clear scientific meaning.
- It directly connects to budget allocation: spend a fixed measurement budget locally or globally.

## Risks

- If `t_span` is too short, the observation may miss global oscillatory behavior.
- If `t_span` is too long, the fixed number of observations may be too sparse.
- Uniform spacing may be too restrictive if the most informative times are irregular.
- BO may exploit windows that are easy for the current model but not globally informative.

## When To Revisit

Revisit after the four-method suitability check is stable and after FMPE calibration issues are better understood.

This idea is likely relevant when moving from:

```text
Can the posterior models learn this simulator?
```

to:

```text
Can BO choose more informative observation windows under a fixed measurement budget?
```
