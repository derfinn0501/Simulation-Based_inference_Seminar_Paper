# Amortization Versus SBI

## Short Version

Amortization is not the definition of SBI, but it is a central feature of many modern neural SBI workflows.

SBI means:

```text
Use simulations to perform inference when the likelihood is unavailable or impractical.
```

Amortized inference means:

```text
Train once, then answer many inference queries cheaply.
```

Modern neural SBI often combines both:

```text
simulate many parameter-observation pairs
train neural posterior/likelihood/ratio estimator
reuse the trained estimator for new observations
```

## Why This Matters

If there is only one dataset and the likelihood is tractable, amortized SBI may be unnecessary overhead. MCMC may be simpler and more reliable.

If there are many datasets, many subjects, many experimental designs, or repeated online inference tasks, the upfront training cost of SBI can be amortized over many posterior queries.

## Working Distinction

- **Non-amortized inference** solves one posterior problem at a time.
- **Amortized inference** learns a reusable map from observations to posterior information.

For neural posterior estimation, the learned object is roughly:

```text
x -> p(theta | x)
```

For BayesFlow-style workflows, this can become:

```text
simulator output -> adapter -> neural posterior approximator
```

The trained network can then produce posterior estimates or samples for new observations without rerunning expensive inference from scratch.
