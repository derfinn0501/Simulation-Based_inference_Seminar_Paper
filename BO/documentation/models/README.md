# Models

This section documents the posterior-estimation models used in the experiments and the budget constraints that matter for each one.

Current model ladder:

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```

Model folders:

- `prior_mean/`
- `abc_knn/`
- `gaussian_npe/`
- `rectified_fmpe/`

The goal of this ladder is diagnostic:

```text
prior_mean     -> trivial baseline
abc_knn        -> simple non-neural posterior-sample baseline
gaussian_npe   -> simple learned posterior baseline
rectified_fmpe -> current flow-matching posterior estimator
```

## Budget Types

Use three budget categories when comparing models:

```text
simulation budget:
  number of simulator calls used to create synthetic data

training/inference budget:
  compute spent fitting or constructing the posterior estimator

posterior-query budget:
  compute spent answering q(theta | x, psi) for validation observations
```

This distinction matters because models spend budget differently. ABC-kNN mostly needs many simulations. NPE mostly needs synthetic data and neural training. FMPE needs synthetic data, vector-field training, and costly posterior sampling by integration.

## Per-Model Budget Summary

| Model | Main simulation budget | Main internal budget | Main query budget |
| --- | --- | --- | --- |
| `prior_mean` | none for training | none | constant |
| `abc_knn` | number of stored simulations `N_sim` | neighborhood size `K` | nearest-neighbor search over `N_sim` |
| `gaussian_npe` | number of training simulations `N_sim` | MLP training iterations/model size | one forward pass plus Gaussian log-probability |
| `rectified_fmpe` | number of training simulations `N_sim` | `flow_samples_per_pair`, training iterations/model size | `posterior_samples * ode_steps` vector-field evaluations |

Detailed notes live in each model's `budget/README.md`.
