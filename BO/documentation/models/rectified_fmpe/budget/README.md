# rectified_fmpe Budget Constraints

## Simulation Budget

Rectified FMPE uses synthetic training pairs:

```text
(theta_i, psi_i, x_i)
```

The main simulator budget is:

```text
N_sim = number of training simulations
```

## Flow-Matching Training Budget

Each simulator pair can produce multiple flow-matching training examples:

```text
flow_samples_per_pair
```

Effective training examples scale like:

```text
N_sim * flow_samples_per_pair
```

Other important training-budget variables:

```text
MLP hidden size
maximum iterations
early stopping settings
learning rate
```

## Posterior-Query Budget

FMPE posterior sampling is more expensive than NPE because sampling requires repeated vector-field evaluations.

Query cost scales with:

```text
posterior_samples * ode_steps
```

For each validation observation, the model must integrate the learned vector field for every posterior sample.

## Main Constraints

- Needs enough simulations to learn the conditional posterior structure.
- Needs enough flow samples and training iterations to learn a stable vector field.
- More `ode_steps` can improve sampling quality but increases query cost.
- More `posterior_samples` improves posterior estimates but increases query cost.
- Exact posterior density is not currently available, so evaluation relies on posterior samples, RMSE, coverage, and posterior predictive checks.
- Current evidence suggests point estimates are competitive, but calibration still needs work.
