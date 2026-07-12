# prior_mean Budget Constraints

## Simulation Budget

`prior_mean` uses no training simulations.

It only needs the prior bounds:

```text
theta ~ p(theta)
```

Validation simulations are still needed to evaluate it, but those are part of the evaluation budget, not the model budget.

## Training Budget

There is no training.

## Posterior-Query Budget

Query cost is constant:

```text
theta_hat = E_p(theta)[theta]
```

## Main Constraint

The method is too weak to be a posterior model. Good coverage can be misleading because prior intervals are broad. It should only be used as a lower sanity baseline.
