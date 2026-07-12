# gaussian_npe Budget Constraints

## Simulation Budget

Gaussian NPE uses synthetic training pairs:

```text
(theta_i, psi_i, x_i)
```

The main simulator budget is:

```text
N_sim = number of training simulations
```

## Training Budget

Training budget is controlled by:

```text
MLP hidden size
maximum iterations
early stopping settings
validation split
learning rate
```

In the current prototype, the model is intentionally simple so that it acts as a baseline rather than the main contribution.

## Posterior-Query Budget

After training, query cost is cheap:

```text
one MLP forward pass per observation
```

Log probability is also cheap because the covariance is a global residual Gaussian covariance.

## Main Constraints

- Needs enough simulations to train the MLP without overfitting.
- The Gaussian residual covariance is simple and may be poorly calibrated.
- It may perform well on posterior means while missing multimodality.
- It is a useful benchmark for whether neural amortization helps before using FMPE.
