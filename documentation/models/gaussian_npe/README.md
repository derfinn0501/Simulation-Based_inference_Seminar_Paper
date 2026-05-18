# gaussian_npe

`gaussian_npe` is the simple learned posterior baseline implemented as `GaussianPosteriorRegressor`.

It predicts a posterior mean with an MLP and uses a residual Gaussian covariance to define an approximate density:

```text
q(theta | x, psi)
```

Budget notes:

- See `budget/README.md`.
