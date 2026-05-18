# abc_knn Budget Constraints

## Simulation Budget

ABC-kNN mainly spends budget on synthetic simulations:

```text
N_sim = number of stored simulator pairs (theta_i, psi_i, x_i)
```

More simulations make the local neighborhood around a query observation more meaningful.

## Neighborhood Budget

The key internal budget is:

```text
K = number of nearest neighbors used as posterior samples
```

Tradeoff:

```text
small K -> sharper but noisy posterior
large K -> smoother but more biased/broader posterior
```

A reasonable first rule is:

```text
K = sqrt(N_sim)
```

clipped to a useful range such as:

```text
10 <= K <= 100
```

## Training Budget

There is no neural training.

There may be a small preprocessing cost for standardizing features and building a nearest-neighbor index.

## Posterior-Query Budget

For brute-force search, one query costs roughly:

```text
O(N_sim * feature_dim)
```

This can be reduced with a nearest-neighbor index, but high-dimensional observations can still make ABC-kNN weak.

## Main Constraints

- Needs enough simulations for local neighborhoods.
- Sensitive to feature scaling and distance metric.
- Suffers as observation dimension grows.
- Posterior sample count is limited by `K`.
- Very useful as a non-neural sanity check: if ABC-kNN works, the simulator setting contains learnable posterior information.
