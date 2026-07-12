# abc_knn

`abc_knn` is an approximate Bayesian computation baseline using nearest neighbors.

For a query `(x_o, psi_o)`, it finds the `K` closest simulated examples in standardized `(x, psi)` feature space and uses their `theta` values as posterior samples.

Budget notes:

- See `budget/README.md`.
