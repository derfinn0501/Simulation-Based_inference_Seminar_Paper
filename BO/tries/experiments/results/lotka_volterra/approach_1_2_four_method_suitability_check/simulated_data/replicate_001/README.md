# Simulated Data Replicate 1

This folder stores the exact random-design synthetic data generated for Approach 1.2.

`train_full.npz` and `train_full.csv` contain the synthetic training data that feeds `gaussian_npe` and `rectified_fmpe`.

Budgeted training sets are prefixes:

```text
train_budget_N = first N rows of train_full
```

`validation.npz` and `validation.csv` contain the held-out validation observations used for evaluation.
