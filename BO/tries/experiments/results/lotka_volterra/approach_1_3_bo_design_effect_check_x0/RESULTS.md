# Approach 1.3 BO Design Effect Check Results

This diagnostic asks whether adaptive BO-selected simulator designs improve parameter estimation, or whether gains mainly come from adding more data.

## Run Configuration

```text
design_space = wide_window
initial = 80
batch = 5
rounds = 20
final_budget = 180
validation = 100
repeats = 2
n_obs = 10
seed = 717
target_mode = fixed_x0
target_seed = 919
target_theta = 1.0 0.05 1.0 0.05
target_psi_unit = 0.5 0.5
bo_candidates = 192
bo_random_fraction = 0.2
marginal_jitter_std = 0.04
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
coverage_weight = 0.25
predictive_weight = 0.1
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
dumb_psi_unit = 0.0 0.0
```

## Design Focus

BO controls design variables `psi`, not the inferred physical parameters `theta`.

For `wide_window`, `psi = (t_span, t_start)`: initial populations and `n_obs` are fixed, while BO can choose between short dense observation windows and longer trend windows.

When `target_mode = fixed_x0`, the reward is evaluated only on one synthetic observed time series. This mimics the real SBI use case more closely than averaging over many validation observations, but it is noisier and should not be interpreted as global posterior quality.

The BO surrogate is fit to round-to-round changes in the configured posterior-quality objective:

```text
negative range-normalized RMSE minus coverage and predictive penalties
```

The fixed dumb baseline uses the same shared initial random data, then adds every later batch at one fixed design:

```text
prey0 = 40.000
pred0 = 15.000
t_start = 0.000
t_span = 2.000
t_end = 2.000
```

## Methods

| Method | Meaning |
| --- | --- |
| random | Uniform random design at every round. |
| fixed_dumb | One fixed naive design at every round after the shared initial data. |
| bo | Adaptive BO-selected design with a small random-design fraction. |
| bo_marginal_random | Random designs sampled from BO's empirical marginal psi distribution after BO finishes. |

Final evaluated round: `20`.

## Final-Round Summary

| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| random | 0.1400 | -0.2918 | 0.1917 | 1.0391 |
| fixed_dumb | 0.1427 | -0.2606 | 0.2333 | 0.5959 |
| bo | 0.1087 | -0.2170 | 0.1417 | 0.7285 |
| bo_marginal_random | 0.1030 | -0.2159 | 0.1583 | 0.7334 |

## Structured Fixed Design Scores

These policies add all post-initial simulations at one interpretable fixed window. They are not adaptive, but they test whether simple design structure can explain or beat the BO choices.

| Design | t_start | t_span | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short early | 0.000 | 2.000 | 0.1159 | -0.2416 | 0.2083 | 0.7363 |
| short late | 38.000 | 2.000 | 0.1474 | -0.3110 | 0.3083 | 0.8652 |
| long early | 0.000 | 24.000 | 0.0837 | -0.1680 | 0.0500 | 0.7182 |
| long late | 16.000 | 24.000 | 0.0759 | -0.1946 | 0.2000 | 0.6876 |
| medium middle | 13.500 | 13.000 | 0.0574 | -0.1405 | 0.2000 | 0.3307 |

## Current Interpretation

At the final round, BO changes range-normalized RMSE by `22.4%` relative to uniform random design.
BO changes range-normalized RMSE by `23.8%` relative to the fixed dumb design.
The BO-marginal random control changes range-normalized RMSE by `26.4%` relative to uniform random design.
BO changes range-normalized RMSE by `-5.5%` relative to the BO-marginal random control.

BO and the BO-marginal random control both beat uniform random design. This suggests the selected design region is useful, but the adaptive sequence itself is not yet clearly better than sampling that region.

## Structured-Design Interpretation

The best fixed structured window by RMSE is `medium middle` with RMSE `0.0574`.
The best fixed structured window by objective is `medium middle` with objective `-0.1405`.

At least one simple structured fixed window beats BO on a final diagnostic. This means BO must be compared against structured non-adaptive designs before claiming adaptive-design value.

For this target run, `medium middle` also matches the target observation design `target_psi_unit = 0.5 0.5`, so simulating at or near the known `psi0` is an especially important baseline.

## Output Files

```text
metrics.csv
summary_by_round.csv
design_trace.csv
design_summary_by_round.csv
bo_trace.csv
design_choice_scores.csv
design_choice_summary.csv
design_choice_trace.csv
design_choice_scores.png
bo_design_effect_summary.png
design_structure_summary.png
```
