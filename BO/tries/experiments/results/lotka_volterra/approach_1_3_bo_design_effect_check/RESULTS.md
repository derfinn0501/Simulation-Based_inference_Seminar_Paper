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
target_mode = validation_set
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
| random | 0.2630 | -0.4429 | 0.3583 | 0.9030 |
| fixed_dumb | 0.3018 | -0.4951 | 0.3933 | 0.9497 |
| bo | 0.2584 | -0.4425 | 0.3629 | 0.9330 |
| bo_marginal_random | 0.2596 | -0.4328 | 0.3050 | 0.9704 |

## Structured Fixed Design Scores

These policies add all post-initial simulations at one interpretable fixed window. They are not adaptive, but they test whether simple design structure can explain or beat the BO choices.

| Design | t_start | t_span | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short early | 0.000 | 2.000 | 0.2907 | -0.4881 | 0.3958 | 0.9848 |
| short late | 38.000 | 2.000 | 0.2713 | -0.4602 | 0.3675 | 0.9703 |
| long early | 0.000 | 24.000 | 0.2621 | -0.4598 | 0.3875 | 1.0076 |
| long late | 16.000 | 24.000 | 0.2655 | -0.4614 | 0.3983 | 0.9636 |
| medium middle | 13.500 | 13.000 | 0.2579 | -0.4403 | 0.3604 | 0.9228 |

## Current Interpretation

At the final round, BO changes range-normalized RMSE by `1.7%` relative to uniform random design.
BO changes range-normalized RMSE by `14.4%` relative to the fixed dumb design.
The BO-marginal random control changes range-normalized RMSE by `1.3%` relative to uniform random design.
BO changes range-normalized RMSE by `0.4%` relative to the BO-marginal random control.

BO is strongest on final range-normalized RMSE, but not on the combined BO objective. This is a mixed signal: more sequential feedback helps point estimation, while coverage/predictive terms still favor another design strategy.

## Structured-Design Interpretation

The best fixed structured window by RMSE is `medium middle` with RMSE `0.2579`.
The best fixed structured window by objective is `medium middle` with objective `-0.4403`.

At least one simple structured fixed window beats BO on a final diagnostic. This means BO must be compared against structured non-adaptive designs before claiming adaptive-design value.

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
