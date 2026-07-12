# Approach 1.3 BO Design Effect Check Results

This diagnostic asks whether adaptive BO-selected simulator designs improve parameter estimation, or whether gains mainly come from adding more data.

## Run Configuration

```text
design_space = wide_window
initial = 80
batch = 14
rounds = 30
final_budget = 500
validation = 100
repeats = 10
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
predictive_timeout_seconds = 0.5
final_score_only = True
structured_policies = medium_mid
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

Final evaluated round: `30`.

## Final-Round Summary

| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| random | 0.2066 | -0.3571 | 0.2342 | 0.9194 |
| fixed_dumb | 0.3282 | -0.5520 | 0.4260 | 1.1733 |
| bo | 0.2179 | -0.3773 | 0.2692 | 0.9210 |
| bo_marginal_random | 0.2168 | -0.3720 | 0.2562 | 0.9113 |

## Structured Fixed Design Scores

These policies add all post-initial simulations at one interpretable fixed window. They are not adaptive, but they test whether simple design structure can explain or beat the BO choices.

| Design | t_start | t_span | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium middle | 13.500 | 13.000 | 0.2626 | -0.4595 | 0.4052 | 0.9561 |

## Current Interpretation

At the final round, BO changes range-normalized RMSE by `-5.4%` relative to uniform random design.
BO changes range-normalized RMSE by `33.6%` relative to the fixed dumb design.
The BO-marginal random control changes range-normalized RMSE by `-4.9%` relative to uniform random design.
BO changes range-normalized RMSE by `-0.5%` relative to the BO-marginal random control.

There is no clear final-budget evidence that BO improves over uniform random design in this run.
The combined objective remains worse for BO than for uniform random design, so the final conclusion should not rely on RMSE alone.

## Paired Repeat Interpretation

This run uses `10` paired repeats. The non-BO baselines are scored only at the
final round; BO is still scored every round because its reward needs the
round-to-round objective change.

At the final round, paired deltas support the negative BO interpretation:

| Comparison | Metric | Mean delta | Approx. 95% interval | BO wins |
| --- | --- | ---: | ---: | ---: |
| BO - random | Range-norm RMSE | 0.0112 | [-0.0002, 0.0227] | 3/10 |
| BO - random | Objective | -0.0202 | [-0.0386, -0.0018] | 4/10 |
| BO - BO-marginal-random | Range-norm RMSE | 0.0011 | [-0.0053, 0.0074] | 5/10 |
| BO - BO-marginal-random | Objective | -0.0053 | [-0.0163, 0.0058] | 4/10 |

Lower RMSE is better; higher objective is better. BO is clearly better than the
fixed dumb policy, but not better than uniform random or the BO-marginal random
control. This suggests that the adaptive BO loop is not adding meaningful value
in this setting.

## Structured-Design Interpretation

The best fixed structured window by RMSE is `medium middle` with RMSE `0.2626`.
The best fixed structured window by objective is `medium middle` with objective `-0.4595`.

BO beats the `medium middle` fixed-window score in this broad-validation run,
but that does not rescue the adaptive-design claim because uniform random is
still stronger than BO.

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
paired_final_deltas.csv
paired_final_delta_summary.csv
design_choice_scores.png
bo_design_effect_summary.png
design_structure_summary.png
```
