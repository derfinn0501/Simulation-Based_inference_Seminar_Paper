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
repeats = 2
n_obs = 10
seed = 717
target_mode = validation_set
bo_design_mode = categorical
bo_candidates = 192
bo_random_fraction = 0.2
bo_category_policies = all
categorical_ucb_weight = 1.0
marginal_jitter_std = 0.04
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
coverage_weight = 0.25
predictive_weight = 0.1
predictive_timeout_seconds = 0.5
final_score_only = True
structured_policies = all
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
dumb_psi_unit = 0.0 0.0
```

## Design Focus

BO controls design variables `psi`, not the inferred physical parameters `theta`.

For `wide_window`, `psi = (t_span, t_start)`: initial populations and `n_obs` are fixed, while BO can choose between short dense observation windows and longer trend windows.

When `bo_design_mode = categorical`, BO and the random baseline choose from the same finite set of named design categories instead of the full continuous `psi` space.

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
| random | 0.2498 | -0.4200 | 0.3063 | 0.9362 |
| fixed_dumb | 0.3621 | -0.5798 | 0.4475 | 1.0582 |
| bo | 0.2384 | -0.4263 | 0.3504 | 1.0023 |
| bo_marginal_random | 0.2537 | -0.4348 | 0.3429 | 0.9535 |

## Structured Fixed Design Scores

These policies add all post-initial simulations at one interpretable fixed window. They are not adaptive, but they test whether simple design structure can explain or beat the BO choices.

| Design | t_start | t_span | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short early | 0.000 | 2.000 | 0.3510 | -0.5664 | 0.4417 | 1.0500 |
| short late | 38.000 | 2.000 | 0.3463 | -0.5530 | 0.4042 | 1.0569 |
| long early | 0.000 | 24.000 | 0.2799 | -0.4841 | 0.4171 | 1.0000 |
| long late | 16.000 | 24.000 | 0.2782 | -0.4712 | 0.3808 | 0.9776 |
| medium middle | 13.500 | 13.000 | 0.2599 | -0.4545 | 0.3792 | 0.9976 |

## Current Interpretation

At the final round, BO changes range-normalized RMSE by `4.5%` relative to uniform random design.
BO changes range-normalized RMSE by `34.2%` relative to the fixed dumb design.
The BO-marginal random control changes range-normalized RMSE by `-1.6%` relative to uniform random design.
BO changes range-normalized RMSE by `6.0%` relative to the BO-marginal random control.

BO is strongest at the final budget. This is initial evidence that adaptive design can help beyond simply sampling the BO-favored design region.
The combined objective remains worse for BO than for uniform random design, so the final conclusion should not rely on RMSE alone.

## Categorical BO Interpretation

This run restricts BO and the random baseline to the same nine named
observation-window categories: short, medium, or long windows crossed with
early, middle, or late starts.

The categorical restriction produces the separation we hoped to test against
the BO-marginal control: BO beats BO-marginal-random on final RMSE in both
paired repeats, and also has the better mean combined objective. However, BO
still has a worse combined objective than uniform categorical random because
coverage and posterior-predictive terms are weaker.

Treat this as a promising categorical-design signal, not yet as evidence that
BO is robustly useful. The next check should repeat this categorical run with
more paired seeds.

## Structured-Design Interpretation

The best fixed structured window by RMSE is `medium middle` with RMSE `0.2599`.
The best fixed structured window by objective is `medium middle` with objective `-0.4545`.

BO beats the structured fixed windows on these final diagnostics, which is stronger evidence for adaptive design than a comparison to random alone.

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
