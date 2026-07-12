# Current State

## Main Conclusion So Far

The Lotka-Volterra simulator is suitable for neural posterior estimation.
Gaussian NPE and the lightweight rectified-FMPE estimator learn useful posterior
signal from simulated data.

The current BO-guided design loop does not yet show robust adaptive value. Some
two-repeat diagnostics made BO look slightly stronger, but a budget-500
`10`-repeat paired check made BO worse than uniform random on mean final RMSE
and combined objective, and effectively tied with the BO-marginal random
control.

A wide-window budget sweep shows that the current BO budget of `230`
simulations is still early for FMPE. Under 1.3-style FMPE settings, the
four-method curve improves clearly after `230`, especially by `500+`
simulations, and keeps improving in point RMSE through `5000`.

## Current Prototype

Location:

```text
experiments/active_fmpe_sbi/
```

Main scripts:

```text
run_lotka_volterra.py
run_lotka_volterra_fmpe.py
evaluate_four_method_suitability.py
evaluate_bo_design_effect.py
posterior_diagnostic_metrics.py
```

## Current Result Bundles

Suitability test:

```text
experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check/
```

This confirms that the chosen Lotka-Volterra simulation design is learnable:
ABC-kNN improves over the prior, and Gaussian NPE / rectified FMPE overtake
ABC-kNN on point-estimate and predictive metrics at the largest tested budget.

BO design-effect test:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check/
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_x0/
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget230_round30/
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_round30/
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_iter500/
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_repeats10_finalonly/
experiments/results/lotka_volterra/approach_1_2_wide_window_budget_saturation/
experiments/results/lotka_volterra/approach_1_2_wide_window_budget_saturation_extended/
```

This compares:

```text
random
fixed_dumb
bo
bo_marginal_random
```

under the same simulation budget. The current run uses `wide_window`, where BO
controls `t_start` and `t_span` with fixed `n_obs`, the lightweight rectified
FMPE estimator, and a sample-based reward:

```text
-range_normalized_rmse - 0.25 * coverage_error - 0.1 * posterior_mean_predictive_rmse
```

With broad validation observations at `rounds = 30` and `final_budget = 230`,
BO is best on final range-normalized RMSE and on the combined objective. Uniform
random still has slightly better coverage error, so the result is promising but
not conclusive.

At `final_budget = 500`, BO remains best on final range-normalized RMSE and the
combined objective. The improvement over random is small, but BO also has
slightly better coverage than random in this larger-budget run.

A `10`-repeat paired budget-500 follow-up does not preserve that apparent BO
advantage. BO has worse mean final range-normalized RMSE than uniform random
(`0.2179` vs `0.2066`) and a worse combined objective (`-0.3773` vs
`-0.3571`). BO is also effectively tied with BO-marginal random, which weakens
the adaptive-sequence claim.

The first categorical BO repair restricts BO and random design to nine named
observation-window categories. In two repeats, categorical BO beats
BO-marginal-random on final RMSE, but categorical random still has the better
combined objective. This is promising enough to repeat, but not a robust BO win.

The matching wide-window Approach 1.2 budget sweep shows FMPE RMSE around
`0.24` at `180-230`, then improving to `0.1963` at `500`, `0.1745` at `1000`,
and `0.1328` at `2000`. This suggests larger-budget BO tests should target at
least the `500` simulation region if the goal is to test BO after FMPE has
enough data.

An extended sweep reaches FMPE RMSE `0.1179` at `3000` and `0.0978` at `5000`.
The final objective is still better for Gaussian NPE because FMPE coverage is
worse, so larger-budget FMPE runs need both point-estimate and calibration
checks.

With one fixed synthetic `x0`, BO and BO-marginal random both beat uniform
random. However, the medium fixed window is strongest overall for that target;
this window matches the target observation design `psi0`.

## Interpretation

Current evidence supports:

- simulation budget matters
- the simulator contains learnable posterior information
- FMPE is promising for point estimates and predictive checks
- FMPE calibration still needs work
- the observation-window design matters
- current BO budgets are below FMPE's stronger wide-window budget region
- FMPE point RMSE keeps improving through `5000` simulations in wide-window runs
- BO clearly avoids the intentionally poor fixed dumb design

Current evidence does not yet support:

- the current BO policy adding robust adaptive value across targets and repeats
- the two-repeat budget-500 BO advantage as stable
- replacing critical posterior checks with visual posterior plots

## Next Useful Questions

1. Does simulating near the known target design `psi0` stay strong across several different `x0` values?
2. Can BO reliably discover the best structured window rather than merely sample a useful region?
3. Would a full posterior-predictive sample reward or simulation-based-calibration reward work better than posterior-mean predictive reward?
4. Can FMPE calibration be improved without losing point-estimate performance?
5. Would a simpler structured or stratified design policy outperform both BO and random?
