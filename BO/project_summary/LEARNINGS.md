# Learnings

This file records stable insights from completed work.

## Learning: The Current Lotka-Volterra Setting Is Learnable

The four-method suitability checks showed that the simulator contains usable
posterior information.

At sufficient simulation budgets:

```text
rectified_fmpe < gaussian_npe < abc_knn < prior_mean
```

for range-normalized RMSE.

Interpretation: the setting is suitable for neural posterior estimation and can
serve as a controlled debugging environment.

## Learning: FMPE Point Estimates Are Stronger Than Its Calibration

The lightweight rectified FMPE estimator performs well on point-estimate and
posterior-predictive metrics, but its coverage error remains worse than simpler
methods in current diagnostics.

Future FMPE work should include calibration checks, not only posterior mean RMSE.

## Learning: Current BO Design Loop Has Mixed But Improving Evidence

Approach 1.3 compared random design, adaptive BO design, and BO-marginal random
design under the same simulation budget.

The first result did not show a clear BO advantage over random design. A later
`wide_window` run gave BO a stronger observation-window choice by letting it
choose between short dense windows and longer trend windows, but still did not
produce a final-budget advantage. A coverage-aware reward,
`validation_log_posterior - 5.0 * coverage_error`, also did not improve the
final result over random.

Adding a fixed dumb baseline showed that BO does outperform an intentionally
poor non-adaptive design, but random design remains stronger on the current
final-budget diagnostics.

Switching Approach 1.3 from Gaussian NPE to lightweight rectified FMPE and using
`-range_normalized_rmse - 0.25 * coverage_error` as the BO objective did not
change the main conclusion: BO still beats the fixed dumb design but does not
beat uniform random.

Adding a posterior-mean predictive term to the FMPE reward also did not create a
clear BO advantage in the few-large-batch setting. Increasing the number of
reward rounds at the same final budget made BO best on broad-validation RMSE,
but the combined objective still favored the BO-marginal random control.

Increasing the default broad-validation run to `rounds = 30` and
`final_budget = 230` produced the clearest BO signal so far: BO was best on
final range-normalized RMSE and on the combined objective, and it beat the
structured fixed-window scores in that run. Coverage remained slightly better
for uniform random, and the run still has only two repeats, so this is
preliminary evidence rather than a settled claim.

A wide-window Approach 1.2 budget sweep with the same FMPE settings as the BO
diagnostic shows that `230` simulations is still early for FMPE. FMPE is noisy
around `180-230`, improves clearly by `500`, and improves further through
`1000-2000`. This means current BO runs are mostly low-budget design
diagnostics, not tests of FMPE at its stronger budget region.

Extending the same wide-window sweep to `3000` and `5000` simulations shows no
clear point-RMSE saturation yet: FMPE reaches `0.1179` at `3000` and `0.0978`
at `5000`. The combined objective improves only slightly from `3000` to `5000`
because coverage error worsens again, reinforcing that calibration remains the
main weakness.

Applying the budget finding to Approach 1.3 with `final_budget = 500` keeps BO
best on broad-validation RMSE and the combined objective. The gain over uniform
random is small, but BO also has slightly better coverage in this run. This
supports trying larger BO budgets, while still treating the adaptive-design
claim as preliminary.

Repeating that budget-500 BO diagnostic with `fmpe_max_iter = 500` instead of
`220` leaves the final-round metrics unchanged. For this setup, increasing the
optimizer iteration cap alone does not explain the small BO-vs-random gap.

Increasing the budget-500 BO diagnostic to `10` paired repeats reverses the
small two-repeat BO advantage. BO is worse than uniform random on mean final
range-normalized RMSE (`0.2179` vs `0.2066`) and combined objective (`-0.3773`
vs `-0.3571`), and is effectively tied with BO-marginal random. This is the
strongest evidence so far that the current adaptive BO loop adds little value
beyond avoiding the intentionally poor fixed dumb design.

Restricting BO to categorical observation-window choices is a plausible repair.
In a first two-repeat budget-500 categorical run, BO separates from
BO-marginal-random on final RMSE, but uniform categorical random still has the
better combined objective. This points toward design-space simplification as a
more promising lever than simply increasing optimizer iterations, while still
requiring more paired repeats.

Reward-landscape diagnostics clarify why the small-batch BO reward is fragile.
With `initial = 80` and an added categorical batch of `14`, most one-step
reward deltas are negative and category rank stability is low. With a larger
`500`-simulation initial set and paired added batches, the category signal
becomes much more stable as the batch grows. For the combined
RMSE-coverage-predictive reward, paired eta squared rises from `0.423` at `+14`
to `0.825` at `+224`, and rank stability rises from `0.028` to `0.711`.
This suggests the current BO loop may be asking the surrogate to learn from
feedback increments that are too small for FMPE.

For one fixed synthetic `x0`, BO and BO-marginal random both beat uniform random,
which suggests that target-specific design can matter. However, a simple medium
fixed observation window beat both BO variants for that target. This medium
window matches the target observation design, so simulating at or near the known
`psi0` is now a strong baseline.

Interpretation: in the current setup, the observation-window choice matters, and
additional BO feedback can help. The adaptive-value claim is still fragile
because it depends on budget, target mode, coverage behavior, and comparison to
structured non-adaptive designs.

## Learning: Result Bundles Need Clear Ownership

Several result folders became redundant as the approach numbering changed.
Future experiments should state whether their output is temporary, canonical, or
safe to delete.
