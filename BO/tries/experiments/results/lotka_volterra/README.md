# Lotka-Volterra Experiment Results

This folder stores Lotka-Volterra result bundles by approach number, so the
project history is visible.

Naming convention:

```text
approach_<number>_<short_description>/
```

Result numbering starts at `approach_1_1` and then increments forward. This numbering is independent of the paper-note section numbers.

Current folders:

```text
approach_1_1_active_design_snapshot/
approach_1_2_four_method_suitability_check/
approach_1_3_bo_design_effect_check/
approach_1_3_bo_design_effect_check_x0/
approach_1_3_bo_design_effect_check_budget230_round30/
approach_1_3_bo_design_effect_check_budget500_round30/
approach_1_3_bo_design_effect_check_budget500_iter500/
approach_1_3_bo_design_effect_check_budget500_repeats10_finalonly/
approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2/
approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2_fullcurve/
approach_1_4_reward_landscape_categorical_psi/
approach_1_4_reward_landscape_batch_sweep_initial500/
approach_1_2_wide_window_budget_saturation/
approach_1_2_wide_window_budget_saturation_extended/
```

`approach_1_1_active_design_snapshot` keeps the earlier BO-vs-random active-design outputs.

`approach_1_2_four_method_suitability_check` tests whether the Lotka-Volterra simulation design is learnable by comparing prior mean, ABC-kNN, Gaussian NPE, and rectified FMPE across simulation budgets.

`approach_1_2_wide_window_budget_saturation` repeats the four-method suitability
check under the `wide_window` design and 1.3-style FMPE settings. It includes
the BO budget points `80`, `180`, and `230`, then extends to `500`, `1000`, and
`2000` to show where FMPE begins improving.

`approach_1_2_wide_window_budget_saturation_extended` extends the same sweep to
`3000` and `5000` simulator calls. FMPE point-estimate RMSE continues improving
through `5000`, while the combined objective is limited by coverage error.

`approach_1_3_bo_design_effect_check` compares uniform random design, a fixed dumb design, adaptive BO design, and BO-marginal random design under the same simulation budget, using a wide observation-window design and an FMPE-compatible sample-based reward to test whether BO can exploit `t_start` and `t_span`.

`approach_1_3_bo_design_effect_check_x0` repeats the BO design-effect check for one fixed synthetic observation `x0`. It tests the more realistic target-observation setting and includes structured fixed-window scores so BO can be compared against simple non-adaptive observation designs.

`approach_1_3_bo_design_effect_check_budget230_round30` repeats the broad-validation BO design-effect check with `rounds = 30` and `final_budget = 230`. In this run, BO is best on final RMSE and the combined objective, but coverage remains slightly better for uniform random.

`approach_1_3_bo_design_effect_check_budget500_round30` applies the
wide-window budget-saturation finding to BO by increasing the final budget to
`500` while keeping `30` reward rounds. BO remains best on final RMSE and the
combined objective, but only by a small margin over uniform random.

`approach_1_3_bo_design_effect_check_budget500_iter500` repeats the same
budget-500 BO diagnostic with `fmpe_max_iter = 500` instead of `220`. The
final-round metrics are unchanged, so the optimizer-iteration cap is not the
active bottleneck for this specific diagnostic.

`approach_1_3_bo_design_effect_check_budget500_repeats10_finalonly` repeats the
budget-500 diagnostic with `10` paired repeats and final-round-only scoring for
non-BO baselines. BO is worse than uniform random on mean final RMSE and
objective, and it is effectively tied with the BO-marginal random control.

`approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2`
restricts BO and random design to the same nine categorical observation-window
choices. BO separates from BO-marginal-random on the two-repeat final RMSE, but
uniform categorical random still has the better combined objective.

`approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2_fullcurve`
reruns the same categorical diagnostic without final-round-only scoring, so all
four methods have full learning curves from `80` to `500` simulations.

`approach_1_4_reward_landscape_categorical_psi` freezes one shared initial
training set per seed, adds one paired batch at each of the nine categorical
wide-window designs, and measures which reward definitions visibly couple to
the chosen `psi`.

`approach_1_4_reward_landscape_batch_sweep_initial500` repeats the reward
landscape diagnostic from a larger `500`-simulation initial set and sweeps
paired added batch sizes `14`, `56`, `112`, and `224`. Larger batches make the
reward-psi signal substantially more stable.
