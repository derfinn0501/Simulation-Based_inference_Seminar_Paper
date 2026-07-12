# Decision Log

This file records methodological and structural decisions.

## Decision: Add Root Research-Agent Files

### Date

2026-05-24

### Context

The project had detailed notes inside `documentation/`, `my_paper/`, and
`experiments/`, but no root-level workflow files for Codex collaboration.

### Decision

Add root operating files for backlog, decisions, instruction changes, prompts,
and Codex workflow. Move stable project-summary material to `project_summary/`.

### Reason

This makes future work easier to coordinate and separates tasks, decisions, and
stable learnings.

### Consequence

Future non-trivial tasks should update the appropriate operating or summary file
when the result is stable.

## Decision: Treat AGENTS.md As The Agreed Instruction Contract

### Date

2026-05-24

### Context

The workflow instructions should stay tight and not drift after every task.

### Decision

Keep `AGENTS.md` concise. Proposed changes go through
`INSTRUCTION_CHANGELOG.md` and are applied only after explicit user agreement.

### Reason

This prevents instruction bloat and keeps collaboration predictable.

### Consequence

Codex may propose instruction changes, but should not silently rewrite stable
workflow rules.

## Decision: Store Project Summary In Dedicated Folder

### Date

2026-05-24

### Context

Root-level markdown was becoming a mix of workflow files and project summary
files.

### Decision

Use `project_summary/` for stable project context, current state, research
style, and learnings.

### Reason

This keeps the repository root clean and makes the summary easier to find.

### Consequence

Root files now focus on operation: instructions, backlog, decisions, prompts,
and instruction changes.

## Decision: Keep Existing Experiment Artifacts In Place For Now

### Date

2026-05-24

### Context

The recommended structure includes a root `results/` folder, but existing
scripts and logs currently write to `experiments/results/`.

### Decision

Create the root `results/` folder for future consolidated exports, but keep
current active experiment bundles under `experiments/results/` until a dedicated
results migration is requested.

### Reason

Moving result paths now would mix a documentation restructure with experiment
path refactoring and could break existing scripts.

### Consequence

New scripts may still default to `experiments/results/` until the project
explicitly standardizes result output paths.

## Decision: Use BO-Marginal Random As A Control

### Date

2026-05-23

### Context

Approach 1.3 tests whether BO improves parameter estimation through design
choice rather than just by adding simulations.

### Decision

Compare uniform random design, adaptive BO design, and a BO-marginal random
control sampled from BO's empirical design distribution.

### Reason

This separates "BO found a useful design region" from "the adaptive BO sequence
itself matters."

### Consequence

The first run showed no clear BO advantage over random design.

## Decision: Add Wide Observation-Window Design Space

### Date

2026-05-24

### Context

The earlier `hard_window` design let BO choose observation timing, but mostly
within short windows. The next most plausible way to make BO useful is to give
it a design choice that affects information content more directly.

### Decision

Add `wide_window`, where BO controls only `t_start` and `t_span`, initial
populations stay fixed, `n_obs` stays fixed, and `t_span` ranges from `2` to
`24`.

### Reason

This exposes the tradeoff between short dense observations and longer trend
observations without letting BO optimize the inferred physical parameters
`theta`.

### Consequence

Approach 1.3 now defaults to `wide_window`. The first run still did not show a
BO advantage, suggesting the reward/objective is the next likely bottleneck.

## Decision: Test Coverage-Aware BO Reward

### Date

2026-05-24

### Context

The pure validation-log-posterior reward can favor sharp local fit without
directly penalizing posterior miscalibration or collapse.

### Decision

Use a combined BO objective for Approach 1.3:

```text
validation_log_posterior - 5.0 * coverage_error
```

and run eight BO rounds so the surrogate has enough observations to use the
reward after initial exploration.

### Reason

This is the simplest calibration-aware reward that directly targets the concern
that a posterior should cover plausible parameter mass instead of only finding
one explanation.

### Consequence

The first run still did not show a BO advantage. Random design had better
coverage at the final budget, so the next reward should likely be more
posterior-predictive or SBC-style rather than only log posterior plus marginal
coverage.

## Decision: Add Fixed Dumb Design Baseline

### Date

2026-05-24

### Context

Random design is a strong baseline. To understand whether BO is failing
entirely or only failing to beat random, the design-effect check needs a very
simple weak baseline.

### Decision

Add `fixed_dumb`, which uses the same shared initial random data as the other
methods, then adds every later simulation batch at one fixed naive design. In
the default `wide_window` setup this is:

```text
t_start = 0
t_span = 2
```

### Reason

This tests whether BO at least avoids a clearly poor non-adaptive design.

### Consequence

BO clearly outperformed the fixed dumb baseline, but still did not outperform
uniform random design. The current conclusion is therefore more nuanced: BO is
better than a bad fixed design, but not yet useful relative to random.

## Decision: Switch BO Design-Effect Check To FMPE Reward

### Date

2026-05-24

### Context

The earlier BO design-effect runs used Gaussian NPE because it exposes
validation log posterior. The project target is FMPE, and the suitability check
showed FMPE can learn the simulator setting.

### Decision

Use the lightweight rectified-FMPE estimator in Approach 1.3 and replace the
log-posterior reward with:

```text
-range_normalized_rmse - 0.25 * coverage_error
```

### Reason

The lightweight FMPE estimator produces posterior samples but not exact
posterior densities. The sample-based reward evaluates posterior mean accuracy
while penalizing poor coverage.

### Consequence

The first FMPE run still did not show a BO advantage over random. BO beat the
fixed dumb design, but random remained stronger across final diagnostics.

## Decision: Add Predictive Term To FMPE BO Reward

### Date

2026-05-24

### Context

The first FMPE reward used posterior mean accuracy and coverage. A better SBI
reward should also ask whether inferred parameters reproduce validation
trajectories through the simulator.

### Decision

Use:

```text
-range_normalized_rmse - 0.25 * coverage_error - 0.1 * posterior_mean_predictive_rmse
```

as the default FMPE BO objective.

### Reason

This adds a simulator-facing term while staying cheap enough for the current BO
loop.

### Consequence

The predictive term did not create a BO advantage. BO still beat the fixed dumb
design, but random remained stronger on the final diagnostics.

## Decision: Increase BO Reward Rounds At Fixed Budget

### Date

2026-05-24

### Context

The BO surrogate only starts becoming useful after several reward observations.
The previous `batch = 20`, `rounds = 5` setup gave few feedback points.

### Decision

Keep the final budget at `180`, but use:

```text
initial = 80
batch = 5
rounds = 20
```

### Reason

This tests whether BO needs more sequential reward feedback rather than more
total simulations.

### Consequence

BO became best on broad-validation range-normalized RMSE, but only slightly.
The combined objective still favored the BO-marginal random control, so the
evidence is mixed rather than a clean BO win.

## Decision: Add Fixed-x0 And Structured Design Scores

### Date

2026-05-24

### Context

Real SBI usually conditions on one observed dataset `x0`, not an average over
many random validation observations. Also, BO needs to beat strong simple design
baselines, not only random and an intentionally dumb design.

### Decision

Add `--target-mode fixed_x0` to the BO design-effect diagnostic and score simple
fixed observation windows:

```text
short_early
short_late
long_early
long_late
medium_mid
```

### Reason

This separates three possibilities:

1. random design is already enough,
2. the target `x0` benefits from a structured observation window,
3. BO adds adaptive value beyond sampling or choosing that structure.

### Consequence

For the chosen `x0`, BO and BO-marginal random beat uniform random, but the
medium fixed window was strongest. This medium window matches the target
observation design `psi0`, so future BO claims must compare against the
baseline of simulating at or near the known observation design.

## Decision: Increase BO Budget And Reward Rounds

### Date

2026-05-24

### Context

The 20-round BO runs left the final BO objective close to random and
BO-marginal random controls. This suggests the current BO cycle may still have
too few reward observations to improve the objective reliably.

### Decision

Keep the initial random data and per-round batch fixed, but increase the default
number of BO rounds:

```text
initial = 80
batch = 5
rounds = 30
final_budget = 230
```

### Reason

This modestly increases the total simulation budget while giving the BO
surrogate ten more sequential feedback points.

### Consequence

The 30-round broad-validation run made BO best on final range-normalized RMSE
and on the combined objective:

```text
random:             RMSE 0.2475, objective -0.4196
bo:                 RMSE 0.2342, objective -0.4041
bo_marginal_random: RMSE 0.2504, objective -0.4308
medium_mid fixed:   RMSE 0.2539, objective -0.4353
```

This is the clearest broad-validation BO signal so far. It is still not a final
claim: there are only two repeats, and uniform random has slightly better
coverage error than BO.

## Decision: Align Approach 1.2 And 1.3 Summary Metrics

### Date

2026-05-24

### Context

Approach 1.2 and Approach 1.3 were being compared by range-normalized RMSE, but
Approach 1.3 also reports the BO-style combined objective:

```text
-range_normalized_rmse
- 0.25 * coverage_error
- 0.1 * posterior_mean_predictive_rmse
```

### Decision

Add the same objective to Approach 1.2 diagnostics and summaries, while keeping
the shared component metrics:

```text
range_normalized_rmse
coverage_error
posterior_mean_predictive_rmse
posterior_quality_objective
```

### Reason

This makes the four-method suitability curve comparable to the BO
design-effect diagnostic on at least three direct measures plus the same
combined objective.

### Consequence

The existing Approach 1.2 result bundle was regenerated from already completed
diagnostics without rerunning simulations. It now reports the aligned objective.
The comparison still requires caution because the existing Approach 1.2 run uses
`hard_window`, higher FMPE training settings, and a larger final budget than the
current Approach 1.3 BO runs.

## Decision: Use Wide-Window 1.2 Sweep To Locate FMPE Budget Region

### Date

2026-05-24

### Context

The 30-round BO design-effect run uses only `230` training simulations. The
earlier Approach 1.2 curve suggested FMPE improves substantially at larger
budgets, but that curve used `hard_window` and stronger FMPE settings.

### Decision

Run Approach 1.2 under the same broad design setting and FMPE settings as the
current BO diagnostic:

```text
design_space = wide_window
budgets = 80 180 230 500 1000 2000
validation = 100
repeats = 2
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
```

### Reason

This translates the BO simulation budget into the four-method suitability
curve. In this setup, `budget = N` in Approach 1.2 is directly comparable to
`simulations = N` in Approach 1.3.

### Consequence

At the BO budget scale, FMPE is still noisy:

```text
FMPE budget 80:   RMSE 0.2735, objective -0.4551
FMPE budget 180:  RMSE 0.2373, objective -0.4029
FMPE budget 230:  RMSE 0.2438, objective -0.4186
FMPE budget 500:  RMSE 0.1963, objective -0.3551
FMPE budget 1000: RMSE 0.1745, objective -0.3179
FMPE budget 2000: RMSE 0.1328, objective -0.2456
```

The current BO budget of `230` lies before the strongest FMPE improvement
region. This suggests future BO tests should either use a larger final budget
or explicitly frame current runs as low-budget design diagnostics.

## Decision: Extend Wide-Window Budget Sweep Beyond 2000

### Date

2026-05-24

### Context

The wide-window budget sweep showed a large FMPE improvement by `2000`
simulations, so the saturation point was still unclear.

### Decision

Run an extended Approach 1.2 wide-window sweep with the same FMPE settings:

```text
budgets = 80 180 230 500 1000 2000 3000 5000
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
```

### Reason

This tests whether the drop at `2000` was a late improvement step or the
beginning of saturation.

### Consequence

FMPE continues improving in point-estimate and predictive metrics:

```text
FMPE budget 2000: RMSE 0.1357, objective -0.2587, predictive 0.7989
FMPE budget 3000: RMSE 0.1179, objective -0.2128, predictive 0.6442
FMPE budget 5000: RMSE 0.0978, objective -0.1993, predictive 0.6094
```

The point-estimate curve has not clearly saturated by `5000`. The combined
objective improves less after `3000` because FMPE coverage error worsens again
at `5000`; Gaussian NPE has the best final objective because it is much better
calibrated.

## Decision: Move BO Diagnostic To The 500-Simulation FMPE Region

### Date

2026-05-24

### Context

The wide-window Approach 1.2 sweeps showed that `230` simulations is still early
for FMPE, while `500` simulations is the first clearly improved region under
the same estimator settings.

### Decision

Run Approach 1.3 with the same number of BO feedback rounds but a larger
per-round batch:

```text
initial = 80
batch = 14
rounds = 30
final_budget = 500
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
```

### Reason

This applies the 1.2 budget finding to BO while keeping the estimator settings
and number of reward observations fixed.

### Consequence

BO remains best on final RMSE and combined objective:

```text
random:             RMSE 0.2206, objective -0.3780
bo:                 RMSE 0.2161, objective -0.3670
bo_marginal_random: RMSE 0.2207, objective -0.3806
medium_mid fixed:   RMSE 0.2573, objective -0.4344
```

The signal is positive but small: BO improves RMSE by only `2.1%` relative to
uniform random. It is stronger evidence than the `230` run because BO also has
better coverage than random here, but more repeats or a higher estimator
training budget are needed before making a robust adaptive-design claim.

## Decision: Do Not Increase FMPE Max Iteration Cap Alone For Budget-500 BO

### Date

2026-05-24

### Context

The budget-500 BO diagnostic used `fmpe_max_iter = 220`. Earlier warning output
raised the question of whether the lightweight rectified FMPE estimator was
being limited by the `MLPRegressor` optimizer iteration cap.

### Decision

Repeat the same Approach 1.3 run while changing only:

```text
fmpe_max_iter = 500
```

The output folder is:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_iter500/
```

### Reason

This isolates optimization budget from simulation budget and BO design choices.

### Consequence

The final-round metrics were unchanged from the `fmpe_max_iter = 220` run:

```text
random:             RMSE 0.2206, objective -0.3780
bo:                 RMSE 0.2161, objective -0.3670
bo_marginal_random: RMSE 0.2207, objective -0.3806
fixed_dumb:         RMSE 0.3294, objective -0.5282
```

Only one intermediate random-design row changed slightly. For this diagnostic,
the max-iteration cap is therefore not the active bottleneck. Future work
should either spend effort on more simulations/repeats or first instrument
`RectifiedFMPE` to record `MLPRegressor.n_iter_` before trying still larger
`fmpe_max_iter` values.

## Decision: Treat Current BO Adaptivity As Unsupported After 10 Paired Repeats

### Date

2026-05-24

### Context

The budget-500 BO diagnostic had only two repeats and showed a small positive
BO signal. To verify whether this was stable, the diagnostic was repeated with
`10` paired repeats at the same final budget.

The first attempted full roundwise repeat run was interrupted after more than
three hours because posterior-predictive scoring got stuck inside a stiff
`solve_ivp` call. A bounded posterior-predictive timeout and final-round-only
baseline scoring were added before rerunning the verification.

### Decision

Use the targeted result bundle:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_repeats10_finalonly/
```

The run keeps:

```text
initial = 80
batch = 14
rounds = 30
final_budget = 500
repeats = 10
fmpe_max_iter = 220
```

It adds:

```text
predictive_timeout_seconds = 0.5
final_score_only = True
structured_policies = medium_mid
```

### Reason

This keeps BO's sequential reward loop intact while avoiding unnecessary
roundwise rescoring of non-BO baselines and preventing one pathological
posterior-predictive ODE solve from stalling the whole experiment.

### Consequence

The `10`-repeat final means do not support a robust BO advantage:

```text
random:             RMSE 0.2066, objective -0.3571
bo:                 RMSE 0.2179, objective -0.3773
bo_marginal_random: RMSE 0.2168, objective -0.3720
fixed_dumb:         RMSE 0.3282, objective -0.5520
medium_mid fixed:   RMSE 0.2626, objective -0.4595
```

Paired deltas make the negative interpretation clearer:

```text
BO - random RMSE:       +0.0112, approx 95% interval [-0.0002, +0.0227]
BO - random objective:  -0.0202, approx 95% interval [-0.0386, -0.0018]
BO - BO-marginal RMSE:  +0.0011, approx 95% interval [-0.0053, +0.0074]
BO - BO-marginal obj.:  -0.0053, approx 95% interval [-0.0163, +0.0058]
```

For this setting, BO clearly beats the intentionally poor fixed dumb design,
but it does not beat uniform random and does not separate from the BO-marginal
random control. The current adaptive BO loop should therefore not be treated as
a useful contribution without changing the design problem, reward, or baseline
structure.

## Decision: Try Categorical BO Over Observation-Window Choices

### Date

2026-05-27

### Context

The continuous BO design space may be too noisy and too flexible for the small
number of reward observations. A simpler categorical design space could make it
easier for BO to learn which observation-window families are useful.

### Decision

Add `bo_design_mode = categorical`, where BO and the random baseline choose from
the same finite set of named `wide_window` categories:

```text
short / medium / long t_span
early / middle / late t_start
```

The categorical adaptive policy uses a finite-arm UCB rule over these named
categories. The BO-marginal-random control samples from the empirical joint
distribution of BO-selected categories rather than independently jittering
continuous dimensions.

### Reason

This tests whether BO can learn a simple design family choice before asking it
to optimize raw continuous `psi` values.

### Consequence

A two-repeat budget-500 run is stored in:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2/
```

Final means:

```text
random categorical:   RMSE 0.2498, objective -0.4200
bo categorical:       RMSE 0.2384, objective -0.4263
bo_marginal_random:   RMSE 0.2537, objective -0.4348
fixed_dumb:           RMSE 0.3621, objective -0.5798
medium_mid fixed:     RMSE 0.2599, objective -0.4545
```

BO now separates from BO-marginal-random on final RMSE in both repeats, which is
the desired sign that adaptivity may matter in the categorical space. However,
uniform categorical random still has the better combined objective because BO
has weaker coverage and posterior-predictive terms. Treat this as a promising
configuration change, not as a robust BO win until it survives more paired
seeds.

## Decision: Probe Reward-Psi Coupling Before Further BO Tuning

### Date

2026-05-28

### Context

The BO design-effect runs suggest that the adaptive loop may be trying to learn
from a weak or noisy reward landscape. Before changing the BO surrogate again,
the project needs to check whether different categorical `psi` choices produce
detectably different one-step reward changes.

### Decision

Add `approach_1_4_reward_landscape_categorical_psi`, a controlled reward
landscape diagnostic. For each seed it:

```text
1. fixes one shared initial training set,
2. scores the initial FMPE baseline,
3. adds one batch at each of the nine categorical wide-window designs,
4. trains one FMPE per augmented dataset,
5. computes multiple higher-is-better reward deltas from the shared baseline.
```

The default batch mode is paired: every category receives the same accepted
`theta` values, so the diagnostic isolates `psi` effects before adding the
realistic noise of independently sampled batches.

### Reason

BO can only help if its chosen design variables are coupled to a reward function
strongly enough for a surrogate model to learn. This diagnostic separates the
reward-design signal question from the adaptive optimization question.

### Consequence

The first three-repeat paired run is stored in:

```text
experiments/results/lotka_volterra/approach_1_4_reward_landscape_categorical_psi/
```

Several rewards show moderate category-level signal, but rank stability is low.
For example, paired eta squared is around `0.44` for the combined
RMSE-coverage-predictive reward and `0.39` for range-normalized RMSE, while
rank stability remains close to zero. This supports further reward-landscape
testing before treating BO performance as a surrogate-model-only problem.

## Decision: Sweep Larger One-Step Reward Batches

### Date

2026-05-28

### Context

The first reward-landscape grid used `initial = 80` and one added batch of
`14` simulations. Most reward deltas were negative, which suggested that the
one-step feedback was below the FMPE noise floor. Approach 1.2 had already
shown that FMPE improves much more clearly at larger budgets.

### Decision

Extend the reward-landscape diagnostic with `--batch-sizes`, where larger
batches are nested prefixes of the same generated category batch. Run the first
larger paired sweep with:

```text
initial = 500
batch_sizes = 14 56 112 224
validation = 100
repeats = 3
```

The result bundle is:

```text
experiments/results/lotka_volterra/approach_1_4_reward_landscape_batch_sweep_initial500/
```

### Reason

This tests whether BO's apparent failure is partly caused by using reward
increments too small for FMPE to react to reliably. It also estimates what
batch scale creates a learnable category-level reward landscape.

### Consequence

Larger batches make the reward landscape much cleaner. For the combined
RMSE-coverage-predictive reward:

```text
+14:  paired eta^2 0.423, rank stability 0.028, best delta 0.0153
+56:  paired eta^2 0.675, rank stability 0.678, best delta 0.0179
+112: paired eta^2 0.618, rank stability 0.283, best delta 0.0260
+224: paired eta^2 0.825, rank stability 0.711, best delta 0.0279
```

For range-normalized RMSE alone, the signal strengthens even more by `+224`
simulations:

```text
+224: paired eta^2 0.894, rank stability 0.778, best category long_late
```

This supports moving BO toward larger per-decision batches, or at least
evaluating BO rewards at a batch scale where FMPE changes are measurable.
