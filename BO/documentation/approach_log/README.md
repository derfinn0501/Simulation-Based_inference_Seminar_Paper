# 4 Approach Log: BO-Guided Simulation Design For FMPE-SBI

## Purpose Of This File

This file records the stepwise evolution of the project approach.

The number `4` is the section number in the overall workflow:

```text
1 -> Practical Guide summaries
2 -> FMPE summaries
3 -> contribution idea
4 -> approach log
```

Inside this section, approaches use their own numbering starting at `1.1`:

```text
Approach 1.1
Approach 1.2
Approach 1.3
...
```

This internal numbering matches the result-folder convention:

```text
experiments/results/lotka_volterra/approach_1_1_*/
experiments/results/lotka_volterra/approach_1_2_*/
```

The earlier notes establish the background and contribution:

- `1_*`: practical SBI foundations,
- `2_*`: FMPE and vector-field posterior estimation,
- `3_*`: possible contribution using BO-guided simulation design,
- `4_*`: tried approaches, diagnostics, and decisions.

This file should track what was actually tried. Each entry should make clear:

- what the approach was,
- why it was tried,
- what was kept fixed,
- what was changed,
- how it was evaluated,
- what was learned,
- what should be tried next.

The goal is to make the project history visible, not just the final result.

## Current Approach Snapshot

Date: 2026-05-18

Current working idea:

> Use Bayesian optimization to choose informative simulator or experiment designs for Lotka-Volterra SBI, while keeping physical parameters sampled from the prior. Compare BO-guided design against random design under the same simulation budget. First use a simple Gaussian posterior baseline, then replace it with a lightweight FMPE-style rectified-flow posterior estimator.

The current prototype lives in:

```text
experiments/active_fmpe_sbi/
```

Main scripts:

```text
run_lotka_volterra.py
run_lotka_volterra_fmpe.py
plot_experiment_results.py
```

## Fixed Principle

The most important design principle is:

```text
BO optimizes psi, not theta.
theta is still sampled from the prior.
```

The inferred physical parameters are:

```text
theta = (alpha, beta, gamma, delta)
```

for the Lotka-Volterra predator-prey system.

BO controls design variables:

```text
psi = simulator or observation design
```

Depending on the design space, `psi` can mean:

- initial prey population,
- initial predator population,
- observation start time,
- observation window length.

The simulator is therefore:

```text
x ~ p_sim(x | theta, psi)
```

and the posterior estimator is conditioned on both observation and design:

```text
q(theta | x, psi)
```

This avoids turning BO into direct parameter optimization. The posterior target stays clean because training parameters still come from:

```text
theta ~ p(theta)
```

## Approach 1.1: Lotka-Volterra Active Design Prototype

### Why This Was Tried

The broad research idea is too large to test all at once. The current approach reduces it to a small question:

> Under a fixed simulation budget, can BO-guided design improve posterior learning compared with random design?

Lotka-Volterra is useful because:

- the simulator is simple and physical,
- the latent parameters are interpretable,
- observation schedules and initial conditions are natural design variables,
- posterior quality can be tested on simulated validation pairs,
- random design can be made weak by choosing a harder observation-window design space.

### What Is Compared

Two design policies are compared:

```text
random:
    psi is sampled randomly.

bo:
    psi is selected by Bayesian optimization.
```

Both policies use:

```text
theta ~ p(theta)
```

for every new simulator call.

The comparison is budget-controlled:

```text
same initial simulations
same batch size
same number of rounds
same validation logic
```

The main question is whether BO reaches better posterior quality with the same number of simulator calls.

### Design Spaces

The prototype currently supports four design spaces.

`full`:

```text
psi = (prey0, pred0, t_span, t_start)
```

BO controls both initial populations and the observation window.

`window`:

```text
psi = (t_span, t_start)
```

Initial populations are fixed:

```text
prey0 = 40
pred0 = 15
```

`hard_window`:

```text
psi = (t_span, t_start)
```

Initial populations are fixed, but the observation-window range is harder. Many random windows can be weakly informative, which makes the usefulness of BO easier to diagnose.

`wide_window`:

```text
psi = (t_span, t_start)
```

Initial populations are fixed, `n_obs` is fixed, and `t_span` ranges from short dense windows to longer trend windows:

```text
t_span in [2, 24]
t_end <= 40
```

This is currently the most focused diagnostic for testing whether BO can exploit observation-window design.

## First Posterior Model Tried: Gaussian NPE-Style Baseline

Script:

```text
experiments/active_fmpe_sbi/run_lotka_volterra.py
```

The first implementation deliberately uses a simple posterior model:

```text
GaussianPosteriorRegressor
```

It predicts a posterior mean with an MLP and uses a residual Gaussian covariance. This is not the final model, but it gives an exact enough density proxy to compute:

```text
validation_log_posterior
```

The BO reward is:

```text
reward = J[q_new] - J[q_old]
```

where:

```text
J[q] = mean log q(theta_val | x_val, psi_val)
```

on held-out simulator-generated validation pairs.

This reward is useful because each validation observation has a known simulator parameter. The reward asks whether the updated posterior assigns more probability to the true generating parameters, not whether it moves toward one hand-picked target.

Additional diagnostic:

```text
coverage_error
```

This is a simple marginal Gaussian coverage proxy. Lower is better.

## Current Posterior Model: Lightweight FMPE-Style Rectified Flow

Script:

```text
experiments/active_fmpe_sbi/run_lotka_volterra_fmpe.py
```

This implementation keeps the same simulator, BO loop, design variables, and comparison structure, but replaces the Gaussian posterior baseline with:

```text
RectifiedFMPE
```

The current FMPE-style model works as follows:

1. Standardize `theta`.
2. Standardize the conditioning features `(x, psi)`.
3. Sample Gaussian noise `theta_0`.
4. Form interpolation points:

```text
theta_t = (1 - t) theta_0 + t theta_1
```

5. Train an MLP to predict the rectified-flow velocity:

```text
theta_1 - theta_0
```

conditioned on:

```text
(theta_t, t, x, psi)
```

6. Generate posterior samples by Euler-integrating the learned vector field from Gaussian noise.

This is not yet a full production FMPE implementation. It is a project-specific bridge:

> Can a conditional vector-field posterior estimator be inserted into the same BO-guided simulation design loop?

## BO Loop In The Current FMPE Prototype

For BO-guided runs:

1. Train the current posterior estimator on the existing dataset.
2. Evaluate posterior quality on a held-out validation set.
3. Let BO propose the next design `psi`.
4. Sample new physical parameters from the prior:

```text
theta_new ~ p(theta)
```

5. Simulate new observations under the BO-selected design:

```text
x_new ~ p_sim(x | theta_new, psi_bo)
```

6. Add the new data to the training set.
7. Retrain and re-evaluate the posterior estimator.
8. Give BO a scalar reward based on posterior improvement.

The BO surrogate is a Gaussian process with a Matern kernel. The acquisition is a simple UCB-style score:

```text
mean + 1.5 * std
```

The BO batch is partly focused on the chosen design and partly random by default. This keeps some exploration in the training set.

## Current FMPE Metrics

The lightweight FMPE model produces posterior samples but does not yet compute exact posterior densities. Therefore the current FMPE experiment does not use `validation_log_posterior`.

Instead it tracks:

```text
posterior_mean_rmse
coverage_error
```

The BO reward is:

```text
reward = old_rmse - new_rmse
```

Equivalently, the trace column is:

```text
reward_delta_negative_rmse
```

Positive reward means the posterior mean RMSE improved after adding the BO-selected simulations.

## How To Run The Current Approach

Gaussian baseline:

```bash
.venv/bin/python experiments/active_fmpe_sbi/run_lotka_volterra.py \
  --design-space hard_window \
  --initial 100 \
  --batch 15 \
  --rounds 6 \
  --validation 200 \
  --repeats 5 \
  --seed 21 \
  --output-dir experiments/active_fmpe_sbi/outputs_hard_window
```

FMPE-style run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/run_lotka_volterra_fmpe.py \
  --design-space hard_window \
  --initial 80 \
  --batch 12 \
  --rounds 5 \
  --validation 120 \
  --repeats 3 \
  --bo-candidates 128 \
  --posterior-samples 64 \
  --ode-steps 16 \
  --seed 404 \
  --output-dir experiments/active_fmpe_sbi/outputs_fmpe_hard_window
```

Summary plot:

```bash
.venv/bin/python experiments/active_fmpe_sbi/plot_experiment_results.py \
  --input-dir experiments/active_fmpe_sbi/outputs_fmpe_hard_window
```

Important output files:

```text
metrics.csv
bo_trace.csv
posterior_quality.png
posterior_rmse.png
experiment_summary.png
```

## How To Interpret The Current Results

Desired pattern:

1. BO improves posterior quality faster than random design.
2. BO does not substantially worsen coverage error.
3. BO rewards become less noisy or more often positive over rounds.
4. BO-selected designs become interpretable, for example by preferring informative observation windows.

Current status:

> This is still a diagnostic prototype, not final evidence for the project claim.

The most important question is still:

> Can posterior-quality improvement be turned into a stable enough scalar reward for BO?

If BO does not beat random design, possible explanations are:

- the design variables are not influential enough,
- random design is already strong,
- the BO reward is too noisy,
- the posterior model cannot exploit the selected simulations,
- the validation set does not match the design objective,
- FMPE training noise is larger than the effect of the design choice.

## What This Approach Has Clarified

The current implementation clarifies several important project decisions:

- BO should act on `psi`, not directly on `theta`.
- The posterior estimator should condition on `psi` when `psi` changes the data-generating process.
- The BO reward should measure posterior quality improvement, not just closeness to one observation.
- A simple Gaussian baseline is useful before FMPE because it provides a density-based reward.
- A lightweight FMPE-style model can be inserted into the same loop, but exact density evaluation is missing.
- The reward design problem is central, not a side detail.

## Current Weaknesses

The current approach still has several limitations:

- The FMPE implementation is lightweight and based on an MLP regressor, not a strong neural ODE or modern conditional flow-matching architecture.
- FMPE evaluation currently uses posterior mean RMSE, which may miss multimodality and uncertainty quality.
- The BO reward is noisy because it depends on simulator randomness, retraining, and validation sampling.
- The BO surrogate sees only one scalar reward per selected design round.
- The current validation setup is synthetic and does not yet test a fixed real observation `x_o`.
- Calibration diagnostics are still simple coverage proxies.

## Approach 1.2: Four-Method Suitability Check

Date: 2026-05-18

Status: implemented and rerun as the canonical suitability test.

### Goal

The next diagnostic should test whether the current Lotka-Volterra simulation setting is suitable for more complicated neural posterior estimators at all.

The comparison should include:

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```

Implementation:

```text
experiments/active_fmpe_sbi/evaluate_four_method_suitability.py
```

Result folder:

```text
experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check/
```

The main question is:

> Do simple non-neural methods already solve this setting better than the neural models?

If simple methods outperform the NN-based methods, that does not necessarily make the setting useless. It can instead become a controlled training environment for calibrating and understanding the more complex models.

### Working Hypothesis

With enough simulations, enough training, and good calibration, the complex models should at least approach the simple baselines.

Expected useful outcomes:

- If `abc_knn` performs well, the simulator contains usable posterior information.
- If `gaussian_npe` performs well but `fmpe` does not, the FMPE implementation or calibration needs work.
- If all methods remain close to `prior_mean`, the simulation setting is probably not informative enough.
- If FMPE has good RMSE but poor coverage, the issue is posterior uncertainty rather than point estimation.

### Budget Sweep

Use multiple simulation budgets instead of one fixed budget:

```text
100
250
500
1000
2000
```

Start smaller if runtime is too high:

```text
100
250
500
1000
```

The budget curve is important because `abc_knn` needs enough simulations to form meaningful local neighborhoods, while neural methods may need enough data and training time to become stable.

### Interrupt-Safe Result Writing

The experiment writes results live, one completed unit at a time.

Recommended unit:

```text
method x budget x replicate
```

After each unit, append to:

```text
diagnostics.csv
```

and flush the file. This makes the run restartable: if it is interrupted, already completed rows remain usable.

The script skips completed rows when restarted, based on:

```text
method
budget
replicate
seed
design_space
```

### Parallelization

The experiment can be parallelized because many units are independent.

Safe parallel options:

- parallelize over replicates,
- parallelize over methods within a budget,
- parallelize over `(method, budget, replicate)` jobs.

However, multiple workers should not write to the same CSV at the same time unless a safe writer/lock is used.

Simpler robust design:

```text
each worker writes its own small result file
main process merges results into diagnostics.csv
```

or:

```text
workers return results
main process is the only writer
```

The first implementation uses a sequential interrupt-safe writer. Parallelization can be added once the metric logic is stable.

### Evaluation

All four methods should be evaluated on the same validation observations and same training data for each budget.

Core metrics:

```text
posterior mean RMSE
per-parameter RMSE
prior-range-normalized RMSE
prior-std-normalized RMSE
coverage error
posterior predictive RMSE
```

For lower-is-better metrics, the ideal error pattern would be:

```text
prior_mean > abc_knn > gaussian_npe > rectified_fmpe
```

or whether simpler methods remain stronger than the NN-based models.

The goal is not only to pick the best method, but to decide whether this simulator is a good calibration and debugging environment for the complex models.

### Quick Smoke Test

Command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_four_method_suitability.py --quick --force
```

Smoke-test configuration:

```text
design_space = hard_window
budgets = 40 70
validation = 40
repeats = 1
seed = 616
```

Final-budget summary at `70` simulator calls:

```text
prior_mean:
  range-normalized RMSE: 0.2822
  prior-std-normalized RMSE: 0.9777
  coverage error: 0.0208
  predictive RMSE: 0.9588

abc_knn:
  range-normalized RMSE: 0.2251
  prior-std-normalized RMSE: 0.7797
  coverage error: 0.0479
  predictive RMSE: 0.8051

gaussian_npe:
  range-normalized RMSE: 0.2455
  prior-std-normalized RMSE: 0.8503
  coverage error: 0.3188
  predictive RMSE: 0.9266

rectified_fmpe:
  range-normalized RMSE: 0.2560
  prior-std-normalized RMSE: 0.8869
  coverage error: 0.3521
  predictive RMSE: 0.9336
```

Smoke-test interpretation:

> ABC-kNN is strongest in the tiny smoke test. This suggests the simulator setting contains learnable local information, while the NN-based methods need more data, better training-budget allocation, or calibration work before they can be trusted.

This smoke test only verifies implementation and output structure. It should not be treated as the full scientific result.

### Larger Default Run

Command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_four_method_suitability.py --force
```

Configuration:

```text
design_space = hard_window
budgets = 100 250 500 1000
validation = 200
repeats = 3
seed = 616
abc_k = sqrt(N), clipped to [10, 100]
flow_samples_per_pair = 4
posterior_samples = 48
ode_steps = 12
fmpe_max_iter = 350
```

Final-budget summary at `1000` simulator calls:

```text
prior_mean:
  range-normalized RMSE: 0.2867
  prior-std-normalized RMSE: 0.9932
  coverage error: 0.0106
  predictive RMSE: 1.0560

abc_knn:
  range-normalized RMSE: 0.2149
  prior-std-normalized RMSE: 0.7444
  coverage error: 0.0250
  predictive RMSE: 0.8705

gaussian_npe:
  range-normalized RMSE: 0.1396
  prior-std-normalized RMSE: 0.4835
  coverage error: 0.0757
  predictive RMSE: 0.7685

rectified_fmpe:
  range-normalized RMSE: 0.1307
  prior-std-normalized RMSE: 0.4527
  coverage error: 0.1635
  predictive RMSE: 0.7277
```

Observed final-budget ordering by range-normalized RMSE:

```text
rectified_fmpe < gaussian_npe < abc_knn < prior_mean
```

Interpretation:

> The simulator setting is learnable. ABC-kNN improves over the prior, which confirms that local simulator neighborhoods contain posterior information. With enough synthetic data, the NN-based methods overtake ABC-kNN on point-estimate and posterior-predictive metrics. Rectified FMPE is strongest at the final budget, but its coverage error remains worse than ABC-kNN and Gaussian NPE, so calibration is still the main weakness.

The exact simulated data used for this run was saved under:

```text
experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check/simulated_data/
```

For each replicate, `train_full` contains the synthetic data that feeds `gaussian_npe` and `rectified_fmpe`. Each simulation budget uses the first `N` rows of that file.

### Longer Observation Variant

A follow-up run keeps the same methods, budgets, validation size, repeats, seed, and `hard_window` design, but increases:

```text
n_obs = 20
```

The result folder used for this variant was:

```text
experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check_n_obs_large/
```

The larger-observation variant is not kept as the canonical result bundle. It can be rerun if the observation-length question becomes central again.

At the final budget of `1000` simulator calls, the range-normalized RMSE ordering remains:

```text
rectified_fmpe < gaussian_npe < abc_knn < prior_mean
```

This supports the same main conclusion as the original run: the setting is learnable, and the neural methods can outperform ABC-kNN on point-estimate metrics once enough synthetic data is available.

## Approach 1.3: BO Design Effect Check

Date: 2026-05-22

Status: implemented.

### Goal

This approach asks whether BO improves parameter estimation because it chooses better simulation designs, or whether it only looks competitive because it generates more data.

The key comparison is:

```text
same posterior estimator
same validation set
same initial training data
same final simulation budget
different design strategy for new simulations
```

### Compared Design Strategies

```text
random
fixed_dumb
bo
bo_marginal_random
```

`random` samples each new design uniformly from the design space.

`fixed_dumb` uses the same shared initial random data, then adds every later simulation batch at one intentionally naive fixed observation window. In the current `wide_window` setup this is:

```text
t_start = 0
t_span = 2
```

`bo` chooses new designs adaptively from previous validation-log-posterior improvements.

`bo_marginal_random` first lets BO reveal its preferred design region, then samples random designs from BO's empirical marginal `psi` distribution without adaptive feedback. This control is intentionally diagnostic: it asks whether the BO-selected region matters even if the stepwise adaptive sequence does not.

### Interpretation Logic

```text
BO better than random and BO-marginal random:
  evidence for adaptive BO design value

BO and BO-marginal random both better than random:
  evidence that the selected design region is useful
  but not yet evidence that stepwise adaptation matters

BO similar to random:
  no clear BO design benefit under this setup

BO better than fixed_dumb but not random:
  active design avoids a bad fixed design
  but does not beat a strong random baseline

BO has structured psi but no better posterior metrics:
  BO changes design, but the chosen structure is not useful for estimation
```

### Implementation

```text
experiments/active_fmpe_sbi/evaluate_bo_design_effect.py
```

Result folder:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check/
```

Default run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py
```

The first implementation uses the Gaussian NPE-style posterior estimator because it is stable, relatively fast, and has a validation log posterior reward. FMPE can be plugged into the same design comparison once the BO effect is understood with the simpler estimator.

Main output files:

```text
metrics.csv
summary_by_round.csv
design_trace.csv
design_summary_by_round.csv
bo_trace.csv
bo_design_effect_summary.png
design_structure_summary.png
RESULTS.md
```

`design_trace.csv` is important for checking whether BO-selected designs follow a recognizable structure instead of behaving like random draws.

### Previous Wide-Window Result

Configuration before adding the coverage-aware reward:

```text
design_space = wide_window
initial = 100
batch = 25
rounds = 4
final_budget = 200
validation = 180
repeats = 3
n_obs = 10
estimator = gaussian_npe
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2290
  validation log posterior: 6.5265
  coverage error: 0.1506
  predictive RMSE: 0.9229

bo:
  range-normalized RMSE: 0.2343
  validation log posterior: 6.1123
  coverage error: 0.1562
  predictive RMSE: 0.9451

bo_marginal_random:
  range-normalized RMSE: 0.2353
  validation log posterior: 5.6445
  coverage error: 0.1804
  predictive RMSE: 0.9039
```

Interpretation:

> There is no clear final-budget evidence that BO improves over uniform random design in this run. Even after making the observation-window choice more consequential, BO is slightly worse than random on range-normalized RMSE and validation log posterior.

The design trace still shows structure: BO-selected windows tend to concentrate in particular `t_start` and `t_span` regions within a replicate. However, that structure does not currently translate into better posterior estimation. This points more toward the BO reward or active-learning objective than toward the observation-window range itself.

### Coverage-Aware Reward Variant

A follow-up changed the BO reward from pure validation log posterior to:

```text
objective = validation_log_posterior - 5.0 * coverage_error
reward = objective_new - objective_old
```

The run also increased the number of BO rounds so the GP surrogate can actually use the reward after the initial exploratory proposals:

```text
design_space = wide_window
initial = 100
batch = 25
rounds = 8
final_budget = 300
validation = 180
repeats = 3
n_obs = 10
reward_mode = log_posterior_coverage
coverage_weight = 5.0
dumb_psi_unit = 0.0 0.0
estimator = gaussian_npe
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2238
  validation log posterior: 6.8409
  coverage error: 0.1196
  predictive RMSE: 0.9008

fixed_dumb:
  range-normalized RMSE: 0.2713
  validation log posterior: 3.7311
  coverage error: 0.2535
  predictive RMSE: 0.9614

bo:
  range-normalized RMSE: 0.2241
  validation log posterior: 5.5863
  coverage error: 0.2310
  predictive RMSE: 0.9158

bo_marginal_random:
  range-normalized RMSE: 0.2324
  validation log posterior: 5.3889
  coverage error: 0.2085
  predictive RMSE: 0.9125
```

Interpretation:

> The coverage-aware reward did not produce a BO advantage over random. BO is essentially tied with random on range-normalized RMSE, but random has better validation log posterior and better coverage. However, BO clearly beats the fixed dumb design, so adaptive design is not useless; it just has not improved over a strong random-design baseline.

### FMPE Reward Variant

The next run changed the posterior estimator from Gaussian NPE to the lightweight rectified-FMPE estimator. Since this FMPE implementation samples posteriors but does not compute exact log densities, the BO reward was changed to:

```text
objective = -range_normalized_rmse - 0.25 * coverage_error
reward = objective_new - objective_old
```

Configuration:

```text
design_space = wide_window
initial = 80
batch = 20
rounds = 5
final_budget = 180
validation = 100
repeats = 2
n_obs = 10
estimator = rectified_fmpe
reward_mode = rmse_coverage
coverage_weight = 0.25
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
dumb_psi_unit = 0.0 0.0
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2636
  objective: -0.3443
  coverage error: 0.3229
  predictive RMSE: 0.8738

fixed_dumb:
  range-normalized RMSE: 0.3017
  objective: -0.4045
  coverage error: 0.4113
  predictive RMSE: 0.9964

bo:
  range-normalized RMSE: 0.2698
  objective: -0.3651
  coverage error: 0.3813
  predictive RMSE: 0.9436

bo_marginal_random:
  range-normalized RMSE: 0.2689
  objective: -0.3524
  coverage error: 0.3342
  predictive RMSE: 0.9669
```

Interpretation:

> Switching to FMPE and using a sample-based reward did not produce a BO advantage over random. BO still beats the fixed dumb design, so it avoids a clearly weak observation strategy, but uniform random remains the stronger baseline on final range-normalized RMSE, objective, coverage, and predictive RMSE.

### FMPE Predictive-Reward Variant

The next reward added a simulator-facing predictive term:

```text
objective = -range_normalized_rmse
            - 0.25 * coverage_error
            - 0.1 * posterior_mean_predictive_rmse
reward = objective_new - objective_old
```

Configuration:

```text
design_space = wide_window
initial = 80
batch = 20
rounds = 5
final_budget = 180
validation = 100
repeats = 2
n_obs = 10
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

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2636
  objective: -0.4317
  coverage error: 0.3229
  predictive RMSE: 0.8738

fixed_dumb:
  range-normalized RMSE: 0.3017
  objective: -0.5042
  coverage error: 0.4113
  predictive RMSE: 0.9964

bo:
  range-normalized RMSE: 0.2708
  objective: -0.4602
  coverage error: 0.3738
  predictive RMSE: 0.9593

bo_marginal_random:
  range-normalized RMSE: 0.2672
  objective: -0.4428
  coverage error: 0.3208
  predictive RMSE: 0.9533
```

Interpretation:

> Adding posterior-mean predictive RMSE to the FMPE reward did not produce a BO advantage. BO still beats the fixed dumb design, but random remains better on final range-normalized RMSE, objective, coverage, and predictive RMSE. The BO-marginal random control is also slightly stronger than BO.

### FMPE Many-Reward-Rounds Variant

The next run kept the final simulation budget fixed at `180`, but changed the
allocation from few large reward updates to many small reward updates:

```text
initial = 80
batch = 5
rounds = 20
final_budget = 180
target_mode = validation_set
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2630
  objective: -0.4429
  coverage error: 0.3583
  predictive RMSE: 0.9030

fixed_dumb:
  range-normalized RMSE: 0.3018
  objective: -0.4951
  coverage error: 0.3933
  predictive RMSE: 0.9497

bo:
  range-normalized RMSE: 0.2584
  objective: -0.4425
  coverage error: 0.3629
  predictive RMSE: 0.9330

bo_marginal_random:
  range-normalized RMSE: 0.2596
  objective: -0.4328
  coverage error: 0.3050
  predictive RMSE: 0.9704
```

Interpretation:

> More reward rounds made BO best on final range-normalized RMSE, but only by a
> small margin. The combined objective still favored the BO-marginal random
> control, mainly because coverage was better there. This is a mixed positive
> signal: reward feedback can help point estimation, but it does not yet prove
> that the adaptive sequence improves posterior quality.

Structured fixed-window scores were also added. In the broad validation run, the
best simple structured design was the medium middle window:

```text
medium middle:
  t_start = 13.5
  t_span = 13.0
  range-normalized RMSE: 0.2579
  objective: -0.4403
```

This slightly beats BO on both RMSE and the combined objective, which means a
simple non-adaptive design can currently explain much of the apparent BO gain.

### Target-x0 Variant

The next step changed the evaluation target from a broad validation set to one
fixed synthetic observation:

```text
target_mode = fixed_x0
target_theta = 1.0 0.05 1.0 0.05
target_psi_unit = 0.5 0.5
```

This better matches the real SBI use case:

```text
given one observed time series x0, choose future simulations that improve
q(theta | x0, psi0)
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.1400
  objective: -0.2918
  coverage error: 0.1917
  predictive RMSE: 1.0391

fixed_dumb:
  range-normalized RMSE: 0.1427
  objective: -0.2606
  coverage error: 0.2333
  predictive RMSE: 0.5959

bo:
  range-normalized RMSE: 0.1087
  objective: -0.2170
  coverage error: 0.1417
  predictive RMSE: 0.7285

bo_marginal_random:
  range-normalized RMSE: 0.1030
  objective: -0.2159
  coverage error: 0.1583
  predictive RMSE: 0.7334
```

Interpretation:

> For a specific `x0`, BO and BO-marginal random both beat uniform random. This
> suggests the design region selected by BO can be useful for a target
> observation. However, BO-marginal random is slightly stronger than BO, so the
> adaptive sequence itself is still not clearly better than sampling from the
> BO-favored region.

The structured fixed-window scores are the most important diagnostic here:

```text
medium middle:
  t_start = 13.5
  t_span = 13.0
  range-normalized RMSE: 0.0574
  objective: -0.1405
  predictive RMSE: 0.3307
```

This fixed medium window beats BO and BO-marginal random for the chosen `x0`.
This is not surprising, because the target observation was generated at:

```text
target_psi_unit = 0.5 0.5
```

which is exactly the medium middle structured design. The important lesson is
therefore not that this window is universally optimal. The lesson is that, for a
target observation with known observation design `psi0`, simulating more data at
or near `psi0` is a strong baseline that BO must beat.

The current implication is:

> The observation-window choice matters, but simple structured windows may be a
> stronger baseline than the current BO policy. The next BO experiment should
> explicitly compare against structured designs and ask whether BO discovers
> them reliably across different `x0` values.

## Approach 1.4: More BO Rounds With Modestly Larger Budget

Date: 2026-05-24

Changed component:

```text
rounds = 30
final_budget = 230
```

Held fixed:

```text
design_space = wide_window
initial = 80
batch = 5
validation = 100
repeats = 2
target_mode = validation_set
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
```

Hypothesis:

> If the BO objective remains close to random because the surrogate has too few
> reward observations, then adding ten feedback rounds and 50 simulations should
> separate BO from random and BO-marginal random.

Run command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget230_round30
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2475
  objective: -0.4196
  coverage error: 0.3033
  predictive RMSE: 0.9624

bo:
  range-normalized RMSE: 0.2342
  objective: -0.4041
  coverage error: 0.3129
  predictive RMSE: 0.9163

bo_marginal_random:
  range-normalized RMSE: 0.2504
  objective: -0.4308
  coverage error: 0.3479
  predictive RMSE: 0.9347

medium middle fixed window:
  range-normalized RMSE: 0.2539
  objective: -0.4353
```

Main result:

> BO is best on final range-normalized RMSE and on the combined objective in
> this broad-validation run. It also beats the structured fixed-window scores,
> which is stronger than the previous 20-round evidence.

Decision:

> Treat this as the clearest broad-validation BO signal so far, but not as a
> settled adaptive-design claim. The run still has only two repeats, and uniform
> random has slightly better coverage error than BO.

Next step:

> Repeat or sweep target observations only after deciding whether the goal is a
> broad average design policy or target-specific simulation design near `psi0`.

## Approach 1.5: Wide-Window Budget Saturation For Approach 1.2

Date: 2026-05-24

Changed component:

```text
Approach 1.2 design_space = wide_window
budgets = 80 180 230 500 1000 2000
```

Held fixed:

```text
validation = 100
repeats = 2
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
reward_mode = rmse_coverage_predictive
```

Hypothesis:

> The BO run at final budget `230` may be operating before FMPE has enough
> training simulations to show its larger-budget behavior.

Run command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_four_method_suitability.py \
  --design-space wide_window \
  --budgets 80 180 230 500 1000 2000 \
  --validation 100 \
  --repeats 2 \
  --flow-samples-per-pair 3 \
  --posterior-samples 32 \
  --ode-steps 8 \
  --fmpe-max-iter 220 \
  --seed 717 \
  --output-dir experiments/results/lotka_volterra/approach_1_2_wide_window_budget_saturation
```

Main result:

```text
FMPE budget 80:   RMSE 0.2735, objective -0.4551
FMPE budget 180:  RMSE 0.2373, objective -0.4029
FMPE budget 230:  RMSE 0.2438, objective -0.4186
FMPE budget 500:  RMSE 0.1963, objective -0.3551
FMPE budget 1000: RMSE 0.1745, objective -0.3179
FMPE budget 2000: RMSE 0.1328, objective -0.2456
```

Decision:

> Treat `230` as an early, low-budget FMPE region. The strongest improvement in
> this wide-window setting starts after the current BO budget.

Next step:

> A larger-budget BO run should target at least the `500` simulation region if
> the question is whether BO helps once FMPE has enough training data.

### Extended Budget Check

The `2000` result still looked like the curve was dropping, so a follow-up
extended the same setup:

```text
budgets = 80 180 230 500 1000 2000 3000 5000
```

Result:

```text
FMPE budget 2000:
  range-normalized RMSE: 0.1357
  objective: -0.2587
  coverage error: 0.1725
  predictive RMSE: 0.7989

FMPE budget 3000:
  range-normalized RMSE: 0.1179
  objective: -0.2128
  coverage error: 0.1221
  predictive RMSE: 0.6442

FMPE budget 5000:
  range-normalized RMSE: 0.0978
  objective: -0.1993
  coverage error: 0.1625
  predictive RMSE: 0.6094
```

Interpretation:

> The point-estimate curve has not clearly saturated by `5000`. However, the
> combined objective improves much less after `3000` because FMPE coverage error
> worsens again. Gaussian NPE has the best final objective despite slightly
> worse RMSE, so calibration remains central.

## Approach 1.6: BO Design Effect At Budget 500

Date: 2026-05-24

Changed component:

```text
initial = 80
batch = 14
rounds = 30
final_budget = 500
```

Held fixed:

```text
design_space = wide_window
validation = 100
repeats = 2
target_mode = validation_set
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
```

Hypothesis:

> If the previous BO run was too early in the FMPE learning curve, moving the
> final budget to `500` should preserve or strengthen the BO advantage.

Run command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --initial 80 \
  --batch 14 \
  --rounds 30 \
  --validation 100 \
  --repeats 2 \
  --flow-samples-per-pair 3 \
  --posterior-samples 32 \
  --ode-steps 8 \
  --fmpe-max-iter 220 \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_round30
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2206
  objective: -0.3780
  coverage error: 0.2796
  predictive RMSE: 0.8752

bo:
  range-normalized RMSE: 0.2161
  objective: -0.3670
  coverage error: 0.2483
  predictive RMSE: 0.8889

bo_marginal_random:
  range-normalized RMSE: 0.2207
  objective: -0.3806
  coverage error: 0.2783
  predictive RMSE: 0.9040
```

Decision:

> BO remains best at budget `500`, including on the combined objective and
> coverage. The improvement over random is still small, so this is positive but
> not decisive evidence.

Next step:

> Before pushing BO to much larger budgets, test whether increasing
> `fmpe_max_iter` reduces MLP non-convergence warnings and improves the 1.2/1.3
> objective at budget `500` or `1000`.

## Approach 1.7: BO Budget 500 With Higher FMPE Iteration Cap

Date: 2026-05-24

Changed component:

```text
fmpe_max_iter = 500
```

Held fixed:

```text
design_space = wide_window
initial = 80
batch = 14
rounds = 30
final_budget = 500
validation = 100
repeats = 2
target_mode = validation_set
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
```

Hypothesis:

> If the budget-500 BO diagnostic is limited by under-training the FMPE
> estimator, increasing the MLP optimizer cap from `220` to `500` should improve
> RMSE, calibration, or the combined objective.

Run command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --initial 80 \
  --batch 14 \
  --rounds 30 \
  --validation 100 \
  --repeats 2 \
  --flow-samples-per-pair 3 \
  --posterior-samples 32 \
  --ode-steps 8 \
  --fmpe-max-iter 500 \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_iter500
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2206
  objective: -0.3780
  coverage error: 0.2796
  predictive RMSE: 0.8752

bo:
  range-normalized RMSE: 0.2161
  objective: -0.3670
  coverage error: 0.2483
  predictive RMSE: 0.8889

bo_marginal_random:
  range-normalized RMSE: 0.2207
  objective: -0.3806
  coverage error: 0.2783
  predictive RMSE: 0.9040
```

Decision:

> The final-round metrics are unchanged from the `fmpe_max_iter = 220` run.
> Only one intermediate random-design row changed slightly. For this specific
> BO diagnostic, increasing the max-iteration cap alone is not the active
> bottleneck.

Next step:

> Spend the next BO experiments on more simulations, more repeats, or sharper
> design baselines. If optimizer budget remains a concern, first record
> `MLPRegressor.n_iter_` in the FMPE diagnostics before trying `1000`.

## Approach 1.8: Budget 500 With 10 Paired Repeats

Date: 2026-05-24

Changed component:

```text
repeats = 10
final_score_only = True
predictive_timeout_seconds = 0.5
structured_policies = medium_mid
```

Held fixed:

```text
design_space = wide_window
initial = 80
batch = 14
rounds = 30
final_budget = 500
target_mode = validation_set
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
```

Hypothesis:

> If the two-repeat budget-500 BO win is real, it should survive paired
> replication and separate from both uniform random and BO-marginal random.

Implementation note:

> A first full roundwise `10`-repeat run was interrupted after more than three
> hours because posterior-predictive scoring got stuck inside one stiff
> `solve_ivp` call. The rerun bounds each posterior-predictive simulator call
> with `predictive_timeout_seconds = 0.5` and scores non-BO baselines only at
> the final round. BO is still scored every round because its reward depends on
> round-to-round objective changes.

Run command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --initial 80 \
  --batch 14 \
  --rounds 30 \
  --validation 100 \
  --repeats 10 \
  --flow-samples-per-pair 3 \
  --posterior-samples 32 \
  --ode-steps 8 \
  --fmpe-max-iter 220 \
  --predictive-timeout-seconds 0.5 \
  --final-score-only \
  --structured-policies medium_mid \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_repeats10_finalonly
```

Final-round summary:

```text
random:
  range-normalized RMSE: 0.2066
  objective: -0.3571
  coverage error: 0.2342
  predictive RMSE: 0.9194

bo:
  range-normalized RMSE: 0.2179
  objective: -0.3773
  coverage error: 0.2692
  predictive RMSE: 0.9210

bo_marginal_random:
  range-normalized RMSE: 0.2168
  objective: -0.3720
  coverage error: 0.2562
  predictive RMSE: 0.9113
```

Paired final deltas:

```text
BO - random RMSE:       +0.0112, approx 95% interval [-0.0002, +0.0227]
BO - random objective:  -0.0202, approx 95% interval [-0.0386, -0.0018]
BO - BO-marginal RMSE:  +0.0011, approx 95% interval [-0.0053, +0.0074]
BO - BO-marginal obj.:  -0.0053, approx 95% interval [-0.0163, +0.0058]
```

Decision:

> The two-repeat budget-500 BO advantage does not survive paired replication.
> BO is worse than uniform random on mean final RMSE and objective, and it is
> effectively tied with BO-marginal random. The current adaptive BO sequence is
> not adding meaningful value in this setting, though it still avoids the
> intentionally poor fixed dumb design.

Next step:

> Shift away from trying to rescue this exact BO loop. Test simpler structured
> or stratified design policies, and use target-`x0` sweeps to understand when
> observation-window design matters.

## Approach 1.9: Categorical BO Over Observation Windows

Date: 2026-05-27

Changed component:

```text
bo_design_mode = categorical
categorical choices = short/medium/long t_span crossed with early/middle/late t_start
```

Held fixed:

```text
design_space = wide_window
initial = 80
batch = 14
rounds = 30
final_budget = 500
validation = 100
repeats = 2
target_mode = validation_set
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
predictive_timeout_seconds = 0.5
final_score_only = True
```

Hypothesis:

> If continuous `psi` is too noisy for BO to learn, restricting BO to a small
> set of interpretable window categories should make adaptive design easier.

Run command:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --initial 80 \
  --batch 14 \
  --rounds 30 \
  --validation 100 \
  --repeats 2 \
  --flow-samples-per-pair 3 \
  --posterior-samples 32 \
  --ode-steps 8 \
  --fmpe-max-iter 220 \
  --predictive-timeout-seconds 0.5 \
  --final-score-only \
  --bo-design-mode categorical \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2
```

Final-round summary:

```text
random categorical:
  range-normalized RMSE: 0.2498
  objective: -0.4200
  coverage error: 0.3063
  predictive RMSE: 0.9362

bo categorical:
  range-normalized RMSE: 0.2384
  objective: -0.4263
  coverage error: 0.3504
  predictive RMSE: 1.0023

bo_marginal_random:
  range-normalized RMSE: 0.2537
  objective: -0.4348
  coverage error: 0.3429
  predictive RMSE: 0.9535
```

Main result:

> Categorical BO beats BO-marginal-random on final RMSE in both paired repeats,
> which is the first sign that adaptivity may matter after simplifying the
> design space. However, uniform categorical random still has the better
> combined objective because BO has weaker coverage and predictive metrics.

Decision:

> Categorizing `psi` is a promising BO lever, but this two-repeat result is not
> yet a robust BO win.

Next step:

> Repeat categorical BO with more paired seeds before changing the reward again.

## Next Tried Approach To Record

The next entry should use the next internal log number and document one concrete
change, not several at once.

Good candidates:

1. Sweep several fixed `x0` values and score whether the same medium window stays strong.
2. Test simple structured or stratified design policies instead of raw continuous BO.
3. Test a full posterior-predictive sample reward or simulation-based-calibration reward.
4. Increase repeats after the target/design baseline question is sharper.

For each next attempt, record:

```text
Approach number:
Date:
Changed component:
Held fixed:
Hypothesis:
Run command:
Main result:
Decision:
Next step:
```
