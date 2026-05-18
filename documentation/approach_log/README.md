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
experiments/results/approach_1_1_*/
experiments/results/approach_1_2_*/
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

The prototype currently supports three design spaces.

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

This is currently the most useful diagnostic setting.

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

## Approach 1.2: Standalone FMPE Quality Check

Date: 2026-05-18

### Question

Before evaluating whether BO helps FMPE, first evaluate:

> Is the current FMPE-style posterior estimator itself good enough to trust?

This should be judged mostly from the `random_fmpe` runs, because those show FMPE under ordinary non-adaptive simulation design. The `bo_fmpe` runs mix posterior-estimator quality with active design quality.

### Why This Check Comes First

If FMPE is weak, then a BO comparison is hard to interpret:

- BO may appear useless because the posterior estimator cannot exploit better simulations.
- BO may appear useful only because training noise dominates the metric.
- RMSE improvements may hide bad uncertainty calibration.
- A design policy cannot be judged cleanly until the posterior estimator is at least reasonable.

So the first gate should be:

```text
Can FMPE learn a usable posterior from random simulator data?
```

Only after this is mostly true should the project ask:

```text
Does BO improve the simulation design?
```

### Minimal Baseline: Prior-Mean Prediction

The prior ranges are:

```text
alpha, gamma: [0.5, 1.5]
beta, delta:  [0.02, 0.08]
```

A trivial estimator that always predicts the prior mean has expected raw RMSE:

```text
prior_mean_raw_rmse = 0.204491
```

This is a weak baseline. FMPE should clearly beat this before it can be called useful.

However, raw RMSE is not enough because the parameter scales are very different. The rates `beta` and `delta` have much smaller numerical ranges than `alpha` and `gamma`, so raw RMSE can be dominated by the large-scale parameters.

The next evaluation should therefore include:

```text
per-parameter RMSE
prior-normalized RMSE
posterior calibration
posterior predictive checks
```

### Current Aggregate Results

Existing FMPE output folders give the following rough picture.

```text
outputs_fmpe_full_design
  random_fmpe final RMSE: 0.179174
  improvement over prior mean: 12.4%
  mean coverage error: 0.377199

outputs_fmpe_full_design_long
  random_fmpe final RMSE: 0.146057
  improvement over prior mean: 28.6%
  mean coverage error: 0.333810

outputs_fmpe_hard_window
  random_fmpe final RMSE: 0.176302
  improvement over prior mean: 13.8%
  mean coverage error: 0.320100
```

The longer full-design run shows the best sign of learning:

```text
RMSE improves from about 0.178 at round 0
to about 0.146 at the final round.
```

This suggests that the FMPE-style model is learning some posterior signal.

### Current Judgment

The current FMPE performance should be described as:

> better than trivial, but not yet clearly good.

Reasons:

- It beats prior-mean prediction, but the improvement is modest in two of the three existing FMPE runs.
- The best run shows about 29% raw-RMSE improvement over the prior-mean baseline, which is encouraging but not decisive.
- Coverage error around `0.32` to `0.38` is high, so posterior uncertainty is not yet well calibrated.
- The current RMSE is averaged over parameters with very different numerical scales.
- There is no exact FMPE log density yet, so the model cannot be evaluated with validation log posterior in the same way as the Gaussian baseline.
- There is no direct comparison yet against the simple Gaussian NPE-style baseline using the same RMSE metric.

### Working Definition Of "Good"

For this project, FMPE should only be called "good" if it satisfies most of the following:

1. It clearly beats the prior-mean baseline in prior-normalized RMSE.
2. It beats or at least matches the simple Gaussian NPE-style baseline.
3. Error decreases reliably as the number of simulations increases.
4. Coverage error is low enough that credible intervals have interpretable meaning.
5. Posterior predictive simulations reproduce the observed trajectory statistics.
6. The result holds across several random seeds, not just one run.

Until then, the safer wording is:

> The current lightweight FMPE prototype is a functional proof of concept, but not yet strong evidence of high-quality posterior inference.

### Implemented Diagnostic

The concrete diagnostic is now implemented in:

```text
experiments/active_fmpe_sbi/evaluate_fmpe_quality.py
```

It evaluates FMPE under random design only and reports:

```text
prior-mean RMSE
Gaussian baseline RMSE
FMPE posterior-mean RMSE
per-parameter RMSE
prior-normalized RMSE
coverage error
posterior predictive error
```

The result folder is:

```text
experiments/results/approach_1_2_fmpe_quality_check/
```

Run command used:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_fmpe_quality.py \
  --design-space hard_window \
  --budgets 60 100 140 \
  --validation 90 \
  --repeats 2 \
  --seed 515 \
  --output-dir experiments/results/approach_1_2_fmpe_quality_check
```

Main output files:

```text
diagnostics.csv
summary_by_budget.csv
fmpe_quality_summary.png
RESULTS.md
```

Final-budget summary at `140` random-design simulator calls:

```text
prior_mean:
  range-normalized RMSE: 0.2898
  prior-std-normalized RMSE: 1.0039
  coverage error: 0.0139
  predictive RMSE: 1.0358

gaussian_npe:
  range-normalized RMSE: 0.2309
  prior-std-normalized RMSE: 0.7997
  coverage error: 0.1426
  predictive RMSE: 0.8960

rectified_fmpe:
  range-normalized RMSE: 0.2246
  prior-std-normalized RMSE: 0.7782
  coverage error: 0.2787
  predictive RMSE: 0.8065
```

Current interpretation:

> FMPE is better than trivial and competitive on point-estimate quality, but not yet well calibrated.

The diagnostic answers:

```text
Is FMPE good enough before adding BO?
```

with:

```text
partly yes for point estimates,
not yet for posterior calibration.
```

## Approach 1.3: Four-Method Suitability Check

Date: 2026-05-18

Status: implemented and smoke-tested.

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
experiments/results/approach_1_3_four_method_suitability_check/
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

The exact simulated data used for this run is saved under:

```text
experiments/results/approach_1_3_four_method_suitability_check/simulated_data/
```

For each replicate, `train_full` contains the synthetic data that feeds `gaussian_npe` and `rectified_fmpe`. Each simulation budget uses the first `N` rows of that file.

## Next Tried Approach To Record

The next entry should use the next internal log number, for example `Approach 1.4`, and document one concrete change, not several at once.

Good candidates:

1. Keep the Gaussian baseline and improve the BO reward stability.
2. Keep the FMPE-style estimator and improve evaluation beyond RMSE.
3. Add a stronger FMPE implementation while keeping the simulator and BO loop fixed.
4. Test whether `hard_window` creates a clearer BO advantage than `full`.
5. Add simulation-based calibration or expected coverage diagnostics.
6. Compare BO against stronger non-adaptive design baselines.

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
