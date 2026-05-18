# Research Guide: BO-Guided Simulation Design For FMPE-SBI

## Final Goal

The final goal is to make a small, defensible contribution to simulation-based inference (SBI) with flow matching:

> Use Bayesian optimization (BO) to improve simulation design and simulator-budget allocation for flow-matching posterior estimation (FMPE), without using BO to directly optimize the inferred physical parameters.

BO should choose design variables, simulator settings, interventions, fidelity levels, or experiment configurations. FMPE should remain responsible for amortized posterior estimation.

The core distinction is:

$$
\text{BO chooses what simulations are informative.}
$$

$$
\text{FMPE learns } q_\phi(\theta \mid x).
$$

BO must not be framed as pushing the posterior toward a desired answer. It is only a tool for spending simulation budget more efficiently.

## Central Research Question

Given a fixed simulation budget, can BO-guided simulation design improve posterior quality or simulation efficiency compared with non-adaptive simulation design?

More concretely:

$$
q_{\phi}^{\text{BO}}(\theta\mid x)
\quad
\text{vs.}
\quad
q_{\phi}^{\text{baseline}}(\theta\mid x)
$$

under the same or smaller simulation budget.

## Principle To Preserve

Physical parameters should usually still be sampled from the prior:

$$
\theta \sim p(\theta).
$$

BO should optimize design variables:

$$
\psi_{t+1}
=
\arg\max_{\psi \in \Psi} a_t(\psi).
$$

Then simulations are generated as

$$
\theta_t \sim p(\theta),
\qquad
x_t \sim p_{\text{sim}}(x\mid \theta_t,\psi_t).
$$

This keeps the posterior target clean. If BO directly chooses $\theta$, then the training distribution changes and one needs sequential SBI, proposal correction, or importance weighting.

## Candidate Benchmark

Start with a simple physical simulator where:

- the simulator is available and cheap enough for repeated experiments,
- the parameter dimension is low,
- the observation dimension can be made simple or moderately complex,
- posterior quality can be evaluated,
- design variables can be introduced naturally.

Good candidates:

- Lotka-Volterra predator-prey dynamics
- SIR epidemic simulator
- damped harmonic oscillator
- projectile or ball-throw simulator
- simple gravitational or orbital toy model
- simple robot/friction/contact simulator, if available

A strong first choice is **Lotka-Volterra or SIR**, because they are standard SBI-style benchmarks and have natural design variables such as observation times, noise levels, number of measurements, or intervention conditions.

## Experimental Roadmap

### Step 1: Choose A Simple Physical Benchmark

Pick one simulator and define:

- parameters $\theta$ to infer,
- observations $x$,
- prior $p(\theta)$,
- design variables $\psi$,
- simulation budget $B$,
- evaluation metric.

Example:

$$
\theta =
(\alpha,\beta,\gamma,\delta)
$$

for Lotka-Volterra rates, and

$$
\psi =
\text{observation schedule, noise level, or initial condition}.
$$

Deliverable:

- one clean simulator interface,
- one prior,
- one fixed observation $x_o$ or test set,
- one baseline simulation dataset.

### Step 2: Apply Very Simple SBI And Evaluate Performance

Begin with a simple posterior estimator before FMPE.

Possible first baselines:

- NPE with a small normalizing flow,
- NPE with a Gaussian or mixture density output,
- existing `sbi` toolbox default NPE.

Train on simulations sampled with a non-adaptive design:

$$
\theta_i \sim p(\theta),
\qquad
\psi_i \sim p_0(\psi),
\qquad
x_i \sim p_{\text{sim}}(x\mid\theta_i,\psi_i).
$$

Evaluate:

- posterior predictive checks,
- coverage or SBC if feasible,
- C2ST against reference posterior if available,
- negative log probability on held-out simulations,
- posterior error as a function of simulation budget.

Deliverable:

- baseline SBI result without BO,
- first performance curve.

### Step 3: Add BO To The Simple Design

Now add BO only over design variables $\psi$.

Keep:

$$
\theta_i \sim p(\theta).
$$

Let BO choose:

$$
\psi_i = \arg\max_\psi a_i(\psi).
$$

Possible BO feedback functions:

1. observation relevance:

$$
u(\psi)
=
-
\mathbb{E}_{\theta,x}
\left[
d(S(x),S(x_o))
\right],
$$

2. posterior predictive fit:

$$
u(\psi)
=
-
d(S(\tilde{x}),S(x_o)),
$$

3. calibration improvement:

$$
u(\psi)
=
\operatorname{ECE}(q_{\phi_t})
-
\operatorname{ECE}(q_{\phi_{t+1}}),
$$

4. ensemble disagreement:

$$
u(\psi)
=
\mathbb{E}
\left[
\operatorname{Var}_m
\left(
\log q_{\phi_t^{(m)}}(\theta\mid x,\psi)
\right)
\right].
$$

Deliverable:

- simple SBI + BO comparison,
- same simulation budget as baseline,
- answer whether BO improves sample efficiency.

### Step 4: Make The Design More Complex Without BO

Increase complexity before adding BO again.

Possible complexity increases:

- higher-dimensional observation $x$,
- longer time series,
- more observation times,
- richer noise model,
- additional nuisance variables,
- more realistic simulator settings,
- more difficult prior.

Do not add BO yet. First establish that the more complex setting is learnable with standard SBI.

Deliverable:

- complex-design SBI baseline,
- clear failure modes or bottlenecks,
- updated evaluation metrics.

### Step 5: Add BO To The More Complex Design

Reintroduce BO over $\psi$.

Now BO should answer:

> Can adaptive design recover performance lost by increasing simulator complexity, or reach the same posterior quality with fewer simulations?

Compare:

$$
\text{complex SBI without BO}
\quad
\text{vs.}
\quad
\text{complex SBI with BO}.
$$

Deliverable:

- performance-vs-budget plot,
- BO acquisition trace,
- posterior diagnostics.

### Step 6: Arrive At Full Flow-Matching Posterior Estimation

Replace the simple posterior estimator with FMPE.

FMPE learns a vector field:

$$
v_{t,x}(\theta)
$$

and generates posterior samples by integrating

$$
\frac{d\theta_t}{dt}
=
v_{t,x}(\theta_t).
$$

At this stage, do not add BO immediately. First verify that FMPE works on the chosen benchmark under the non-adaptive design.

Evaluate:

- posterior quality,
- simulation efficiency,
- training time,
- sampling time,
- calibration,
- mass coverage if possible.

Deliverable:

- FMPE baseline without BO,
- comparison to simple NPE baseline.

### Step 7: Add BO To The Final FMPE Model

Finally combine both ideas:

$$
\text{BO-guided design}
+
\text{FMPE posterior estimation}.
$$

The final loop is:

$$
\psi_t
=
\arg\max_\psi a_t(\psi),
$$

$$
\theta_t \sim p(\theta),
\qquad
x_t \sim p_{\text{sim}}(x\mid\theta_t,\psi_t),
$$

$$
\mathcal{D}_{t+1}
=
\mathcal{D}_t
\cup
\{(\theta_t,\psi_t,x_t)\},
$$

$$
q_{\phi_{t+1}}(\theta\mid x,\psi)
\leftarrow
\text{FMPE training on } \mathcal{D}_{t+1}.
$$

Final comparison:

- simple SBI without BO,
- simple SBI with BO,
- complex SBI without BO,
- complex SBI with BO,
- FMPE without BO,
- FMPE with BO.

The contribution is successful if FMPE + BO reaches comparable or better posterior quality with fewer simulations, better calibration, or better posterior predictive fit.

## Minimum Viable Contribution

The smallest publishable/seminar-defensible version is:

1. one benchmark simulator,
2. one design variable $\psi$,
3. one baseline SBI method,
4. one FMPE implementation,
5. one BO acquisition function,
6. one budget-controlled comparison,
7. clear diagnostics.

The contribution does not need to prove that BO always helps. It is enough to show when it helps, when it fails, and why the design variable matters.

## What To Measure

Use at least three metric categories.

Posterior quality:

- C2ST against reference posterior,
- posterior log probability on held-out simulations,
- MMD or Wasserstein distance if reference samples exist.

Calibration:

- SBC,
- expected coverage,
- TARP,
- local coverage if feasible.

Simulation efficiency:

- posterior quality vs. number of simulator calls,
- number of simulations needed to reach a target error,
- wall-clock cost if simulator is expensive.

Scientific fit:

- posterior predictive checks,
- distance between summary statistics of posterior predictives and $x_o$,
- parameter recovery on synthetic observations.

## Main Risks

1. **BO optimizes the wrong thing.** A design may make simulations look close to $x_o$ but not improve posterior learning.

2. **BO overfits to one observation.** If the goal is amortized inference, the BO design objective must not become too specific to a single $x_o$ unless this is stated clearly.

3. **Design variables affect the posterior target.** If $\psi$ changes the data-generating process, the posterior may need to be conditioned on $\psi$:

$$
q_\phi(\theta\mid x,\psi).
$$

4. **BO overhead dominates.** If simulations are cheap, BO may not be worth it. The benchmark should either have an artificial budget limit or a simulator where calls are meaningfully costly.

5. **FMPE complexity hides the contribution.** The work should first show the BO idea in a simple SBI model before moving to FMPE.

## Suggested Project Narrative

The paper or seminar story can be:

1. SBI with FMPE is powerful but simulation budgets remain important.
2. BO should not estimate the posterior and should not directly optimize physical parameters.
3. BO can instead guide simulator or experiment design.
4. On a simple physical benchmark, BO-guided design improves simulation efficiency.
5. The same design principle can be combined with FMPE.
6. The learned FMPE vector field may provide useful uncertainty signals for future BO acquisition functions.

## Final One-Sentence Thesis

Bayesian optimization can make FMPE-based SBI more simulation-efficient by adaptively selecting informative simulator or experiment designs, while preserving FMPE as the amortized posterior estimator.
