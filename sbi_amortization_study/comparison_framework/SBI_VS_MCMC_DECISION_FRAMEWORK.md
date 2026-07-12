# SBI Versus MCMC Decision Framework

## Purpose

Identify whether a problem is worth solving with SBI, especially amortized neural SBI, instead of standard posterior computation such as MCMC.

## Key Parameters

### 1. Likelihood Availability

Ask:

- Can the likelihood be evaluated?
- Is it differentiable?
- Is it numerically stable?
- Is it cheap enough to evaluate many times?

If the likelihood is available and cheap, MCMC remains a strong default. If the likelihood is unavailable but simulation is possible, SBI becomes much more relevant.

### 2. Simulation Cost

Ask:

- How expensive is one simulation?
- Can simulations be parallelized?
- Is simulator output reusable?
- Can simulation budgets support neural training?

SBI needs many simulations. It is attractive when simulation is feasible at scale or when the trained network will be reused enough to justify the upfront cost.

### 3. Number of Posterior Queries

Ask:

- Is there one observation or many?
- Will inference be repeated across subjects, experiments, time points, or design candidates?
- Is fast inference needed after training?

Amortized SBI is strongest when many posterior queries share the same simulator.

### 4. Parameter Dimension

Ask:

- How many inferred parameters `theta` are there?
- Are parameters identifiable from observations?
- Are there strong posterior correlations or multimodalities?

Very high-dimensional parameter spaces can challenge both SBI and MCMC. SBI needs careful diagnostics to ensure posterior approximation quality.

### 5. Observation Dimension and Structure

Ask:

- Is the observation low-dimensional and tabular?
- Is it a time series, image, spatial field, or set?
- Is a useful summary statistic known?
- Can a neural embedding learn useful summaries?

SBI becomes attractive when observations are complex but simulatable, and when neural summary networks can extract information better than hand-crafted statistics.

### 6. Reuse of Simulator Knowledge

Ask:

- Will the same simulator be used repeatedly?
- Will posterior inference be needed for many possible observations?
- Will experimental design or active learning repeatedly query posteriors?

Reuse is the main amortization argument.

### 7. Accuracy and Calibration Requirements

Ask:

- Is an approximate posterior acceptable?
- Can calibration be tested with synthetic ground truth?
- Are coverage, recovery, and posterior predictive checks feasible?

MCMC is preferable when exactness is important and the likelihood is tractable. SBI needs validation because neural posterior approximations can be biased or miscalibrated.

### 8. Online or Real-Time Constraints

Ask:

- Does inference need to run quickly after observations arrive?
- Is the expensive computation allowed upfront?

If yes, amortized SBI can beat MCMC by moving cost from inference time to training time.

## When SBI Can Beat MCMC

SBI is promising when many of these are true:

- likelihood unavailable or very expensive
- simulator available and reasonably parallelizable
- many posterior queries are needed
- same simulator is reused
- observation structure benefits from neural summaries
- approximate posterior is acceptable if well calibrated
- diagnostics can be run on synthetic ground truth
- fast inference after training matters

## When MCMC Is Probably Better

MCMC is probably better when many of these are true:

- likelihood is available and cheap
- only one or a few datasets need inference
- posterior dimension is moderate
- exactness and transparent diagnostics matter
- simulation is expensive
- neural training budget is not justified
- amortization cannot be reused

## Minimal Decision Rule

Use this first-pass rule:

```text
If likelihood is tractable and inference is needed only a few times, start with MCMC.

If likelihood is unavailable and simulation is cheap enough, consider SBI.

If many posterior queries share one simulator, amortized SBI becomes especially attractive.
```

## Empirical Checks to Design

To make this framework concrete, run toy comparisons varying:

- number of posterior queries
- simulation cost
- parameter dimension
- observation dimension
- posterior complexity
- required calibration quality

For each setting compare:

- upfront cost
- per-query inference cost
- posterior recovery
- calibration
- posterior predictive quality
