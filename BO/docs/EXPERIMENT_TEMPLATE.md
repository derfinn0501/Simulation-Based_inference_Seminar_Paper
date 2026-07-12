# Experiment Template

## Experiment ID

Example: `exp_001_gaussian_npe_baseline`

## Research Question

What do we want to learn?

## Hypothesis

What do we expect before running the experiment?

## Simulator

Define:

- parameter `theta`
- prior `p(theta)`
- simulator `p(x | theta)`
- observed data `x_0`

## Method

Define:

- model
- training objective
- conditioning variable
- sampling procedure

## Baseline

What do we compare against?

## Metrics

Use fixed metric names where possible:

- posterior mean error
- posterior variance error
- coverage
- posterior predictive error
- calibration diagnostic
- runtime
- number of simulations

## Result

What happened?

## Interpretation

What does the result mean?

## Failure Modes

What could make the conclusion invalid?

## Next Step

What should be tried next?
