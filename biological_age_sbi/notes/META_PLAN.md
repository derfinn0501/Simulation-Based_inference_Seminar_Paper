# Biological Age SBI Workflow Meta Plan

## Objective

Build a small BayesFlow-based SBI workflow for inferring biological age from simulated bioindicator measurements.

The first goal is not realism at full scale. The first goal is a simple, inspectable simulator that can later be made more realistic using evidence from biological-age datasets.

## Initial Simulator Idea

Create a simulator with:

- `theta`: biological age
- observed variables: a small vector of bioindicators
- simulator relation: bioindicators are sampled conditionally on biological age

Initial structure:

```text
biological_age ~ prior over realistic human ages
bioindicators ~ p(bioindicators | biological_age)
```

The first version can use simple conditional Gaussian relationships. Later versions should introduce more realistic features such as nonlinear trends, indicator correlations, measurement noise, and population heterogeneity.

## Data Grounding

Before making the simulator too complex, inspect common biological-age datasets and papers to identify:

- typical age ranges
- common bioindicators
- plausible marginal ranges
- age-dependent trends
- correlation structure between indicators
- missingness or measurement-noise patterns

The simulator should be adjusted only when there is a concrete reason from data or domain knowledge.

## Prior

The prior should be simple at first.

It only needs to represent realistic biological ages across humans, for example:

```text
biological_age ~ Uniform(min_age, max_age)
```

or a weakly informative distribution over adult ages. The key requirement is that simulated ages stay within plausible human ranges.

## Inference Network

Use BayesFlow to build the inference workflow.

The network choice should be calibrated by:

- trial and error on simulator-generated validation data
- posterior recovery diagnostics
- calibration diagnostics
- prior knowledge from BayesFlow examples about stable architectures

Do not optimize architecture before the toy simulator works. Start with the simplest BayesFlow setup that can learn the one-dimensional posterior over biological age.

## Development Stages

1. Define a minimal simulator with a few bioindicators.
2. Generate prior predictive samples and check whether ranges look plausible.
3. Train a simple BayesFlow posterior approximator.
4. Evaluate recovery and calibration on synthetic ground truth.
5. Add realistic indicator structure from dataset inspection.
6. Repeat diagnostics after each simulator change.

## Main Research Standard

Posterior plots alone are not sufficient evidence. Use synthetic ground-truth recovery, calibration, and posterior predictive checks before trusting the workflow.
