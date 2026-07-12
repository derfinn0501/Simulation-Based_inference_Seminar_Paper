# Meta Plan: Amortization in SBI

## Core Question

Investigate whether amortized learning is merely a side effect of SBI or a core feature that explains much of its practical value.

Working hypothesis:

```text
SBI is not inherently amortized, but many modern neural SBI methods become powerful because they amortize inference across many simulated datasets.
```

The key distinction is:

- **Simulation-based inference**: infer parameters when likelihood evaluation is unavailable or impractical, but simulation is possible.
- **Amortized inference**: pay an upfront training cost so future posterior queries are cheap.

These concepts overlap, but they are not identical.

## Motivation

The original neural networks in SBI are often trained to approximate good posteriors. Amortization can appear as a training strategy, but it can also become the main reason to prefer SBI:

```text
many observations or many design settings
-> repeated posterior inference needed
-> amortized network can reuse simulation/training cost
```

This study should clarify when amortization is central and when it is only a convenience.

## Main Questions

1. What exactly is amortized in neural SBI?
2. Which SBI methods are amortized, sequential but partly amortized, or non-amortized?
3. When can SBI overtake MCMC in practice?
4. When is MCMC still the better default?
5. What problem features indicate that SBI is worth trying?

## SBI Versus MCMC Focus

MCMC is usually strong when:

- the likelihood is available and reasonably cheap
- posterior dimension is manageable
- only a small number of datasets must be analyzed
- exactness or asymptotic guarantees matter

SBI may become attractive when:

- the likelihood is unavailable, implicit, or expensive
- simulation is easier than likelihood evaluation
- many posterior queries are needed
- inference must be fast after training
- the same simulator is reused across tasks, subjects, or experimental designs
- posterior approximation quality can be validated with simulation-based diagnostics

## Output Goal

Create a practical decision framework:

```text
Given an inference problem, decide whether SBI is likely worth the cost.
```

The framework should identify key parameters such as simulation cost, likelihood availability, number of posterior queries, parameter dimension, observation dimension, simulator reuse, acceptable approximation error, and calibration requirements.

## Development Steps

1. Write a conceptual note separating SBI from amortized inference.
2. Summarize how BayesFlow examples use amortization.
3. Compare amortized SBI, sequential SBI, MCMC, ABC, and variational inference.
4. Define a checklist for when SBI can beat MCMC.
5. Design small toy comparisons where amortization clearly helps or clearly does not help.
6. Convert the checklist into a seminar-paper-friendly decision table.
