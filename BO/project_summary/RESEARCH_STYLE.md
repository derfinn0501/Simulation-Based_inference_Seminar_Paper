# Research Style

## Preferred Thinking Pattern

Build understanding in layers:

1. intuitive explanation
2. mathematical formulation
3. minimal toy example
4. implementation
5. evaluation
6. critical limitations
7. possible research extension

## Preferred Explanations

Use precise but accessible language.
Connect equations to intuition.
Do not skip the statistical meaning of objects like prior, likelihood,
posterior, conditional density, simulator, vector field, and amortized inference.

## Preferred Coding Style

Start with small scripts or notebooks.
Use simple synthetic examples before complex benchmarks.
Prefer readable code over clever code.
Add comments explaining the statistical role of each step.

## Preferred Research Attitude

Be critical.
Do not assume that a neural posterior is valid just because it samples.

Always ask:

- Does the posterior have correct coverage?
- Does it recover known parameters in synthetic settings?
- Does it capture multimodality?
- Does it generalize to observations not well represented in training?
- Is the simulator misspecified?
- Is the prior too restrictive?
- Is the evaluation circular?

## Seminar Paper Orientation

The final output should support a seminar paper.
Code should produce figures, tables, and arguments that can be reused in writing.
