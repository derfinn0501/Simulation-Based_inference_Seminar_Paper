# Reusable Codex Prompts

## Concept-To-Code Prompt

Explain the statistical idea behind this method first.
Then implement the smallest possible Python example.
Keep the code readable.
After coding, explain what each component corresponds to mathematically.

## Debugging Prompt

Find the conceptual or implementation bug.
Do not only fix syntax.
Check whether the code still matches the intended statistical model.

## Experiment Prompt

Create an experiment following `docs/EXPERIMENT_TEMPLATE.md`.
Before coding, state the research question, hypothesis, and evaluation metric.

## Refactoring Prompt

Refactor only for readability and reuse.
Do not change the statistical behavior unless explicitly requested.
After refactoring, state what changed and what stayed equivalent.

## Critical Review Prompt

Critically evaluate this experiment.
Focus on posterior validity, simulator assumptions, prior sensitivity, and
whether the evaluation actually supports the claim.
