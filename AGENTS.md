# AGENTS.md

## Contract

This is the agreed working instruction file for Codex in this repository.
Keep it tight.
Do not rewrite it unless the user explicitly asks for an instruction change or
approves a proposal recorded in `INSTRUCTION_CHANGELOG.md`.

## Project

Research project on Simulation-Based Inference, Neural Posterior Estimation,
Flow Matching, and BO-guided simulation design.

Goal: build small, inspectable experiments that support understanding and a
seminar-paper contribution.

## Operating Loop

For non-trivial tasks:

1. Restate the task and research objective briefly.
2. Ask only decision-critical clarifications.
3. State assumptions when proceeding without clarification.
4. Consider at most three approaches.
5. Choose the simplest useful approach and execute it.
6. Verify with the smallest meaningful check.
7. Record stable outcomes in the right file.

Avoid long planning loops. Prefer small experiments that reduce uncertainty.

## Recording Rules

- `TASK_BACKLOG.md`: actionable tasks.
- `DECISION_LOG.md`: methodological or structural decisions.
- `project_summary/LEARNINGS.md`: stable findings.
- `INSTRUCTION_CHANGELOG.md`: proposed or approved instruction changes.
- `docs/EXPERIMENT_TEMPLATE.md`: structure for non-trivial experiments.

Update these only when the entry is useful beyond the current message.

## Research Standards

- Separate `theta` inferred parameters from `psi` design variables.
- Do not treat posterior plots as sufficient evidence.
- Prefer coverage, predictive checks, synthetic ground-truth recovery, and
  baseline comparisons.
- Keep experiments reproducible: save config, metrics, result summary, and
  interpretation.

## Coding Standards

- Use Python.
- Prefer NumPy, pandas, matplotlib, scikit-learn, and PyTorch when needed.
- Keep notebooks exploratory; move reusable logic into `src/`.
- Do not add complex frameworks or abstractions before the toy case works.

## Verification

For code changes, run relevant quick checks when possible.
Report what was tested and what remains untested.
Never claim correctness without evidence.
