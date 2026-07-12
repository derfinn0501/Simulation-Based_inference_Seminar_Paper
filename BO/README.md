# SBI Flow Matching Research Project

This repository supports a Data Science research project on Simulation-Based
Inference, Neural Posterior Estimation, and Flow Matching for posterior
inference when likelihoods are unavailable or impractical.

The project is intentionally research-driven:

```text
research question
-> mathematical formulation
-> minimal simulator
-> inference method
-> posterior samples
-> evaluation
-> interpretation
-> next question
```

## Current Status

See the dedicated project summary:

```text
project_summary/CURRENT_STATE.md
```

## Repository Map

```text
AGENTS.md                  research-agent instructions
TASK_BACKLOG.md            actionable tasks
DECISION_LOG.md            methodological and structural decisions
INSTRUCTION_CHANGELOG.md   proposed and applied instruction updates
PROMPTS.md                 reusable Codex prompts

project_summary/           stable project context, current state, and learnings
docs/                      reusable templates and reference notes
documentation/             existing detailed project notes and approach logs
experiments/               runnable experiment scripts and active results
my_paper/                  seminar-paper notes and summaries
papers/                    source papers and literature assets
src/                       future reusable package code
notebooks/                 future exploratory notebooks
results/                   future consolidated result exports
tests/                     future automated tests
```

## Important Current Locations

- Approach log: `documentation/approach_log/README.md`
- Project summary: `project_summary/CURRENT_STATE.md`
- Active prototype: `experiments/active_fmpe_sbi/`
- Current results: `experiments/results/`
- Experiment template: `docs/EXPERIMENT_TEMPLATE.md`
- Evaluation principles: `docs/EVALUATION.md`

## Working With Codex

For non-trivial tasks, Codex should follow the workflow in `AGENTS.md`.
The short version is:

1. restate the task and research objective
2. identify decision-critical ambiguity
3. choose the simplest useful approach
4. execute a small, inspectable change or experiment
5. evaluate and record results
6. update backlog, decisions, and project summary files only when the result is stable

## Setup

The current experiments use Python and the existing `.venv/` environment.
From the repository root:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py --quick
```

Use full experiment runs only when the research question and expected output are clear.
