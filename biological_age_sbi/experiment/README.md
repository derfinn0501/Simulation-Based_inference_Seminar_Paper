# Biological Age SBI Experiment

Runnable experiment area for the biological-age BayesFlow workflow.

## Intended Flow

```text
prior over biological age
-> simulator for bioindicator observations
-> BayesFlow posterior approximation
-> recovery, calibration, and posterior predictive checks
```

## Folders

- `src/bioage_sbi/`: reusable Python code for the simulator, adapters, training, and diagnostics.
- `scripts/`: command-line entry points for running simulations and training.
- `configs/`: experiment configurations.
- `data/raw/`: external dataset extracts or metadata used for simulator grounding.
- `data/processed/`: processed data summaries used by the simulator.
- `results/`: generated metrics, plots, and summaries.

Keep notebooks exploratory only; move reusable logic into `src/bioage_sbi/`.

## MLflow Tracking

Use the shared guided-research MLflow server with a separate experiment page for
this seminar project:

```bash
python biological_age_sbi/experiment/scripts/init_mlflow_project.py
```

If the active Python environment does not have MLflow installed, either install
it or run the script with an environment that already has MLflow. The config is
stored in `configs/mlflow_bioage.json`.

The project experiment name is `seminar-bioage-sbi`. Stepwise code-reading or
training work can use `bioage_sbi.mlflow_utils.start_bioage_run` and
`bioage_sbi.mlflow_utils.log_step` to log progress, metrics, and artifacts into
that experiment.
