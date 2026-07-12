#!/usr/bin/env python3
"""Create the seminar bioage MLflow experiment and optionally log a smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = EXPERIMENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioage_sbi.mlflow_utils import configure_mlflow, load_mlflow_config, log_step, start_bioage_run


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "mlflow_bioage.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-smoke-run",
        action="store_true",
        help="Only create/select the MLflow experiment; do not log a setup run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_mlflow_config(args.config)
    experiment_name = args.experiment_name or config.get("experiment_name", "seminar-bioage-sbi")

    mlflow, _ = configure_mlflow(
        args.config,
        tracking_uri=args.tracking_uri,
        experiment_name=experiment_name,
    )
    experiment = mlflow.get_experiment_by_name(str(experiment_name))
    if experiment is None:
        raise RuntimeError(f"MLflow did not create or find experiment {experiment_name!r}")

    tracking_uri = mlflow.get_tracking_uri()
    experiment_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}"
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Experiment page: {experiment_url}")

    if args.no_smoke_run:
        return

    run_name = args.run_name or "bioage-mlflow-setup-smoke-test"
    with start_bioage_run(
        run_name=run_name,
        config_path=args.config,
        tracking_uri=args.tracking_uri,
        experiment_name=experiment_name,
        tags={"setup": "true"},
    ) as run:
        mlflow.log_params(
            {
                "project": config.get("project", "2026-seminar"),
                "experiment_name": experiment_name,
                "tracking_uri": tracking_uri,
            }
        )
        log_step(
            mlflow,
            "mlflow_setup",
            0,
            metrics={"reachable": 1.0},
            artifacts=[args.config],
        )
        run_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
        print(f"Smoke run ID: {run.info.run_id}")
        print(f"Smoke run page: {run_url}")


if __name__ == "__main__":
    main()
