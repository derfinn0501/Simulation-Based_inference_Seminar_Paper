"""Small MLflow helpers for the biological-age SBI workflow."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator

DEFAULT_MLFLOW_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mlflow_bioage.json"


def load_mlflow_config(config_path: Path | str = DEFAULT_MLFLOW_CONFIG_PATH) -> dict[str, Any]:
    """Load the project MLflow config."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def import_mlflow() -> Any:
    """Import MLflow with an actionable error message."""

    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow is not installed in this Python environment. Install it with "
            "`python -m pip install mlflow` or run this script with an environment "
            "that already has MLflow."
        ) from exc
    return mlflow


def configure_mlflow(
    config_path: Path | str = DEFAULT_MLFLOW_CONFIG_PATH,
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Configure MLflow and select the seminar bioage experiment."""

    config = load_mlflow_config(config_path)
    mlflow = import_mlflow()

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or config.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(str(uri))

    experiment = experiment_name or config.get("experiment_name", "seminar-bioage-sbi")
    mlflow.set_experiment(str(experiment))
    return mlflow, config


@contextmanager
def start_bioage_run(
    run_name: str | None = None,
    *,
    config_path: Path | str = DEFAULT_MLFLOW_CONFIG_PATH,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    tags: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Start a configured MLflow run for stepwise bioage work."""

    mlflow, config = configure_mlflow(
        config_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    merged_tags = {**config.get("tags", {}), **(tags or {})}

    with mlflow.start_run(run_name=run_name or config.get("run_name", "bioage-code-reading")) as run:
        if merged_tags:
            mlflow.set_tags({str(key): str(value) for key, value in merged_tags.items()})
        yield run


def log_step(
    mlflow: Any,
    step_name: str,
    step_index: int,
    *,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[Path | str] | None = None,
) -> None:
    """Log one named workflow step to the active MLflow run."""

    safe_step_name = step_name.replace(" ", "_").replace("/", "_")
    mlflow.set_tag(f"step.{step_index}.name", step_name)
    mlflow.log_metric(f"step.{safe_step_name}.completed", 1.0, step=step_index)

    for key, value in (params or {}).items():
        mlflow.log_param(f"{safe_step_name}.{key}", value)

    for key, value in (metrics or {}).items():
        mlflow.log_metric(f"{safe_step_name}.{key}", float(value), step=step_index)

    for artifact in artifacts or []:
        path = Path(artifact)
        artifact_path = f"steps/{step_index:02d}_{safe_step_name}"
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
