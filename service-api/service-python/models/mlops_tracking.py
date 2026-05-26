from contextlib import contextmanager
from typing import Any, Dict, Optional

from config.settings import (
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    WANDB_ENABLED,
    WANDB_PROJECT,
)
from utils.logger import logger


class ExperimentTracker:
    def __init__(self):
        self.mlflow = None
        self.wandb = None
        self._load_clients()

    def _load_clients(self):
        if MLFLOW_ENABLED:
            try:
                import mlflow

                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
                self.mlflow = mlflow
            except Exception as exc:
                logger.warning(f"MLflow disabled at runtime: {exc}")

        if WANDB_ENABLED:
            try:
                import wandb

                self.wandb = wandb
            except Exception as exc:
                logger.warning(f"Weights & Biases disabled at runtime: {exc}")

    @contextmanager
    def run(self, name: str, params: Optional[Dict[str, Any]] = None):
        active_mlflow = None
        active_wandb = None
        params = params or {}

        try:
            if self.mlflow:
                active_mlflow = self.mlflow.start_run(run_name=name)
                self.mlflow.log_params(_flatten(params))
            if self.wandb:
                active_wandb = self.wandb.init(project=WANDB_PROJECT, name=name, config=params, reinit=True)
            yield self
        finally:
            if active_wandb:
                active_wandb.finish()
            if active_mlflow:
                self.mlflow.end_run()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        flat = _flatten(metrics)
        if self.mlflow:
            for key, value in flat.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(key, float(value), step=step)
        if self.wandb:
            self.wandb.log(flat, step=step)

    def log_artifact(self, path: str):
        if self.mlflow:
            try:
                self.mlflow.log_artifact(path)
            except Exception as exc:
                logger.warning(f"Failed to log artifact '{path}': {exc}")


def _flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[full_key] = value
    return flattened
