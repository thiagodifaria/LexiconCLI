from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from models.mlops_tracking import ExperimentTracker
from utils.logger import logger


@dataclass
class TrainingJobResult:
    symbol: str
    status: str
    started_at: str
    finished_at: str
    metrics: dict


class ContinuousTrainingPipeline:
    def __init__(self, data_controller, prediction_controller, tracker: Optional[ExperimentTracker] = None):
        self.data_controller = data_controller
        self.prediction_controller = prediction_controller
        self.tracker = tracker or ExperimentTracker()

    def train_symbols(self, symbols: Iterable[str], period: str = "5y") -> list[TrainingJobResult]:
        results: list[TrainingJobResult] = []
        for symbol in symbols:
            started = datetime.utcnow().isoformat()
            metrics = {}
            status = "failed"
            try:
                historical = self.data_controller.buscar_dados_historicos(symbol, periodo=period)
                if historical.dataframe.empty:
                    raise RuntimeError(f"No historical data available for {symbol}")

                with self.tracker.run(f"lstm-{symbol}", {"symbol": symbol, "period": period}):
                    _, _, _, metrics = self.prediction_controller.treinar_avaliar_modelo_lstm_bayesian(
                        historical.dataframe.copy(),
                        "close",
                        symbol,
                    )
                    self.tracker.log_metrics(metrics or {})

                status = "completed"
            except Exception as exc:
                logger.error(f"Continuous training failed for {symbol}: {exc}", exc_info=True)
                metrics = {"error": str(exc)}
            finally:
                results.append(
                    TrainingJobResult(
                        symbol=symbol,
                        status=status,
                        started_at=started,
                        finished_at=datetime.utcnow().isoformat(),
                        metrics=metrics or {},
                    )
                )
        return results
