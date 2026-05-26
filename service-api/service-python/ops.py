import argparse
import json
import os
import sys
import time


SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVICE_ROOT, "..", ".."))
CLIENT_ROOT = os.path.join(PROJECT_ROOT, "client-cli")
for path in (SERVICE_ROOT, CLIENT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from config.settings import DEFAULT_ASSETS_MONITOR
from controllers.data_controller import DataController
from controllers.prediction_controller import PredictionController
from models.database import inicializar_db
from models.timeseries_store import TimescaleStore
from models.training_pipeline import ContinuousTrainingPipeline
from utils.logger import configurar_logger


def _prediction_config() -> dict:
    from config.settings import LSTM_ACTIVATION_DENSE, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_LOOKBACK_PERIOD

    return {
        "lookback_period": LSTM_LOOKBACK_PERIOD,
        "epochs": LSTM_EPOCHS,
        "batch_size": LSTM_BATCH_SIZE,
        "activation_dense": LSTM_ACTIVATION_DENSE,
    }


def init_db(_args):
    inicializar_db()
    TimescaleStore().initialize()
    print("databases initialized")


def train(args):
    data_controller = DataController()
    prediction_controller = PredictionController(model_config=_prediction_config())
    prediction_controller.data_controller = data_controller
    pipeline = ContinuousTrainingPipeline(data_controller, prediction_controller)
    result = pipeline.train_symbols(args.symbols, period=args.period)
    print(json.dumps([item.__dict__ for item in result], indent=2, default=str))


def portfolio(args):
    data_controller = DataController()
    result = data_controller.sugerir_alocacao_portfolio(args.symbols, args.period, args.risk_aversion)
    print(json.dumps(result, indent=2, default=str))


def stream(args):
    data_controller = DataController()
    symbols = args.symbols or DEFAULT_ASSETS_MONITOR
    started = data_controller.iniciar_stream_tempo_real(symbols)
    print(json.dumps({"started": started, "symbols": symbols}))
    if not started:
        return 1
    try:
        if args.seconds <= 0:
            while True:
                time.sleep(1)
        else:
            time.sleep(args.seconds)
    finally:
        data_controller.parar_stream_tempo_real()
    return 0


def latest_ticks(args):
    data_controller = DataController()
    df = data_controller.obter_ticks_recentes(args.symbols, args.limit)
    if df.empty:
        print("[]")
    else:
        print(df.to_json(orient="records", date_format="iso"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexicon-ops")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-db")
    init_parser.set_defaults(func=init_db)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("symbols", nargs="+")
    train_parser.add_argument("--period", default="5y")
    train_parser.set_defaults(func=train)

    portfolio_parser = sub.add_parser("portfolio")
    portfolio_parser.add_argument("symbols", nargs="+")
    portfolio_parser.add_argument("--period", default="2y")
    portfolio_parser.add_argument("--risk-aversion", type=float, default=2.0)
    portfolio_parser.set_defaults(func=portfolio)

    stream_parser = sub.add_parser("stream")
    stream_parser.add_argument("symbols", nargs="*")
    stream_parser.add_argument("--seconds", type=int, default=0)
    stream_parser.set_defaults(func=stream)

    ticks_parser = sub.add_parser("latest-ticks")
    ticks_parser.add_argument("symbols", nargs="+")
    ticks_parser.add_argument("--limit", type=int, default=1)
    ticks_parser.set_defaults(func=latest_ticks)

    return parser


def main(argv=None):
    configurar_logger()
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
