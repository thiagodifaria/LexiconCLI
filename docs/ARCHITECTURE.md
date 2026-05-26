# Architecture

LexiconCLI is now organized around service boundaries instead of the previous repository-level layered layout.

## Repository Layout

```text
client-cli/
  main.py
  app_controller.py
  views/

service-api/
  service-python/
    config/
    controllers/
    models/
    utils/
    tests/
  service-postgresql/
    migrations/
    seeds/

scripts/
docs/
infra/
```

## Client CLI

`client-cli` owns terminal interaction. It loads the backend package path at startup and delegates data, analysis, forecasting, simulation and persistence work to `service-api/service-python`.

The former application controller was moved to the client layer because it coordinates menus, Rich views, background alert checks and OS notifications.

## Python Service

`service-api/service-python` contains the backend capabilities:

- market data providers for Finnhub, yfinance, Alpha Vantage, BCB, FRED, Investpy and Nasdaq Data Link;
- local SQLite persistence for preferences, cache and alerts;
- optional TimescaleDB persistence for ticks and candles;
- optional Finnhub WebSocket ingestion for realtime trade ticks;
- optional RabbitMQ broker for distributed event delivery;
- technical indicators, backtesting, Monte Carlo simulation and forecasting;
- MLflow/W&B experiment tracking adapter;
- continuous training orchestration;
- portfolio allocation research primitives.

The current internal package names were preserved to keep the migration stable and avoid a large import rewrite in the same change.

## Persistence

SQLite remains the default local store for cache, watchlist, preferences and alert definitions.

TimescaleDB is optional and enabled through:

```env
TIMESCALE_ENABLED=true
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/lexicon
```

Schema assets live in `service-api/service-postgresql/migrations`. Runtime PostgreSQL settings live in `service-api/service-postgresql/postgresql.conf`.

## Infrastructure

Container assets live in `infra`.

- `infra/Dockerfile` builds the Python application image.
- `infra/compose.yaml` defines the CLI container, operational command container, alert worker, TimescaleDB and RabbitMQ.

The Compose stack mounts `service-api/service-postgresql/postgresql.conf` into PostgreSQL and applies migrations from `service-api/service-postgresql/migrations`.

## Events

The event layer is abstracted behind `models.message_broker`.

When `RABBITMQ_ENABLED=false`, the backend uses a local in-process broker. When enabled, events are published to a topic exchange named `lexicon.events`.

Currently emitted event keys:

- `market.tick`
- `alert.triggered`

`service-api/service-python/workers/alert_worker.py` consumes `alert.triggered` and dispatches operating-system notifications.

## Realtime Data

Realtime ingestion is implemented through Finnhub WebSocket support. It is disabled by default and activated with:

```env
REALTIME_ENABLED=true
FINNHUB_API_KEY=...
```

Ticks are written to TimescaleDB when available and always published to the broker abstraction.

## MLOps

`models.mlops_tracking.ExperimentTracker` supports MLflow by default and W&B when explicitly enabled. LSTM training logs model parameters and metrics through this abstraction.

MLflow defaults to a local file store under `service-api/service-python/mlruns`.

## Continuous Training

`models.training_pipeline.ContinuousTrainingPipeline` provides an orchestration layer for retraining a list of symbols and logging the resulting metrics. It is intentionally explicit: scheduled automation can call this pipeline without changing the forecasting code.

## Portfolio Research

`models.portfolio_rl` contains a lightweight portfolio environment and a risk/return policy agent. It gives the project a controlled bridge from prediction toward allocation research without treating the current model as a production trading agent.

The operational command is:

```powershell
.\scripts\portfolio.ps1 AAPL MSFT NVDA --period 2y --risk-aversion 2.0
```
