# LexiconCLI - Financial Analysis System

LexiconCLI is an interactive financial analysis system developed in Python for terminal-based workflows. It is designed for investors, traders, quantitative analysts and financial market researchers who need market monitoring, technical analysis, macroeconomic context, forecasting, scenario simulation and alerting in a single tool.

The project now follows a service-oriented repository structure. The terminal client lives in `client-cli`, backend capabilities live in `service-api/service-python`, PostgreSQL and TimescaleDB assets live in `service-api/service-postgresql`, infrastructure files live in `infra`, operational scripts live in `scripts`, and technical documentation lives in `docs`.

The application can still run locally as a CLI with SQLite. When the optional infrastructure is enabled, it can also ingest realtime market ticks through WebSockets, persist time-series data in TimescaleDB, publish events through RabbitMQ, track experiments with MLflow or Weights & Biases, and run explicit retraining and portfolio allocation commands.

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
    workers/
    tests/
  service-postgresql/
    migrations/
    postgresql.conf
    seeds/

infra/
  Dockerfile
  compose.yaml

scripts/
docs/
```

## Features

- Market monitoring through an interactive dashboard.
- Custom watchlist for user-selected assets and indices.
- Macroeconomic indicators from BCB, FRED and Nasdaq Data Link.
- Technical analysis with trend, momentum, volume and volatility indicators.
- Bayesian LSTM and Prophet forecasting workflows.
- Uncertainty quantification with Monte Carlo Dropout.
- Monte Carlo scenario simulation with Geometric Brownian Motion.
- Fundamental analysis through Finnhub financial statements.
- Backtesting for strategy validation.
- Proactive alerts for price thresholds, RSI conditions and MACD crossovers.
- CSV and TXT exports.
- Rich terminal interface with menus, tables and terminal charts.

## Application Surfaces

LexiconCLI exposes the same financial engine through three practical surfaces:

- The interactive terminal client in `client-cli`, used for dashboards, analysis flows, model execution, alerts and exports.
- The operational command runner in `service-api/service-python/ops.py`, used for database initialization, retraining, realtime ingestion checks, latest tick inspection and portfolio allocation.
- The worker process in `service-api/service-python/workers`, currently used for alert delivery through the event broker.

This split keeps the terminal experience focused while making backend routines callable from scripts and containers.

## Data Sources

LexiconCLI integrates with:

- Finnhub for quotes, symbol search, company news and financial reports.
- yfinance for historical market data.
- Alpha Vantage for complementary historical data and news sentiment.
- python-bcb for Brazilian macroeconomic series such as IPCA, SELIC and PTAX.
- FRED for United States macroeconomic series.
- Nasdaq Data Link for alternative datasets and commodities.
- Investpy for additional historical data coverage.

## Technical Indicators

The technical analysis module includes:

- Trend: SMA, EMA, MACD, Bollinger Bands and ADX.
- Momentum: RSI, Stochastic Oscillator, ROC and Williams %R.
- Volume: OBV and volume moving averages.
- Volatility: ATR, rolling volatility and return standard deviation.

The indicator workflow is centralized in `AnalysisController` and `IndicadoresTecnicos`. Historical OHLCV data is normalized before calculation so providers with different column naming conventions can feed the same analysis path.

## Forecasting Models

### Bayesian LSTM

The LSTM pipeline supports directional classification, technical features, temporal features, sentiment score, focal loss, class weighting, adaptive feature selection, threshold optimization and uncertainty-aware inference.

Tracked metrics include accuracy, weighted F1-score, balanced accuracy, confidence metrics, class distributions and overfitting indicators.

### Prophet

The Prophet workflow supports trend and seasonality modeling, external regressors from technical indicators, confidence intervals, temporal backtesting and saved model loading.

### Monte Carlo Simulation

The Monte Carlo simulator uses Geometric Brownian Motion with dynamically calculated drift and volatility. It produces simulated price paths, percentiles, risk statistics and terminal visualizations.

## Backtesting and Simulation

The backtesting module currently includes an SMA crossover strategy with configurable windows, initial capital and commission. It can run a direct backtest or optimize the short and long moving average windows under a simple constraint.

The Monte Carlo flow validates inputs before simulation, checks that price data is usable, limits extreme simulation parameters and returns both the matrix of simulated paths and metadata for reporting.

## Advanced Data and Modeling Capabilities

The project includes:

- Data augmentation for small time-series datasets with jittering, time warping and magnitude warping.
- Adaptive feature selection using correlation filtering, variance checks and mutual information.
- Dynamic LSTM complexity management based on dataset size and feature count.
- Focal Loss and class weighting for imbalanced directional classes.
- Threshold optimization to improve F1-score on imbalanced classes.
- Ensemble-style workflows combining LSTM outputs with simulation context.

## Realtime and Event Architecture

Realtime market ingestion is available through Finnhub WebSocket support. When enabled, the stream subscribes to selected symbols, reconnects after disconnections, writes ticks to TimescaleDB when available and publishes `market.tick` events through the broker abstraction.

RabbitMQ can be enabled for distributed event delivery. When RabbitMQ is disabled, the backend uses a local in-process broker so the CLI remains functional without external infrastructure.

The alert worker consumes `alert.triggered` events and dispatches operating-system notifications.

## Persistence

SQLite remains the default local persistence layer for:

- API cache.
- Watchlist.
- Visual preferences.
- Alert definitions.

TimescaleDB is optional and used for:

- Realtime market ticks.
- Historical OHLCV candles.
- Time-series queries for market data.

The TimescaleDB schema is located in `service-api/service-postgresql/migrations`, and PostgreSQL runtime settings are defined in `service-api/service-postgresql/postgresql.conf`.

## Data Flow

The typical data flow is:

1. `DataController` requests quotes, historical data or macroeconomic series.
2. `MarketDataProvider` queries the preferred provider and falls back when possible.
3. `CacheAPIManager` stores API responses with source-specific TTLs.
4. Historical candles can be written to TimescaleDB when enabled.
5. Analysis, forecasting, simulation or backtesting modules receive normalized DataFrames.
6. CLI views render the result as Rich tables or Plotext charts.

Realtime ticks follow a separate path: `FinnhubRealtimeStream` receives WebSocket trade messages, persists ticks through `TimescaleStore` when available and publishes `market.tick` through the broker abstraction.

## MLOps

The project includes an experiment tracking adapter for MLflow and optional Weights & Biases.

MLflow is enabled by default with a local file backend. Training runs can log parameters, metrics and artifacts. This gives the LSTM and Prophet workflows a traceable experiment history instead of only saving local model files.

## Continuous Training

Continuous Training is exposed as an explicit operational command rather than an automatic hidden background process. Operators can retrain one or more symbols, capture metrics and track the run through MLflow.

Example:

```powershell
.\scripts\train.ps1 AAPL MSFT --period 5y
```

The command uses the same `DataController` and `PredictionController` used by the CLI. This avoids separate training logic and keeps model behavior consistent between interactive and operational execution.

## Portfolio Allocation Research

The project includes a lightweight portfolio research module. It estimates annualized return and volatility from historical close prices and suggests normalized weights using a risk/return policy.

This is a research tool and does not execute trades.

Example:

```powershell
.\scripts\portfolio.ps1 AAPL MSFT NVDA --period 2y --risk-aversion 2.0
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\service-api\service-python\requirements.txt
```

## Running the CLI

```powershell
.\scripts\run-cli.ps1
```

Direct execution:

```powershell
$env:PYTHONPATH = "$PWD\service-api\service-python;$PWD\client-cli"
python .\client-cli\main.py
```

## Container Execution

Start infrastructure:

```powershell
docker compose -f .\infra\compose.yaml up -d postgres rabbitmq
```

Initialize databases:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-ops init-db
```

Run the CLI:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-cli
```

Run the alert worker:

```powershell
docker compose -f .\infra\compose.yaml --profile workers up lexicon-worker-alerts
```

## Configuration

Create `.env` in the repository root.

```env
FINNHUB_API_KEY=
ALPHA_VANTAGE_API_KEY=
FRED_API_KEY=
NASDAQ_DATA_LINK_API_KEY=

REALTIME_ENABLED=false
TIMESCALE_ENABLED=false
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/lexicon

RABBITMQ_ENABLED=false
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=service-api/service-python/mlruns
MLFLOW_EXPERIMENT_NAME=lexiconcli

WANDB_ENABLED=false
WANDB_PROJECT=lexiconcli
```

## Usage

After starting the CLI, the main menu provides:

1. Dashboard with monitored assets and indices.
2. Asset analysis with historical charts, technical indicators, Monte Carlo simulation, fundamental data and backtesting.
3. Macroeconomic indicators.
4. Forecasting with LSTM or Prophet.
5. Watchlist, alert configuration, visual preferences and export tools.

## Validation and Test Coverage

The test suite covers cache behavior, data controller operations, SQLite persistence, technical indicators, alert checks, LSTM/Prophet model preparation and evaluator metrics.

The current validation command is:

```powershell
.\scripts\run-tests.ps1
```

Container configuration can be validated with:

```powershell
docker compose -f .\infra\compose.yaml config --quiet
```

## Operational Boundaries

The project is designed for analysis and research. It does not place broker orders, manage live brokerage accounts or claim production trading execution. Portfolio allocation and continuous training are explicit tools for investigation and model management, not automatic trading agents.

## Tests

```powershell
.\scripts\run-tests.ps1
```

Equivalent direct command:

```powershell
$env:PYTHONPATH = "$PWD\service-api\service-python;$PWD\client-cli"
pytest .\service-api\service-python\tests
```

## Screenshots

### Main Menu

![Main Menu](docs/assets/images/menu_principal.png)

### Dashboard

![Dashboard](docs/assets/images/dashboard_principal.png)

### Asset Analysis

![AMZN Analysis](docs/assets/images/analise_amzn.png)

### LSTM Forecast

![LSTM Forecast](docs/assets/images/previsao_lstm_amzn.png)

## Executable Packaging

PyInstaller packaging assets are kept in `scripts/pyinstaller`. The previous standalone executable flow remains available as a packaging target, but the current source layout should be reflected in the PyInstaller spec before producing a new release artifact.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact

Thiago Di Faria - [thiagodifaria@gmail.com](mailto:thiagodifaria@gmail.com)

Project link: [https://github.com/thiagodifaria/LexiconCLI](https://github.com/thiagodifaria/LexiconCLI)
