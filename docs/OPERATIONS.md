# Operations

## Install

Use Python 3.11 or another version supported by the scientific dependencies in `service-api/service-python/requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\service-api\service-python\requirements.txt
```

## Run CLI

```powershell
.\scripts\run-cli.ps1
```

## Run Tests

```powershell
.\scripts\run-tests.ps1
```

## Service Commands

Initialize SQLite and TimescaleDB when enabled:

```powershell
.\scripts\init-db.ps1
```

Train one or more symbols with experiment tracking:

```powershell
.\scripts\train.ps1 AAPL MSFT --period 5y
```

Generate a risk/return portfolio allocation:

```powershell
.\scripts\portfolio.ps1 AAPL MSFT NVDA --period 2y --risk-aversion 2.0
```

Run a bounded realtime ingestion session:

```powershell
.\scripts\stream.ps1 AAPL MSFT --seconds 60
```

Run the alert worker:

```powershell
.\scripts\alert-worker.ps1
```

## Enable TimescaleDB

The local infrastructure stack can be started from `infra/compose.yaml`:

```powershell
docker compose -f .\infra\compose.yaml up -d postgres rabbitmq
```

Provision PostgreSQL with TimescaleDB, create the target database and apply:

```powershell
psql $env:POSTGRES_DSN -f .\service-api\service-postgresql\migrations\001_timescale_market_data.sql
```

Then set:

```env
TIMESCALE_ENABLED=true
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/lexicon
```

## Enable RabbitMQ

Start RabbitMQ locally or provide a reachable broker URL:

```env
RABBITMQ_ENABLED=true
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
```

The application publishes market tick and alert events to the broker abstraction.

## Enable Realtime Stream

```env
REALTIME_ENABLED=true
FINNHUB_API_KEY=...
```

Realtime streaming subscribes to the active watchlist when the CLI starts.

For a standalone ingestion session, use:

```powershell
.\scripts\stream.ps1 AAPL MSFT --seconds 60
```

## Enable Experiment Tracking

Local MLflow is enabled by default:

```env
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=service-api/service-python/mlruns
MLFLOW_EXPERIMENT_NAME=lexiconcli
```

W&B is opt-in:

```env
WANDB_ENABLED=true
WANDB_PROJECT=lexiconcli
```

## Portfolio Allocation

Portfolio allocation is exposed as a research command, not an automated execution engine:

```powershell
.\scripts\portfolio.ps1 AAPL MSFT NVDA --period 2y --risk-aversion 2.0
```

The command reads historical close prices, estimates annualized return and volatility, and returns normalized weights.

## Container Commands

Run the CLI in a container:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-cli
```

Run operational commands in a container:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-ops init-db
docker compose -f .\infra\compose.yaml run --rm lexicon-ops train AAPL MSFT --period 5y
docker compose -f .\infra\compose.yaml run --rm lexicon-ops portfolio AAPL MSFT NVDA --period 2y --risk-aversion 2.0
```

Run the alert worker:

```powershell
docker compose -f .\infra\compose.yaml --profile workers up lexicon-worker-alerts
```
