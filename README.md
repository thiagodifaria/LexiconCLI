# LexiconCLI

![LexiconCLI](https://img.shields.io/badge/LexiconCLI-Financial%20Analysis-111827?style=for-the-badge&logo=python&logoColor=white)

**LexiconCLI is a terminal-first financial analysis platform built in Python. It combines market monitoring, technical analysis, macroeconomic indicators, probabilistic forecasting, Monte Carlo simulation, backtesting, proactive alerts and experiment tracking in a service-oriented project layout.**

[![Version](https://img.shields.io/badge/Version-2.2.0-2563EB?style=flat)](README.md)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](service-api/service-python)
[![CLI](https://img.shields.io/badge/Interface-terminal-111827?style=flat)](client-cli)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-optional%20persistence-FDB515?style=flat&logo=timescale&logoColor=111827)](service-api/service-postgresql)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-optional%20events-FF6600?style=flat&logo=rabbitmq&logoColor=white)](infra)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?style=flat&logo=mlflow&logoColor=white)](service-api/service-python)
[![Runtime](https://img.shields.io/badge/Runtime-Docker%20Compose-2496ED?style=flat&logo=docker&logoColor=white)](infra)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat)](LICENSE)

---

## Documentation / Documentacao

**Leia em Portugues:** [README_PT.md](README_PT.md)  
**Read the detailed English README:** [README_EN.md](README_EN.md)  
**Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)  
**Operations:** [docs/OPERATIONS.md](docs/OPERATIONS.md)

---

## Visual Preview

### Main Menu

![LexiconCLI main menu](docs/assets/images/menu_principal.png)

### Dashboard

![LexiconCLI dashboard](docs/assets/images/dashboard_principal.png)

### Asset Analysis

![LexiconCLI asset analysis](docs/assets/images/analise_amzn.png)

### LSTM Forecast

![LexiconCLI LSTM forecast](docs/assets/images/previsao_lstm_amzn.png)

The project was reorganized from a compact layered CLI structure into a clearer runtime separation:

- `client-cli`: terminal client, application controller and Rich-based views.
- `service-api/service-python`: Python backend, data providers, models, workers and tests.
- `service-api/service-postgresql`: TimescaleDB migrations, seeds and PostgreSQL configuration.
- `infra`: Dockerfile and Compose stack.
- `scripts`: local operational commands.
- `docs`: architecture, operations and screenshots.

## Main Capabilities

- Interactive market dashboard with stocks, indices and custom watchlist.
- Historical data ingestion from Finnhub, yfinance, Alpha Vantage, BCB, FRED, Investpy and Nasdaq Data Link.
- Complete technical analysis with SMA, EMA, MACD, Bollinger Bands, ADX, RSI, Stochastic, ROC, Williams %R, OBV, ATR and volatility metrics.
- Bayesian LSTM and Prophet forecasting with uncertainty-aware evaluation.
- Monte Carlo simulation using Geometric Brownian Motion.
- Fundamental data through Finnhub financial statements.
- Backtesting for trading strategies.
- Price, RSI and MACD alert rules with OS notifications.
- Optional Finnhub WebSocket ingestion for realtime ticks.
- Optional TimescaleDB persistence for ticks and candles.
- Optional RabbitMQ event delivery with local fallback.
- MLflow tracking and optional Weights & Biases logging.
- Continuous training command for explicit retraining runs.
- Portfolio allocation research based on risk/return scoring.

## Quick Start

Install locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\service-api\service-python\requirements.txt
```

Run the CLI:

```powershell
.\scripts\run-cli.ps1
```

Run tests:

```powershell
.\scripts\run-tests.ps1
```

## Container Stack

Start TimescaleDB and RabbitMQ:

```powershell
docker compose -f .\infra\compose.yaml up -d postgres rabbitmq
```

Initialize persistence:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-ops init-db
```

Run the CLI in a container:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-cli
```

## Operational Commands

```powershell
.\scripts\init-db.ps1
.\scripts\train.ps1 AAPL MSFT --period 5y
.\scripts\portfolio.ps1 AAPL MSFT NVDA --period 2y --risk-aversion 2.0
.\scripts\stream.ps1 AAPL MSFT --seconds 60
.\scripts\alert-worker.ps1
```

## Configuration

Create a `.env` file in the repository root when external integrations are needed.

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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact

Thiago Di Faria - [thiagodifaria@gmail.com](mailto:thiagodifaria@gmail.com)

Project link: [https://github.com/thiagodifaria/LexiconCLI](https://github.com/thiagodifaria/LexiconCLI)
