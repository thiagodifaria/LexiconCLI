# LexiconCLI - Sistema de Analise Financeira

LexiconCLI e um sistema interativo de analise financeira desenvolvido em Python para operacao no terminal. O projeto foi pensado para investidores, traders, analistas quantitativos e pesquisadores de mercado que precisam reunir monitoramento, analise tecnica, indicadores macroeconomicos, previsao, simulacao de cenarios e alertas em uma unica ferramenta.

O projeto agora segue uma organizacao orientada a servicos. O cliente de terminal fica em `client-cli`, as capacidades de backend ficam em `service-api/service-python`, os ativos de PostgreSQL e TimescaleDB ficam em `service-api/service-postgresql`, a infraestrutura fica em `infra`, os scripts operacionais ficam em `scripts` e a documentacao tecnica fica em `docs`.

A aplicacao ainda pode rodar localmente como CLI com SQLite. Quando a infraestrutura opcional e habilitada, ela tambem pode ingerir ticks em tempo real via WebSocket, persistir series temporais no TimescaleDB, publicar eventos no RabbitMQ, rastrear experimentos com MLflow ou Weights & Biases e executar comandos explicitos de retreinamento e alocacao de portfolio.

## Estrutura do Repositorio

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

## Funcionalidades

- Monitoramento de mercado em dashboard interativo.
- Watchlist personalizada para ativos e indices escolhidos pelo usuario.
- Indicadores macroeconomicos do BCB, FRED e Nasdaq Data Link.
- Analise tecnica com indicadores de tendencia, momentum, volume e volatilidade.
- Fluxos de previsao com Bayesian LSTM e Prophet.
- Quantificacao de incerteza com Monte Carlo Dropout.
- Simulacao Monte Carlo com Geometric Brownian Motion.
- Analise fundamentalista por demonstrativos financeiros via Finnhub.
- Backtesting para validacao de estrategias.
- Alertas proativos por preco, RSI e cruzamentos MACD.
- Exportacao de dados em CSV e TXT.
- Interface rica no terminal com menus, tabelas e graficos.

## Superficies da Aplicacao

O LexiconCLI expoe o mesmo motor financeiro por tres superficies praticas:

- O cliente interativo de terminal em `client-cli`, usado para dashboards, fluxos de analise, execucao de modelos, alertas e exportacoes.
- O executor operacional em `service-api/service-python/ops.py`, usado para inicializacao de banco, retreinamento, verificacoes de ingestao em tempo real, consulta de ticks recentes e alocacao de portfolio.
- O processo worker em `service-api/service-python/workers`, atualmente usado para entrega de alertas por meio do broker de eventos.

Essa separacao mantem a experiencia de terminal focada e torna rotinas de backend acionaveis por scripts e containers.

## Fontes de Dados

O LexiconCLI integra:

- Finnhub para cotacoes, busca de simbolos, noticias corporativas e demonstrativos financeiros.
- yfinance para dados historicos de mercado.
- Alpha Vantage para dados historicos complementares e sentimento de noticias.
- python-bcb para series macroeconomicas brasileiras como IPCA, SELIC e PTAX.
- FRED para series macroeconomicas dos Estados Unidos.
- Nasdaq Data Link para datasets alternativos e commodities.
- Investpy para cobertura historica complementar.

## Indicadores Tecnicos

O modulo de analise tecnica inclui:

- Tendencia: SMA, EMA, MACD, Bandas de Bollinger e ADX.
- Momentum: RSI, Oscilador Estocastico, ROC e Williams %R.
- Volume: OBV e medias moveis de volume.
- Volatilidade: ATR, volatilidade movel e desvio padrao dos retornos.

O fluxo de indicadores e centralizado em `AnalysisController` e `IndicadoresTecnicos`. Dados historicos OHLCV sao normalizados antes dos calculos para permitir que provedores com nomes de colunas diferentes alimentem o mesmo caminho de analise.

## Modelos de Previsao

### Bayesian LSTM

O pipeline LSTM suporta classificacao direcional, features tecnicas, features temporais, score de sentimento, Focal Loss, pesos de classe, selecao adaptativa de features, otimizacao de thresholds e inferencia com incerteza.

As metricas rastreadas incluem accuracy, F1 ponderado, balanced accuracy, metricas de confianca, distribuicao de classes e indicadores de overfitting.

### Prophet

O fluxo Prophet suporta modelagem de tendencia e sazonalidade, regressores externos derivados de indicadores tecnicos, intervalos de confianca, backtesting temporal e carregamento de modelo salvo.

### Simulacao Monte Carlo

O simulador Monte Carlo usa Geometric Brownian Motion com drift e volatilidade calculados dinamicamente. Ele gera trajetorias simuladas, percentis, estatisticas de risco e visualizacoes no terminal.

## Backtesting e Simulacao

O modulo de backtesting inclui uma estrategia de cruzamento de medias moveis simples com janelas configuraveis, capital inicial e comissao. Ele pode executar um backtest direto ou otimizar as janelas curta e longa sob uma restricao simples.

O fluxo de Monte Carlo valida parametros antes da simulacao, verifica se os dados de preco sao utilizaveis, limita parametros extremos e retorna tanto a matriz de trajetorias simuladas quanto metadados para relatorio.

## Recursos Avancados de Dados e Modelagem

O projeto inclui:

- Data augmentation para series temporais pequenas com jittering, time warping e magnitude warping.
- Selecao adaptativa de features com filtros de correlacao, variancia e mutual information.
- Controle dinamico de complexidade da LSTM com base no tamanho do dataset e quantidade de features.
- Focal Loss e class weighting para classes direcionais desbalanceadas.
- Otimizacao de thresholds para melhorar F1-score em classes desbalanceadas.
- Fluxos de ensemble combinando saidas da LSTM com contexto de simulacao.

## Arquitetura de Tempo Real e Eventos

A ingestao em tempo real esta disponivel via WebSocket da Finnhub. Quando habilitado, o stream assina simbolos selecionados, reconecta apos quedas, grava ticks no TimescaleDB quando disponivel e publica eventos `market.tick` pela abstracao de broker.

RabbitMQ pode ser habilitado para entrega distribuida de eventos. Quando RabbitMQ esta desabilitado, o backend usa um broker local em memoria para manter a CLI funcional sem infraestrutura externa.

O worker de alertas consome eventos `alert.triggered` e dispara notificacoes do sistema operacional.

## Persistencia

SQLite continua sendo a camada padrao de persistencia local para:

- Cache de APIs.
- Watchlist.
- Preferencias visuais.
- Definicoes de alertas.

TimescaleDB e opcional e usado para:

- Ticks de mercado em tempo real.
- Candles OHLCV historicos.
- Consultas de series temporais de mercado.

O schema do TimescaleDB fica em `service-api/service-postgresql/migrations`, e as configuracoes de runtime do PostgreSQL ficam em `service-api/service-postgresql/postgresql.conf`.

## Fluxo de Dados

O fluxo tipico de dados e:

1. `DataController` solicita cotacoes, historico ou series macroeconomicas.
2. `MarketDataProvider` consulta o provedor preferencial e aplica fallback quando possivel.
3. `CacheAPIManager` armazena respostas de API com TTL por fonte.
4. Candles historicos podem ser gravados no TimescaleDB quando habilitado.
5. Modulos de analise, previsao, simulacao ou backtesting recebem DataFrames normalizados.
6. As views da CLI renderizam os resultados como tabelas Rich ou graficos Plotext.

Ticks em tempo real seguem um caminho separado: `FinnhubRealtimeStream` recebe mensagens WebSocket de trades, persiste ticks via `TimescaleStore` quando disponivel e publica `market.tick` pela abstracao de broker.

## MLOps

O projeto inclui um adaptador de rastreamento de experimentos para MLflow e suporte opcional a Weights & Biases.

MLflow e habilitado por padrao com backend local em arquivo. Treinos podem registrar parametros, metricas e artefatos. Isso da aos fluxos de LSTM e Prophet um historico rastreavel de experimentos, em vez de apenas salvar modelos locais.

## Continuous Training

Continuous Training e exposto como comando operacional explicito, e nao como processo automatico oculto. O operador pode retreinar um ou mais simbolos, capturar metricas e registrar a execucao no MLflow.

Exemplo:

```powershell
.\scripts\train.ps1 AAPL MSFT --period 5y
```

O comando usa o mesmo `DataController` e `PredictionController` usados pela CLI. Isso evita uma logica paralela de treino e mantem o comportamento dos modelos consistente entre execucao interativa e operacional.

## Pesquisa de Alocacao de Portfolio

O projeto inclui um modulo leve de pesquisa de portfolio. Ele estima retorno anualizado e volatilidade a partir de precos historicos de fechamento e sugere pesos normalizados usando uma politica baseada em risco e retorno.

Essa funcionalidade e uma ferramenta de pesquisa e nao executa ordens.

Exemplo:

```powershell
.\scripts\portfolio.ps1 AAPL MSFT NVDA --period 2y --risk-aversion 2.0
```

## Instalacao

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\service-api\service-python\requirements.txt
```

## Executando a CLI

```powershell
.\scripts\run-cli.ps1
```

Execucao direta:

```powershell
$env:PYTHONPATH = "$PWD\service-api\service-python;$PWD\client-cli"
python .\client-cli\main.py
```

## Execucao em Container

Subir infraestrutura:

```powershell
docker compose -f .\infra\compose.yaml up -d postgres rabbitmq
```

Inicializar bancos:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-ops init-db
```

Rodar a CLI:

```powershell
docker compose -f .\infra\compose.yaml run --rm lexicon-cli
```

Rodar o worker de alertas:

```powershell
docker compose -f .\infra\compose.yaml --profile workers up lexicon-worker-alerts
```

## Configuracao

Crie `.env` na raiz do repositorio.

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

## Uso

Apos iniciar a CLI, o menu principal oferece:

1. Dashboard com ativos e indices monitorados.
2. Analise de ativo com grafico historico, indicadores tecnicos, simulacao Monte Carlo, dados fundamentalistas e backtesting.
3. Indicadores macroeconomicos.
4. Previsoes com LSTM ou Prophet.
5. Watchlist, configuracao de alertas, preferencias visuais e exportacao.

## Validacao e Cobertura de Testes

A suite de testes cobre comportamento de cache, operacoes do controlador de dados, persistencia SQLite, indicadores tecnicos, verificacao de alertas, preparacao de modelos LSTM/Prophet e metricas do avaliador.

O comando atual de validacao e:

```powershell
.\scripts\run-tests.ps1
```

A configuracao dos containers pode ser validada com:

```powershell
docker compose -f .\infra\compose.yaml config --quiet
```

## Limites Operacionais

O projeto e voltado para analise e pesquisa. Ele nao envia ordens para corretoras, nao gerencia contas reais e nao se apresenta como motor de execucao automatica de trading. Alocacao de portfolio e Continuous Training sao ferramentas explicitas de investigacao e gestao de modelos, nao agentes autonomos de negociacao.

## Testes

```powershell
.\scripts\run-tests.ps1
```

Comando direto equivalente:

```powershell
$env:PYTHONPATH = "$PWD\service-api\service-python;$PWD\client-cli"
pytest .\service-api\service-python\tests
```

## Screenshots

### Menu Principal

![Menu Principal](docs/assets/images/menu_principal.png)

### Dashboard

![Dashboard](docs/assets/images/dashboard_principal.png)

### Analise de Ativo

![Analise AMZN](docs/assets/images/analise_amzn.png)

### Previsao LSTM

![Previsao LSTM](docs/assets/images/previsao_lstm_amzn.png)

## Empacotamento Executavel

Os ativos de empacotamento com PyInstaller ficam em `scripts/pyinstaller`. O fluxo anterior de executavel standalone continua disponivel como alvo de empacotamento, mas o spec deve refletir a estrutura atual antes de gerar um novo artefato de release.

## Licenca

Este projeto esta licenciado sob a licenca MIT. Veja [LICENSE](LICENSE).

## Contato

Thiago Di Faria - [thiagodifaria@gmail.com](mailto:thiagodifaria@gmail.com)

Link do projeto: [https://github.com/thiagodifaria/LexiconCLI](https://github.com/thiagodifaria/LexiconCLI)
