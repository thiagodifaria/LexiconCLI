CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    PRIMARY KEY (time, symbol, source)
);

SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS market_candles (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    interval TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (time, symbol, interval, source)
);

SELECT create_hypertable('market_candles', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_time ON market_ticks (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_market_candles_symbol_time ON market_candles (symbol, interval, time DESC);
