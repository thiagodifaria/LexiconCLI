from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from config.settings import POSTGRES_DSN, TIMESCALE_ENABLED
from utils.logger import logger


@dataclass
class TickPoint:
    symbol: str
    price: float
    timestamp_ms: int
    volume: float = 0.0
    source: str = "finnhub"


class TimescaleStore:
    def __init__(self, dsn: Optional[str] = None, enabled: Optional[bool] = None):
        self.dsn = dsn or POSTGRES_DSN
        self.enabled = TIMESCALE_ENABLED if enabled is None else enabled

    def is_available(self) -> bool:
        return bool(self.enabled and self.dsn)

    def _connect(self):
        try:
            import psycopg2
        except ImportError as exc:
            raise RuntimeError("psycopg2-binary is required for TimescaleDB support") from exc
        return psycopg2.connect(self.dsn)

    def initialize(self) -> bool:
        if not self.is_available():
            logger.info("TimescaleDB disabled or POSTGRES_DSN missing.")
            return False

        statements = [
            "CREATE EXTENSION IF NOT EXISTS timescaledb;",
            """
            CREATE TABLE IF NOT EXISTS market_ticks (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                PRIMARY KEY (time, symbol, source)
            );
            """,
            "SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);",
            """
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
            """,
            "SELECT create_hypertable('market_candles', 'time', if_not_exists => TRUE);",
        ]

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for statement in statements:
                        cur.execute(statement)
            logger.info("TimescaleDB schema initialized.")
            return True
        except Exception as exc:
            logger.error(f"Failed to initialize TimescaleDB schema: {exc}", exc_info=True)
            return False

    def write_ticks(self, ticks: Iterable[TickPoint]) -> int:
        tick_list = list(ticks)
        if not tick_list or not self.is_available():
            return 0

        rows = [
            (
                pd.to_datetime(t.timestamp_ms, unit="ms", utc=True).to_pydatetime(),
                t.symbol,
                t.price,
                t.volume,
                t.source,
            )
            for t in tick_list
        ]

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO market_ticks (time, symbol, price, volume, source)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        rows,
                    )
            return len(rows)
        except Exception as exc:
            logger.error(f"Failed to write ticks to TimescaleDB: {exc}", exc_info=True)
            return 0

    def write_candles(self, symbol: str, frame: pd.DataFrame, interval: str, source: str) -> int:
        if frame.empty or not self.is_available():
            return 0

        rows = []
        for idx, row in frame.iterrows():
            timestamp = pd.to_datetime(idx, utc=True).to_pydatetime()
            rows.append((
                timestamp,
                symbol,
                float(row.get("Open", row.get("open", 0)) or 0),
                float(row.get("High", row.get("high", 0)) or 0),
                float(row.get("Low", row.get("low", 0)) or 0),
                float(row.get("Close", row.get("close", 0)) or 0),
                float(row.get("Volume", row.get("volume", 0)) or 0),
                interval,
                source,
            ))

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO market_candles
                        (time, symbol, open, high, low, close, volume, interval, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        rows,
                    )
            return len(rows)
        except Exception as exc:
            logger.error(f"Failed to write candles to TimescaleDB: {exc}", exc_info=True)
            return 0

    def fetch_recent_candles(self, symbol: str, interval: str = "1d", limit: int = 500) -> pd.DataFrame:
        if not self.is_available():
            return pd.DataFrame()
        try:
            with self._connect() as conn:
                df = pd.read_sql_query(
                    """
                    SELECT time, open, high, low, close, volume, source
                    FROM market_candles
                    WHERE symbol = %s AND interval = %s
                    ORDER BY time DESC
                    LIMIT %s
                    """,
                    conn,
                    params=(symbol, interval, limit),
                )
            if df.empty:
                return df
            df = df.sort_values("time").set_index("time")
            df.rename(
                columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"},
                inplace=True,
            )
            return df
        except Exception as exc:
            logger.error(f"Failed to fetch candles from TimescaleDB: {exc}", exc_info=True)
            return pd.DataFrame()

    def fetch_latest_ticks(self, symbols: Iterable[str], limit_per_symbol: int = 1) -> pd.DataFrame:
        symbol_list = [s for s in symbols if s]
        if not symbol_list or not self.is_available():
            return pd.DataFrame()
        try:
            with self._connect() as conn:
                df = pd.read_sql_query(
                    """
                    SELECT time, symbol, price, volume, source
                    FROM (
                        SELECT *,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY time DESC) AS rank
                        FROM market_ticks
                        WHERE symbol = ANY(%s)
                    ) ranked
                    WHERE rank <= %s
                    ORDER BY symbol, time DESC
                    """,
                    conn,
                    params=(symbol_list, limit_per_symbol),
                )
            return df
        except Exception as exc:
            logger.error(f"Failed to fetch latest ticks from TimescaleDB: {exc}", exc_info=True)
            return pd.DataFrame()
