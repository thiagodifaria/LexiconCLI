import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from config.settings import FINNHUB_API_KEY, REALTIME_ENABLED
from utils.logger import logger


@dataclass
class RealtimeTick:
    symbol: str
    price: float
    timestamp_ms: int
    volume: float = 0.0
    source: str = "finnhub"


class FinnhubRealtimeStream:
    def __init__(self, api_key: Optional[str] = None, reconnect_delay_seconds: int = 5):
        self.api_key = api_key or FINNHUB_API_KEY
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._symbols: list[str] = []
        self._last_error: Optional[str] = None

    def is_available(self) -> bool:
        return bool(self.api_key and REALTIME_ENABLED)

    def start(self, symbols: Iterable[str], on_tick: Callable[[RealtimeTick], None]) -> bool:
        if not self.is_available():
            logger.info("Realtime stream disabled or Finnhub API key missing.")
            return False

        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client is not installed; realtime stream unavailable.")
            return False

        self._symbols = sorted({s for s in symbols if s})
        if not self._symbols:
            logger.info("Realtime stream not started because no symbols were provided.")
            return False
        if self._thread and self._thread.is_alive():
            logger.info("Realtime stream already running; updating subscription list.")
            return True

        def handle_message(_, message):
            try:
                payload = json.loads(message)
                if payload.get("type") != "trade":
                    return
                for item in payload.get("data", []):
                    tick = RealtimeTick(
                        symbol=item.get("s", ""),
                        price=float(item.get("p", 0.0)),
                        timestamp_ms=int(item.get("t", 0)),
                        volume=float(item.get("v", 0.0) or 0.0),
                    )
                    if tick.symbol and tick.price > 0:
                        on_tick(tick)
            except Exception as exc:
                logger.error(f"Failed to process realtime tick: {exc}", exc_info=True)

        def handle_error(_, error):
            self._last_error = str(error)
            logger.error(f"Finnhub realtime stream error: {error}")

        def handle_close(*_):
            logger.info("Finnhub realtime stream closed.")

        def handle_open(ws):
            for symbol in self._symbols:
                ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
            logger.info(f"Finnhub realtime stream subscribed to: {', '.join(self._symbols)}")

        def run_loop():
            while self._running:
                self._ws = websocket.WebSocketApp(
                    f"wss://ws.finnhub.io?token={self.api_key}",
                    on_message=handle_message,
                    on_error=handle_error,
                    on_close=handle_close,
                )
                self._ws.on_open = handle_open
                self._ws.run_forever()
                if self._running:
                    logger.warning("Realtime stream disconnected; reconnecting.")
                    time.sleep(self.reconnect_delay_seconds)

        self._running = True
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        return True

    def status(self) -> dict:
        return {
            "running": bool(self._running and self._thread and self._thread.is_alive()),
            "symbols": list(self._symbols),
            "last_error": self._last_error,
        }

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
