import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from config.settings import RABBITMQ_ENABLED, RABBITMQ_URL
from utils.logger import logger


class MessageBroker:
    def publish(self, routing_key: str, payload: Any) -> bool:
        raise NotImplementedError

    def consume(self, queue_name: str, callback: Callable[[dict], None]):
        raise NotImplementedError

    def close(self):
        return None


class LocalMessageBroker(MessageBroker):
    def __init__(self):
        self.messages: list[tuple[str, dict]] = []

    def publish(self, routing_key: str, payload: Any) -> bool:
        data = _normalize_payload(payload)
        self.messages.append((routing_key, data))
        logger.info(f"Local event published: {routing_key}")
        return True

    def consume(self, queue_name: str, callback: Callable[[dict], None]):
        for routing_key, payload in list(self.messages):
            if routing_key == queue_name or queue_name == "#":
                callback(payload)


class RabbitMQBroker(MessageBroker):
    def __init__(self, url: Optional[str] = None, exchange: str = "lexicon.events"):
        self.url = url or RABBITMQ_URL
        self.exchange = exchange

    def _connection(self):
        try:
            import pika
        except ImportError as exc:
            raise RuntimeError("pika is required for RabbitMQ support") from exc
        params = pika.URLParameters(self.url)
        return pika.BlockingConnection(params)

    def publish(self, routing_key: str, payload: Any) -> bool:
        try:
            body = json.dumps(_normalize_payload(payload), default=str).encode("utf-8")
            with self._connection() as conn:
                channel = conn.channel()
                channel.exchange_declare(exchange=self.exchange, exchange_type="topic", durable=True)
                channel.basic_publish(
                    exchange=self.exchange,
                    routing_key=routing_key,
                    body=body,
                    properties=_message_properties(),
                )
            logger.info(f"RabbitMQ event published: {routing_key}")
            return True
        except Exception as exc:
            logger.error(f"Failed to publish RabbitMQ event: {exc}", exc_info=True)
            return False

    def consume(self, queue_name: str, callback: Callable[[dict], None]):
        with self._connection() as conn:
            channel = conn.channel()
            channel.exchange_declare(exchange=self.exchange, exchange_type="topic", durable=True)
            result = channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(exchange=self.exchange, queue=result.method.queue, routing_key=queue_name)

            def handle_message(ch, method, properties, body):
                del properties
                callback(json.loads(body.decode("utf-8")))
                ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_consume(queue=result.method.queue, on_message_callback=handle_message)
            channel.start_consuming()


def create_message_broker() -> MessageBroker:
    if RABBITMQ_ENABLED:
        broker = RabbitMQBroker()
        logger.info("RabbitMQ broker configured.")
        return broker
    logger.info("Local in-process event broker configured.")
    return LocalMessageBroker()


def _normalize_payload(payload: Any) -> dict:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _message_properties():
    import pika

    return pika.BasicProperties(content_type="application/json", delivery_mode=2)
