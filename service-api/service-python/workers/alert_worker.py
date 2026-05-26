import os
import sys


SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from models.message_broker import create_message_broker
from utils.logger import configurar_logger, logger
from utils.notifications import enviar_notificacao_sistema


def handle_alert(payload: dict):
    message = payload.get("message") or payload.get("value") or str(payload)
    enviar_notificacao_sistema("LexiconCLI - Alerta de Mercado", message)
    logger.info(f"Alert worker delivered notification: {message}")


def main():
    configurar_logger()
    broker = create_message_broker()
    logger.info("Alert worker started.")
    broker.consume("alert.triggered", handle_alert)


if __name__ == "__main__":
    main()
