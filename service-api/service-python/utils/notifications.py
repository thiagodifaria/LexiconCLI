from plyer import notification
from utils.logger import logger

def enviar_notificacao_sistema(titulo: str, mensagem: str):
    """
    Envia uma notificação nativa para o sistema operacional.

    Args:
        titulo (str): O título da notificação.
        mensagem (str): O corpo da mensagem da notificação.
    """
    try:
        notification.notify(
            title=titulo,
            message=mensagem,
            app_name='LexiconCLI',
            timeout=10
        )
        logger.info(f"Notificação do sistema enviada: '{titulo}'")
    except Exception as e:
        logger.error(f"Falha ao enviar notificação do sistema: {e}")