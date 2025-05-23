import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.logger import logger

class ModelEvaluator:
    @staticmethod
    def calcular_metricas(y_real, y_previsto):
        if len(y_real) != len(y_previsto):
            logger.error("y_real e y_previsto devem ter o mesmo tamanho para calcular métricas.")
            return {}
        
        if len(y_real) == 0:
            logger.warning("y_real está vazio, não é possível calcular métricas.")
            return {}

        mae = mean_absolute_error(y_real, y_previsto)
        rmse = np.sqrt(mean_squared_error(y_real, y_previsto))       
        
        acertos_direcionais = 0
        total_comparacoes_direcionais = 0

        if len(y_real) > 1:
            mudanca_real = np.diff(y_real)
            mudanca_prevista = np.diff(y_previsto) 
            
            sinal_mudanca_real = np.sign(mudanca_real)
            sinal_mudanca_prevista = np.sign(mudanca_prevista)
            
            acertos_direcionais = np.sum(sinal_mudanca_real == sinal_mudanca_prevista)
            total_comparacoes_direcionais = len(mudanca_real)
        
        taxa_acerto_direcional = (acertos_direcionais / total_comparacoes_direcionais) if total_comparacoes_direcionais > 0 else 0.0

        metricas = {
            "mae": mae,
            "rmse": rmse,
            "taxa_acerto_direcional": taxa_acerto_direcional,
            "num_acertos_direcionais": acertos_direcionais,
            "total_comparacoes_direcionais": total_comparacoes_direcionais
        }
        logger.info(f"Métricas de avaliação calculadas: {metricas}")
        return metricas