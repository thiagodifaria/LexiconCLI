import numpy as np
import pandas as pd
from utils.logger import logger
from typing import Optional


class MonteCarloSimulator:
    """
    Classe responsável pela execução da lógica matemática da simulação de Monte Carlo
    utilizando o modelo de Movimento Browniano Geométrico (GBM) para previsão de preços de ativos.
    
    A lógica matemática foi isolada nesta classe para manter o código modular, testável e reutilizável,
    seguindo o padrão de separação de responsabilidades do projeto.
    """
    
    def __init__(self):
        logger.info("MonteCarloSimulator inicializado.")
    
    def run_simulation(self, df_historico: pd.DataFrame, dias_simulacao: int, num_simulacoes: int) -> Optional[np.ndarray]:
        """
        Executa a simulação de Monte Carlo utilizando Movimento Browniano Geométrico.
        
        O modelo GBM é amplamente utilizado em finanças para modelar comportamento de preços de ativos,
        assumindo que os retornos logarítmicos são normalmente distribuídos e que o preço segue
        um processo estocástico com drift constante mais choques aleatórios.
        
        Fórmula utilizada: S(t+1) = S(t) * exp((μ - σ²/2)Δt + σ√(Δt)Z)
        Onde:
        - S(t) = preço no tempo t
        - μ = média dos retornos logarítmicos (drift)
        - σ = desvio padrão dos retornos logarítmicos (volatilidade)
        - Δt = incremento de tempo (assumido como 1 dia = 1/252 anos)
        - Z = número aleatório da distribuição normal padrão
        
        Args:
            df_historico (pd.DataFrame): DataFrame com dados históricos contendo coluna 'close'
            dias_simulacao (int): Número de dias para simular no futuro
            num_simulacoes (int): Número de trajetórias de simulação a gerar
            
        Returns:
            Optional[np.ndarray]: Matriz NumPy (dias_simulacao + 1, num_simulacoes) com as trajetórias
                                 simuladas, ou None se houver erro
        """
        try:
            if df_historico.empty:
                logger.error("DataFrame histórico está vazio. Não é possível executar simulação.")
                return None
                
            if 'close' not in df_historico.columns:
                close_columns = ['Close', 'CLOSE', 'close', 'adj_close', 'Adj Close']
                close_col = None
                for col in close_columns:
                    if col in df_historico.columns:
                        close_col = col
                        break
                
                if close_col is None:
                    logger.error("Coluna 'close' não encontrada no DataFrame histórico.")
                    return None
                    
                df_historico = df_historico.copy()
                df_historico['close'] = df_historico[close_col]
                
            if dias_simulacao <= 0 or num_simulacoes <= 0:
                logger.error(f"Parâmetros inválidos: dias_simulacao={dias_simulacao}, num_simulacoes={num_simulacoes}")
                return None
                
            logger.info(f"Iniciando simulação Monte Carlo: {num_simulacoes} simulações para {dias_simulacao} dias")
            
            precos_close = df_historico['close'].dropna()
            
            if len(precos_close) < 2:
                logger.error("Dados históricos insuficientes para calcular retornos.")
                return None
                
            retornos_log = np.log(precos_close / precos_close.shift(1)).dropna()
            
            if len(retornos_log) == 0:
                logger.error("Não foi possível calcular retornos logarítmicos válidos.")
                return None
            
            mu = retornos_log.mean()
            sigma = retornos_log.std()
            preco_inicial = float(precos_close.iloc[-1])
            
            logger.info(f"Parâmetros calculados - mu (drift): {mu:.6f}, sigma (volatilidade): {sigma:.6f}, "
                       f"Preço inicial: {preco_inicial}")
            
            matriz_resultados = np.zeros((dias_simulacao + 1, num_simulacoes))
            matriz_resultados[0, :] = preco_inicial
            
            for dia in range(1, dias_simulacao + 1):
                choques_aleatorios = np.random.normal(mu, sigma, num_simulacoes)
                
                matriz_resultados[dia, :] = matriz_resultados[dia-1, :] * np.exp(choques_aleatorios)
                
                if dia % max(1, dias_simulacao // 4) == 0:
                    progresso = (dia / dias_simulacao) * 100
                    logger.info(f"Progresso da simulação: {progresso:.1f}% ({dia}/{dias_simulacao} dias)")
            
            if np.any(np.isnan(matriz_resultados)) or np.any(np.isinf(matriz_resultados)):
                logger.warning("Resultados contêm valores NaN ou infinitos. Verificar parâmetros de entrada.")
                
            precos_finais = matriz_resultados[-1, :]
            logger.info(f"Simulação concluída. Preço final médio: {np.mean(precos_finais):.2f}, "
                       f"Min: {np.min(precos_finais):.2f}, Max: {np.max(precos_finais):.2f}")
            
            return matriz_resultados
            
        except Exception as e:
            logger.error(f"Erro durante execução da simulação Monte Carlo: {str(e)}", exc_info=True)
            return None
    
    def calcular_estatisticas_simulacao(self, matriz_simulacao: np.ndarray) -> dict:
        """
        Calcula estatísticas descritivas dos resultados da simulação.
        
        Args:
            matriz_simulacao (np.ndarray): Matriz com resultados da simulação
            
        Returns:
            dict: Dicionário com estatísticas calculadas
        """
        try:
            if matriz_simulacao is None or matriz_simulacao.size == 0:
                return {}
                
            precos_finais = matriz_simulacao[-1, :]
            
            estatisticas = {
                'preco_inicial': float(matriz_simulacao[0, 0]),
                'preco_final_medio': float(np.mean(precos_finais)),
                'preco_final_mediano': float(np.median(precos_finais)),
                'preco_final_min': float(np.min(precos_finais)),
                'preco_final_max': float(np.max(precos_finais)),
                'desvio_padrao_final': float(np.std(precos_finais)),
                'percentil_5': float(np.percentile(precos_finais, 5)),
                'percentil_95': float(np.percentile(precos_finais, 95)),
                'retorno_esperado_pct': float(((np.mean(precos_finais) / matriz_simulacao[0, 0]) - 1) * 100),
                'volatilidade_simulacao': float(np.std(precos_finais) / np.mean(precos_finais))
            }
            
            logger.info(f"Estatísticas calculadas: retorno esperado {estatisticas['retorno_esperado_pct']:.2f}%")
            return estatisticas
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas da simulação: {str(e)}")
            return {}