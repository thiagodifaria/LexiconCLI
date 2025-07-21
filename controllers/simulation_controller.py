from models.simulation.monte_carlo_simulator import MonteCarloSimulator
from controllers.data_controller import DataController
from utils.logger import logger
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class SimulationController:
    """
    Controlador responsável por gerenciar o fluxo de trabalho da funcionalidade de
    simulação de Monte Carlo. Atua como ponte entre a interface do usuário e a
    lógica de simulação, seguindo o padrão MVC estabelecido no projeto.
    
    Esta classe coordena a obtenção de dados históricos, execução da simulação
    e processamento dos resultados, mantendo a separação de responsabilidades.
    """
    
    def __init__(self, data_controller: DataController):
        """
        Inicializa o controlador de simulação com injeção de dependência.
        
        Args:
            data_controller (DataController): Instância do controlador de dados
                                            para buscar informações históricas
        """
        self.data_controller = data_controller
        self.monte_carlo_simulator = MonteCarloSimulator()
        logger.info("SimulationController inicializado com DataController injetado.")
    
    def executar_simulacao_monte_carlo(
        self, 
        simbolo: str, 
        periodo_historico: str, 
        dias_simulacao: int, 
        num_simulacoes: int
    ) -> Tuple[Optional[np.ndarray], Optional[float], Dict[str, Any]]:
        """
        Executa uma simulação de Monte Carlo completa para um ativo específico.
        
        Este método coordena todo o fluxo de trabalho:
        1. Validação dos parâmetros de entrada
        2. Busca dos dados históricos via DataController  
        3. Validação dos dados obtidos
        4. Execução da simulação via MonteCarloSimulator
        5. Cálculo de estatísticas dos resultados
        
        Args:
            simbolo (str): Símbolo do ativo a ser simulado (ex: "AAPL", "PETR4.SA")
            periodo_historico (str): Período dos dados históricos (ex: "1y", "6mo", "2y")
            dias_simulacao (int): Número de dias para simular no futuro
            num_simulacoes (int): Quantidade de trajetórias de simulação
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[float], Dict[str, Any]]: 
                - Matriz de simulações (dias x simulações) ou None se houver erro
                - Preço inicial usado na simulação ou None se houver erro  
                - Dicionário com estatísticas e metadados da simulação
        """
        try:
            logger.info(f"Iniciando simulação Monte Carlo para {simbolo}: "
                       f"{num_simulacoes} simulações, {dias_simulacao} dias, período histórico: {periodo_historico}")
            
            resultado_validacao = self._validar_parametros_entrada(
                simbolo, periodo_historico, dias_simulacao, num_simulacoes
            )
            
            if not resultado_validacao['valido']:
                logger.error(f"Parâmetros inválidos: {resultado_validacao['erro']}")
                return None, None, {
                    'erro': resultado_validacao['erro'],
                    'sucesso': False,
                    'simbolo': simbolo
                }
            
            logger.info(f"Buscando dados históricos para {simbolo} (período: {periodo_historico})")
            dados_historicos = self.data_controller.buscar_dados_historicos(
                simbolo=simbolo,
                periodo=periodo_historico,
                intervalo="1d"
            )
            
            if dados_historicos is None or dados_historicos.dataframe.empty:
                erro_msg = f"Não foi possível obter dados históricos para {simbolo}"
                logger.error(erro_msg)
                return None, None, {
                    'erro': erro_msg,
                    'sucesso': False,
                    'simbolo': simbolo
                }
            
            df_historico = dados_historicos.dataframe
            
            validacao_dados = self._validar_dados_historicos(df_historico, simbolo)
            if not validacao_dados['valido']:
                logger.error(f"Dados históricos inválidos: {validacao_dados['erro']}")
                return None, None, {
                    'erro': validacao_dados['erro'],
                    'sucesso': False,
                    'simbolo': simbolo,
                    'dados_disponiveis': len(df_historico)
                }
            
            logger.info(f"Dados históricos validados: {len(df_historico)} pontos de dados disponíveis")
            
            logger.info("Iniciando execução da simulação Monte Carlo...")
            matriz_simulacoes = self.monte_carlo_simulator.run_simulation(
                df_historico=df_historico,
                dias_simulacao=dias_simulacao,
                num_simulacoes=num_simulacoes
            )
            
            if matriz_simulacoes is None:
                erro_msg = "Falha na execução da simulação Monte Carlo"
                logger.error(erro_msg)
                return None, None, {
                    'erro': erro_msg,
                    'sucesso': False,
                    'simbolo': simbolo
                }
            
            preco_inicial = float(matriz_simulacoes[0, 0])
            
            estatisticas = self.monte_carlo_simulator.calcular_estatisticas_simulacao(matriz_simulacoes)
            
            metadados_simulacao = {
                'simbolo': simbolo,
                'periodo_historico': periodo_historico,
                'dias_simulacao': dias_simulacao,
                'num_simulacoes': num_simulacoes,
                'pontos_historicos_utilizados': len(df_historico),
                'data_inicio_historico': df_historico.index.min().strftime('%Y-%m-%d'),
                'data_fim_historico': df_historico.index.max().strftime('%Y-%m-%d'),
                'sucesso': True,
                'estatisticas': estatisticas
            }
            
            logger.info(f"Simulação Monte Carlo concluída com sucesso para {simbolo}. "
                       f"Preço inicial: {preco_inicial:.2f}")
            
            return matriz_simulacoes, preco_inicial, metadados_simulacao
            
        except Exception as e:
            erro_msg = f"Erro inesperado durante simulação Monte Carlo para {simbolo}: {str(e)}"
            logger.error(erro_msg, exc_info=True)
            return None, None, {
                'erro': erro_msg,
                'sucesso': False,
                'simbolo': simbolo
            }
    
    def _validar_parametros_entrada(self, simbolo: str, periodo_historico: str, 
                                   dias_simulacao: int, num_simulacoes: int) -> Dict[str, Any]:
        """
        Valida os parâmetros de entrada para a simulação.
        
        Args:
            simbolo (str): Símbolo do ativo
            periodo_historico (str): Período histórico solicitado
            dias_simulacao (int): Dias para simulação
            num_simulacoes (int): Número de simulações
            
        Returns:
            Dict[str, Any]: Resultado da validação com flag 'valido' e 'erro' se aplicável
        """
        if not simbolo or not isinstance(simbolo, str) or len(simbolo.strip()) == 0:
            return {'valido': False, 'erro': 'Símbolo do ativo é obrigatório e deve ser uma string válida'}
        
        periodos_validos = ['7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        if periodo_historico not in periodos_validos:
            return {
                'valido': False, 
                'erro': f'Período histórico inválido. Valores aceitos: {periodos_validos}'
            }
        
        if not isinstance(dias_simulacao, int) or dias_simulacao <= 0:
            return {'valido': False, 'erro': 'Dias de simulação deve ser um número inteiro positivo'}
        
        if dias_simulacao > 1000:
            return {'valido': False, 'erro': 'Dias de simulação não pode exceder 1000 dias'}
        
        if not isinstance(num_simulacoes, int) or num_simulacoes <= 0:
            return {'valido': False, 'erro': 'Número de simulações deve ser um número inteiro positivo'}
        
        if num_simulacoes > 10000:
            return {'valido': False, 'erro': 'Número de simulações não pode exceder 10.000'}
        
        if num_simulacoes < 100:
            logger.warning(f"Número baixo de simulações ({num_simulacoes}). "
                          "Recomenda-se pelo menos 100 simulações para resultados confiáveis.")
        
        return {'valido': True}
    
    def _validar_dados_historicos(self, df_historico: pd.DataFrame, simbolo: str) -> Dict[str, Any]:
        """
        Valida a qualidade e adequação dos dados históricos para simulação.
        
        Args:
            df_historico (pd.DataFrame): DataFrame com dados históricos
            simbolo (str): Símbolo do ativo para contexto na mensagem
            
        Returns:
            Dict[str, Any]: Resultado da validação
        """
        if len(df_historico) < 30:
            return {
                'valido': False,
                'erro': f'Dados históricos insuficientes para {simbolo}. '
                       f'Necessário pelo menos 30 pontos, encontrados: {len(df_historico)}'
            }
        
        colunas_preco = ['close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']
        coluna_preco_encontrada = any(col in df_historico.columns for col in colunas_preco)
        
        if not coluna_preco_encontrada:
            return {
                'valido': False,
                'erro': f'Nenhuma coluna de preço válida encontrada nos dados de {simbolo}. '
                       f'Colunas disponíveis: {list(df_historico.columns)}'
            }
        
        coluna_preco = None
        for col in colunas_preco:
            if col in df_historico.columns:
                coluna_preco = col
                break
        
        precos = df_historico[coluna_preco].dropna()
        if len(precos) < len(df_historico) * 0.8:
            return {
                'valido': False,
                'erro': f'Muitos valores nulos na coluna de preços de {simbolo}. '
                       f'Dados válidos: {len(precos)}/{len(df_historico)}'
            }
        
        if (precos <= 0).any():
            return {
                'valido': False,
                'erro': f'Preços inválidos (negativos ou zero) encontrados nos dados de {simbolo}'
            }
        
        logger.info(f"Dados históricos de {simbolo} validados: {len(df_historico)} pontos, "
                   f"{len(precos)} preços válidos")
        
        return {'valido': True}
    
    def gerar_relatorio_simulacao(self, matriz_simulacoes: np.ndarray, 
                                  metadados: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera um relatório detalhado dos resultados da simulação.
        
        Args:
            matriz_simulacoes (np.ndarray): Resultados da simulação
            metadados (Dict[str, Any]): Metadados da simulação
            
        Returns:
            Dict[str, Any]: Relatório formatado com análises e insights
        """
        try:
            if matriz_simulacoes is None or not metadados.get('sucesso', False):
                return {'erro': 'Não é possível gerar relatório sem simulação válida'}
            
            estatisticas = metadados.get('estatisticas', {})
            
            retorno_esperado = estatisticas.get('retorno_esperado_pct', 0)
            volatilidade = estatisticas.get('volatilidade_simulacao', 0)
            
            if volatilidade < 0.1:
                classificacao_risco = "Baixo"
            elif volatilidade < 0.3:
                classificacao_risco = "Moderado"
            else:
                classificacao_risco = "Alto"
            
            preco_inicial = estatisticas.get('preco_inicial', 0)
            precos_finais = matriz_simulacoes[-1, :]
            prob_lucro = (precos_finais > preco_inicial).mean() * 100
            
            relatorio = {
                'resumo_executivo': {
                    'simbolo': metadados['simbolo'],
                    'retorno_esperado_pct': round(retorno_esperado, 2),
                    'classificacao_risco': classificacao_risco,
                    'probabilidade_lucro_pct': round(prob_lucro, 1),
                    'intervalo_confianca_95': {
                        'minimo': round(estatisticas.get('percentil_5', 0), 2),
                        'maximo': round(estatisticas.get('percentil_95', 0), 2)
                    }
                },
                'parametros_simulacao': {
                    'dias_simulados': metadados['dias_simulacao'],
                    'numero_simulacoes': metadados['num_simulacoes'],
                    'periodo_historico': metadados['periodo_historico'],
                    'pontos_historicos': metadados['pontos_historicos_utilizados']
                },
                'estatisticas_detalhadas': estatisticas,
                'metadata': {
                    'data_geracao': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'versao': '1.0'
                }
            }
            
            logger.info(f"Relatório de simulação gerado para {metadados['simbolo']}")
            return relatorio
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de simulação: {str(e)}")
            return {'erro': f'Falha na geração do relatório: {str(e)}'}