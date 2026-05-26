import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from controllers.data_controller import DataController
from utils.logger import logger
import webbrowser
import os
import warnings

class SmaCross(Strategy):
    """
    Estratégia de cruzamento de SMA.
    Os parâmetros n1 e n2 agora podem ser definidos dinamicamente.
    Se não forem fornecidos durante a otimização, usarão os valores padrão.
    """
    n1 = 10
    n2 = 30

    def init(self):
        close_price = self.data.Close
        self.sma1 = self.I(lambda x, n: pd.Series(x).rolling(n).mean(), close_price, self.n1)
        self.sma2 = self.I(lambda x, n: pd.Series(x).rolling(n).mean(), close_price, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

class BacktestingController:
    """Controlador para gerenciar a execução de backtests de estratégias."""
    def __init__(self, data_controller: DataController):
        self.data_controller = data_controller
        logger.info("BacktestingController inicializado.")

    def executar_backtest_sma_cross(self, simbolo: str, periodo: str = "1y", 
                                     n1: int = 10, n2: int = 30,
                                     capital_inicial: float = 100000, comissao: float = 0.002,
                                     plotar_grafico: bool = True):
        """
        Executa um backtest para a estratégia SmaCross com parâmetros customizáveis.
        """
        logger.info(f"Iniciando backtest SmaCross para {simbolo} com n1={n1}, n2={n2}.")
        dados_historicos = self.data_controller.buscar_dados_historicos(simbolo, periodo=periodo)

        if dados_historicos is None or dados_historicos.dataframe.empty:
            logger.warning(f"Não foi possível obter dados históricos para {simbolo}.")
            return None, None

        df = dados_historicos.dataframe.copy()
        df.columns = [col.capitalize() for col in df.columns]

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"O DataFrame para {simbolo} não contém as colunas necessárias: {required_cols}")
            return None, None

        bt = Backtest(df, SmaCross, cash=capital_inicial, commission=comissao)
        
        stats = bt.run(n1=n1, n2=n2)
        
        logger.info(f"Backtest para {simbolo} concluído. Retorno: {stats['Return [%]']:.2f}%")
        
        caminho_plot = None
        if plotar_grafico:
            caminho_plot = f"backtest_{simbolo}_{n1}_{n2}.html"
            bt.plot(filename=caminho_plot, open_browser=False)
            logger.info(f"Gráfico do backtest salvo em: {caminho_plot}")

        return stats, caminho_plot

    def otimizar_estrategia_sma_cross(self, simbolo: str, periodo: str = "1y"):
        """
        NOVO MÉTODO: Executa uma otimização para encontrar os melhores parâmetros n1 e n2.
        """
        logger.info(f"Iniciando OTIMIZAÇÃO SmaCross para {simbolo}.")
        dados_historicos = self.data_controller.buscar_dados_historicos(simbolo, periodo=periodo)
        
        if dados_historicos is None or dados_historicos.dataframe.empty:
            logger.warning(f"Não foi possível obter dados históricos para {simbolo} para otimização.")
            return None

        df = dados_historicos.dataframe.copy()
        df.columns = [col.capitalize() for col in df.columns]

        bt = Backtest(df, SmaCross, cash=100000, commission=.002)

        stats = bt.optimize(
            n1=range(5, 31, 5),
            n2=range(10, 71, 5),
            constraint=lambda p: p.n1 < p.n2,
            maximize='Equity Final [$]'
        )
        
        logger.info(f"Otimização para {simbolo} concluída.")
        return stats
    
warnings.filterwarnings(
    'ignore', 
    message='.*insufficient margin.*',
    category=UserWarning
)

warnings.filterwarnings(
    'ignore',
    message='.*Broker canceled.*relative-sized order.*',
    category=UserWarning
)