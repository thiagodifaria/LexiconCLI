import pandas as pd
import ta
from utils.logger import logger

class IndicadoresTecnicos:
    def __init__(self, df_ohlcv: pd.DataFrame):
        if not isinstance(df_ohlcv, pd.DataFrame):
            raise ValueError("Entrada para IndicadoresTecnicos deve ser um DataFrame pandas.")
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df_ohlcv.columns]
        if missing_cols:
            raise ValueError(f"DataFrame de entrada não contém as colunas obrigatórias: {', '.join(missing_cols)}")

        self.df = df_ohlcv.copy()        
        self.df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        if 'Volume' in self.df.columns:
            self.df['Volume'] = self.df['Volume'].fillna(0)


    def adicionar_sma(self, periodo: int = 20, coluna_origem: str = 'Close'):
        try:
            if len(self.df) >= periodo and coluna_origem in self.df.columns and not self.df[coluna_origem].isnull().all():
                self.df[f'SMA_{periodo}'] = ta.trend.SMAIndicator(close=self.df[coluna_origem], window=periodo, fillna=False).sma_indicator()
            else:
                logger.warning(f"Dados insuficientes ou coluna '{coluna_origem}' ausente/vazia para SMA_{periodo}. Preenchendo com NA.")
                self.df[f'SMA_{periodo}'] = pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular SMA_{periodo}: {e}")
            self.df[f'SMA_{periodo}'] = pd.NA
        return self

    def adicionar_ema(self, periodo: int = 20, coluna_origem: str = 'Close'):
        try:
            if len(self.df) >= periodo and coluna_origem in self.df.columns and not self.df[coluna_origem].isnull().all():
                self.df[f'EMA_{periodo}'] = ta.trend.EMAIndicator(close=self.df[coluna_origem], window=periodo, fillna=False).ema_indicator()
            else:
                logger.warning(f"Dados insuficientes ou coluna '{coluna_origem}' ausente/vazia para EMA_{periodo}. Preenchendo com NA.")
                self.df[f'EMA_{periodo}'] = pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular EMA_{periodo}: {e}")
            self.df[f'EMA_{periodo}'] = pd.NA
        return self

    def adicionar_macd(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9, coluna_origem: str = 'Close'):
        try:
            if len(self.df) >= window_slow and coluna_origem in self.df.columns and not self.df[coluna_origem].isnull().all():
                indicador_macd = ta.trend.MACD(close=self.df[coluna_origem], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=False)
                self.df['MACD'] = indicador_macd.macd()
                self.df['MACD_Signal'] = indicador_macd.macd_signal()
                self.df['MACD_Hist'] = indicador_macd.macd_diff()
            else:
                logger.warning(f"Dados insuficientes ou coluna '{coluna_origem}' ausente/vazia para MACD. Preenchendo com NA.")
                self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = pd.NA, pd.NA, pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular MACD: {e}")
            self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = pd.NA, pd.NA, pd.NA
        return self

    def adicionar_bandas_bollinger(self, periodo: int = 20, num_desvios: int = 2, coluna_origem: str = 'Close'):
        try:
            if len(self.df) >= periodo and coluna_origem in self.df.columns and not self.df[coluna_origem].isnull().all():
                indicador_bb = ta.volatility.BollingerBands(close=self.df[coluna_origem], window=periodo, window_dev=num_desvios, fillna=False)
                self.df['BB_High'] = indicador_bb.bollinger_hband()
                self.df['BB_Mid'] = indicador_bb.bollinger_mavg()
                self.df['BB_Low'] = indicador_bb.bollinger_lband()
                self.df['BB_Width'] = indicador_bb.bollinger_wband() 
                self.df['BB_P बैंड'] = indicador_bb.bollinger_pband() 
            else:
                logger.warning(f"Dados insuficientes ou coluna '{coluna_origem}' ausente/vazia para Bandas de Bollinger. Preenchendo com NA.")
                self.df['BB_High'], self.df['BB_Mid'], self.df['BB_Low'], self.df['BB_Width'], self.df['BB_P बैंड'] = pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular Bandas de Bollinger: {e}")
            self.df['BB_High'], self.df['BB_Mid'], self.df['BB_Low'], self.df['BB_Width'], self.df['BB_P बैंड'] = pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
        return self

    def adicionar_adx(self, periodo: int = 14):
        try:
            required_cols_adx = ['High', 'Low', 'Close']
            if len(self.df) >= periodo and all(col in self.df.columns for col in required_cols_adx) and not self.df[required_cols_adx].isnull().all().any():
                indicador_adx = ta.trend.ADXIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=periodo, fillna=False)
                self.df['ADX'] = indicador_adx.adx()
                self.df['ADX_Pos'] = indicador_adx.adx_pos() 
                self.df['ADX_Neg'] = indicador_adx.adx_neg() 
            else:
                logger.warning(f"Dados insuficientes ou colunas High/Low/Close ausentes/vazias para ADX com período {periodo}. Preenchendo com NA.")
                self.df['ADX'], self.df['ADX_Pos'], self.df['ADX_Neg'] = pd.NA, pd.NA, pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular ADX: {e}")
            self.df['ADX'], self.df['ADX_Pos'], self.df['ADX_Neg'] = pd.NA, pd.NA, pd.NA
        return self

    def adicionar_rsi(self, periodo: int = 14, coluna_origem: str = 'Close'):
        try:
            if len(self.df) >= periodo and coluna_origem in self.df.columns and not self.df[coluna_origem].isnull().all():
                self.df[f'RSI_{periodo}'] = ta.momentum.RSIIndicator(close=self.df[coluna_origem], window=periodo, fillna=False).rsi()
            else:
                logger.warning(f"Dados insuficientes ou coluna '{coluna_origem}' ausente/vazia para RSI_{periodo}. Preenchendo com NA.")
                self.df[f'RSI_{periodo}'] = pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular RSI_{periodo}: {e}")
            self.df[f'RSI_{periodo}'] = pd.NA
        return self

    def adicionar_estocastico(self, periodo_k: int = 14, periodo_d: int = 3, smoothing_s: int = 3):
        try:
            required_cols_stoch = ['High', 'Low', 'Close']
            if len(self.df) >= periodo_k and all(col in self.df.columns for col in required_cols_stoch) and not self.df[required_cols_stoch].isnull().all().any():
                indicador_stoch = ta.momentum.StochasticOscillator(
                    high=self.df['High'], low=self.df['Low'], close=self.df['Close'], 
                    window=periodo_k, smooth_window=periodo_d, 
                    fillna=False
                )
                self.df['Stoch_K'] = indicador_stoch.stoch()
                self.df['Stoch_D'] = indicador_stoch.stoch_signal()
            else:
                logger.warning(f"Dados insuficientes ou colunas High/Low/Close ausentes/vazias para Estocástico. Preenchendo com NA.")
                self.df['Stoch_K'], self.df['Stoch_D'] = pd.NA, pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular Oscilador Estocástico: {e}")
            self.df['Stoch_K'], self.df['Stoch_D'] = pd.NA, pd.NA
        return self

    def adicionar_obv(self, coluna_origem: str = 'Close'):
        try:
            if 'Volume' not in self.df.columns or self.df['Volume'].isnull().all():
                logger.warning("Coluna 'Volume' não encontrada ou vazia para cálculo do OBV. Adicionando como NA.")
                self.df['OBV'] = pd.NA
                return self
            if coluna_origem not in self.df.columns or self.df[coluna_origem].isnull().all():
                logger.warning(f"Coluna '{coluna_origem}' não encontrada ou vazia para cálculo do OBV. Adicionando como NA.")
                self.df['OBV'] = pd.NA
                return self

            self.df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=self.df[coluna_origem], volume=self.df['Volume'].fillna(0), fillna=False).on_balance_volume()
        except Exception as e:
            logger.error(f"Erro ao calcular OBV: {e}")
            self.df['OBV'] = pd.NA
        return self

    def adicionar_atr(self, periodo: int = 14):
        try:
            required_cols_atr = ['High', 'Low', 'Close']
            if len(self.df) >= periodo and all(col in self.df.columns for col in required_cols_atr) and not self.df[required_cols_atr].isnull().all().any():
                self.df[f'ATR_{periodo}'] = ta.volatility.AverageTrueRange(
                    high=self.df['High'], low=self.df['Low'], close=self.df['Close'], 
                    window=periodo, fillna=False
                ).average_true_range()
            else:
                logger.warning(f"Dados insuficientes ou colunas High/Low/Close ausentes/vazias para ATR_{periodo}. Preenchendo com NA.")
                self.df[f'ATR_{periodo}'] = pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular ATR_{periodo}: {e}")
            self.df[f'ATR_{periodo}'] = pd.NA
        return self

    def adicionar_desvio_padrao_retornos(self, periodo: int = 20, coluna_origem: str = 'Close'):
        try:
            if len(self.df) >= periodo and coluna_origem in self.df.columns and not self.df[coluna_origem].isnull().all():
                retornos_diarios = self.df[coluna_origem].pct_change()
                self.df[f'StdDev_Ret_{periodo}'] = retornos_diarios.rolling(window=periodo).std() * 100 
            else:
                logger.warning(f"Dados insuficientes ou coluna '{coluna_origem}' ausente/vazia para Desvio Padrão dos Retornos. Preenchendo com NA.")
                self.df[f'StdDev_Ret_{periodo}'] = pd.NA
        except Exception as e:
            logger.error(f"Erro ao calcular Desvio Padrão dos Retornos: {e}")
            self.df[f'StdDev_Ret_{periodo}'] = pd.NA
        return self

    def obter_df_com_indicadores(self):
        if not isinstance(self.df.index, pd.DatetimeIndex) and 'Date' in self.df.columns:
            try:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.df.set_index('Date', inplace=True)
            except: pass 
        elif not isinstance(self.df.index, pd.DatetimeIndex) and self.df.index.name == 'Date':
            try:
                self.df.index = pd.to_datetime(self.df.index)
            except: pass

        return self.df