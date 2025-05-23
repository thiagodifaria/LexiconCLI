import pytest
import pandas as pd
import numpy as np
from models.indicators import IndicadoresTecnicos
from controllers.analysis_controller import AnalysisController
from utils.logger import logger 

@pytest.fixture
def ohlcv_data_fixture():
    data = {
        'Open': [10, 11, 12, 11, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18, 19, 20],
        'High': [11, 12, 13, 12, 13, 14, 15, 16, 15, 14, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21],
        'Low': [9, 10, 11, 10, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 17, 16, 15, 14, 13, 14, 15, 16, 17, 18, 19],
        'Close': [11, 12, 11, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18, 19, 20, 19],
        'Volume': [100, 110, 120, 110, 120, 130, 140, 150, 140, 130, 120, 130, 140, 150, 160, 170, 180, 190, 200, 190, 180, 170, 160, 150, 140, 150, 160, 170, 180, 190, 200]
    }
    index = pd.date_range(start='2023-01-01', periods=len(data['Open']))
    return pd.DataFrame(data, index=index)

@pytest.fixture
def ohlcv_data_fixture_small():
    data = {
        'Open': [10, 11, 12, 11],
        'High': [11, 12, 13, 12],
        'Low': [9, 10, 11, 10],
        'Close': [11, 12, 11, 12],
        'Volume': [100, 110, 120, 110]
    }
    index = pd.date_range(start='2023-01-01', periods=len(data['Open']))
    return pd.DataFrame(data, index=index)

@pytest.fixture
def ohlcv_data_with_nans():
    data = {
        'Open': [10, np.nan, 12, 11, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18, 19, 20],
        'High': [11, 12, 13, 12, 13, 14, 15, 16, 15, 14, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20, 21],
        'Low': [9, 10, 11, 10, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 17, 16, 15, 14, 13, 14, 15, 16, 17, 18, 19],
        'Close': [11, 12, np.nan, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18, 19, 20, 19],
        'Volume': [100, 110, 120, 110, 120, 130, 140, 150, 140, 130, 120, 130, 140, 150, 160, 170, 180, 190, 200, 190, 180, 170, 160, 150, 140, 150, 160, 170, 180, 190, 200]
    }
    index = pd.date_range(start='2023-01-01', periods=len(data['Open']))
    return pd.DataFrame(data, index=index)


class TestIndicadoresTecnicos:

    def test_indicador_sma(self, ohlcv_data_fixture):
        it = IndicadoresTecnicos(ohlcv_data_fixture.copy())
        it.adicionar_sma(periodo=5)
        assert 'SMA_5' in it.df.columns
        assert not it.df['SMA_5'].isnull().all()
        assert it.df['SMA_5'].iloc[4] == pytest.approx((11+12+11+12+13)/5)
        assert pd.isna(it.df['SMA_5'].iloc[3])

    def test_indicador_ema(self, ohlcv_data_fixture):
        it = IndicadoresTecnicos(ohlcv_data_fixture.copy())
        it.adicionar_ema(periodo=5)
        assert 'EMA_5' in it.df.columns
        assert not it.df['EMA_5'].isnull().all()
        assert pd.isna(it.df['EMA_5'].iloc[3]) 

    def test_indicador_macd(self, ohlcv_data_fixture):
        it = IndicadoresTecnicos(ohlcv_data_fixture.copy())
        it.adicionar_macd()
        assert 'MACD' in it.df.columns
        assert 'MACD_Signal' in it.df.columns
        assert 'MACD_Hist' in it.df.columns
        assert not it.df['MACD'].isnull().all()

    def test_indicador_rsi(self, ohlcv_data_fixture):
        it = IndicadoresTecnicos(ohlcv_data_fixture.copy())
        it.adicionar_rsi(periodo=14)
        assert 'RSI_14' in it.df.columns
        assert not it.df['RSI_14'].isnull().all()
        assert pd.isna(it.df['RSI_14'].iloc[13]) 

    def test_indicador_adx(self, ohlcv_data_fixture):
        it = IndicadoresTecnicos(ohlcv_data_fixture.copy())
        it.adicionar_adx(periodo=14)
        assert 'ADX' in it.df.columns
        assert not it.df['ADX'].isnull().all()
        assert pd.isna(it.df['ADX'].iloc[26]) 

    def test_indicador_sma_dados_insuficientes(self, ohlcv_data_fixture_small):
        it = IndicadoresTecnicos(ohlcv_data_fixture_small.copy())
        it.adicionar_sma(periodo=5)
        assert 'SMA_5' in it.df.columns
        assert it.df['SMA_5'].isnull().all()

    def test_indicador_adx_dados_insuficientes(self, ohlcv_data_fixture_small):
        it = IndicadoresTecnicos(ohlcv_data_fixture_small.copy())
        it.adicionar_adx(periodo=14)
        assert 'ADX' in it.df.columns
        assert it.df['ADX'].isnull().all()
        assert 'ADX_Pos' in it.df.columns
        assert it.df['ADX_Pos'].isnull().all()
        assert 'ADX_Neg' in it.df.columns
        assert it.df['ADX_Neg'].isnull().all()

    def test_indicadores_com_nans_na_entrada(self, ohlcv_data_with_nans):
        it = IndicadoresTecnicos(ohlcv_data_with_nans.copy())
        original_len = len(ohlcv_data_with_nans)
        it.adicionar_sma(5).adicionar_ema(5).adicionar_macd().adicionar_rsi(5).adicionar_adx(5)
        
        assert len(it.df) < original_len 
        assert 'SMA_5' in it.df.columns
        assert 'RSI_5' in it.df.columns
    
    def test_construtor_com_colunas_faltando(self):
        data_incompleta = pd.DataFrame({'Open': [10, 11], 'High': [11,12]})
        with pytest.raises(ValueError):
            IndicadoresTecnicos(data_incompleta)


class TestAnalysisController:
    @pytest.fixture
    def analysis_controller_fixture(self):
        return AnalysisController()

    def test_calcular_todos_indicadores_principais_df_valido(self, analysis_controller_fixture, ohlcv_data_fixture):
        df_com_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(ohlcv_data_fixture.copy())
        assert not df_com_indicadores.empty
        assert 'SMA_9' in df_com_indicadores.columns
        assert 'EMA_21' in df_com_indicadores.columns
        assert 'MACD_Hist' in df_com_indicadores.columns
        assert 'RSI_14' in df_com_indicadores.columns
        assert 'ADX' in df_com_indicadores.columns
        assert 'OBV' in df_com_indicadores.columns
        assert len(df_com_indicadores) == len(ohlcv_data_fixture)


    def test_calcular_todos_indicadores_principais_df_vazio(self, analysis_controller_fixture):
        df_vazio = pd.DataFrame()
        df_com_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(df_vazio)
        assert df_com_indicadores.empty

    def test_calcular_todos_indicadores_principais_colunas_faltando(self, analysis_controller_fixture):
        df_incompleto = pd.DataFrame({'Open': [1,2,3], 'High': [2,3,4]})
        df_com_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(df_incompleto)
        assert df_com_indicadores.empty
    
    def test_verificar_sinais_para_alertas_preco_acima(self, analysis_controller_fixture, ohlcv_data_fixture):
        df_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(ohlcv_data_fixture.copy())
        df_indicadores.iloc[-1, df_indicadores.columns.get_loc('Close')] = 100 

        alertas_config = [{
            'id_alerta': 1, 'simbolo': "TEST", 'tipo_alerta': 'preco_acima', 
            'condicao': {'valor_referencia': 90.0}, 'ativo': True, 'mensagem_customizada': "Preço subiu!"
        }]
        sinais = analysis_controller_fixture.verificar_sinais_para_alertas("TEST", df_indicadores, alertas_config)
        assert len(sinais) == 1
        assert "Preço (100.00) ACIMA de 90.00!" in sinais[0]
        assert "Preço subiu!" in sinais[0]

    def test_verificar_sinais_para_alertas_rsi_sobrecompra(self, analysis_controller_fixture, ohlcv_data_fixture):
        df_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(ohlcv_data_fixture.copy())
        df_indicadores.iloc[-1, df_indicadores.columns.get_loc('RSI_14')] = 75 

        alertas_config = [{
            'id_alerta': 2, 'simbolo': "TEST", 'tipo_alerta': 'rsi_sobrecompra', 
            'condicao': {'limiar_rsi': 70}, 'ativo': True, 'mensagem_customizada': None
        }]
        sinais = analysis_controller_fixture.verificar_sinais_para_alertas("TEST", df_indicadores, alertas_config)
        assert len(sinais) == 1
        assert "RSI (75.00) em SOBRECOMPRA (> 70)!" in sinais[0]

    def test_verificar_sinais_para_alertas_nenhum_sinal(self, analysis_controller_fixture, ohlcv_data_fixture):
        df_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(ohlcv_data_fixture.copy())
        df_indicadores.iloc[-1, df_indicadores.columns.get_loc('Close')] = 50 
        df_indicadores.iloc[-1, df_indicadores.columns.get_loc('RSI_14')] = 50

        alertas_config = [
            {'id_alerta':1, 'simbolo': "TEST", 'tipo_alerta': 'preco_acima', 'condicao': {'valor_referencia': 90.0}, 'ativo': True, 'mensagem_customizada':None},
            {'id_alerta':2, 'simbolo': "TEST", 'tipo_alerta': 'rsi_sobrecompra', 'condicao': {'limiar_rsi': 70}, 'ativo': True, 'mensagem_customizada':None}
        ]
        sinais = analysis_controller_fixture.verificar_sinais_para_alertas("TEST", df_indicadores, alertas_config)
        assert len(sinais) == 0

    def test_verificar_sinais_para_alertas_alerta_inativo(self, analysis_controller_fixture, ohlcv_data_fixture):
        df_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(ohlcv_data_fixture.copy())
        df_indicadores.iloc[-1, df_indicadores.columns.get_loc('Close')] = 100

        alertas_config = [{'id_alerta':1, 'simbolo': "TEST", 'tipo_alerta': 'preco_acima', 'condicao': {'valor_referencia': 90.0}, 'ativo': False, 'mensagem_customizada':None}]
        sinais = analysis_controller_fixture.verificar_sinais_para_alertas("TEST", df_indicadores, alertas_config)
        assert len(sinais) == 0

    def test_verificar_sinais_para_alertas_simbolo_diferente(self, analysis_controller_fixture, ohlcv_data_fixture):
        df_indicadores = analysis_controller_fixture.calcular_todos_indicadores_principais(ohlcv_data_fixture.copy())
        df_indicadores.iloc[-1, df_indicadores.columns.get_loc('Close')] = 100

        alertas_config = [{'id_alerta':1, 'simbolo': "OTHER", 'tipo_alerta': 'preco_acima', 'condicao': {'valor_referencia': 90.0}, 'ativo': True, 'mensagem_customizada':None}]
        sinais = analysis_controller_fixture.verificar_sinais_para_alertas("TEST", df_indicadores, alertas_config)
        assert len(sinais) == 0