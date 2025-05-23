import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import time
import sqlite3
import os
import json
from datetime import datetime
from models.market_data import CacheAPIManager, MarketDataProvider
from controllers.data_controller import DataController
from models.data_model import CotacaoAtivo, PreferenciasVisualizacao, AlertaConfigurado
import models.database as db_module
from config.settings import DB_NAME, DEFAULT_HISTORICAL_PERIOD, DEFAULT_INDICATORS_VIEW, DEFAULT_USER_ID_PREFERENCES

@pytest.fixture
def cache_manager_fixture():
    test_db_path = ":memory:" 
    manager = CacheAPIManager(db_path=test_db_path)
    manager._criar_tabela_se_nao_existir() 
    return manager

@pytest.fixture
def market_data_provider_fixture(cache_manager_fixture):
    provider = MarketDataProvider()
    provider.cache_manager = cache_manager_fixture 
    return provider

@pytest.fixture
def data_controller_fixture(market_data_provider_fixture):
    controller = DataController()
    controller.provider = market_data_provider_fixture
    return controller

@pytest.fixture
def in_memory_db():
    original_db_name = db_module.DB_NAME
    db_module.DB_NAME = ":memory:"
    conn = db_module.conectar_db()
    db_module.inicializar_db() 
    yield conn 
    conn.close()
    db_module.DB_NAME = original_db_name


class TestCacheAPIManager:
    def test_gerar_chave_cache_consistencia(self, cache_manager_fixture):
        chave1 = cache_manager_fixture._gerar_chave_cache("api_test", "endpoint1", {"param": "valor"})
        chave2 = cache_manager_fixture._gerar_chave_cache("api_test", "endpoint1", {"param": "valor"})
        assert chave1 == chave2

    def test_gerar_chave_cache_diferentes_params(self, cache_manager_fixture):
        chave1 = cache_manager_fixture._gerar_chave_cache("api_test", "endpoint1", {"param": "valor1"})
        chave2 = cache_manager_fixture._gerar_chave_cache("api_test", "endpoint1", {"param": "valor2"})
        assert chave1 != chave2

    def test_set_e_get_cache_hit(self, cache_manager_fixture):
        api_origem = "test_api"
        endpoint = "test_endpoint"
        params = {"test": "data"}
        valor = {"result": "success"}
        expiracao = 3600

        cache_manager_fixture.set_cache(api_origem, endpoint, valor, expiracao, params)
        resultado_cache = cache_manager_fixture.get_cache(api_origem, endpoint, params)
        assert resultado_cache == valor

    def test_get_cache_miss(self, cache_manager_fixture):
        resultado_cache = cache_manager_fixture.get_cache("miss_api", "miss_endpoint")
        assert resultado_cache is None

    def test_get_cache_expired(self, cache_manager_fixture):
        api_origem = "test_api_expired"
        endpoint = "test_endpoint_expired"
        valor = {"result": "will_expire"}
        expiracao_curta = 1 
        params = {"p":1}

        cache_manager_fixture.set_cache(api_origem, endpoint, valor, expiracao_curta, params)
        
        with patch('time.time', return_value=time.time() + expiracao_curta + 1):
            resultado_cache = cache_manager_fixture.get_cache(api_origem, endpoint, params)
            assert resultado_cache is None

    def test_delete_cache(self, cache_manager_fixture):
        api_origem = "test_api_delete"
        endpoint = "test_endpoint_delete"
        valor = {"result": "to_delete"}
        expiracao = 3600
        params = {"d":1}

        cache_manager_fixture.set_cache(api_origem, endpoint, valor, expiracao, params)
        chave_cache = cache_manager_fixture._gerar_chave_cache(api_origem, endpoint, params)
        cache_manager_fixture.delete_cache(chave_cache)
        
        resultado_cache = cache_manager_fixture.get_cache(api_origem, endpoint, params)
        assert resultado_cache is None


class TestMarketDataProvider:
    @patch('finnhub.Client')
    def test_obter_cotacao_finnhub_sucesso_cache_miss(self, mock_finnhub_client_class, market_data_provider_fixture):
        mock_client_instance = MagicMock()
        mock_client_instance.quote.return_value = {'c': 150.0, 'dp': 1.0, 'd': 1.5, 'o': 149.0, 'h': 151.0, 'l': 148.0, 'pc': 148.5, 't': 1678886400}
        mock_finnhub_client_class.return_value = mock_client_instance
        
        market_data_provider_fixture.finnhub_client = mock_client_instance 
        
        resultado = market_data_provider_fixture.obter_cotacao_finnhub("AAPL")
        
        assert resultado['c'] == 150.0
        mock_client_instance.quote.assert_called_once_with("AAPL")
        
        cache_result = market_data_provider_fixture.cache_manager.get_cache("finnhub", "quote_AAPL", params={"simbolo": "AAPL"})
        assert cache_result['c'] == 150.0

    @patch('finnhub.Client')
    def test_obter_cotacao_finnhub_sucesso_cache_hit(self, mock_finnhub_client_class, market_data_provider_fixture):
        mock_client_instance = MagicMock()
        mock_finnhub_client_class.return_value = mock_client_instance
        market_data_provider_fixture.finnhub_client = mock_client_instance 

        dados_mock_cache = {'c': 155.0, 'dp': 1.2, 'd': 1.8, 'o': 154.0, 'h': 156.0, 'l': 153.0, 'pc': 153.2, 't': 1678886500}
        market_data_provider_fixture.cache_manager.set_cache("finnhub", "quote_MSFT", dados_mock_cache, 3600, params={"simbolo":"MSFT"})

        resultado = market_data_provider_fixture.obter_cotacao_finnhub("MSFT")
        
        assert resultado['c'] == 155.0
        mock_client_instance.quote.assert_not_called()


    @patch('yfinance.Ticker')
    def test_obter_dados_historicos_yf_sucesso(self, mock_yf_ticker, market_data_provider_fixture):
        mock_ticker_instance = MagicMock()
        mock_df = pd.DataFrame({'Open': [100], 'High': [105], 'Low': [99], 'Close': [102], 'Volume': [1000]})
        mock_df.index = pd.to_datetime(['2023-01-01'])
        mock_ticker_instance.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_ticker_instance

        resultado_df = market_data_provider_fixture.obter_dados_historicos_yf("TEST", "2023-01-01", "2023-01-01")
        
        assert not resultado_df.empty
        assert resultado_df.iloc[0]['Close'] == 102
        mock_ticker_instance.history.assert_called_once_with(start="2023-01-01", end="2023-01-01", interval="1d")

    @patch('models.market_data.bcb_sgs')
    def test_obter_indicador_sgs_bcb_sucesso(self, mock_bcb_sgs, market_data_provider_fixture):
        mock_df = pd.DataFrame({'IPCA_Test': [0.5]})
        mock_df.index = pd.to_datetime(['2023-03-01'])
        mock_df.index.name = 'data'
        mock_bcb_sgs.get.return_value = mock_df
        
        resultado = market_data_provider_fixture.obter_indicador_sgs_bcb(123, "IPCA_Test", ultimos_n=1)
        
        assert not resultado.empty
        assert 'IPCA_Test' in resultado.columns
        assert resultado.iloc[0]['IPCA_Test'] == 0.5
        mock_bcb_sgs.get.assert_called_once()


class TestDataController:
    @patch.object(MarketDataProvider, 'obter_cotacao_finnhub')
    @patch.object(MarketDataProvider, 'obter_dados_historicos_yf')
    def test_buscar_cotacao_ativo_fallback_finnhub_falha(self, mock_obter_dados_historicos_yf, mock_obter_cotacao_finnhub, data_controller_fixture):
        mock_obter_cotacao_finnhub.return_value = {'c': 0} 
        
        mock_hist_df = pd.DataFrame({
            'Open': [140.0, 142.0], 'High': [141.0, 143.0], 'Low': [139.0, 141.0], 
            'Close': [140.5, 142.5], 'Volume': [1000, 1200]
        }, index=pd.to_datetime(['2023-03-14', '2023-03-15']))
        mock_obter_dados_historicos_yf.return_value = mock_hist_df
        
        data_controller_fixture.provider.obter_dados_historicos_yf = MagicMock(return_value=mock_hist_df)

        resultado = data_controller_fixture.buscar_cotacao_ativo("FAIL.SA")
        
        assert isinstance(resultado, CotacaoAtivo)
        assert resultado.preco_atual == 142.5
        assert resultado.preco_fechamento_anterior == 140.5
        mock_obter_cotacao_finnhub.assert_called_once_with("FAIL.SA")
        data_controller_fixture.provider.obter_dados_historicos_yf.assert_called_once()


    @patch('models.database.adicionar_item_watchlist_db')
    def test_add_to_watchlist_chama_db(self, mock_add_db, data_controller_fixture):
        mock_add_db.return_value = True
        data_controller_fixture.add_to_watchlist("AAPL", "asset")
        mock_add_db.assert_called_once_with("AAPL", "asset")

    @patch('models.database.salvar_preferencias_visualizacao_db')
    def test_salvar_preferencias_chama_db(self, mock_salvar_prefs_db, data_controller_fixture):
        prefs = {"id_usuario": 0, "periodo_historico_padrao": "6mo", "indicadores_tecnicos_padrao": ["SMA_50"]}
        mock_salvar_prefs_db.return_value = True
        
        data_controller_fixture.salvar_preferencias_visualizacao_db(prefs)
        
        mock_salvar_prefs_db.assert_called_once_with(0, "6mo", "SMA_50")

    @patch('models.database.carregar_preferencias_visualizacao_db')
    def test_carregar_preferencias_retorna_dados_corretos(self, mock_carregar_prefs_db, data_controller_fixture):
        mock_db_return = {"id_usuario": 0, "periodo_historico_padrao": "3mo", "indicadores_tecnicos_padrao": "RSI_14,MACD"}
        mock_carregar_prefs_db.return_value = mock_db_return
        
        resultado = data_controller_fixture.carregar_preferencias_visualizacao_db(0)
        
        assert resultado["periodo_historico_padrao"] == "3mo"
        assert "RSI_14" in resultado["indicadores_tecnicos_padrao"]
        assert "MACD" in resultado["indicadores_tecnicos_padrao"]
        mock_carregar_prefs_db.assert_called_once_with(0)

    @patch('models.database.adicionar_alerta_db')
    def test_adicionar_alerta_chama_db(self, mock_adicionar_alerta, data_controller_fixture):
        alerta_data = {"simbolo": "MSFT", "tipo_alerta": "preco_abaixo", "condicao": {"valor_referencia": 200.0}, "ativo": True, "mensagem_customizada": "MSFT low"}
        alerta_obj = AlertaConfigurado(**alerta_data)
        mock_adicionar_alerta.return_value = True
        
        data_controller_fixture.adicionar_alerta_db(alerta_data)
        
        mock_adicionar_alerta.assert_called_once_with(alerta_obj)


class TestDatabaseOperations:
    def test_inicializar_db_cria_tabelas(self, in_memory_db):
        cursor = in_memory_db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "cache_api" in tables
        assert "dados_historicos_ohlcv" in tables
        assert "dados_macro_bcb" in tables
        assert "user_watchlist" in tables
        assert "user_visual_preferences" in tables
        assert "user_alerts" in tables

    def test_watchlist_crud(self, in_memory_db):
        assert db_module.adicionar_item_watchlist_db("GOOG", "asset")
        watchlist = db_module.obter_watchlist_do_db()
        assert len(watchlist) == 1
        assert watchlist[0]['simbolo'] == "GOOG"
        assert db_module.remover_item_watchlist_db("GOOG")
        assert not db_module.obter_watchlist_do_db()
        assert not db_module.remover_item_watchlist_db("NONEXIST")

    def test_visual_preferences_crud(self, in_memory_db):
        user_id = DEFAULT_USER_ID_PREFERENCES
        periodo = "6mo"
        indicadores = "SMA_10,EMA_10"
        assert db_module.salvar_preferencias_visualizacao_db(user_id, periodo, indicadores)
        
        prefs = db_module.carregar_preferencias_visualizacao_db(user_id)
        assert prefs is not None
        assert prefs['periodo_historico_padrao'] == periodo
        assert prefs['indicadores_tecnicos_padrao'] == indicadores
        
        assert db_module.salvar_preferencias_visualizacao_db(user_id, "1y", "RSI_14")
        prefs_updated = db_module.carregar_preferencias_visualizacao_db(user_id)
        assert prefs_updated['periodo_historico_padrao'] == "1y"

    def test_alerts_crud(self, in_memory_db):
        alerta1_data = AlertaConfigurado(simbolo="TSLA", tipo_alerta="preco_acima", condicao={"valor_referencia": 700.0}, ativo=True)
        alerta2_data = AlertaConfigurado(simbolo="NVDA", tipo_alerta="rsi_sobrecompra", condicao={"limiar_rsi": 75}, ativo=True)

        assert db_module.adicionar_alerta_db(alerta1_data)
        assert db_module.adicionar_alerta_db(alerta2_data)

        alertas = db_module.listar_alertas_db()
        assert len(alertas) == 2
        assert alertas[0]['simbolo'] == "TSLA"
        assert alertas[1]['condicao']['limiar_rsi'] == 75
        
        id_alerta1 = alertas[0]['id_alerta']
        
        novos_dados = {"ativo": False, "mensagem_customizada": "TSLA alvo atingido"}
        assert db_module.atualizar_alerta_db(id_alerta1, novos_dados)
        
        alertas_atualizados = db_module.listar_alertas_db()
        alerta1_atualizado = next(a for a in alertas_atualizados if a['id_alerta'] == id_alerta1)
        assert not alerta1_atualizado['ativo']
        assert alerta1_atualizado['mensagem_customizada'] == "TSLA alvo atingido"

        assert db_module.remover_alerta_db(id_alerta1)
        alertas_final = db_module.listar_alertas_db()
        assert len(alertas_final) == 1
        assert not any(a['id_alerta'] == id_alerta1 for a in alertas_final)
        assert not db_module.remover_alerta_db(999)