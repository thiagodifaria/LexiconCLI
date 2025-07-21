import finnhub
import yfinance as yf
import pandas as pd
from bcb import sgs as bcb_sgs
from bcb import PTAX
from bcb import Expectativas
import sqlite3
import time
import json
import hashlib
from datetime import datetime, timedelta, date
import requests
import investpy
from fredapi import Fred
import nasdaqdatalink
from config.settings import (
    FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY, FRED_API_KEY, NASDAQ_DATA_LINK_API_KEY,
    CACHE_EXPIRATION_SGS, CACHE_EXPIRATION_PTAX,
    CACHE_EXPIRATION_FINNHUB_QUOTE, CACHE_EXPIRATION_YFINANCE_HISTORY,
    CACHE_EXPIRATION_ALPHA_VANTAGE_TIMESERIES, CACHE_EXPIRATION_INVESTPY,
    CACHE_EXPIRATION_FRED, CACHE_EXPIRATION_NASDAQ,
    BCB_SERIES_IPCA, BCB_SERIES_SELIC,
    DB_NAME, INVESTPY_COUNTRY_MAP
)
from utils.logger import logger
from typing import List, Dict, Any, Optional

class CacheAPIManager:
    def __init__(self, db_path: str, existing_conn: Optional[sqlite3.Connection] = None):
        self.db_path = db_path
        self.existing_conn = existing_conn
        if not self.existing_conn or self.db_path != ":memory:":
            self._criar_tabela_se_nao_existir()

    def _get_connection(self) -> sqlite3.Connection:
        if self.existing_conn and self.db_path == ":memory:":
            return self.existing_conn
        conn = sqlite3.connect(self.db_path)
        return conn

    def _criar_tabela_se_nao_existir(self):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_api (
                chave TEXT PRIMARY KEY,
                valor TEXT,
                timestamp REAL,
                api_origem TEXT,
                expiracao INTEGER
            )
            """)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Erro ao criar/verificar tabela de cache: {e}")
        finally:
            if not (self.existing_conn and self.db_path == ":memory:"):
                if conn:
                    conn.close()

    def _gerar_chave_cache(self, api_origem: str, endpoint_ou_simbolo: str, params: dict = None) -> str:
        params_str = json.dumps(params, sort_keys=True) if params else ""
        hash_input = f"{api_origem}:{endpoint_ou_simbolo}:{params_str}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    def _json_converter_default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)): 
            return obj.isoformat()
        if hasattr(obj, 'isoformat'):
            return obj.isoformat() 
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def get_cache(self, api_origem: str, endpoint_ou_simbolo: str, params: dict = None):
        chave = self._gerar_chave_cache(api_origem, endpoint_ou_simbolo, params)
        conn = self._get_connection()
        valor_retorno = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT valor, timestamp, expiracao FROM cache_api WHERE chave = ?", (chave,))
            resultado = cursor.fetchone()
            if resultado:
                valor_json, timestamp_cache, expiracao = resultado
                if time.time() - timestamp_cache < expiracao:
                    logger.info(f"Cache HIT para {api_origem} - {endpoint_ou_simbolo} com params {params}")
                    valor_retorno = json.loads(valor_json)
                else:
                    logger.info(f"Cache EXPIRED para {api_origem} - {endpoint_ou_simbolo} com params {params}")
                    self.delete_cache_with_conn(chave, conn) 
            else:
                logger.info(f"Cache MISS para {api_origem} - {endpoint_ou_simbolo} com params {params}")
        except sqlite3.Error as e:
            logger.error(f"Erro ao buscar do cache ({chave}): {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON do cache ({chave}): {e}")
            self.delete_cache_with_conn(chave, conn)
        finally:
            if not (self.existing_conn and self.db_path == ":memory:"):
                if conn:
                    conn.close()
        return valor_retorno

    def set_cache(self, api_origem: str, endpoint_ou_simbolo: str, valor, expiracao: int, params: dict = None):
        chave = self._gerar_chave_cache(api_origem, endpoint_ou_simbolo, params)
        try:
            valor_json = json.dumps(valor, default=self._json_converter_default)
        except TypeError as e:
            logger.error(f"Erro de TypeError ao serializar valor para cache ({chave}): {e}. Valor: {type(valor)}")
            return 
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO cache_api (chave, valor, timestamp, api_origem, expiracao)
            VALUES (?, ?, ?, ?, ?)
            """, (chave, valor_json, time.time(), api_origem, expiracao))
            conn.commit()
            logger.info(f"Cache SET para {api_origem} - {endpoint_ou_simbolo} com params {params}")
        except sqlite3.Error as e:
            logger.error(f"Erro ao salvar no cache ({chave}): {e}")
        finally:
            if not (self.existing_conn and self.db_path == ":memory:"):
                if conn:
                    conn.close()
    
    def delete_cache_with_conn(self, chave: str, conn: sqlite3.Connection):
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache_api WHERE chave = ?", (chave,))
            conn.commit()
            logger.info(f"Cache DELETED (with conn) para chave: {chave}")
        except sqlite3.Error as e:
            logger.error(f"Erro ao deletar do cache ({chave}) (with conn): {e}")

    def delete_cache(self, chave: str): 
        conn = self._get_connection()
        try:
            self.delete_cache_with_conn(chave, conn)
        finally:
            if not (self.existing_conn and self.db_path == ":memory:"):
                if conn:
                    conn.close()

class MarketDataProvider:
    def __init__(self, cache_manager_instance: Optional[CacheAPIManager] = None):
        self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None
        self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
        self.fred_client = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        if NASDAQ_DATA_LINK_API_KEY:
            try:
                nasdaqdatalink.ApiConfig.api_key = NASDAQ_DATA_LINK_API_KEY
                logger.info("Chave da API Nasdaq Data Link configurada.")
            except Exception as e:
                logger.error(f"Erro ao configurar chave da API Nasdaq Data Link: {e}")
        else:
            logger.warning("Chave da API Nasdaq Data Link não encontrada no .env.")
        
        if cache_manager_instance:
            self.cache_manager = cache_manager_instance
        else:
            self.cache_manager = CacheAPIManager(DB_NAME)


    def _get_investpy_country_and_symbol(self, simbolo_original: str) -> tuple[str, str, str]:
        if simbolo_original == "^BVSP":
            return "brazil", "Bovespa", "index"
        if simbolo_original == "^GSPC":
            return "united states", "S&P 500", "index"
        if simbolo_original == "^IXIC":
            return "united states", "Nasdaq Composite", "index"

        for sufixo, pais in INVESTPY_COUNTRY_MAP.items():
            if sufixo and simbolo_original.endswith(sufixo):
                simbolo_limpo = simbolo_original[:-len(sufixo)]
                return pais, simbolo_limpo, "stock"
        return INVESTPY_COUNTRY_MAP.get("", "united states"), simbolo_original, "stock"
    
    def obter_balanco_patrimonial_anual(self, simbolo: str) -> Optional[Dict]:
        """Obtém o balanço patrimonial anual (balance sheet) de uma empresa via Finnhub."""
        if not self.finnhub_client:
            logger.warning("Chave da API Finnhub não configurada.")
            return None

        params = {"simbolo": simbolo, "freq": "annual", "statement": "bs"}
        cache_key_suffix = f"finnhub_financials_bs_{simbolo}"
        dados_cache = self.cache_manager.get_cache("finnhub_financials", cache_key_suffix, params=params)
        if dados_cache:
            return dados_cache

        try:
            logger.info(f"Buscando balanço patrimonial para {simbolo} na Finnhub (sem cache).")
            balance_sheet = self.finnhub_client.financials_reported(symbol=simbolo, freq='annual', statement='bs')
            
            if not balance_sheet or 'data' not in balance_sheet or not balance_sheet['data']:
                logger.warning(f"Finnhub retornou dados de balanço patrimonial vazios para {simbolo}.")
                self.cache_manager.set_cache("finnhub_financials", cache_key_suffix, {}, CACHE_EXPIRATION_FINNHUB_QUOTE, params=params)
                return None

            self.cache_manager.set_cache("finnhub_financials", cache_key_suffix, balance_sheet, CACHE_EXPIRATION_YFINANCE_HISTORY, params=params)
            return balance_sheet
        except Exception as e:
            logger.error(f"Erro ao obter balanço patrimonial da Finnhub para {simbolo}: {e}")
            return None

    def obter_dre_anual(self, simbolo: str) -> Optional[Dict]:
        """Obtém a demonstração de resultados anual (income statement) de uma empresa via Finnhub."""
        if not self.finnhub_client:
            logger.warning("Chave da API Finnhub não configurada.")
            return None

        params = {"simbolo": simbolo, "freq": "annual", "statement": "ic"}
        cache_key_suffix = f"finnhub_financials_ic_{simbolo}"
        dados_cache = self.cache_manager.get_cache("finnhub_financials", cache_key_suffix, params=params)
        if dados_cache:
            return dados_cache

        try:
            logger.info(f"Buscando DRE para {simbolo} na Finnhub (sem cache).")
            income_statement = self.finnhub_client.financials_reported(symbol=simbolo, freq='annual', statement='ic')

            if not income_statement or 'data' not in income_statement or not income_statement['data']:
                logger.warning(f"Finnhub retornou dados de DRE vazios para {simbolo}.")
                self.cache_manager.set_cache("finnhub_financials", cache_key_suffix, {}, CACHE_EXPIRATION_FINNHUB_QUOTE, params=params)
                return None
            
            self.cache_manager.set_cache("finnhub_financials", cache_key_suffix, income_statement, CACHE_EXPIRATION_YFINANCE_HISTORY, params=params)
            return income_statement
        except Exception as e:
            logger.error(f"Erro ao obter DRE da Finnhub para {simbolo}: {e}")
            return None

    def obter_fluxo_caixa_anual(self, simbolo: str) -> Optional[Dict]:
        """Obtém o fluxo de caixa anual (cash flow) de uma empresa via Finnhub."""
        if not self.finnhub_client:
            logger.warning("Chave da API Finnhub não configurada.")
            return None

        params = {"simbolo": simbolo, "freq": "annual", "statement": "cf"}
        cache_key_suffix = f"finnhub_financials_cf_{simbolo}"
        dados_cache = self.cache_manager.get_cache("finnhub_financials", cache_key_suffix, params=params)
        if dados_cache:
            return dados_cache

        try:
            logger.info(f"Buscando fluxo de caixa para {simbolo} na Finnhub (sem cache).")
            cash_flow = self.finnhub_client.financials_reported(symbol=simbolo, freq='annual', statement='cf')
            
            if not cash_flow or 'data' not in cash_flow or not cash_flow['data']:
                logger.warning(f"Finnhub retornou dados de fluxo de caixa vazios para {simbolo}.")
                self.cache_manager.set_cache("finnhub_financials", cache_key_suffix, {}, CACHE_EXPIRATION_FINNHUB_QUOTE, params=params)
                return None
            
            self.cache_manager.set_cache("finnhub_financials", cache_key_suffix, cash_flow, CACHE_EXPIRATION_YFINANCE_HISTORY, params=params)
            return cash_flow
        except Exception as e:
            logger.error(f"Erro ao obter fluxo de caixa da Finnhub para {simbolo}: {e}")
            return None

    def obter_cotacao_finnhub(self, simbolo: str):
        if not self.finnhub_client:
            logger.warning("Chave da API Finnhub não configurada.")
            return None
        params={"simbolo": simbolo}
        dados_cache = self.cache_manager.get_cache("finnhub", f"quote_{simbolo}", params=params)
        if dados_cache:
            return dados_cache
        try:
            quote = self.finnhub_client.quote(simbolo)
            if quote.get('c') is None and quote.get('pc') is None : 
                logger.warning(f"Finnhub retornou dados vazios para {simbolo}: {quote}")
                return None
            self.cache_manager.set_cache("finnhub", f"quote_{simbolo}", quote, CACHE_EXPIRATION_FINNHUB_QUOTE, params=params)
            return quote
        except Exception as e:
            logger.error(f"Erro ao obter cotação da Finnhub para {simbolo}: {e}")
            return None

    def obter_dados_historicos_yf(self, simbolo: str, start_date: str, end_date: str, interval="1d"):
        params = {"simbolo": simbolo, "start_date": start_date, "end_date": end_date, "interval": interval}
        dados_cache = self.cache_manager.get_cache("yfinance", f"history_{simbolo}", params=params)
        if dados_cache:
            try:
                df = pd.DataFrame(dados_cache)
                date_col_name = 'Date' if 'Date' in df.columns else 'index'
                if date_col_name not in df.columns and df.index.name and 'date' in df.index.name.lower():
                     df = df.reset_index()
                     date_col_name = df.columns[0]

                if date_col_name in df.columns:
                    if df[date_col_name].dtype == 'int64' or str(df[date_col_name].dtype).startswith('datetime64[ns'):
                        try:
                            df[date_col_name] = pd.to_datetime(df[date_col_name], unit='ms' if df[date_col_name].dtype == 'int64' else None)
                        except Exception:
                             df[date_col_name] = pd.to_datetime(df[date_col_name])
                    
                    if df[date_col_name].dt.tz:
                        df[date_col_name] = df[date_col_name].dt.tz_convert('UTC') 
                    df = df.set_index(date_col_name)
                    df.index.name = 'Date'
                else:
                    logger.warning(f"Coluna de data não encontrada ao carregar yfinance do cache para {simbolo}")
                    self.cache_manager.delete_cache(self.cache_manager._gerar_chave_cache("yfinance", f"history_{simbolo}", params=params))
                    return pd.DataFrame()

                expected_cols = ['Open', 'High', 'Low', 'Close'] 
                if not all(col in df.columns for col in expected_cols):
                    logger.warning(f"Colunas OHLC ausentes ao carregar yfinance do cache para {simbolo}. Cache invalidado.")
                    self.cache_manager.delete_cache(self.cache_manager._gerar_chave_cache("yfinance", f"history_{simbolo}", params=params))
                    return pd.DataFrame()
                else: 
                    return df
            except Exception as e:
                logger.error(f"Erro ao processar dados históricos do yfinance do cache para {simbolo}: {e}.")
                self.cache_manager.delete_cache(self.cache_manager._gerar_chave_cache("yfinance", f"history_{simbolo}", params=params))
        try:
            ticker = yf.Ticker(simbolo)
            hist = ticker.history(start=start_date, end=end_date, interval=interval)
            if hist.empty: return pd.DataFrame()
            
            hist_to_cache = hist.reset_index()
            date_column_name_to_serialize = hist_to_cache.columns[0] 

            if pd.api.types.is_datetime64_any_dtype(hist_to_cache[date_column_name_to_serialize]):
                if hist_to_cache[date_column_name_to_serialize].dt.tz is not None:
                    hist_to_cache[date_column_name_to_serialize] = hist_to_cache[date_column_name_to_serialize].dt.tz_convert('UTC')
                hist_to_cache[date_column_name_to_serialize] = (hist_to_cache[date_column_name_to_serialize].astype('int64') // 10**6)
            
            self.cache_manager.set_cache("yfinance", f"history_{simbolo}", hist_to_cache.to_dict(orient='list'), CACHE_EXPIRATION_YFINANCE_HISTORY, params=params)
            return hist
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos do yfinance para {simbolo}: {e}")
            return pd.DataFrame()

    def obter_dados_historicos_investpy(self, simbolo_original: str, de_data: str, ate_data: str):
        pais, simbolo_investpy, tipo_ativo = self._get_investpy_country_and_symbol(simbolo_original)
        params = {"simbolo_investpy": simbolo_investpy, "pais": pais, "de_data": de_data, "ate_data": ate_data, "tipo_ativo": tipo_ativo}
        cache_key_suffix = f"investpy_hist_{simbolo_investpy}_{tipo_ativo}"
        dados_cache = self.cache_manager.get_cache("investpy", cache_key_suffix, params=params)
        if dados_cache:
            df = pd.DataFrame(dados_cache)
            if 'Date' in df.columns :
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            return df
        try:
            de_data_fmt = datetime.strptime(de_data, "%Y-%m-%d").strftime("%d/%m/%Y")
            ate_data_fmt = datetime.strptime(ate_data, "%Y-%m-%d").strftime("%d/%m/%Y")
            df = pd.DataFrame()
            if tipo_ativo == "stock":
                df = investpy.get_stock_historical_data(stock=simbolo_investpy, country=pais, from_date=de_data_fmt, to_date=ate_data_fmt)
            elif tipo_ativo == "etf":
                df = investpy.get_etf_historical_data(etf=simbolo_investpy, country=pais, from_date=de_data_fmt, to_date=ate_data_fmt)
            elif tipo_ativo == "index":
                df = investpy.get_index_historical_data(index=simbolo_investpy, country=pais, from_date=de_data_fmt, to_date=ate_data_fmt)
            else:
                logger.warning(f"Tipo de ativo '{tipo_ativo}' para '{simbolo_investpy}' não é diretamente suportado para histórico investpy ou mapeamento pendente.")
                return pd.DataFrame()
            
            if df.empty: return pd.DataFrame()
            df_para_cache = df.reset_index()
            if 'Date' in df_para_cache.columns:
                df_para_cache['Date'] = pd.to_datetime(df_para_cache['Date']).dt.strftime('%Y-%m-%d') 
            self.cache_manager.set_cache("investpy", cache_key_suffix, df_para_cache.to_dict(orient='list'), CACHE_EXPIRATION_INVESTPY, params=params)
            return df
        except RuntimeError as re:
            if "country not found" in str(re) or "not found" in str(re):
                 logger.warning(f"Investpy: {simbolo_investpy} ({tipo_ativo}) não encontrado em {pais}. Detalhe: {re}")
            else:
                 logger.error(f"Erro Runtime ao obter dados históricos do investpy para {simbolo_investpy} ({pais}): {re}")
        except ConnectionError as ce:
            logger.error(f"Erro de Conexão ao obter dados históricos do investpy para {simbolo_investpy} ({pais}): {ce}")
        except Exception as e:
            logger.error(f"Erro Geral ao obter dados históricos do investpy para {simbolo_investpy} ({pais}): {e}")
        return pd.DataFrame()

    def obter_info_recente_investpy(self, simbolo_original: str):
        pais, simbolo_investpy, tipo_ativo = self._get_investpy_country_and_symbol(simbolo_original)
        params = {"simbolo_investpy": simbolo_investpy, "pais": pais, "tipo_ativo": tipo_ativo}
        cache_key_suffix = f"investpy_info_{simbolo_investpy}_{tipo_ativo}"
        dados_cache = self.cache_manager.get_cache("investpy", cache_key_suffix, params=params)
        if dados_cache: return dados_cache
        try:
            data_dict = None
            if tipo_ativo == "stock":
                data_str = investpy.get_stock_recent_data(stock=simbolo_investpy, country=pais, as_json=True)
                data_dict = json.loads(data_str) if isinstance(data_str, str) else data_str
            elif tipo_ativo == "index": 
                info_dict = investpy.get_index_information(index=simbolo_investpy, country=pais, as_json=True)
                info_dict = json.loads(info_dict) if isinstance(info_dict, str) else info_dict
                data_dict = { 
                    "name": info_dict.get("name"),
                    "last": info_dict.get("Stock Exchange"), 
                    "changeAbsolute": info_dict.get("Change"),
                    "changePercent": info_dict.get("Change %")
                }
            else:
                logger.warning(f"Tipo de ativo '{tipo_ativo}' para '{simbolo_investpy}' não suportado para info recente via investpy.")
                return None
            
            if data_dict:
                self.cache_manager.set_cache("investpy", cache_key_suffix, data_dict, CACHE_EXPIRATION_FINNHUB_QUOTE, params=params)
            return data_dict
        except Exception as e:
            logger.error(f"Erro ao obter dados recentes do investpy para {simbolo_investpy} ({pais}): {e}")
            return None

    def obter_dados_historicos_alpha_vantage(self, simbolo: str, outputsize="compact", series_type="TIME_SERIES_DAILY_ADJUSTED"):
        if not self.alpha_vantage_key: logger.warning("Chave Alpha Vantage não configurada."); return pd.DataFrame()
        params_api = {"function": series_type, "symbol": simbolo, "outputsize": outputsize, "apikey": self.alpha_vantage_key, "datatype": "json"}
        cache_params = {"simbolo": simbolo, "outputsize": outputsize, "series_type": series_type}
        dados_cache = self.cache_manager.get_cache("alpha_vantage", f"timeseries_{simbolo}", params=cache_params)
        if dados_cache:
            try:
                df = pd.DataFrame.from_dict(dados_cache, orient='index', dtype=float)
                df.index = pd.to_datetime(df.index) 
                df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. adjusted close': 'Adj Close', '6. volume': 'Volume'})
                return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].sort_index()
            except Exception as e:
                logger.error(f"Erro ao processar dados Alpha Vantage do cache para {simbolo}: {e}.")
                self.cache_manager.delete_cache(self.cache_manager._gerar_chave_cache("alpha_vantage", f"timeseries_{simbolo}", params=cache_params))
        try:
            response = requests.get("https://www.alphavantage.co/query", params=params_api)
            response.raise_for_status(); data = response.json()
            time_series_key = next((k for k in data if "Time Series" in k or "Adjusted Time Series" in k), None)
            if not time_series_key or not data.get(time_series_key):
                logger.error(f"Formato inesperado Alpha Vantage para {simbolo}: {data.get('Information', data.get('Note', ''))}")
                return pd.DataFrame()

            self.cache_manager.set_cache("alpha_vantage", f"timeseries_{simbolo}", data[time_series_key], CACHE_EXPIRATION_ALPHA_VANTAGE_TIMESERIES, params=cache_params)
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index', dtype=float)
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. adjusted close': 'Adj Close', '6. volume': 'Volume'})
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].sort_index()
        except Exception as e: logger.error(f"Erro na requisição Alpha Vantage para {simbolo}: {e}"); return pd.DataFrame()

    def obter_indicador_sgs_bcb(self, codigo_serie: int, nome_serie: str, ultimos_n: int = None, data_inicio: str = None, data_fim: str = None):
        params = {"codigo_serie": codigo_serie, "nome_serie": nome_serie, "ultimos_n": ultimos_n, "data_inicio": data_inicio, "data_fim": data_fim}
        cache_key_suffix = f"sgs_{codigo_serie}"
        dados_cache = self.cache_manager.get_cache("bcb_sgs", cache_key_suffix, params=params)
        if dados_cache:
            df = pd.DataFrame(dados_cache)
            if 'data' in df.columns:
                df['data'] = pd.to_datetime(df['data'])
                df = df.set_index('data')
            if nome_serie not in df.columns and len(df.columns) == 1: df.rename(columns={df.columns[0]: nome_serie}, inplace=True)
            return df
        try:
            serie_df = bcb_sgs.get({nome_serie: codigo_serie}, last=ultimos_n, start=data_inicio, end=data_fim)
            if isinstance(serie_df, dict) and nome_serie in serie_df: serie_df = pd.DataFrame(serie_df[nome_serie]) 
            elif isinstance(serie_df, dict): serie_df = pd.DataFrame() 
            if isinstance(serie_df, pd.Series): serie_df = pd.DataFrame({nome_serie: serie_df})
            
            if not serie_df.empty:
                serie_para_cache = serie_df.reset_index()
                date_col_name = serie_para_cache.columns[0]
                serie_para_cache.rename(columns={date_col_name: 'data'}, inplace=True)
                if pd.api.types.is_datetime64_any_dtype(serie_para_cache['data']):
                    serie_para_cache['data'] = serie_para_cache['data'].dt.strftime('%Y-%m-%d')
                self.cache_manager.set_cache("bcb_sgs", cache_key_suffix, serie_para_cache.to_dict(orient='list'), CACHE_EXPIRATION_SGS, params=params)
                serie_df.index.name = 'data'; return serie_df
            return pd.DataFrame()
        except Exception as e: logger.error(f"Erro SGS BCB série {codigo_serie} ({nome_serie}): {e}"); return pd.DataFrame()

    def obter_cotacao_dolar_ptax_periodo(self, data_inicio: str, data_fim: str):
        params = {"data_inicio": data_inicio, "data_fim": data_fim}
        dados_cache = self.cache_manager.get_cache("bcb_ptax", "ptax_usd_periodo", params=params)
        if dados_cache:
            df = pd.DataFrame(dados_cache)
            if 'dataHoraCotacao' in df.columns:
                try:
                    df['dataHoraCotacao'] = pd.to_datetime(df['dataHoraCotacao'], errors='coerce')
                except Exception as e_load_conv:
                    logger.warning(f"Erro ao converter 'dataHoraCotacao' do cache para datetime: {e_load_conv}")
            return df
        
        try: 
            ptax_client = PTAX()
            ep = ptax_client.get_endpoint('CotacaoMoedaPeriodo')
            dt_inicio_obj = datetime.strptime(data_inicio, '%Y-%m-%d')
            dt_fim_obj = datetime.strptime(data_fim, '%Y-%m-%d')
            data_inicio_ptax = dt_inicio_obj.strftime('%m/%d/%Y')
            data_fim_ptax = dt_fim_obj.strftime('%m/%d/%Y')
            
            cotacoes = ep.query().parameters(moeda='USD', dataInicial=data_inicio_ptax, dataFinalCotacao=data_fim_ptax).collect()
            
            if not cotacoes.empty:
                cotacoes_para_cache = cotacoes.copy()
                for col in cotacoes_para_cache.columns:
                    if pd.api.types.is_datetime64_any_dtype(cotacoes_para_cache[col].dtype) or \
                       (cotacoes_para_cache[col].dtype == 'object' and not cotacoes_para_cache[col].empty and isinstance(cotacoes_para_cache[col].iloc[0], (datetime, date))):
                        try:
                            cotacoes_para_cache[col] = pd.to_datetime(cotacoes_para_cache[col], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
                        except Exception as e_conv:
                            logger.warning(f"Não foi possível converter coluna PTAX '{col}' para string ISO: {e_conv}. Tentando str().")
                            cotacoes_para_cache[col] = cotacoes_para_cache[col].astype(str)
                    elif cotacoes_para_cache[col].dtype == 'object': 
                        try:
                            cotacoes_para_cache[col] = pd.to_datetime(cotacoes_para_cache[col], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
                        except: 
                            logger.info(f"Coluna PTAX '{col}' do tipo object, convertendo para string para cache.")
                            cotacoes_para_cache[col] = cotacoes_para_cache[col].astype(str)
                
                self.cache_manager.set_cache("bcb_ptax", "ptax_usd_periodo", cotacoes_para_cache.to_dict(orient='list'), CACHE_EXPIRATION_PTAX, params=params)
            return cotacoes
        except Exception as e:
            logger.error(f"Erro PTAX USD: {e}")
            return pd.DataFrame()

    def obter_expectativas_focus_mercado(self, indicador: str, ultimos_n_anos: int = 1):
        params = {"indicador": indicador, "ultimos_n_anos": ultimos_n_anos}
        dados_cache = self.cache_manager.get_cache("bcb_focus", f"focus_{indicador}", params=params)
        if dados_cache:
            return pd.DataFrame(dados_cache)
        
        try:
            em_client = Expectativas()
            ep = em_client.get_endpoint('ExpectativasMercadoAnuais')
            
            data_atual = datetime.now()
            anos_referencia_int = [data_atual.year + i for i in range(ultimos_n_anos)]
            
            query = ep.query().filter(ep.Indicador == indicador)
            
            todos_dados_indicador = query.collect()

            if todos_dados_indicador.empty:
                return pd.DataFrame()

            todos_dados_indicador['DataReferencia'] = pd.to_numeric(todos_dados_indicador['DataReferencia'], errors='coerce')
            query_result = todos_dados_indicador[todos_dados_indicador['DataReferencia'].isin(anos_referencia_int)].copy()

            if not query_result.empty:
                query_result_cache = query_result.copy() 
                for col in query_result_cache.columns:
                    if pd.api.types.is_datetime64_any_dtype(query_result_cache[col].dtype) or \
                       (query_result_cache[col].dtype == 'object' and not query_result_cache[col].empty and isinstance(query_result_cache[col].iloc[0], (datetime, date))):
                        try:
                            query_result_cache[col] = pd.to_datetime(query_result_cache[col], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
                        except Exception as e_conv:
                            logger.warning(f"Não foi possível converter coluna Focus '{col}' para string ISO: {e_conv}. Tentando str().")
                            query_result_cache[col] = query_result_cache[col].astype(str)
                    elif query_result_cache[col].dtype == 'object': 
                        try:
                            query_result_cache[col] = pd.to_datetime(query_result_cache[col], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
                        except:
                            logger.info(f"Coluna Focus '{col}' do tipo object, convertendo para string para cache.")
                            query_result_cache[col] = query_result_cache[col].astype(str)
                
                self.cache_manager.set_cache("bcb_focus", f"focus_{indicador}", query_result_cache.to_dict(orient='list'), CACHE_EXPIRATION_SGS, params=params)
            return query_result 
        except Exception as e:
            logger.error(f"Erro Focus {indicador}: {e}")
            return pd.DataFrame()

    def obter_serie_fred(self, id_serie: str, data_inicio: str = None, data_fim: str = None):
        if not self.fred_client: logger.warning("Chave FRED não configurada."); return pd.DataFrame()
        params = {"id_serie": id_serie, "data_inicio": data_inicio, "data_fim": data_fim}
        dados_cache = self.cache_manager.get_cache("fred", f"serie_{id_serie}", params=params)
        if dados_cache:
            df = pd.DataFrame(dados_cache)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            return df
        try:
            serie_data = self.fred_client.get_series(id_serie, observation_start=data_inicio, observation_end=data_fim)
            df = pd.DataFrame(serie_data, columns=['value']).dropna(); df.index.name = 'date'
            df_cache = df.reset_index()
            if pd.api.types.is_datetime64_any_dtype(df_cache['date']):
                 df_cache['date'] = df_cache['date'].dt.strftime('%Y-%m-%d')
            self.cache_manager.set_cache("fred", f"serie_{id_serie}", df_cache.to_dict(orient='list'), CACHE_EXPIRATION_FRED, params=params)
            return df
        except Exception as e: logger.error(f"Erro FRED {id_serie}: {e}"); return pd.DataFrame()

    def obter_dataset_nasdaq(self, codigo_dataset: str, data_inicio: str = None, data_fim: str = None, **kwargs):
        if not NASDAQ_DATA_LINK_API_KEY: logger.warning("Chave Nasdaq não configurada."); return pd.DataFrame()
        params = {"codigo_dataset": codigo_dataset, "data_inicio": data_inicio, "data_fim": data_fim, "kwargs": str(kwargs)}
        cache_key = f"dataset_{codigo_dataset.replace('/','_')}"
        dados_cache = self.cache_manager.get_cache("nasdaq", cache_key, params=params)
        if dados_cache:
            df = pd.DataFrame(dados_cache)
            date_col_name = df.columns[0] 
            if date_col_name in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col_name].dtype) : 
                df[date_col_name] = pd.to_datetime(df[date_col_name])
            if date_col_name in df.columns: 
                df = df.set_index(date_col_name)
            return df
        try:
            data = nasdaqdatalink.get(codigo_dataset, start_date=data_inicio, end_date=data_fim, **kwargs)
            df = pd.DataFrame(data)
            df_cache = df.reset_index()
            date_col_cache = df_cache.columns[0] 
            if pd.api.types.is_datetime64_any_dtype(df_cache[date_col_cache].dtype):
                 df_cache[date_col_cache] = df_cache[date_col_cache].dt.strftime('%Y-%m-%d')
            self.cache_manager.set_cache("nasdaq", cache_key, df_cache.to_dict(orient='list'), CACHE_EXPIRATION_NASDAQ, params=params)
            return df
        except Exception as e: logger.error(f"Erro Nasdaq {codigo_dataset}: {e}"); return pd.DataFrame()

    def buscar_simbolos_finnhub(self, query: str) -> List[Dict[str, str]]:
        if not self.finnhub_client:
            logger.warning("Finnhub client não configurado para busca de símbolos.")
            return []
        params = {"query": query}
        cache_key = f"finnhub_symbol_search_{query.lower().replace(' ','_')}"
        dados_cache = self.cache_manager.get_cache("finnhub_search", cache_key, params=params)
        if dados_cache:
            return dados_cache
        
        try:
            resultados = self.finnhub_client.symbol_lookup(query)
            simbolos_formatados = []
            if resultados and isinstance(resultados, dict) and 'result' in resultados:
                for item in resultados['result']:
                    simbolos_formatados.append({
                        "symbol": item.get("symbol"),
                        "description": item.get("description"),
                        "type": item.get("type")
                    })
            self.cache_manager.set_cache("finnhub_search", cache_key, simbolos_formatados, CACHE_EXPIRATION_FINNHUB_QUOTE * 48, params=params) 
            return simbolos_formatados
        except Exception as e:
            logger.error(f"Erro ao buscar símbolos na Finnhub para '{query}': {e}")
            return []