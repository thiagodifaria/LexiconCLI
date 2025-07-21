from models.market_data import MarketDataProvider
from models.data_model import CotacaoAtivo, IndicadorMacroeconomico, DadosHistoricos, SerieEconomica, PreferenciasVisualizacao, AlertaConfigurado
from models.database import (
    obter_watchlist_do_db, 
    adicionar_item_watchlist_db, 
    remover_item_watchlist_db,
    salvar_preferencias_visualizacao_db,
    carregar_preferencias_visualizacao_db,
    adicionar_alerta_db,
    listar_alertas_db,
    remover_alerta_db,
    atualizar_alerta_db 
)
from config.settings import BCB_SERIES_IPCA, BCB_SERIES_SELIC, DEFAULT_HISTORICAL_PERIOD, DEFAULT_INDICATORS_VIEW
from datetime import datetime, timedelta
import pandas as pd
from utils.logger import logger
from typing import Optional, Dict, List, Any

class DataController:
    def __init__(self):
        self.provider = MarketDataProvider()

    def _identificar_tipo_ativo_original(self, simbolo: str) -> str:
        if simbolo.startswith("^"): return "index"
        if ".SA" in simbolo: return "stock_br" 
        return "stock_us" 

    def buscar_cotacao_ativo(self, simbolo: str) -> CotacaoAtivo:
        dados_finnhub = self.provider.obter_cotacao_finnhub(simbolo)
        if dados_finnhub and dados_finnhub.get('c') is not None and dados_finnhub.get('c') != 0:
            return CotacaoAtivo(
                simbolo=simbolo, preco_atual=dados_finnhub.get('c'),
                variacao_percentual=dados_finnhub.get('dp'), variacao_absoluta=dados_finnhub.get('d'),
                preco_abertura=dados_finnhub.get('o'), preco_maximo=dados_finnhub.get('h'),
                preco_minimo=dados_finnhub.get('l'), preco_fechamento_anterior=dados_finnhub.get('pc'),
                timestamp_ultima_atualizacao=dados_finnhub.get('t')
            )
        
        logger.info(f"Finnhub sem dados para {simbolo} ou preço zerado. Tentando fallback com histórico recente.")
        dados_hist = self.buscar_dados_historicos(simbolo, periodo="7d", intervalo="1d")
        
        if not dados_hist.dataframe.empty and len(dados_hist.dataframe) >= 2:
            df_hist = dados_hist.dataframe.sort_index(ascending=True) 
            ultima_cotacao = df_hist.iloc[-1]
            penultima_cotacao = df_hist.iloc[-2]
            
            preco_atual = ultima_cotacao['Close']
            preco_fech_ant = penultima_cotacao['Close']
            
            if pd.isna(preco_atual) or pd.isna(preco_fech_ant):
                logger.warning(f"Preço atual ou fechamento anterior N/A para {simbolo} no histórico recente.")
                return CotacaoAtivo(simbolo=simbolo)

            variacao_abs = preco_atual - preco_fech_ant
            variacao_perc = (variacao_abs / preco_fech_ant) * 100 if preco_fech_ant != 0 else 0.0
            return CotacaoAtivo(
                simbolo=simbolo, preco_atual=preco_atual, variacao_percentual=variacao_perc,
                variacao_absoluta=variacao_abs, preco_abertura=ultima_cotacao['Open'],
                preco_maximo=ultima_cotacao['High'], preco_minimo=ultima_cotacao['Low'],
                preco_fechamento_anterior=preco_fech_ant,
                volume=int(ultima_cotacao['Volume']) if 'Volume' in ultima_cotacao and pd.notna(ultima_cotacao['Volume']) else None,
                timestamp_ultima_atualizacao=int(datetime.combine(ultima_cotacao.name.date(), datetime.min.time()).timestamp()) if hasattr(ultima_cotacao.name, 'date') else int(datetime.now().timestamp())
            )
        logger.warning(f"Não foi possível obter cotação detalhada para {simbolo} via fallback."); return CotacaoAtivo(simbolo=simbolo)

    def buscar_dados_historicos(self, simbolo: str, periodo="1y", intervalo="1d", data_inicio:Optional[str] = None, data_fim:Optional[str] = None) -> DadosHistoricos:
        """
        🔧 BUG FIX: Busca dados históricos com normalização robusta de colunas
        """
        df_historico = pd.DataFrame()
        
        if not data_inicio or not data_fim:
            data_fim_dt = datetime.now()
            num_anos = 0
            num_meses = 0
            num_dias = 0

            if "y" in periodo:
                num_anos = int(periodo.replace("y",""))
            elif "mo" in periodo:
                num_meses = int(periodo.replace("mo",""))
            elif "d" in periodo:
                num_dias = int(periodo.replace("d",""))
            else: 
                num_dias = 365
                
            data_inicio_dt = data_fim_dt - timedelta(days=(num_anos*365 + num_meses*30 + num_dias))
            data_inicio_str = data_inicio_dt.strftime("%Y-%m-%d"); data_fim_str = data_fim_dt.strftime("%Y-%m-%d")
        else: 
            data_inicio_str = data_inicio
            data_fim_str = data_fim
            
        logger.info(f"Tentando yfinance para dados históricos de {simbolo} de {data_inicio_str} a {data_fim_str}")
        df_historico = self.provider.obter_dados_historicos_yf(simbolo, start_date=data_inicio_str, end_date=data_fim_str, interval=intervalo)
        
        if df_historico.empty:
            logger.info(f"yfinance falhou para {simbolo}. Tentando investpy.")
            df_historico = self.provider.obter_dados_historicos_investpy(simbolo, de_data=data_inicio_str, ate_data=data_fim_str)

        if df_historico.empty and self.provider.alpha_vantage_key:
            logger.info(f"investpy falhou para {simbolo}. Tentando Alpha Vantage.")
            series_type = "TIME_SERIES_DAILY_ADJUSTED"; output_size = "full" 
            if intervalo == "1wk": series_type = "TIME_SERIES_WEEKLY_ADJUSTED"
            elif intervalo == "1mo": series_type = "TIME_SERIES_MONTHLY_ADJUSTED"
            df_historico_av = self.provider.obter_dados_historicos_alpha_vantage(simbolo, outputsize=output_size, series_type=series_type)
            if not df_historico_av.empty:
                 df_historico_av.index = pd.to_datetime(df_historico_av.index) 
                 df_historico = df_historico_av[(df_historico_av.index >= pd.to_datetime(data_inicio_str)) & (df_historico_av.index <= pd.to_datetime(data_fim_str))]

        cols_necessarias = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not df_historico.empty:
            logger.debug(f"Colunas originais dos dados históricos de {simbolo}: {list(df_historico.columns)}")
            
            column_mapping = {}
            cols_esperadas_lower = ['open', 'high', 'low', 'close', 'volume']
            
            for col in df_historico.columns:
                col_lower = str(col).lower()
                if col_lower in cols_esperadas_lower:
                    column_mapping[col] = col_lower.capitalize()
            
            if column_mapping:
                df_historico = df_historico.rename(columns=column_mapping)
                logger.debug(f"Colunas após normalização: {list(df_historico.columns)}")
            
            for col in cols_necessarias:
                if col not in df_historico.columns:
                    if col == 'Volume':
                        df_historico[col] = 0
                        logger.warning(f"Coluna {col} não encontrada para {simbolo}, preenchendo com zeros")
                    else:
                        logger.error(f"Coluna crítica {col} não encontrada para {simbolo}")
                        df_historico[col] = pd.NA
            
            outras_colunas = [col for col in df_historico.columns if col not in cols_necessarias]
            df_historico = df_historico[cols_necessarias + outras_colunas]
            
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df_historico.columns:
                    df_historico[col] = pd.to_numeric(df_historico[col], errors='coerce')
            
            if 'Volume' in df_historico.columns:
                df_historico['Volume'] = pd.to_numeric(df_historico['Volume'], errors='coerce').fillna(0)
        
        else: 
            df_historico = pd.DataFrame(columns=cols_necessarias)
            logger.warning(f"Não foi possível obter dados históricos para {simbolo}")
        
        return DadosHistoricos(simbolo=simbolo, dataframe=df_historico.sort_index())

    def buscar_indicadores_macro_bcb(self) -> Dict[str, IndicadorMacroeconomico]:
        indicadores: Dict[str, IndicadorMacroeconomico] = {}
        df_ipca = self.provider.obter_indicador_sgs_bcb(BCB_SERIES_IPCA, "IPCA", ultimos_n=2)
        if not df_ipca.empty and "IPCA" in df_ipca.columns and not df_ipca["IPCA"].empty:
            ultimo_ipca = df_ipca.iloc[-1]
            indicadores["IPCA"] = IndicadorMacroeconomico(nome="IPCA (Mensal)", valor=ultimo_ipca["IPCA"], data_referencia=ultimo_ipca.name.strftime('%Y-%m-%d'), unidade="%")
        df_selic = self.provider.obter_indicador_sgs_bcb(BCB_SERIES_SELIC, "SELIC Meta", ultimos_n=2)
        if not df_selic.empty and "SELIC Meta" in df_selic.columns and not df_selic["SELIC Meta"].empty:
            ultima_selic = df_selic.iloc[-1]
            indicadores["SELIC"] = IndicadorMacroeconomico(nome="SELIC (Meta)", valor=ultima_selic["SELIC Meta"], data_referencia=ultima_selic.name.strftime('%Y-%m-%d'), unidade="% a.a.")
        hoje = datetime.now(); inicio_datas_ptax = hoje - timedelta(days=35)
        df_dolar_ptax = self.provider.obter_cotacao_dolar_ptax_periodo(data_inicio=inicio_datas_ptax.strftime('%Y-%m-%d'), data_fim=hoje.strftime('%Y-%m-%d'))
        if not df_dolar_ptax.empty:
            df_dolar_ptax_sorted = df_dolar_ptax.sort_values(by='dataHoraCotacao', ascending=False)
            ultima_cotacao_dolar_valida = next((row for _, row in df_dolar_ptax_sorted.iterrows() if pd.notna(row['cotacaoVenda']) and row['cotacaoVenda'] > 0), None)
            if ultima_cotacao_dolar_valida is not None:
                data_ref_ptax_obj = pd.to_datetime(ultima_cotacao_dolar_valida['dataHoraCotacao'])
                data_ref_ptax = data_ref_ptax_obj.strftime('%Y-%m-%d') if pd.notna(data_ref_ptax_obj) else "N/A"
                indicadores["USD_BRL_PTAX"] = IndicadorMacroeconomico(nome="USD/BRL (PTAX Venda)", valor=ultima_cotacao_dolar_valida['cotacaoVenda'], data_referencia=data_ref_ptax, unidade="R$")
        return indicadores

    def buscar_expectativas_mercado_focus(self, indicador_nome: str, ultimos_n_anos: int = 1) -> pd.DataFrame:
        return self.provider.obter_expectativas_focus_mercado(indicador_nome, ultimos_n_anos)

    def buscar_serie_fred(self, id_serie: str, nome_exibicao: str, data_inicio: Optional[str] = None, data_fim: Optional[str] = None) -> Optional[SerieEconomica]:
        df_fred = self.provider.obter_serie_fred(id_serie, data_inicio, data_fim)
        if not df_fred.empty:
            unidade, frequencia, notas, nome_serie_api = None, None, None, nome_exibicao
            if self.provider.fred_client:
                try:
                    info = self.provider.fred_client.get_series_info(id_serie)
                    unidade = getattr(info, 'units_short', getattr(info, 'units', None))
                    frequencia = getattr(info, 'frequency_short', getattr(info, 'frequency', None))
                    notas = getattr(info, 'notes', None)
                    nome_serie_api = getattr(info, 'title', nome_exibicao)
                except Exception as e: logger.warning(f"Não obter metadados FRED para {id_serie}: {e}")
            return SerieEconomica(id_serie=id_serie, nome_serie=nome_serie_api, dataframe=df_fred, fonte="FRED", unidade=unidade, frequencia=frequencia, notas=str(notas) if notas else None)
        return None

    def buscar_dataset_nasdaq(self, codigo_dataset: str, nome_exibicao: str, data_inicio: Optional[str] = None, data_fim: Optional[str] = None, **kwargs) -> Optional[SerieEconomica]:
        df_nasdaq = self.provider.obter_dataset_nasdaq(codigo_dataset, data_inicio, data_fim, **kwargs)
        if not df_nasdaq.empty:
            return SerieEconomica(id_serie=codigo_dataset, nome_serie=nome_exibicao, dataframe=df_nasdaq, fonte="Nasdaq Data Link")
        return None

    def get_watchlist(self) -> List[Dict[str, Any]]:
        return obter_watchlist_do_db()

    def add_to_watchlist(self, simbolo: str, tipo: str) -> bool:
        return adicionar_item_watchlist_db(simbolo, tipo)

    def remove_from_watchlist(self, simbolo: str) -> bool:
        return remover_item_watchlist_db(simbolo)
    
    def buscar_balanco_patrimonial(self, simbolo: str) -> Optional[Dict]:
        """Busca os dados do balanço patrimonial anual."""
        logger.info(f"DataController solicitando balanço patrimonial para {simbolo}.")
        return self.provider.obter_balanco_patrimonial_anual(simbolo)

    def buscar_demonstrativo_resultados(self, simbolo: str) -> Optional[Dict]:
        """Busca os dados da demonstração de resultados anual."""
        logger.info(f"DataController solicitando demonstrativo de resultados para {simbolo}.")
        return self.provider.obter_dre_anual(simbolo)

    def buscar_fluxo_caixa(self, simbolo: str) -> Optional[Dict]:
        """Busca os dados do fluxo de caixa anual."""
        logger.info(f"DataController solicitando fluxo de caixa para {simbolo}.")
        return self.provider.obter_fluxo_caixa_anual(simbolo)

    def pesquisar_simbolos(self, termo_busca: str) -> List[Dict[str, str]]:
        resultados = self.provider.buscar_simbolos_finnhub(termo_busca)
        simbolos_formatados = []
        if resultados and isinstance(resultados, dict) and 'result' in resultados:
            for item in resultados['result']:
                simbolos_formatados.append({
                    "symbol": item.get("symbol"),
                    "description": item.get("description"),
                    "type": item.get("type")
                })
        elif isinstance(resultados, list): 
            simbolos_formatados = resultados

        return simbolos_formatados[:10] 

    def salvar_preferencias_visualizacao_db(self, preferencias: Dict[str, Any]) -> bool:
        id_usuario = preferencias.get('id_usuario', 0) 
        periodo_historico = preferencias.get('periodo_historico_padrao', DEFAULT_HISTORICAL_PERIOD)
        indicadores_tecnicos_str = ",".join(preferencias.get('indicadores_tecnicos_padrao', DEFAULT_INDICATORS_VIEW))
        
        return salvar_preferencias_visualizacao_db(id_usuario, periodo_historico, indicadores_tecnicos_str)

    def carregar_preferencias_visualizacao_db(self, id_usuario: int = 0) -> Optional[Dict[str, Any]]:
        prefs_raw = carregar_preferencias_visualizacao_db(id_usuario)
        if prefs_raw:
            return {
                "id_usuario": prefs_raw["id_usuario"],
                "periodo_historico_padrao": prefs_raw["periodo_historico_padrao"],
                "indicadores_tecnicos_padrao": prefs_raw["indicadores_tecnicos_padrao"].split(',') if prefs_raw["indicadores_tecnicos_padrao"] else []
            }
        return None
        
    def adicionar_alerta_db(self, alerta_data: Dict[str, Any]) -> bool:
        alerta = AlertaConfigurado(**alerta_data)
        return adicionar_alerta_db(alerta)

    def listar_alertas_db(self) -> List[Dict[str, Any]]:
        alertas_raw = listar_alertas_db()
        return [dict(alerta) for alerta in alertas_raw]


    def remover_alerta_db(self, id_alerta: int) -> bool:
        return remover_alerta_db(id_alerta)

    def atualizar_alerta_db(self, id_alerta: int, novos_dados: Dict[str, Any]) -> bool:
        return atualizar_alerta_db(id_alerta, novos_dados)