from models.indicators import IndicadoresTecnicos
import pandas as pd
from utils.logger import logger
from typing import List, Dict, Any

class AnalysisController:
    def __init__(self):
        pass

    def calcular_todos_indicadores_principais(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if df_ohlcv is None or df_ohlcv.empty:
            logger.warning("DataFrame OHLCV vazio fornecido para cálculo de indicadores.")
            return pd.DataFrame()
        
        try:
            logger.debug(f"Colunas originais do DataFrame: {list(df_ohlcv.columns)}")
            
            column_mapping = {}
            required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
            
            for col in df_ohlcv.columns:
                col_lower = col.lower()
                if col_lower in required_cols_lower:
                    column_mapping[col] = col_lower.capitalize()
            
            if column_mapping:
                df_ohlcv = df_ohlcv.rename(columns=column_mapping)
                logger.debug(f"Colunas após mapeamento: {list(df_ohlcv.columns)}")
            
            required_cols = ['Open', 'High', 'Low', 'Close'] 
            missing_cols = [col for col in required_cols if col not in df_ohlcv.columns]
            
            if missing_cols:
                logger.error(f"DataFrame não contém todas as colunas OHLCV necessárias. Faltando: {missing_cols}")
                logger.error(f"Colunas disponíveis: {list(df_ohlcv.columns)}")
                return pd.DataFrame()
            
            if 'Volume' not in df_ohlcv.columns:
                logger.warning("Coluna 'Volume' não encontrada. Criando com valores zero.")
                df_ohlcv['Volume'] = 0 
            
            if len(df_ohlcv) < 30:
                logger.warning(f"Dados históricos insuficientes ({len(df_ohlcv)} dias). Mínimo recomendado: 30 dias.")
            
            indicadores_obj = IndicadoresTecnicos(df_ohlcv)

            indicadores_obj.adicionar_sma(periodo=9)
            indicadores_obj.adicionar_sma(periodo=21)
            indicadores_obj.adicionar_sma(periodo=50)
            indicadores_obj.adicionar_sma(periodo=200)
            indicadores_obj.adicionar_ema(periodo=9)
            indicadores_obj.adicionar_ema(periodo=21)
            indicadores_obj.adicionar_ema(periodo=50)
            indicadores_obj.adicionar_ema(periodo=200)
            indicadores_obj.adicionar_macd()
            indicadores_obj.adicionar_bandas_bollinger()
            indicadores_obj.adicionar_adx()
            indicadores_obj.adicionar_rsi()
            indicadores_obj.adicionar_estocastico()
            
            if 'Volume' in df_ohlcv.columns and df_ohlcv['Volume'].sum() > 0:
                indicadores_obj.adicionar_obv()
            else:
                indicadores_obj.df['OBV'] = pd.NA

            indicadores_obj.adicionar_atr()
            indicadores_obj.adicionar_desvio_padrao_retornos()
            
            df_final = indicadores_obj.obter_df_com_indicadores()            
            
            logger.info(f"Indicadores calculados com sucesso. DataFrame final: {df_final.shape}")
            return df_final

        except KeyError as e:
            logger.error(f"Erro de coluna não encontrada: {e}. Colunas disponíveis: {list(df_ohlcv.columns)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro geral ao calcular indicadores: {e}", exc_info=True)
            return pd.DataFrame()

    def verificar_sinais_para_alertas(self, simbolo: str, df_com_indicadores: pd.DataFrame, alertas_configurados: List[Dict[str, Any]]) -> List[str]:
        sinais_disparados = []
        if df_com_indicadores.empty or not alertas_configurados:
            return sinais_disparados

        ultimo_registro = df_com_indicadores.iloc[-1]
        preco_fechamento_atual = ultimo_registro.get('Close')

        for alerta in alertas_configurados:            
            if alerta.get('simbolo') == simbolo and alerta.get('ativo', False):
                tipo_alerta = alerta.get('tipo_alerta')
                condicao = alerta.get('condicao', {})
                mensagem_customizada = alerta.get('mensagem_customizada')
                id_alerta = alerta.get('id_alerta', 'N/A')
                
                mensagem_base = f"ID:{id_alerta} - {simbolo}"
                if mensagem_customizada:
                    mensagem_final = f"ALERTA: {mensagem_base} - {mensagem_customizada}"
                else:
                    mensagem_final = f"ALERTA: {mensagem_base}"

                disparado = False
                if tipo_alerta == 'preco_acima' and preco_fechamento_atual is not None:                           
                    valor_ref = condicao.get('valor_referencia')
                    if valor_ref is not None and preco_fechamento_atual > valor_ref:
                        mensagem_final += f" - Preço ({preco_fechamento_atual:.2f}) ACIMA de {valor_ref:.2f}!"
                        disparado = True
                
                elif tipo_alerta == 'preco_abaixo' and preco_fechamento_atual is not None:
                    valor_ref = condicao.get('valor_referencia')
                    if valor_ref is not None and preco_fechamento_atual < valor_ref:
                        mensagem_final += f" - Preço ({preco_fechamento_atual:.2f}) ABAIXO de {valor_ref:.2f}!"
                        disparado = True

                elif tipo_alerta == 'rsi_sobrecompra':                
                    rsi_val = ultimo_registro.get('RSI_14')
                    limiar = condicao.get('limiar_rsi', 70)
                    if rsi_val is not None and rsi_val > limiar:                                  
                        mensagem_final += f" - RSI ({rsi_val:.2f}) em SOBRECOMPRA (> {limiar})!"
                        disparado = True
                
                elif tipo_alerta == 'rsi_sobrevenda':                
                    rsi_val = ultimo_registro.get('RSI_14')
                    limiar = condicao.get('limiar_rsi', 30)
                    if rsi_val is not None and rsi_val < limiar:                                  
                        mensagem_final += f" - RSI ({rsi_val:.2f}) em SOBREVENDA (< {limiar})!"
                        disparado = True
                
                if disparado:
                    sinais_disparados.append(mensagem_final)
        
        if sinais_disparados:
            logger.info(f"Verificação de alertas para {simbolo} concluída. Sinais: {sinais_disparados}")
        return sinais_disparados