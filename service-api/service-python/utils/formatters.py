from datetime import datetime
import pandas as pd

def formatar_valor_monetario(valor, moeda="R$"):
    if valor is None:
        return f"{moeda} N/A"
    return f"{moeda} {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_percentual(valor):
    if valor is None:
        return "N/A %"
    return f"{valor:.2f}%"

def formatar_data_ptbr(data_iso):
    if isinstance(data_iso, str):
        try:
            data_obj = datetime.fromisoformat(data_iso.split(' ')[0])
            return data_obj.strftime('%d/%m/%Y')
        except ValueError:
            try: 
                data_obj = datetime.strptime(data_iso, '%Y-%m-%d')
                return data_obj.strftime('%d/%m/%Y')
            except ValueError:
                return data_iso 
    elif isinstance(data_iso, (datetime, pd.Timestamp)):
        return data_iso.strftime('%d/%m/%Y')
    return str(data_iso)

def formatar_variacao_diaria(fechamento_atual, fechamento_anterior):
    if fechamento_atual is None or fechamento_anterior is None or fechamento_anterior == 0:
        return "N/A"
    variacao = ((fechamento_atual - fechamento_anterior) / fechamento_anterior) * 100
    sinal = "+" if variacao > 0 else ""
    return f"{sinal}{variacao:.2f}%"

def formatar_dataframe_para_csv(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    
    df_export = df.copy()
    if isinstance(df_export.index, pd.DatetimeIndex):
        df_export.index = df_export.index.strftime('%Y-%m-%d')
    
    if 'Date' in df_export.columns and isinstance(df_export['Date'].iloc[0], pd.Timestamp):
         df_export['Date'] = pd.to_datetime(df_export['Date']).dt.strftime('%Y-%m-%d')
    elif 'ds' in df_export.columns and isinstance(df_export['ds'].iloc[0], pd.Timestamp):
         df_export['ds'] = pd.to_datetime(df_export['ds']).dt.strftime('%Y-%m-%d')

    return df_export.to_csv(index=True, sep=';', decimal=',')

def formatar_dados_para_txt_simples(titulo: str, dados: dict) -> str:
    report_lines = [f"--- {titulo} ---"]
    
    if "titulo" in dados and dados["titulo"] == titulo:
        del dados["titulo"]

    for chave, valor in dados.items():
        report_lines.append(f"{str(chave).replace('_', ' ').capitalize()}: {valor}")
    report_lines.append("--------------------")
    return "\n".join(report_lines)