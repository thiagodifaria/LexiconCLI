from rich.table import Table
from rich.text import Text
from utils.formatters import formatar_valor_monetario, formatar_percentual, formatar_data_ptbr
from models.data_model import CotacaoAtivo, IndicadorMacroeconomico, SerieEconomica 
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import json

def criar_tabela_ativos_monitorados(ativos_data: list[CotacaoAtivo]):
    tabela = Table(title="Ativos Monitorados", show_header=True, header_style="bold magenta")
    tabela.add_column("Símbolo", style="dim cyan", width=12)
    tabela.add_column("Preço Atual", justify="right")
    tabela.add_column("Mud. $", justify="right") 
    tabela.add_column("Mud. %", justify="right")
    tabela.add_column("Abertura", justify="right")
    tabela.add_column("Máxima", justify="right")
    tabela.add_column("Mínima", justify="right")
    tabela.add_column("Fech. Ant.", justify="right")

    for ativo in ativos_data:
        cor_variacao = "green" if ativo.variacao_percentual is not None and ativo.variacao_percentual >= 0 else "red"
        preco_str = formatar_valor_monetario(ativo.preco_atual, "") if ativo.preco_atual is not None else "N/A"
        var_abs_str = formatar_valor_monetario(ativo.variacao_absoluta, "") if ativo.variacao_absoluta is not None else "N/A"
        var_perc_str = formatar_percentual(ativo.variacao_percentual) if ativo.variacao_percentual is not None else "N/A"
        abertura_str = formatar_valor_monetario(ativo.preco_abertura, "") if ativo.preco_abertura is not None else "N/A"
        maxima_str = formatar_valor_monetario(ativo.preco_maximo, "") if ativo.preco_maximo is not None else "N/A"
        minima_str = formatar_valor_monetario(ativo.preco_minimo, "") if ativo.preco_minimo is not None else "N/A"
        fech_ant_str = formatar_valor_monetario(ativo.preco_fechamento_anterior, "") if ativo.preco_fechamento_anterior is not None else "N/A"

        tabela.add_row(
            ativo.simbolo,
            Text(preco_str, style=cor_variacao if ativo.variacao_percentual is not None else "white"),
            Text(var_abs_str, style=cor_variacao if ativo.variacao_absoluta is not None and ativo.variacao_absoluta !=0 else "white"),
            Text(var_perc_str, style=cor_variacao if ativo.variacao_percentual is not None and ativo.variacao_percentual !=0 else "white"),
            abertura_str,
            maxima_str,
            minima_str,
            fech_ant_str
        )
    return tabela

def criar_tabela_indices_mercado(indices_data: list[CotacaoAtivo]):
    tabela = Table(title="Principais Índices", show_header=True, header_style="bold blue")
    tabela.add_column("Índice", style="dim yellow", width=15)
    tabela.add_column("Pontuação", justify="right")
    tabela.add_column("Mud. $", justify="right") 
    tabela.add_column("Mud. %", justify="right")
    
    for indice in indices_data:
        cor_variacao = "green" if indice.variacao_percentual is not None and indice.variacao_percentual >= 0 else "red"
        pontuacao_str = f"{indice.preco_atual:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if indice.preco_atual is not None else "N/A"
        var_abs_str = f"{indice.variacao_absoluta:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if indice.variacao_absoluta is not None else "N/A"
        var_perc_str = formatar_percentual(indice.variacao_percentual) if indice.variacao_percentual is not None else "N/A"
        
        tabela.add_row(
            indice.simbolo,
            Text(pontuacao_str, style=cor_variacao if indice.variacao_percentual is not None else "white"),
            Text(var_abs_str, style=cor_variacao if indice.variacao_absoluta is not None and indice.variacao_absoluta != 0 else "white"),
            Text(var_perc_str, style=cor_variacao if indice.variacao_percentual is not None and indice.variacao_percentual !=0 else "white")
        )
    return tabela
    
def criar_tabela_indicadores_macro(indicadores_macro_data: dict[str, IndicadorMacroeconomico]):
    tabela = Table(title="Indicadores Macroeconômicos (BCB)", show_header=True, header_style="bold green")
    tabela.add_column("Indicador", style="dim magenta")
    tabela.add_column("Valor", justify="right")
    tabela.add_column("Data Referência", justify="center")
    tabela.add_column("Unidade", justify="left")

    for nome, dados in indicadores_macro_data.items():
        valor_str = f"{dados.valor:.2f}" if dados.valor is not None else "N/A"
        data_ref_str = formatar_data_ptbr(dados.data_referencia) if dados.data_referencia else "N/A"
        unidade_str = dados.unidade if dados.unidade else ""
        
        tabela.add_row(dados.nome, valor_str, data_ref_str, unidade_str)
    return tabela

def criar_tabela_serie_economica(serie: SerieEconomica, ultimos_n=10):
    if serie.dataframe.empty:
        return Text(f"Dados não disponíveis para {serie.nome_serie} ({serie.id_serie}) da fonte {serie.fonte}.", style="yellow")

    titulo = f"{serie.nome_serie} ({serie.id_serie}) - Fonte: {serie.fonte}"
    if serie.unidade: titulo += f" - Unidade: {serie.unidade}"
    if serie.frequencia: titulo += f" - Frequência: {serie.frequencia}"

    tabela = Table(title=titulo, show_header=True, header_style="bold cyan")
    tabela.add_column("Data", style="dim", justify="center")
    col_valor = 'value' if 'value' in serie.dataframe.columns else serie.dataframe.columns[0] if len(serie.dataframe.columns) > 0 else "Valor"
    tabela.add_column(col_valor.capitalize(), justify="right")

    df_exibir = serie.dataframe.sort_index(ascending=False).head(ultimos_n)

    for data_idx, row in df_exibir.iterrows():
        data_str = formatar_data_ptbr(data_idx)
        valor = row[col_valor]
        valor_str = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(valor) and isinstance(valor, (int, float)) else str(valor)
        tabela.add_row(data_str, valor_str)
    
    if serie.notas:
        tabela.caption = f"Notas: {serie.notas[:200]}{'...' if len(serie.notas) > 200 else ''}"
    return tabela


def criar_tabela_indicadores_tecnicos(indicadores_df):
    if indicadores_df is None or indicadores_df.empty:
        return Table(title="Indicadores Técnicos (Últimos Valores)")
    tabela = Table(title="Indicadores Técnicos (Últimos Valores)", show_header=True, header_style="bold cyan")
    tabela.add_column("Data", style="dim", justify="center")
    
    colunas_validas = [col for col in indicadores_df.columns if col.lower() != 'data']
    for coluna in colunas_validas:
        tabela.add_column(coluna, justify="right")

    ultimos_registros = indicadores_df.tail(5).iloc[::-1] # Pega os últimos 5 e inverte a ordem
    
    for index, row in ultimos_registros.iterrows():
        valores_linha = [index.strftime('%d/%m/%Y') if isinstance(index, pd.Timestamp) else str(index)]
        for coluna in colunas_validas:
            valor = row[coluna]
            if pd.isna(valor): valores_linha.append("N/A")
            elif isinstance(valor, float): valores_linha.append(f"{valor:.2f}")
            else: valores_linha.append(str(valor))
        tabela.add_row(*valores_linha)
    return tabela

def criar_tabela_previsoes_lstm(df_previsoes):
    tabela = Table(title="Previsões LSTM vs Real", show_header=True, header_style="bold yellow")
    tabela.add_column("Data", style="dim", justify="center")
    tabela.add_column("Preço Real", justify="right")
    tabela.add_column("Preço Previsto (LSTM)", justify="right")
    tabela.add_column("Diferença", justify="right")
    tabela.add_column("Erro %", justify="right")
    for _, row in df_previsoes.iterrows():
        data_str = row['Data'].strftime('%d/%m/%Y') if isinstance(row['Data'], (datetime, pd.Timestamp)) else str(row['Data'])
        real_str = formatar_valor_monetario(row['Real'], "")
        previsto_str = formatar_valor_monetario(row['Previsto'], "")
        diferenca = row['Real'] - row['Previsto']
        erro_perc = (diferenca / row['Real']) * 100 if row['Real'] != 0 else float('inf')
        cor_erro = "red" if abs(erro_perc) > 5 else "yellow" if abs(erro_perc) > 2 else "green"
        diferenca_str = formatar_valor_monetario(diferenca, "")
        erro_perc_str = formatar_percentual(erro_perc)
        tabela.add_row(data_str, real_str, previsto_str, Text(diferenca_str, style=cor_erro if diferenca !=0 else "white"), Text(erro_perc_str, style=cor_erro if erro_perc !=0 else "white"))
    return tabela

def criar_tabela_previsoes_prophet(df_previsoes: pd.DataFrame):
    if df_previsoes is None or df_previsoes.empty:
        return Table(title="Previsões Prophet")
        
    tabela = Table(title="Previsões Prophet", show_header=True, header_style="bold magenta")
    tabela.add_column("Data", style="dim", justify="center")
    
    col_real_presente = 'Real' in df_previsoes.columns
    if col_real_presente:
        tabela.add_column("Real", justify="right")

    tabela.add_column("Previsto (yhat)", justify="right")
    tabela.add_column("Prev. Mín (yhat_lower)", justify="right")
    tabela.add_column("Prev. Máx (yhat_upper)", justify="right")

    for _, row in df_previsoes.iterrows():
        data_str = row['ds'].strftime('%d/%m/%Y') if isinstance(row['ds'], (datetime, pd.Timestamp)) else str(row['ds'])
        
        valores_linha = [data_str]

        if col_real_presente:
            real_val = row.get('Real')
            real_str = formatar_valor_monetario(real_val, "") if pd.notna(real_val) else "N/A"
            valores_linha.append(real_str)

        previsto_val = row.get('Previsto') 
        previsto_str = formatar_valor_monetario(previsto_val, "") if pd.notna(previsto_val) else "N/A"
        valores_linha.append(previsto_str)
        
        min_val = row.get('yhat_lower')
        min_str = formatar_valor_monetario(min_val, "") if pd.notna(min_val) else "N/A"
        valores_linha.append(min_str)

        max_val = row.get('yhat_upper')
        max_str = formatar_valor_monetario(max_val, "") if pd.notna(max_val) else "N/A"
        valores_linha.append(max_str)
        
        tabela.add_row(*valores_linha)
    return tabela

def criar_tabela_watchlist(watchlist_items: List[Dict[str, Any]]):
    tabela = Table(title="Minha Watchlist", show_header=True, header_style="bold green")
    tabela.add_column("Símbolo", style="dim cyan", width=15)
    tabela.add_column("Tipo", width=10)
    
    if not watchlist_items:
        tabela.add_row(Text("Sua watchlist está vazia.", justify="center", span=2))
        return tabela

    for item in watchlist_items:
        tipo_display = "Ação" if item.get('tipo') == 'asset' else "Índice" if item.get('tipo') == 'index' else item.get('tipo', 'N/A')
        tabela.add_row(item.get('simbolo', 'N/A'), tipo_display)
    return tabela

def criar_tabela_resultados_busca_simbolos(resultados: List[Dict[str, Any]]):
    if not resultados:
        return Text("Nenhum símbolo encontrado.", style="yellow")
        
    tabela = Table(title="Resultados da Busca de Símbolos", show_header=True, header_style="bold blue")
    tabela.add_column("#", style="dim", width=3, justify="right")
    tabela.add_column("Símbolo", style="cyan", width=15)
    tabela.add_column("Descrição", width=40)
    tabela.add_column("Tipo", width=10)

    for i, item in enumerate(resultados):
        tabela.add_row(
            str(i + 1),
            item.get('symbol', 'N/A'),
            item.get('description', 'N/A'),
            item.get('type', 'N/A')
        )
    return tabela

def criar_tabela_preferencias_visualizacao(preferencias: Dict[str, Any]):
    tabela = Table(title="Preferências de Visualização Atuais", show_header=True, header_style="bold yellow")
    tabela.add_column("Parâmetro", style="dim cyan", width=35)
    tabela.add_column("Valor Atual", width=40)

    tabela.add_row("Período Histórico Padrão", str(preferencias.get('periodo_historico_padrao', 'N/A')))
    
    indicadores_str = ", ".join(preferencias.get('indicadores_tecnicos_padrao', [])) if preferencias.get('indicadores_tecnicos_padrao') else "Nenhum"
    tabela.add_row("Indicadores Técnicos Padrão", indicadores_str)
    
    return tabela

def criar_tabela_alertas_configurados(alertas: List[Dict[str, Any]]):
    if not alertas:
        return Text("Nenhum alerta configurado.", style="yellow")

    tabela = Table(title="Alertas Configurados", show_header=True, header_style="bold yellow")
    tabela.add_column("ID", style="dim", width=5, justify="right")
    tabela.add_column("Símbolo", style="cyan", width=12)
    tabela.add_column("Tipo", width=20)
    tabela.add_column("Condição", width=30)
    tabela.add_column("Ativo", width=7, justify="center")
    tabela.add_column("Mensagem", width=30)

    for alerta in alertas:
        condicao_str_list = []
        if isinstance(alerta.get('condicao'), dict):
            for k, v in alerta.get('condicao', {}).items():
                condicao_str_list.append(f"{k.replace('_',' ').capitalize()}: {v}")
        condicao_display = "; ".join(condicao_str_list) if condicao_str_list else "N/A"
        
        tabela.add_row(
            str(alerta.get('id_alerta', 'N/A')),
            alerta.get('simbolo', 'N/A'),
            alerta.get('tipo_alerta', 'N/A').replace('_',' ').capitalize(),
            condicao_display,
            "Sim" if alerta.get('ativo', False) else "Não",
            alerta.get('mensagem_customizada', '')
        )
    return tabela