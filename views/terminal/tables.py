from rich.table import Table
from rich.text import Text
from utils.formatters import formatar_valor_monetario, formatar_percentual, formatar_data_ptbr
from utils.logger import configurar_logger
from models.data_model import CotacaoAtivo, IndicadorMacroeconomico, SerieEconomica 
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
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

    ultimos_registros = indicadores_df.tail(5).iloc[::-1]
    
    for index, row in ultimos_registros.iterrows():
        valores_linha = [index.strftime('%d/%m/%Y') if isinstance(index, pd.Timestamp) else str(index)]
        for coluna in colunas_validas:
            valor = row[coluna]
            if pd.isna(valor): valores_linha.append("N/A")
            elif isinstance(valor, float): valores_linha.append(f"{valor:.2f}")
            else: valores_linha.append(str(valor))
        tabela.add_row(*valores_linha)
    return tabela

def criar_tabela_estatisticas_monte_carlo(metadados: dict): 
    if not metadados or not metadados.get('sucesso', False):
        return Text("Erro: Metadados da simulação não disponíveis ou simulação falhou.", style="bold red")
    
    estatisticas = metadados.get('estatisticas', {})
    
    tabela = Table(title=f"Estatísticas da Simulação Monte Carlo - {metadados.get('simbolo', 'N/A')}", 
                   show_header=True, header_style="bold magenta")
    tabela.add_column("Métrica", style="dim cyan", width=35)
    tabela.add_column("Valor", justify="right", width=25)
    
    tabela.add_row("Símbolo", metadados.get('simbolo', 'N/A'))
    tabela.add_row("Número de Simulações", f"{metadados.get('num_simulacoes', 'N/A'):,}")
    tabela.add_row("Dias Simulados", str(metadados.get('dias_simulacao', 'N/A')))
    tabela.add_row("Período Histórico Usado", metadados.get('periodo_historico', 'N/A'))
    tabela.add_row("Pontos de Dados Históricos", f"{metadados.get('pontos_historicos_utilizados', 'N/A'):,}")
    
    tabela.add_row("", "")
    tabela.add_row("[bold]RESULTADOS DA SIMULAÇÃO[/bold]", "")
    
    preco_inicial = estatisticas.get('preco_inicial', 0)
    preco_final_medio = estatisticas.get('preco_final_medio', 0)
    retorno_esperado = estatisticas.get('retorno_esperado_pct', 0)
    
    tabela.add_row("Preço Inicial", formatar_valor_monetario(preco_inicial, "$"))
    
    cor_retorno = "green" if retorno_esperado >= 0 else "red"
    preco_final_str = Text(formatar_valor_monetario(preco_final_medio, "$"), style=cor_retorno)
    retorno_str = Text(formatar_percentual(retorno_esperado), style=cor_retorno)
    
    tabela.add_row("Preço Final Médio", preco_final_str)
    tabela.add_row("Retorno Esperado", retorno_str)
    
    preco_final_mediano = estatisticas.get('preco_final_mediano', 0)
    tabela.add_row("Preço Final Mediano", formatar_valor_monetario(preco_final_mediano, "$"))
    
    tabela.add_row("", "")
    tabela.add_row("[bold]INTERVALO DE CONFIANÇA (90%)[/bold]", "")
    
    percentil_5 = estatisticas.get('percentil_5', 0)
    percentil_95 = estatisticas.get('percentil_95', 0)
    
    tabela.add_row("Pior Cenário (P5)", formatar_valor_monetario(percentil_5, "$"))
    tabela.add_row("Melhor Cenário (P95)", formatar_valor_monetario(percentil_95, "$"))
    
    prob_lucro = ((preco_final_medio > preco_inicial) and retorno_esperado > 0)
    prob_lucro_str = "Alta" if prob_lucro else "Baixa"
    cor_prob = "green" if prob_lucro else "red"
    
    tabela.add_row("", "")
    tabela.add_row("Probabilidade de Lucro", Text(prob_lucro_str, style=cor_prob))
    
    tabela.add_row("", "")
    tabela.add_row("[bold]MÉTRICAS DE RISCO[/bold]", "")
    
    volatilidade = estatisticas.get('volatilidade_simulacao', 0)
    tabela.add_row("Volatilidade da Simulação", f"{volatilidade:.3f}")
    
    if volatilidade < 0.1:
        classificacao_risco = Text("Baixo", style="green")
    elif volatilidade < 0.3:
        classificacao_risco = Text("Moderado", style="yellow")
    else:
        classificacao_risco = Text("Alto", style="red")
    
    tabela.add_row("Classificação de Risco", classificacao_risco)
    
    preco_min = estatisticas.get('preco_final_min', 0)
    preco_max = estatisticas.get('preco_final_max', 0)
    tabela.add_row("Preço Mínimo Simulado", formatar_valor_monetario(preco_min, "$"))
    tabela.add_row("Preço Máximo Simulado", formatar_valor_monetario(preco_max, "$"))
    
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

def criar_tabela_relatorio_financeiro(titulo: str, dados_relatorio: Optional[Dict]) -> Table | Text:
          
    if not dados_relatorio:
        return Text(f"Dados para '{titulo}' não fornecidos.", style="yellow")
    
    if isinstance(dados_relatorio, str):
        try:
            dados_relatorio = json.loads(dados_relatorio)
        except json.JSONDecodeError:
            return Text(f"Dados para '{titulo}' em formato inválido (JSON malformado).", style="red")
    
    if not isinstance(dados_relatorio, dict):
        return Text(f"Dados para '{titulo}' em formato inesperado: {type(dados_relatorio).__name__}", style="red")
    
    if 'data' not in dados_relatorio:
        return Text(f"Estrutura de dados inválida para '{titulo}' - campo 'data' não encontrado.", style="yellow")
    
    report_data = dados_relatorio['data']
    
    if isinstance(report_data, str):
        try:
            report_data = json.loads(report_data)
        except json.JSONDecodeError:
            return Text(f"Campo 'data' em formato JSON inválido para '{titulo}'.", style="red")
    
    if not isinstance(report_data, list):
        return Text(f"Formato de dados inesperado para '{titulo}' - 'data' deve ser lista.", style="yellow")
    
    if not report_data:
        return Text(f"Nenhum dado financeiro disponível para '{titulo}'.", style="yellow")

    tabela = Table(title=titulo, show_header=True, header_style="bold magenta", min_width=100)

    try:
        periodos_recentes = report_data[:3] if len(report_data) >= 3 else report_data
    except TypeError:
        return Text(f"Erro ao processar dados para '{titulo}' - estrutura inválida.", style="red")
    
    if not periodos_recentes:
        return Text(f"Nenhum período de relatório encontrado para '{titulo}'.", style="yellow")

    periodos_validos = []
    for periodo in periodos_recentes:
        if isinstance(periodo, dict) and 'endDate' in periodo:
            periodos_validos.append(periodo)
    
    if not periodos_validos:
        return Text(f"Nenhum período válido encontrado para '{titulo}'.", style="yellow")

    try:
        datas_periodos = []
        for periodo in periodos_validos:
            end_date = periodo['endDate']
            if isinstance(end_date, str):
                data_formatada = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            else:
                data_formatada = str(end_date)
            datas_periodos.append(data_formatada)
    except Exception:
        return Text(f"Erro ao processar datas para '{titulo}'.", style="red")
    
    tabela.add_column("Conceito Contábil", style="dim cyan", no_wrap=False, width=45)
    for data in datas_periodos:
        tabela.add_column(data, justify="right")

    primeiro_periodo = periodos_validos[0]
    if 'report' not in primeiro_periodo or not primeiro_periodo['report']:
        return Text(f"Relatório para o período mais recente de '{titulo}' está vazio.", style="yellow")
    
    todos_conceitos = set()
    for periodo in periodos_validos:
        report_items = periodo.get('report', [])
        if isinstance(report_items, list):
            for item in report_items:
                if isinstance(item, dict) and 'concept' in item:
                    todos_conceitos.add(item['concept'])
    
    if not todos_conceitos:
        return Text(f"Nenhum conceito contábil encontrado para '{titulo}'.", style="yellow")
    
    conceitos_unicos = sorted(list(todos_conceitos))

    for conceito in conceitos_unicos:
        valores_linha = [conceito]
        
        for periodo in periodos_validos:
            valor_item = None
            report_items = periodo.get('report', [])
            
            if isinstance(report_items, list):
                for item in report_items:
                    if isinstance(item, dict) and item.get('concept') == conceito:
                        valor_item = item
                        break
            
            if valor_item:
                valor = valor_item.get('value')
                if valor is None or valor == 'None':
                    valor_formatado = "N/A"
                else:
                    try:
                        if isinstance(valor, (int, float)):
                            valor_formatado = f"{valor:,.0f}"
                        else:
                            valor_num = float(str(valor).replace(',', ''))
                            valor_formatado = f"{valor_num:,.0f}"
                    except (ValueError, TypeError):
                        valor_formatado = str(valor)
            else:
                valor_formatado = "N/A"
            
            valores_linha.append(valor_formatado)
        
        tabela.add_row(*valores_linha)

    return tabela

def criar_tabela_backtest(stats: Optional[pd.Series]) -> Table | Text:
    """Cria uma tabela Rich para exibir as estatísticas de um resultado de backtest."""
    if stats is None or stats.empty:
        return Text("Não foi possível gerar as estatísticas do backtest.", style="yellow")
    
    tabela = Table(title="Resultados do Backtest (Estratégia: SmaCross)", show_header=True, header_style="bold magenta")
    tabela.add_column("Métrica", style="cyan", width=30)
    tabela.add_column("Valor", justify="right", style="green")

    format_map = {
        "Return [%]": "{:.2f}%",
        "Buy & Hold Return [%]": "{:.2f}%",
        "Max. Drawdown [%]": "{:.2f}%",
        "Win Rate [%]": "{:.2f}%",
        "Profit Factor": "{:.2f}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Avg. Trade [%]": "{:.2f}%"
    }

    for index, value in stats.items():
        if index.startswith("_"):
            continue
        
        valor_formatado = str(value)
        if isinstance(value, float):
            valor_formatado = format_map.get(str(index), "{:,.2f}").format(value)
            
        tabela.add_row(str(index), valor_formatado)
        
    return tabela

def criar_tabela_previsoes_lstm_classificacao(df_previsoes):
    """
    Cria tabela específica para exibir resultados de classificação LSTM.
    """
    tabela = Table(title="Previsões LSTM - Classificação de Direção", show_header=True, header_style="bold yellow")
    tabela.add_column("Data", style="dim", justify="center")
    tabela.add_column("Classe Real", justify="center")
    tabela.add_column("Classe Prevista", justify="center")
    tabela.add_column("Prob. BAIXA", justify="right")
    tabela.add_column("Prob. NEUTRO", justify="right")
    tabela.add_column("Prob. ALTA", justify="right")
    tabela.add_column("Acerto", justify="center")
    
    mapeamento_classes = {0: "BAIXA", 1: "NEUTRO", 2: "ALTA"}
    
    for _, row in df_previsoes.iterrows():
        data_str = row['Data'].strftime('%d/%m/%Y') if isinstance(row['Data'], (datetime, pd.Timestamp)) else str(row['Data'])
        
        classe_real_str = mapeamento_classes.get(row['Classe_Real'], str(row['Classe_Real']))
        classe_prevista_str = mapeamento_classes.get(row['Classe_Prevista'], str(row['Classe_Prevista']))
        
        acertou = row['Classe_Real'] == row['Classe_Prevista']
        cor_acerto = "green" if acertou else "red"
        acerto_str = "SIM" if acertou else "NAO"
        
        prob_baixa = f"{row['Prob_Baixa']:.3f}"
        prob_neutro = f"{row['Prob_Neutro']:.3f}"
        prob_alta = f"{row['Prob_Alta']:.3f}"
        
        prob_max = max(row['Prob_Baixa'], row['Prob_Neutro'], row['Prob_Alta'])
        cor_confianca = "green" if prob_max > 0.6 else "yellow" if prob_max > 0.4 else "red"
        
        tabela.add_row(
            data_str,
            classe_real_str,
            Text(classe_prevista_str, style=cor_confianca),
            prob_baixa,
            prob_neutro,
            prob_alta,
            Text(acerto_str, style=cor_acerto)
        )
    return tabela

def criar_tabela_metricas_classificacao(metricas):
    """
    Cria tabela para exibir métricas de classificação de forma organizada.
    """
    tabela = Table(title="Métricas de Classificação LSTM", show_header=True, header_style="bold magenta")
    tabela.add_column("Métrica", style="dim cyan", width=25)
    tabela.add_column("Valor", justify="right", width=15)
    tabela.add_column("Detalhes por Classe", width=40)
    
    tabela.add_row("Acurácia Geral", f"{metricas.get('accuracy', 0):.4f}", "")
    tabela.add_row("Acurácia Balanceada", f"{metricas.get('balanced_accuracy', 0):.4f}", "")
    tabela.add_row("F1-Score (Weighted)", f"{metricas.get('f1_score_weighted', 0):.4f}", "")
    tabela.add_row("Precisão (Weighted)", f"{metricas.get('precision_weighted', 0):.4f}", "")
    tabela.add_row("Recall (Weighted)", f"{metricas.get('recall_weighted', 0):.4f}", "")