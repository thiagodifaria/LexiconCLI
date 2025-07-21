import plotext as plt
import pandas as pd
import numpy as np
from rich.text import Text
from datetime import datetime
from utils.logger import logger

def plotar_historico_preco_volume(df_historico: pd.DataFrame, simbolo: str):
    if df_historico is None or df_historico.empty or 'Close' not in df_historico.columns:
        logger.warning(f"Dados históricos insuficientes ou ausentes para {simbolo} para plotar gráfico de histórico.")
        return Text(f"Dados históricos insuficientes para {simbolo} para plotar gráfico.", style="bold red")

    df_plot = df_historico.copy()
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        try:
            df_plot.index = pd.to_datetime(df_plot.index)
        except Exception as e:
            logger.error(f"Não foi possível converter índice para DatetimeIndex para plotagem de histórico de {simbolo}: {e}")
            return Text(f"Formato de data inválido para plotagem de histórico de {simbolo}.", style="bold red")

    datas_str = df_plot.index.strftime('%d/%m').tolist()
    
    num_ticks = min(len(datas_str), 10) 
    tick_step = max(1, len(datas_str) // num_ticks if num_ticks > 0 else 1)
    
    x_ticks_positions = list(range(len(datas_str)))
    selected_x_ticks_positions = x_ticks_positions[::tick_step]
    selected_datas_str = datas_str[::tick_step]


    precos = df_plot['Close'].tolist()
    volumes = df_plot['Volume'].fillna(0).tolist() if 'Volume' in df_plot.columns else [] 

    plt.clear_figure()
    plt.limit_size(True, True) 
    
    num_subplots = 1
    if volumes and any(v > 0 for v in volumes): 
        num_subplots = 2

    if num_subplots == 2:
        plt.subplots(2, 1) 
        plt.subplot(1, 1) 
    else:
        plt.subplots(1, 1) 
        plt.subplot(1, 1)
    
    plt.plot(x_ticks_positions, precos, marker="braille", color="blue") 
    plt.xticks(selected_x_ticks_positions, selected_datas_str)
    plt.title(f"Histórico de Preços (Fechamento) - {simbolo}")
    plt.ylabel("Preço")
    plt.grid(True, False)
    plt.theme('pro')

    if num_subplots == 2:
        plt.subplot(2, 1) 
        plt.bar(x_ticks_positions, volumes, color="green", width=0.7)
        plt.xticks(selected_x_ticks_positions, selected_datas_str)
        plt.title(f"Volume - {simbolo}") 
        plt.ylabel("Volume")
        plt.grid(False, True) 
        plt.theme('pro')

    return plt.build()

def plotar_simulacao_monte_carlo(matriz_simulacoes: np.ndarray, simbolo: str):
    """
    Passo 3a: Criação da visualização gráfica para simulação Monte Carlo.
    
    Decisões técnicas:
    1. Limitei a 50 trajetórias para evitar sobrecarregar o gráfico e manter legibilidade
    2. Usei plotext para manter consistência com o resto da aplicação
    3. Implementei diferentes cores para destacar quartis de resultados
    4. Adicionei configurações de tema e grid para melhor visualização
    """
    if matriz_simulacoes is None or matriz_simulacoes.size == 0:
        logger.warning(f"Matriz de simulações vazia ou None para {simbolo}")
        return Text(f"Dados de simulação insuficientes para {simbolo}.", style="bold red")
    
    try:
        plt.clear_figure()
        plt.limit_size(True, True)
        
        dias_simulacao = matriz_simulacoes.shape[0] - 1
        num_simulacoes_total = matriz_simulacoes.shape[1]
        
        max_simulacoes_plot = min(50, num_simulacoes_total)
        step_simulacoes = max(1, num_simulacoes_total // max_simulacoes_plot)
        indices_simulacoes = range(0, num_simulacoes_total, step_simulacoes)[:max_simulacoes_plot]
        
        x_axis = list(range(dias_simulacao + 1))
        
        cores_disponiveis = ["blue", "red", "green", "yellow", "magenta", "cyan"]
        for i, idx_sim in enumerate(indices_simulacoes):
            trajetoria = matriz_simulacoes[:, idx_sim]
            cor = cores_disponiveis[i % len(cores_disponiveis)]
            
            if i < 10:
                plt.plot(x_axis, trajetoria.tolist(), marker="braille", color=cor)
            else:
                plt.plot(x_axis, trajetoria.tolist(), color=cor)
        
        precos_finais = matriz_simulacoes[-1, :]
        idx_mediana = np.argsort(precos_finais)[len(precos_finais)//2]
        trajetoria_mediana = matriz_simulacoes[:, idx_mediana]
        plt.plot(x_axis, trajetoria_mediana.tolist(), marker="braille", color="white", label="Mediana")
        
        if dias_simulacao <= 30:
            tick_step = 5
        elif dias_simulacao <= 90:
            tick_step = 15
        else:
            tick_step = 30
            
        x_ticks_pos = list(range(0, dias_simulacao + 1, tick_step))
        x_ticks_labels = [f"Dia {x}" for x in x_ticks_pos]
        
        plt.xticks(x_ticks_pos, x_ticks_labels)
        plt.title(f"Simulação Monte Carlo - {simbolo} ({max_simulacoes_plot}/{num_simulacoes_total} trajetórias)")
        plt.xlabel("Dias de Simulação")
        plt.ylabel("Preço Simulado")
        plt.grid(True, True)
        plt.theme('pro')
        
        logger.info(f"Gráfico Monte Carlo gerado para {simbolo}: {max_simulacoes_plot} trajetórias de {num_simulacoes_total}")
        return plt.build()
        
    except Exception as e:
        logger.error(f"Erro ao plotar simulação Monte Carlo para {simbolo}: {str(e)}", exc_info=True)
        return Text(f"Erro ao gerar gráfico de simulação para {simbolo}: {str(e)}", style="bold red")

def plotar_previsoes_lstm(df_comparacao: pd.DataFrame, simbolo: str):
    if df_comparacao is None or df_comparacao.empty or 'Data' not in df_comparacao.columns:
        return Text(f"Dados de previsão LSTM insuficientes para {simbolo}.", style="bold red")

    try:
        if not isinstance(df_comparacao['Data'].iloc[0], (datetime, pd.Timestamp)):
            df_comparacao['Data'] = pd.to_datetime(df_comparacao['Data'])
    except Exception:
        logger.error(f"Coluna 'Data' em previsões LSTM não é conversível para datetime para {simbolo}")
        return Text(f"Formato de data inválido nas previsões LSTM para {simbolo}.", style="bold red")

    datas_str = df_comparacao['Data'].dt.strftime('%d/%m/%y').tolist()
    
    num_ticks = min(len(datas_str), 10)
    tick_step = max(1, len(datas_str) // num_ticks if num_ticks > 0 else 1)

    x_ticks_positions = list(range(len(datas_str)))
    selected_x_ticks_positions = x_ticks_positions[::tick_step]
    selected_datas_str = datas_str[::tick_step]

    reais = df_comparacao['Real'].tolist()
    previstos = df_comparacao['Previsto'].tolist()

    plt.clear_figure()
    plt.limit_size(True, True)
    
    plt.plot(x_ticks_positions, reais, label="Preço Real", marker="braille", color="blue")
    plt.plot(x_ticks_positions, previstos, label="Preço Previsto (LSTM)", marker="braille", color="red")
    
    plt.xticks(selected_x_ticks_positions, selected_datas_str)
    
    plt.title(f"Comparação Previsão LSTM vs Real - {simbolo}")
    plt.ylabel("Preço")
    plt.grid(True, True)
    plt.theme('pro')

    return plt.build()

def plotar_previsoes_prophet(df_historico_prophet_formatado: pd.DataFrame, df_previsao_prophet: pd.DataFrame, simbolo: str):
    if df_previsao_prophet is None or df_previsao_prophet.empty or 'ds' not in df_previsao_prophet.columns or 'Previsto' not in df_previsao_prophet.columns:
        return Text(f"Dados de previsão Prophet insuficientes para {simbolo}.", style="bold red")

    plt.clear_figure()
    plt.limit_size(True, True)
    
    df_plot_hist = pd.DataFrame()
    if df_historico_prophet_formatado is not None and not df_historico_prophet_formatado.empty and 'ds' in df_historico_prophet_formatado.columns and 'y' in df_historico_prophet_formatado.columns:
        df_plot_hist = df_historico_prophet_formatado.copy()
        df_plot_hist['ds'] = pd.to_datetime(df_plot_hist['ds'])
        df_plot_hist.sort_values(by='ds', inplace=True)
        
        datas_hist_str = df_plot_hist['ds'].dt.strftime('%d/%m/%y').tolist()
        x_hist_ticks_pos = list(range(len(datas_hist_str)))
        plt.plot(x_hist_ticks_pos, df_plot_hist['y'].tolist(), label="Histórico Real", color="gray", marker=".")


    df_plot_fcst = df_previsao_prophet.copy()
    df_plot_fcst['ds'] = pd.to_datetime(df_plot_fcst['ds'])
    df_plot_fcst.sort_values(by='ds', inplace=True)

    offset_previsao = 0
    if not df_plot_hist.empty:
        if df_plot_fcst['ds'].min() >= df_plot_hist['ds'].min():            
            
            todas_datas = pd.concat([df_plot_hist['ds'], df_plot_fcst['ds']]).drop_duplicates().sort_values()
            map_data_para_x = {data: i for i, data in enumerate(todas_datas)}

            x_hist_plot = [map_data_para_x[d] for d in df_plot_hist['ds']]
            x_fcst_plot = [map_data_para_x[d] for d in df_plot_fcst['ds']]
            
            if not df_plot_hist.empty:
                plt.clear_data() 
                plt.plot(x_hist_plot, df_plot_hist['y'].tolist(), label="Histórico Real", color="gray", marker="braille")

        else: 
            x_fcst_plot = list(range(len(df_plot_fcst)))
    else: 
        x_fcst_plot = list(range(len(df_plot_fcst)))
        todas_datas = df_plot_fcst['ds'].drop_duplicates().sort_values() 
        map_data_para_x = {data: i for i, data in enumerate(todas_datas)}


    plt.plot(x_fcst_plot, df_plot_fcst['Previsto'].tolist(), label="Previsto (Prophet)", color="red", marker="braille")

    if 'yhat_lower' in df_plot_fcst.columns and 'yhat_upper' in df_plot_fcst.columns:
        plt.plot(x_fcst_plot, df_plot_fcst['yhat_lower'].tolist(), label="Prev. Mín.", color="lightred", marker="braille")
        plt.plot(x_fcst_plot, df_plot_fcst['yhat_upper'].tolist(), label="Prev. Máx.", color="lightred", marker="braille")
        
    datas_str_combinadas = todas_datas.dt.strftime('%d/%m/%y').tolist()
    x_ticks_positions_combinados = [map_data_para_x[d] for d in todas_datas]

    num_ticks = min(len(datas_str_combinadas), 15)
    tick_step = max(1, len(datas_str_combinadas) // num_ticks if num_ticks > 0 else 1)
    
    selected_x_ticks_positions = x_ticks_positions_combinados[::tick_step]
    selected_datas_str = datas_str_combinadas[::tick_step]
    
    plt.xticks(selected_x_ticks_positions, selected_datas_str)
    
    plt.title(f"Previsão com Prophet - {simbolo}")
    plt.ylabel("Preço")
    plt.grid(True, True)
    plt.theme('pro')

    return plt.build()