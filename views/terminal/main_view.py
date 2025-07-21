from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.markup import escape
import time
import pandas as pd
from datetime import datetime
import os

from views.terminal.tables import (
    criar_tabela_ativos_monitorados, 
    criar_tabela_indices_mercado,
    criar_tabela_indicadores_macro,
    criar_tabela_indicadores_tecnicos,
    criar_tabela_previsoes_lstm,
    criar_tabela_serie_economica,
    criar_tabela_previsoes_prophet,
    criar_tabela_watchlist,
    criar_tabela_resultados_busca_simbolos,
    criar_tabela_alertas_configurados,
    criar_tabela_preferencias_visualizacao,
    criar_tabela_relatorio_financeiro,
    criar_tabela_backtest,
    criar_tabela_estatisticas_monte_carlo,
    criar_tabela_previsoes_lstm_classificacao,
    criar_tabela_metricas_classificacao
)
from views.terminal.charts import (
    plotar_historico_preco_volume, 
    plotar_previsoes_lstm,
    plotar_previsoes_prophet,
    plotar_simulacao_monte_carlo
)
from config.settings import DEFAULT_ASSETS_MONITOR, DEFAULT_INDICES_MONITOR, DEFAULT_HISTORICAL_PERIOD, DEFAULT_INDICATORS_VIEW, DEFAULT_EXPORT_PATH
from utils.logger import logger
from utils.formatters import formatar_dataframe_para_csv, formatar_dados_para_txt_simples
from models.data_model import PreferenciasVisualizacao 

class MainView:
    def __init__(self, data_controller, analysis_controller=None, prediction_controller=None, backtesting_controller=None):
        self.console = Console()
        self.data_controller = data_controller
        self.analysis_controller = analysis_controller
        self.prediction_controller = prediction_controller
        self.backtesting_controller = backtesting_controller
        self.layout = self._criar_layout_base()
        self.user_assets_monitor = list(DEFAULT_ASSETS_MONITOR)
        self.user_indices_monitor = list(DEFAULT_INDICES_MONITOR)
        self.preferencias_visualizacao = self._carregar_preferencias_visualizacao()
        self.carregar_watchlist_inicial()

    def _carregar_preferencias_visualizacao(self) -> PreferenciasVisualizacao:
        prefs_db = self.data_controller.carregar_preferencias_visualizacao_db()
        if prefs_db:
            logger.info(f"Preferências de visualização carregadas do DB: {prefs_db}")
            return PreferenciasVisualizacao(**prefs_db)
        logger.info("Nenhuma preferência de visualização no DB, usando padrões.")
        return PreferenciasVisualizacao(
            id_usuario=0,
            periodo_historico_padrao=DEFAULT_HISTORICAL_PERIOD,
            indicadores_tecnicos_padrao=list(DEFAULT_INDICATORS_VIEW)
        )

    def carregar_watchlist_inicial(self):
        try:
            watchlist_db = self.data_controller.get_watchlist()
            if watchlist_db:
                self.user_assets_monitor = [item['simbolo'] for item in watchlist_db if item['tipo'] == 'asset']
                self.user_indices_monitor = [item['simbolo'] for item in watchlist_db if item['tipo'] == 'index']
                if not self.user_assets_monitor and not DEFAULT_ASSETS_MONITOR: self.user_assets_monitor = []
                elif not self.user_assets_monitor : self.user_assets_monitor = list(DEFAULT_ASSETS_MONITOR)

                if not self.user_indices_monitor and not DEFAULT_INDICES_MONITOR: self.user_indices_monitor = []
                elif not self.user_indices_monitor : self.user_indices_monitor = list(DEFAULT_INDICES_MONITOR)

            else: 
                self.user_assets_monitor = list(DEFAULT_ASSETS_MONITOR)
                self.user_indices_monitor = list(DEFAULT_INDICES_MONITOR)
            logger.info(f"Watchlist carregada: Ativos {self.user_assets_monitor}, Índices {self.user_indices_monitor}")
        except Exception as e:
            logger.error(f"Erro ao carregar watchlist inicial do banco de dados: {e}. Usando padrões.")
            self.user_assets_monitor = list(DEFAULT_ASSETS_MONITOR)
            self.user_indices_monitor = list(DEFAULT_INDICES_MONITOR)


    def _criar_layout_base(self):
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main_content"),
            Layout(name="footer", size=3)
        )
        layout["main_content"].split_row(
            Layout(name="painel_esq", ratio=1), 
            Layout(name="painel_dir", ratio=2)  
        )
        layout["painel_esq"].split_column(
            Layout(name="indices_mercado", ratio=1, minimum_size=5),
            Layout(name="indicadores_macro_bcb", ratio=1, minimum_size=5) 
        )
        layout["painel_dir"].split_column(
            Layout(name="ativos_monitorados", ratio=2, minimum_size=10),
            Layout(name="dados_economicos_adicionais", ratio=1, minimum_size=5), 
            Layout(name="mensagens_feedback", size=3, minimum_size=3)
        )
        return layout

    def _atualizar_header_footer(self):
        self.layout["header"].update(Panel(Text("LexiconCLI - Terminal de Análise Financeira", justify="center", style="bold white on blue")))
        self.layout["footer"].update(Panel(Text(f"Atualizado em: {time.strftime('%H:%M:%S %d/%m/%Y')} - Pressione Ctrl+C para Menu", justify="center", style="dim")))

    def exibir_simulacao_monte_carlo(self, simbolo: str, simulation_controller):
        """
        Passo 4: Orquestração completa do fluxo da simulação Monte Carlo.
        
        Decisões técnicas:
        1. Seguir o padrão estabelecido: título → inputs → status → execução → resultados
        2. Valores padrão sugeridos baseados em práticas da indústria (30 dias, 1000 simulações)
        3. Validação robusta dos inputs do usuário
        4. Feedback visual contínuo durante execução
        5. Tratamento de erros com mensagens claras
        6. Exibição dos resultados em múltiplos formatos (tabela + gráfico)
        """
        simbolo_safe = escape(simbolo)
        
        try:
            self.console.clear()
            self.console.print(Panel(
                Text(f"Simulação de Monte Carlo para [bold cyan]{simbolo}[/bold cyan]", 
                     justify="center", style="bold yellow"), 
                title="Análise de Cenários Futuros"
            ))
            
            self.console.print("")
            self.console.print(Text(
                "A Simulação de Monte Carlo utiliza o modelo de Movimento Browniano Geométrico "
                "para gerar milhares de cenários possíveis de evolução do preço do ativo.",
                style="dim"
            ))
            self.console.print("")
            
            dias_simulacao = IntPrompt.ask(
                "Quantos dias deseja simular para o futuro?", 
                default=30,
                show_default=True
            )
            
            if dias_simulacao <= 0 or dias_simulacao > 365:
                self.console.print("[red]Número de dias deve estar entre 1 e 365. Usando valor padrão de 30 dias.[/red]")
                dias_simulacao = 30
            
            num_simulacoes = IntPrompt.ask(
                "Quantas simulações executar? (mais simulações = resultados mais precisos)", 
                default=1000,
                show_default=True
            )
            
            if num_simulacoes <= 0 or num_simulacoes > 10000:
                self.console.print("[red]Número de simulações deve estar entre 1 e 10.000. Usando valor padrão de 1000.[/red]")
                num_simulacoes = 1000
            elif num_simulacoes < 100:
                self.console.print("[yellow]Aviso: Número baixo de simulações pode afetar a precisão dos resultados.[/yellow]")
            
            self.console.print(f"\n[green]Configuração da simulação:[/green]")
            self.console.print(f"  • Ativo: [bold]{simbolo}[/bold]")
            self.console.print(f"  • Dias a simular: [bold]{dias_simulacao}[/bold]")
            self.console.print(f"  • Número de simulações: [bold]{num_simulacoes:,}[/bold]")
            
            if not Confirm.ask("\nDeseja prosseguir com a simulação?", default=True):
                self.console.print("[yellow]Simulação cancelada pelo usuário.[/yellow]")
                self.console.input("\nPressione Enter para continuar...")
                return
            
            with self.console.status(
                f"[bold green]Executando simulação Monte Carlo para {simbolo}...\n"
                f"Processando {num_simulacoes:,} simulações de {dias_simulacao} dias...", 
                spinner="earth"
            ) as status:
                
                status.update("[bold green]Obtendo dados históricos...")
                time.sleep(0.5)
                
                status.update("[bold green]Calculando parâmetros estatísticos...")
                time.sleep(0.5)
                
                status.update(f"[bold green]Gerando {num_simulacoes:,} trajetórias de preços...")
                
                matriz_simulacoes, preco_inicial, metadados_simulacao = simulation_controller.executar_simulacao_monte_carlo(
                    simbolo=simbolo,
                    periodo_historico="2y",
                    dias_simulacao=dias_simulacao,
                    num_simulacoes=num_simulacoes
                )
            
            if not metadados_simulacao.get('sucesso', False):
                erro_msg = metadados_simulacao.get('erro', 'Erro desconhecido na simulação')
                self.console.print(Panel(
                    Text(f"Falha na simulação: {erro_msg}", style="bold red"),
                    title="Erro na Simulação Monte Carlo"
                ))
                self.console.input("\nPressione Enter para continuar...")
                return
            
            self.console.clear()
            self.console.print(Panel(
                Text(f"Simulação Monte Carlo Concluída - {simbolo}", 
                     justify="center", style="bold green"),
                subtitle=f"{num_simulacoes:,} simulações • {dias_simulacao} dias"
            ))
            
            tabela_estatisticas = criar_tabela_estatisticas_monte_carlo(metadados_simulacao)
            self.console.print(tabela_estatisticas)
            
            self.console.print("\n")
            
            self.console.print("[bold]Trajetórias de Preços Simuladas:[/bold]\n")
            
            grafico_simulacao = plotar_simulacao_monte_carlo(matriz_simulacoes, simbolo)
            if isinstance(grafico_simulacao, str):
                self.console.print(Text.from_ansi(grafico_simulacao))
            elif isinstance(grafico_simulacao, Text):
                self.console.print(grafico_simulacao)
            else:
                self.console.print(str(grafico_simulacao))
            
            self.console.print(f"\n[dim]• Gráfico mostra até 50 trajetórias representativas de {num_simulacoes:,} simulações totais[/dim]")
            self.console.print("[dim]• Linha branca destaca a trajetória mediana[/dim]")
            self.console.print("[dim]• Cada trajetória representa um cenário possível de evolução do preço[/dim]")
            
            if Confirm.ask("\nDeseja gerar um relatório detalhado da simulação?", default=False):
                relatorio = simulation_controller.gerar_relatorio_simulacao(matriz_simulacoes, metadados_simulacao)
                if 'erro' not in relatorio:
                    self.console.print("\n[green]Relatório gerado com sucesso![/green]")
                else:
                    self.console.print(f"[red]Erro ao gerar relatório: {relatorio['erro']}[/red]")
            
            logger.info(f"Simulação Monte Carlo concluída para {simbolo}: "
                       f"{num_simulacoes} simulações, {dias_simulacao} dias, "
                       f"sucesso: {metadados_simulacao.get('sucesso', False)}")
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Simulação interrompida pelo usuário.[/yellow]")
            logger.info(f"Simulação Monte Carlo para {simbolo} interrompida pelo usuário")
        except Exception as e:
            erro_msg = f"Erro inesperado durante simulação Monte Carlo: {str(e)}"
            self.console.print(Panel(Text(erro_msg, style="bold red"), title="Erro Inesperado"))
            logger.error(f"Erro inesperado na simulação Monte Carlo para {simbolo}: {str(e)}", exc_info=True)
        
        self.console.input("\nPressione Enter para continuar...")

    def exibir_dashboard_inicial(self):
        with Live(self.layout, console=self.console, refresh_per_second=0.5, screen=True, transient=True) as live:
            try:
                while True:
                    self.carregar_watchlist_inicial() 
                    self._atualizar_header_footer()
                    
                    ativos_data_list = [self.data_controller.buscar_cotacao_ativo(simbolo) for simbolo in self.user_assets_monitor]
                    indices_data_list = [self.data_controller.buscar_cotacao_ativo(simbolo) for simbolo in self.user_indices_monitor]
                    indicadores_macro_bcb_dict = self.data_controller.buscar_indicadores_macro_bcb()
                    
                    serie_desemprego_eua = self.data_controller.buscar_serie_fred("UNRATE", "Taxa Desemprego EUA")
                    
                    self.layout["indices_mercado"].update(Panel(criar_tabela_indices_mercado(indices_data_list), title="Mercados Globais"))
                    self.layout["indicadores_macro_bcb"].update(Panel(criar_tabela_indicadores_macro(indicadores_macro_bcb_dict), title="Economia Brasil (BCB)"))
                    self.layout["ativos_monitorados"].update(Panel(criar_tabela_ativos_monitorados(ativos_data_list), title="Minha Carteira"))
                    
                    painel_dados_econ_add_content = []
                    if serie_desemprego_eua and not serie_desemprego_eua.dataframe.empty:
                        painel_dados_econ_add_content.append(criar_tabela_serie_economica(serie_desemprego_eua, ultimos_n=5))
                    
                    if painel_dados_econ_add_content:
                         self.layout["dados_economicos_adicionais"].update(Panel(*painel_dados_econ_add_content, title="Outros Dados Econômicos"))
                    else:
                         self.layout["dados_economicos_adicionais"].update(Panel(Text("Nenhum dado adicional disponível.", justify="center"), title="Outros Dados Econômicos"))

                    self.layout["mensagens_feedback"].update(Panel(Text("Bem-vindo! Use o menu para mais opções.", style="italic green"), title="Status"))
                    
                    live.update(self.layout)
                    time.sleep(120) 
            except KeyboardInterrupt:
                logger.info("Dashboard interrompido pelo usuário.")
                live.stop()
                return

    def exibir_grafico_historico_ativo(self, simbolo: str):
        periodo_grafico = self.preferencias_visualizacao.periodo_historico_padrao
        with self.console.status(f"[bold green]Buscando dados históricos ({periodo_grafico}) para {simbolo}...", spinner="dots"):
            dados_hist = self.data_controller.buscar_dados_historicos(simbolo, periodo=periodo_grafico, intervalo="1d")
        if dados_hist.dataframe.empty:
            self.console.print(Panel(Text(f"Não foi possível obter dados históricos para {simbolo}.", style="bold red"), title="Erro")); return
        
        self.console.clear()
        self.console.print(Panel(Text(f"Gráfico de Preço e Volume - {simbolo} ({periodo_grafico})", justify="center", style="bold blue")))
        
        grafico_output_str = plotar_historico_preco_volume(dados_hist.dataframe, simbolo)

        if isinstance(grafico_output_str, str):
            self.console.print(Text.from_ansi(grafico_output_str))
        elif isinstance(grafico_output_str, Text): 
            self.console.print(grafico_output_str)
        else: 
            self.console.print(str(grafico_output_str))
            
        self.console.input("\nPressione Enter para continuar...")

    def exibir_indicadores_tecnicos_ativo(self, simbolo: str):
        if not self.analysis_controller:
            self.console.print(Panel(Text("Módulo de Análise Técnica não inicializado.", style="bold red"), title="Erro")); return
        
        periodo_calculo = self.preferencias_visualizacao.periodo_historico_padrao 
        num_dias_aprox = 250
        if "y" in periodo_calculo: num_dias_aprox = int(periodo_calculo.replace("y","")) * 252
        elif "mo" in periodo_calculo: num_dias_aprox = int(periodo_calculo.replace("mo","")) * 21
        elif "d" in periodo_calculo: num_dias_aprox = int(periodo_calculo.replace("d",""))

        with self.console.status(f"[bold green]Calculando indicadores técnicos para {simbolo} (dados de {periodo_calculo})...", spinner="dots"):
            dados_hist = self.data_controller.buscar_dados_historicos(simbolo, periodo=f"{num_dias_aprox + 60}d", intervalo="1d")
            if dados_hist.dataframe.empty or len(dados_hist.dataframe) < 60: 
                self.console.print(Panel(Text(f"Dados históricos insuficientes para calcular indicadores para {simbolo}.", style="bold red"), title="Erro")); return
            
            df_indicadores_todos = self.analysis_controller.calcular_todos_indicadores_principais(dados_hist.dataframe)
            
            indicadores_para_exibir = self.preferencias_visualizacao.indicadores_tecnicos_padrao
            if not df_indicadores_todos.empty and indicadores_para_exibir:
                cols_para_manter = [col for col in indicadores_para_exibir if col in df_indicadores_todos.columns]
                if not all(col in ['Open', 'High', 'Low', 'Close', 'Volume'] for col in cols_para_manter):
                    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    cols_para_manter = base_cols + [c for c in cols_para_manter if c not in base_cols]

                df_indicadores_filtrados = df_indicadores_todos[cols_para_manter]
            else:
                df_indicadores_filtrados = df_indicadores_todos


        if df_indicadores_filtrados.empty:
            self.console.print(Panel(Text(f"Não foi possível calcular ou filtrar indicadores técnicos para {simbolo}.", style="bold red"), title="Erro")); return
        
        self.console.clear()
        self.console.print(Panel(criar_tabela_indicadores_tecnicos(df_indicadores_filtrados), title=f"Indicadores Técnicos - {simbolo}"))
        self.console.input("\nPressione Enter para continuar...")

    def exibir_previsao_lstm_ativo(self, simbolo: str):
        if not self.prediction_controller or not self.analysis_controller:
            self.console.print(Panel(Text("Módulo de Previsão ou Análise não inicializado.", style="bold red"), title="Erro")); return

        modelo_carregado = self.prediction_controller.carregar_modelo_lstm(simbolo)
        usar_modelo_salvo = False

        if modelo_carregado:
            self.console.print(f"[green]✓ Modelo LSTM de classificação treinado para [bold]{simbolo}[/bold] encontrado![/green]")
            usar_modelo_salvo = Confirm.ask("Deseja usar este modelo para fazer uma nova previsão?", default=True)

        if usar_modelo_salvo:
            self.console.print("[cyan]Usando modelo de classificação salvo para gerar previsão...[/cyan]")
            with self.console.status(f"[bold green]Buscando dados recentes para {simbolo}...", spinner="dots"):
                dados_recentes = self.data_controller.buscar_dados_historicos(simbolo, periodo="1y", intervalo="1d")
                if dados_recentes.dataframe.empty or len(dados_recentes.dataframe) < self.prediction_controller.model_config.get('lookback_period', 60):
                    self.console.print(Panel(Text(f"Dados recentes insuficientes para fazer previsão para {simbolo}.", style="bold red"), title="Erro")); return
                
                df_com_indicadores = self.analysis_controller.calcular_todos_indicadores_principais(dados_recentes.dataframe.copy())

            predicao_resultado = self.prediction_controller.prever_proximos_passos_lstm(df_com_indicadores)
            
            self.console.clear()
            self.console.print(Panel(Text(f"Previsão LSTM - Classificação de Direção - {simbolo}", justify="center", style="bold yellow")))
            
            if predicao_resultado is not None:
                cotacao_atual = dados_recentes.dataframe['close'].iloc[-1]
                classe_predita = predicao_resultado['classe_predita']
                probabilidades = predicao_resultado['probabilidades']
                
                self.console.print(f"\nÚltimo preço de fechamento: [bold cyan]${cotacao_atual:.2f}[/bold cyan]")
                self.console.print(f"\n[bold]Previsão de Direção para o Próximo Dia:[/bold]")
                
                if classe_predita == "ALTA":
                    cor_classe = "bold green"
                elif classe_predita == "BAIXA":
                    cor_classe = "bold red" 
                else:
                    cor_classe = "bold yellow"
                
                self.console.print(f"Direção Prevista: [{cor_classe}]{classe_predita}[/{cor_classe}]")
                
                self.console.print(f"\n[bold]Probabilidades por Classe:[/bold]")
                self.console.print(f"  BAIXA:  {probabilidades['BAIXA']:.1%}")
                self.console.print(f"  NEUTRO: {probabilidades['NEUTRO']:.1%}")
                self.console.print(f"  ALTA:   {probabilidades['ALTA']:.1%}")
                
                prob_maxima = max(probabilidades.values())
                if prob_maxima > 0.6:
                    confianca = "[bold green]Alta Confiança[/bold green]"
                elif prob_maxima > 0.4:
                    confianca = "[bold yellow]Confiança Moderada[/bold yellow]"
                else:
                    confianca = "[bold red]Baixa Confiança[/bold red]"
                
                self.console.print(f"\nNível de Confiança: {confianca} ({prob_maxima:.1%})")
                
                self.console.print(f"\n[dim]Interpretação: O modelo prevê movimento {classe_predita} com {prob_maxima:.1%} de confiança.[/dim]")
                if prob_maxima < 0.5:
                    self.console.print("[dim]Atenção: Confiança baixa pode indicar movimento lateral ou incerteza do mercado.[/dim]")
                    
            else:
                self.console.print("[red]Não foi possível gerar uma previsão com o modelo salvo.[/red]")

        else:
            if modelo_carregado:
                self.console.print("[yellow]Ok, vamos treinar um novo modelo de classificação do zero.[/yellow]\n")
            
            with self.console.status(f"[bold green]Preparando dados e treinando modelo LSTM de classificação para {simbolo}...", spinner="earth") as status:
                status.update(f"[bold green]Buscando dados históricos (3 anos) para {simbolo}...")
                dados_hist_completos = self.data_controller.buscar_dados_historicos(simbolo, periodo="3y", intervalo="1d")
                
                if dados_hist_completos.dataframe.empty or len(dados_hist_completos.dataframe) < (self.prediction_controller.model_config.get('lookback_period', 60) * 2):
                    self.console.print(Panel(Text(f"Dados históricos insuficientes para treinar modelo LSTM para {simbolo}.", style="bold red"), title="Erro")); return
                
                status.update(f"[bold green]Calculando features (indicadores técnicos) para {simbolo}...")
                df_com_indicadores = self.analysis_controller.calcular_todos_indicadores_principais(dados_hist_completos.dataframe.copy())
                
                if df_com_indicadores.empty:
                    self.console.print(Panel(Text(f"Falha ao calcular features para LSTM para {simbolo}.", style="bold red"), title="Erro")); return

                status.update(f"[bold green]Treinando modelo LSTM de CLASSIFICAÇÃO para {simbolo}... Isso pode levar alguns minutos.")
                modelo, _, df_comparacao, metricas = self.prediction_controller.treinar_avaliar_modelo_lstm_bayesian(
                    df_com_indicadores, 
                    coluna_target='close', 
                    simbolo=simbolo
                )
            
            self.console.clear()
            if modelo and not df_comparacao.empty:
                self.console.print(Panel(Text(f"Resultados LSTM - Classificação de Direção - {simbolo}", justify="center", style="bold yellow")))
                
                self.console.print("\n[bold]Métricas de Avaliação (Classificação):[/bold]")
                tabela_metricas = criar_tabela_metricas_classificacao(metricas)
                self.console.print(tabela_metricas)

                self.console.print("\n[bold]Últimas Previsões vs Real (Classificação):[/bold]"); 
                tabela_previsoes = criar_tabela_previsoes_lstm_classificacao(df_comparacao.tail(15))
                self.console.print(tabela_previsoes)
                
                accuracy = metricas.get('accuracy', 0)
                balanced_accuracy = metricas.get('balanced_accuracy', 0)
                
                self.console.print(f"\n[bold]Resumo de Performance:[/bold]")
                if accuracy > 0.6:
                    performance_cor = "green"
                    performance_status = "EXCELENTE"
                elif accuracy > 0.45:
                    performance_cor = "yellow" 
                    performance_status = "MODERADA"
                else:
                    performance_cor = "red"
                    performance_status = "BAIXA"
                
                self.console.print(f"Performance: [{performance_cor}]{performance_status}[/{performance_cor}] - Acurácia Geral: [{performance_cor}]{accuracy:.1%}[/{performance_cor}]")
                self.console.print(f"Acurácia Balanceada: {balanced_accuracy:.1%}")
                
                if accuracy > 0.5:
                    self.console.print(f"\n[green]Modelo apresenta performance superior ao acaso (>50%)[/green]")
                else:
                    self.console.print(f"\n[red]Modelo com performance próxima ao acaso - considere mais dados ou features[/red]")
                    
            else:
                self.console.print(Panel(Text(f"Falha ao treinar ou avaliar o modelo LSTM de classificação para {simbolo}.", style="bold red"), title="Erro na Previsão LSTM"))
        
        self.console.input("\nPressione Enter para continuar...")

    def exibir_previsao_prophet_ativo(self, simbolo: str):
        if not self.prediction_controller:
            self.console.print(Panel(Text("Módulo de Previsão não inicializado.", style="bold red"), title="Erro")); return

        modelo_carregado = self.prediction_controller.carregar_modelo_prophet(simbolo)
        usar_modelo_salvo = False

        if modelo_carregado:
            self.console.print(f"[green]✓ Modelo Prophet treinado para [bold]{simbolo}[/bold] encontrado![/green]")
            usar_modelo_salvo = Confirm.ask("Deseja usar este modelo para fazer uma nova previsão?", default=True)

        if usar_modelo_salvo:
            self.console.print("[cyan]Usando modelo Prophet salvo para gerar previsão...[/cyan]")
            with self.console.status(f"[bold green]Gerando previsões futuras para {simbolo}...", spinner="dots"):
                periodos_previsao = 30
                forecast_df = self.prediction_controller.modelo_prophet_instancia.prever_futuro(
                    periodos=periodos_previsao,
                    frequencia='B'
                )
            
            self.console.clear()
            if not forecast_df.empty:
                self.console.print(Panel(Text(f"Previsões Futuras com Modelo Prophet Salvo - {simbolo}", justify="center", style="bold magenta")))
                self.console.print(criar_tabela_previsoes_prophet(forecast_df.tail(periodos_previsao)))
                
                self.console.print("\n[yellow]Gráfico de previsão com modelo salvo ainda não implementado.[/yellow]")
            else:
                self.console.print(Panel(Text(f"Falha ao gerar previsões com o modelo Prophet salvo para {simbolo}.", style="bold red"), title="Erro na Previsão Prophet"))

        else:
            if modelo_carregado:
                self.console.print("[yellow]Ok, vamos treinar um novo modelo Prophet do zero.[/yellow]\n")

            with self.console.status(f"[bold green]Preparando dados e treinando modelo Prophet para {simbolo}...", spinner="dots") as status:
                status.update(f"[bold green]Buscando dados históricos (3 anos) para {simbolo}...")
                dados_hist_completos = self.data_controller.buscar_dados_historicos(simbolo, periodo="3y", intervalo="1d")

                if dados_hist_completos.dataframe.empty:
                    self.console.print(Panel(Text(f"Dados históricos insuficientes para treinar modelo Prophet para {simbolo}.", style="bold red"), title="Erro")); return
                
                df_para_prophet = dados_hist_completos.dataframe.copy()
                
                status.update(f"[bold green]Treinando modelo Prophet para {simbolo}... Isso pode ser rápido.")
                modelo_prophet, df_previsao_prophet, df_historico_usado_prophet = self.prediction_controller.treinar_avaliar_modelo_prophet(
                    df_dados_completos=df_para_prophet.reset_index(), 
                    simbolo=simbolo,
                    coluna_data_prophet='date',
                    coluna_target_prophet='close',
                    periodos_previsao=30 
                )
            
            self.console.clear()
            if modelo_prophet and not df_previsao_prophet.empty:
                self.console.print(Panel(Text(f"Resultados da Previsão Prophet - {simbolo}", justify="center", style="bold magenta")))
                
                self.console.print("\nPrevisões Futuras (Prophet):"); 
                self.console.print(criar_tabela_previsoes_prophet(df_previsao_prophet.tail(30))) 

                grafico_prophet_str = plotar_previsoes_prophet(df_historico_usado_prophet, df_previsao_prophet, simbolo)
                if isinstance(grafico_prophet_str, str): self.console.print(Text.from_ansi(grafico_prophet_str))
                else: self.console.print(str(grafico_prophet_str))
            else:
                self.console.print(Panel(Text(f"Falha ao treinar ou gerar previsões com o modelo Prophet para {simbolo}.", style="bold red"), title="Erro na Previsão Prophet"))

        self.console.input("\nPressione Enter para continuar...")
        
    def exibir_analise_fundamentalista(self, simbolo: str):
        """Orquestra a exibição dos relatórios de análise fundamentalista."""
        self.console.clear()
        self.console.print(Panel(Text(f"Análise Fundamentalista para [bold cyan]{simbolo}[/bold cyan]", justify="center"), expand=False))

        with self.console.status("[bold green]Buscando dados fundamentalistas...", spinner="dots"):
            dados_bp = self.data_controller.buscar_balanco_patrimonial(simbolo)
            tabela_bp = criar_tabela_relatorio_financeiro(f"Balanço Patrimonial Anual ({simbolo})", dados_bp)
            
            dados_dre = self.data_controller.buscar_demonstrativo_resultados(simbolo)
            tabela_dre = criar_tabela_relatorio_financeiro(f"Demonstração de Resultados Anual ({simbolo})", dados_dre)

            dados_fc = self.data_controller.buscar_fluxo_caixa(simbolo)
            tabela_fc = criar_tabela_relatorio_financeiro(f"Fluxo de Caixa Anual ({simbolo})", dados_fc)

        self.console.print(tabela_bp)
        self.console.print(tabela_dre)
        self.console.print(tabela_fc)
        
        self.console.input("\n[bold]Pressione Enter para voltar ao menu anterior[/bold]")
        
    def exibir_backtest_ativo(self, simbolo: str):
        """Orquestra a execução e exibição do resultado de um backtest."""
        self.console.clear()
        self.console.print(Panel(Text(f"Backtesting para [bold cyan]{simbolo}[/bold cyan]", justify="center"), expand=False))
        
        if not self.backtesting_controller:
            self.console.print(Panel(Text("Módulo de Backtesting não inicializado.", style="bold red"), title="Erro"))
            self.console.input("\n[bold]Pressione Enter para voltar ao menu anterior[/bold]")
            return

        with self.console.status("[bold green]Executando simulação... (Isso pode levar um momento)", spinner="earth"):
            stats, caminho_plot = self.backtesting_controller.executar_backtest_sma_cross(simbolo)

        tabela_resultados = criar_tabela_backtest(stats)
        self.console.print(tabela_resultados)
        
        self.console.input("\n[bold]Pressione Enter para voltar ao menu anterior[/bold]")

    def exibir_indicadores_macro_detalhados(self):
        self.console.clear()
        self.console.print(Panel(Text("Indicadores Macroeconômicos Detalhados", justify="center", style="bold green")))
        
        with self.console.status("[bold green]Buscando dados do BCB (SGS e Focus)...", spinner="dots"):
            indicadores_bcb = self.data_controller.buscar_indicadores_macro_bcb()
            focus_ipca = self.data_controller.buscar_expectativas_mercado_focus("IPCA", ultimos_n_anos=2)
            focus_pib = self.data_controller.buscar_expectativas_mercado_focus("PIB Total", ultimos_n_anos=2)

        self.console.print(Panel(criar_tabela_indicadores_macro(indicadores_bcb), title="Indicadores Macroeconômicos (BCB)"))
        if not focus_ipca.empty:
            t_focus_ipca = Table(title="Focus - Expectativas IPCA (Anual)", show_header=True, header_style="bold magenta")
            for col in focus_ipca.columns: t_focus_ipca.add_column(str(col))
            for _, row in focus_ipca.iterrows(): t_focus_ipca.add_row(*[str(x) for x in row.values])
            self.console.print(t_focus_ipca)
        if not focus_pib.empty:
            t_focus_pib = Table(title="Focus - Expectativas PIB Total (Anual)", show_header=True, header_style="bold magenta")
            for col in focus_pib.columns: t_focus_pib.add_column(str(col))
            for _, row in focus_pib.iterrows(): t_focus_pib.add_row(*[str(x) for x in row.values])
            self.console.print(t_focus_pib)

        with self.console.status("[bold green]Buscando dados do FRED (Taxa de Desemprego EUA)...", spinner="dots"):
            serie_desemprego_eua = self.data_controller.buscar_serie_fred("UNRATE", "Taxa Desemprego EUA")
        if serie_desemprego_eua and not serie_desemprego_eua.dataframe.empty: 
            self.console.print(criar_tabela_serie_economica(serie_desemprego_eua, ultimos_n=12))
        else: 
            self.console.print(Text("Não foi possível buscar Taxa de Desemprego EUA do FRED.", style="yellow"))

        with self.console.status("[bold green]Buscando dados do Nasdaq Data Link (Petróleo Brent)...", spinner="dots"):
            serie_petroleo_brent = self.data_controller.buscar_dataset_nasdaq("FRED/DCOILBRENTEU", "Petróleo Brent (Diário)")
        if serie_petroleo_brent and not serie_petroleo_brent.dataframe.empty: 
            self.console.print(criar_tabela_serie_economica(serie_petroleo_brent, ultimos_n=12))
        else: 
            self.console.print(Text("Não foi possível buscar Petróleo Brent do Nasdaq Data Link.", style="yellow"))
        
        self.console.input("\nPressione Enter para continuar...")

    def exibir_gerenciamento_watchlist(self, menu_instance):
        rodando_watchlist = True
        while rodando_watchlist:
            self.console.clear()
            escolha_watchlist = menu_instance.exibir_submenu_gerenciar_watchlist()
            if escolha_watchlist == 1:
                watchlist_items = self.data_controller.get_watchlist()
                self.console.clear()
                if watchlist_items:
                    self.console.print(Panel(criar_tabela_watchlist(watchlist_items), title="Minha Watchlist"))
                else:
                    self.console.print(Panel(Text("Sua watchlist está vazia.", justify="center"), title="Minha Watchlist"))
                self.console.input("\nPressione Enter para continuar...")
            elif escolha_watchlist == 2: 
                simbolo = menu_instance.solicitar_simbolo_ativo(self.data_controller)
                if simbolo:
                    tipo_ativo = menu_instance.solicitar_tipo_ativo_watchlist()
                    if tipo_ativo:
                        if self.data_controller.add_to_watchlist(simbolo, tipo_ativo):
                            self.console.print(f"[green]'{simbolo}' adicionado à watchlist como {tipo_ativo}.[/green]")
                            self.carregar_watchlist_inicial() 
                        else:
                            self.console.print(f"[red]Erro ao adicionar '{simbolo}' (pode já existir).[/red]")
                    else:
                        self.console.print("[yellow]Tipo de ativo inválido. Operação cancelada.[/yellow]")
                else:
                    self.console.print("[yellow]Símbolo inválido. Operação cancelada.[/yellow]")
                self.console.input("\nPressione Enter para continuar...")
            elif escolha_watchlist == 3: 
                simbolo_para_remover = menu_instance.solicitar_simbolo_ativo() 
                if simbolo_para_remover:
                    if self.data_controller.remove_from_watchlist(simbolo_para_remover):
                        self.console.print(f"[green]'{simbolo_para_remover}' removido da watchlist.[/green]")
                        self.carregar_watchlist_inicial() 
                    else:
                        self.console.print(f"[red]Erro ao remover '{simbolo_para_remover}' (pode não existir).[/red]")
                else:
                    self.console.print("[yellow]Símbolo inválido. Operação cancelada.[/yellow]")
                self.console.input("\nPressione Enter para continuar...")
            elif escolha_watchlist == 0: 
                rodando_watchlist = False
            else:
                self.console.print("[red]Opção inválida.[/red]")
                self.console.input("\nPressione Enter para continuar...")


    def exibir_configuracao_parametros_visualizacao(self, menu_instance):
        self.console.clear()
        self.console.print(Panel(Text("Configuração de Parâmetros de Visualização", justify="center", style="bold yellow")))
        
        prefs_atuais_dict = {
            "periodo_historico_padrao": self.preferencias_visualizacao.periodo_historico_padrao,
            "indicadores_tecnicos_padrao": list(self.preferencias_visualizacao.indicadores_tecnicos_padrao)
        }
        self.console.print(criar_tabela_preferencias_visualizacao(prefs_atuais_dict))

        if Confirm.ask("\nDeseja alterar esses parâmetros?", default=True):
            novas_prefs_dict = menu_instance.solicitar_parametros_visualizacao(prefs_atuais_dict)
            if novas_prefs_dict:
                if self.data_controller.salvar_preferencias_visualizacao_db(novas_prefs_dict):
                    self.console.print("[green]Preferências de visualização salvas com sucesso![/green]")
                    self.preferencias_visualizacao = self._carregar_preferencias_visualizacao() 
                else:
                    self.console.print("[red]Erro ao salvar preferências de visualização.[/red]")
        self.console.input("\nPressione Enter para continuar...")

    def exibir_configuracao_alertas(self, menu_instance):
        self.console.clear()
        rodando_alertas = True
        while rodando_alertas:
            self.console.clear()
            escolha_alertas = menu_instance.exibir_submenu_configurar_alertas()

            if escolha_alertas == 1: 
                simbolo_ativo_selecionado = None
                if Confirm.ask("Deseja buscar o símbolo do ativo para o alerta?", default=True):
                    simbolo_ativo_selecionado = menu_instance.solicitar_simbolo_ativo(self.data_controller)
                
                novo_alerta_data = menu_instance.solicitar_detalhes_novo_alerta(simbolo_padrao=simbolo_ativo_selecionado)
                if novo_alerta_data:
                    if self.data_controller.adicionar_alerta_db(novo_alerta_data): 
                        self.console.print(f"[green]Alerta para '{novo_alerta_data['simbolo']}' ({novo_alerta_data['tipo_alerta']}) criado com sucesso![/green]")
                    else:
                        self.console.print("[red]Erro ao criar o alerta.[/red]")
                self.console.input("Pressione Enter para continuar...")

            elif escolha_alertas == 2: 
                alertas = self.data_controller.listar_alertas_db() 
                self.console.clear()
                if alertas:
                    self.console.print(Panel(criar_tabela_alertas_configurados(alertas), title="Alertas Configurados"))
                else:
                    self.console.print(Panel(Text("Nenhum alerta configurado.", justify="center"), title="Alertas Configurados"))
                self.console.input("Pressione Enter para continuar...")
            
            elif escolha_alertas == 3: 
                 self.console.print("\nFuncionalidade 'Modificar Alerta' ainda não implementada.")
                 self.console.print("Aqui você poderá selecionar um alerta existente e alterar seus parâmetros (símbolo, tipo, condição, mensagem).")
                 self.console.input("Pressione Enter para continuar...")

            elif escolha_alertas == 4: 
                alertas = self.data_controller.listar_alertas_db()
                if not alertas:
                    self.console.print("[yellow]Não há alertas para remover.[/yellow]")
                else:
                    id_alerta_remover = menu_instance.solicitar_id_alerta_para_remover(alertas)
                    if id_alerta_remover is not None:
                        if self.data_controller.remover_alerta_db(id_alerta_remover): 
                            self.console.print(f"[green]Alerta ID {id_alerta_remover} removido com sucesso![/green]")
                        else:
                            self.console.print(f"[red]Erro ao remover o alerta ID {id_alerta_remover}.[/red]")
                self.console.input("Pressione Enter para continuar...")

            elif escolha_alertas == 0: 
                rodando_alertas = False
            
            else:
                self.console.print("[red]Opção inválida.[/red]")
                self.console.input("\nPressione Enter para continuar...")

    def exibir_exportacao_relatorio(self, menu_instance):
        self.console.clear()
        rodando_exportacao = True
        while rodando_exportacao:
            self.console.clear()
            self.console.print(Panel(Text("Exportação de Relatórios", justify="center", style="bold blue")))
            tipo_exportacao = menu_instance.exibir_submenu_exportar_relatorio()

            if tipo_exportacao == 0: 
                rodando_exportacao = False
                continue

            detalhes_export = menu_instance.solicitar_detalhes_exportacao(tipo_exportacao, self.data_controller)
            if not detalhes_export:
                self.console.input("Pressione Enter para continuar...")
                continue

            nome_arquivo_base = detalhes_export["nome_arquivo"]
            caminho_completo = ""
            conteudo_exportar = ""

            try:
                if not os.path.exists(DEFAULT_EXPORT_PATH):
                    os.makedirs(DEFAULT_EXPORT_PATH)
                    logger.info(f"Diretório de exportação criado: {DEFAULT_EXPORT_PATH}")

                if tipo_exportacao == 1: 
                    df_hist = self.data_controller.buscar_dados_historicos(detalhes_export["simbolo"], periodo=detalhes_export["periodo"]).dataframe
                    if not df_hist.empty:
                        conteudo_exportar = formatar_dataframe_para_csv(df_hist)
                        caminho_completo = os.path.join(DEFAULT_EXPORT_PATH, f"{nome_arquivo_base}_historico.csv")
                    else:
                        self.console.print(f"[red]Não foi possível obter dados históricos para {detalhes_export['simbolo']}.[/red]")
                
                elif tipo_exportacao == 2: 
                    df_hist_indicadores = self.data_controller.buscar_dados_historicos(detalhes_export["simbolo"], periodo=detalhes_export["periodo"]).dataframe
                    if not df_hist_indicadores.empty:
                        df_indicadores = self.analysis_controller.calcular_todos_indicadores_principais(df_hist_indicadores)
                        if not df_indicadores.empty:
                            conteudo_exportar = formatar_dataframe_para_csv(df_indicadores)
                            caminho_completo = os.path.join(DEFAULT_EXPORT_PATH, f"{nome_arquivo_base}_indicadores.csv")
                        else:
                             self.console.print(f"[red]Não foi possível calcular indicadores para {detalhes_export['simbolo']}.[/red]")
                    else:
                        self.console.print(f"[red]Não foi possível obter dados históricos para indicadores de {detalhes_export['simbolo']}.[/red]")

                elif tipo_exportacao == 3: 
                    watchlist_items = self.data_controller.get_watchlist()
                    if watchlist_items:
                        dados_formatar = {"titulo": "Minha Watchlist LexiconCLI"}
                        for i, item in enumerate(watchlist_items):
                            dados_formatar[f"Item {i+1}"] = f"Símbolo: {item['simbolo']}, Tipo: {item['tipo']}"
                        conteudo_exportar = formatar_dados_para_txt_simples("Minha Watchlist LexiconCLI", dados_formatar)
                        caminho_completo = os.path.join(DEFAULT_EXPORT_PATH, f"{nome_arquivo_base}_watchlist.txt")
                    else:
                        self.console.print("[yellow]Watchlist está vazia. Nada para exportar.[/yellow]")

                if conteudo_exportar and caminho_completo:
                    with open(caminho_completo, "w", encoding="utf-8") as f:
                        f.write(conteudo_exportar)
                    self.console.print(f"[green]Relatório exportado com sucesso para: {caminho_completo}[/green]")
                elif not caminho_completo and tipo_exportacao !=0 :
                    self.console.print("[red]Falha ao preparar dados para exportação.[/red]")

            except Exception as e:
                logger.error(f"Erro ao exportar relatório '{nome_arquivo_base}': {e}", exc_info=True)
                self.console.print(f"[red]Ocorreu um erro durante a exportação: {e}[/red]")
            
            self.console.input("Pressione Enter para continuar...")