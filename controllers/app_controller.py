from views.terminal.main_view import MainView
from views.terminal.menu import MenuPrincipal
from controllers.data_controller import DataController
from controllers.analysis_controller import AnalysisController 
from controllers.prediction_controller import PredictionController 
from config.settings import LSTM_LOOKBACK_PERIOD, LSTM_EPOCHS, LSTM_BATCH_SIZE, ALERT_CHECK_INTERVAL_SECONDS
from utils.logger import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time # Para o loop de verificação de alertas
import threading # Para o loop de verificação de alertas

class AppController:
    def __init__(self):
        self.console = Console()
        self.data_controller = DataController()
        self.analysis_controller = AnalysisController() 
               
        lstm_config = {
            'lookback_period': LSTM_LOOKBACK_PERIOD,
            'epochs': LSTM_EPOCHS,
            'batch_size': LSTM_BATCH_SIZE,
            'lstm_units_1': 50,
            'lstm_units_2': 50,
            'dense_units': 25,
            'activation_dense': 'relu',
            'optimizer': 'adam',
            'loss_function': 'mean_squared_error',
            'prophet_configs': {}
        }
        self.prediction_controller = PredictionController(model_config=lstm_config)

        self.view = MainView(self.data_controller, self.analysis_controller, self.prediction_controller)
        self.menu = MenuPrincipal(self.console)
        self._rodando_verificador_alertas = False
        self._thread_verificador_alertas = None
        self._ultimos_alertas_disparados = [] 


    def _iniciar_verificador_alertas(self):
        if not self._rodando_verificador_alertas:
            self._rodando_verificador_alertas = True
            self._thread_verificador_alertas = threading.Thread(target=self._loop_verificador_alertas, daemon=True)
            self._thread_verificador_alertas.start()
            logger.info("Thread de verificação de alertas iniciada.")

    def _parar_verificador_alertas(self):
        self._rodando_verificador_alertas = False
        if self._thread_verificador_alertas and self._thread_verificador_alertas.is_alive():
            logger.info("Aguardando thread de verificação de alertas finalizar...")
            self._thread_verificador_alertas.join(timeout=5) # Espera um pouco para a thread terminar
            if self._thread_verificador_alertas.is_alive():
                logger.warning("Thread de verificação de alertas não finalizou a tempo.")
        logger.info("Thread de verificação de alertas parada.")


    def _loop_verificador_alertas(self):
        while self._rodando_verificador_alertas:
            try:
                logger.debug("Iniciando ciclo de verificação de alertas.")
                alertas_config = self.data_controller.listar_alertas_db()
                simbolos_com_alertas = list(set(alerta['simbolo'] for alerta in alertas_config if alerta.get('ativo')))

                novos_sinais_disparados_neste_ciclo = []

                for simbolo in simbolos_com_alertas:
                    alertas_do_simbolo = [a for a in alertas_config if a['simbolo'] == simbolo and a.get('ativo')]
                    if not alertas_do_simbolo:
                        continue
                    
                    dados_hist_recentes = self.data_controller.buscar_dados_historicos(simbolo, periodo="260d", intervalo="1d").dataframe
                    if dados_hist_recentes.empty or len(dados_hist_recentes) < 60:
                        logger.warning(f"Dados insuficientes para verificar alertas de {simbolo}")
                        continue
                    
                    df_com_indicadores = self.analysis_controller.calcular_todos_indicadores_principais(dados_hist_recentes)
                    if df_com_indicadores.empty:
                        logger.warning(f"Não foi possível calcular indicadores para verificar alertas de {simbolo}")
                        continue
                    
                    sinais = self.analysis_controller.verificar_sinais_para_alertas(simbolo, df_com_indicadores, alertas_do_simbolo)
                    novos_sinais_disparados_neste_ciclo.extend(sinais)
                
                # Lógica para exibir apenas novos alertas ou atualizar a view de alguma forma
                # Esta é uma implementação simples, pode precisar de refinamento para integração com a MainView
                alertas_realmente_novos = [s for s in novos_sinais_disparados_neste_ciclo if s not in self._ultimos_alertas_disparados]
                if alertas_realmente_novos:
                    self._ultimos_alertas_disparados.extend(alertas_realmente_novos) # Adiciona para não repetir imediatamente
                    # Idealmente, passaria isso para a MainView atualizar o painel de mensagens
                    for alerta_msg in alertas_realmente_novos:
                        self.console.print(f"[bold blink red]ALERTA DISPARADO:[/bold blink red] {alerta_msg}", style="on dark_red")
                        logger.info(f"ALERTA DISPARADO E EXIBIDO: {alerta_msg}")


                if len(self._ultimos_alertas_disparados) > 50 : # Limpa buffer antigo
                    self._ultimos_alertas_disparados = self._ultimos_alertas_disparados[-25:]


            except Exception as e:
                logger.error(f"Erro no loop de verificação de alertas: {e}", exc_info=True)
            
            for _ in range(ALERT_CHECK_INTERVAL_SECONDS):
                if not self._rodando_verificador_alertas:
                    break
                time.sleep(1)


    def iniciar(self):
        logger.info("Sistema de Análise Financeira - LexiconCLI iniciado.")
        self.console.clear()
        self._iniciar_verificador_alertas()
        
        rodando = True
        while rodando:
            try:
                self.console.clear()
                escolha_menu = self.menu.exibir_menu_principal()

                if escolha_menu == 1: 
                    logger.info("Usuário selecionou Dashboard Principal.")                    
                    self.view.exibir_dashboard_inicial() 
                
                elif escolha_menu == 2: 
                    logger.info("Usuário selecionou Analisar Ativo Específico.")
                    simbolo = self.menu.solicitar_simbolo_ativo(self.data_controller)
                    if simbolo:
                        self.console.clear()
                        escolha_analise = self.menu.exibir_menu_analise_ativo(simbolo)
                        if escolha_analise == 1: 
                             logger.info(f"Usuário selecionou Gráfico Histórico para {simbolo}.")
                             self.view.exibir_grafico_historico_ativo(simbolo)
                        elif escolha_analise == 2:
                             logger.info(f"Usuário selecionou Indicadores Técnicos para {simbolo}.")
                             self.view.exibir_indicadores_tecnicos_ativo(simbolo)
                        
                elif escolha_menu == 3:
                    logger.info("Usuário selecionou Ver Indicadores Macroeconômicos Detalhados.")
                    self.view.exibir_indicadores_macro_detalhados()

                elif escolha_menu == 4:
                    logger.info("Usuário selecionou Realizar Previsão para Ativo.")
                    simbolo = self.menu.solicitar_simbolo_ativo(self.data_controller)
                    if simbolo:
                        self.console.clear()
                        escolha_modelo_previsao = self.menu.exibir_menu_escolha_modelo_previsao(simbolo)
                        if escolha_modelo_previsao == 1: 
                            logger.info(f"Usuário selecionou Previsão com LSTM para {simbolo}.")
                            self.view.exibir_previsao_lstm_ativo(simbolo)
                        elif escolha_modelo_previsao == 2: 
                            logger.info(f"Usuário selecionou Previsão com Prophet para {simbolo}.")
                            self.view.exibir_previsao_prophet_ativo(simbolo)
                
                elif escolha_menu == 5:
                    logger.info("Usuário selecionou Watchlist, Configurações e Ferramentas.")
                    self.console.clear()
                    escolha_config = self.menu.exibir_menu_watchlist_config_ferramentas()
                    if escolha_config == 1: 
                        self.view.exibir_gerenciamento_watchlist(self.menu)
                    elif escolha_config == 2: 
                        self.view.exibir_configuracao_parametros_visualizacao(self.menu)
                    elif escolha_config == 3: 
                        self.view.exibir_configuracao_alertas(self.menu)
                    elif escolha_config == 4: 
                        self.view.exibir_exportacao_relatorio(self.menu)

                elif escolha_menu == 0:
                    logger.info("Usuário selecionou Sair. Encerrando aplicação.")
                    rodando = False
                
                else:
                    self.console.print("[bold red]Opção inválida. Tente novamente.[/bold red]")
                    self.console.input("Pressione Enter para continuar...")

            except KeyboardInterrupt:
                logger.warning("Aplicação interrompida por Ctrl+C no menu principal. Encerrando.")
                rodando = False
            except Exception as e:
                logger.exception(f"Erro inesperado no AppController: {e}")
                self.console.print(f"[bold red]Ocorreu um erro inesperado: {e}[/bold red]")
                self.console.input("Pressione Enter para continuar ou Ctrl+C para sair...")
        
        self._parar_verificador_alertas()
        self.console.print(Panel(Text("LexiconCLI encerrado. Até logo!", justify="center", style="bold yellow on blue")))