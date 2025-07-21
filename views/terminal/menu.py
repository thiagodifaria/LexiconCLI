from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm, FloatPrompt
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict, Optional, Any

class MenuPrincipal:
    def __init__(self, console: Console):
        self.console = console

    def exibir_menu_principal(self):
        self.console.print(Panel(Text("Sistema de Análise Financeira - LexiconCLI", justify="center", style="bold green"), title="Menu Principal"))
        self.console.print("1. Dashboard Principal")
        self.console.print("2. Analisar Ativo Específico")
        self.console.print("3. Ver Indicadores Macroeconômicos Detalhados")
        self.console.print("4. Realizar Previsão para Ativo")
        self.console.print("5. Watchlist, Configurações e Ferramentas")
        self.console.print("0. Sair")
        
        escolha = IntPrompt.ask("Escolha uma opção", choices=[str(i) for i in range(6)], show_choices=False)
        return escolha

    def solicitar_simbolo_ativo(self, data_controller_para_busca = None) -> Optional[str]:
        if data_controller_para_busca and Confirm.ask("Deseja pesquisar o símbolo do ativo?", default=False):
            termo_busca = Prompt.ask("Digite parte do nome ou símbolo para pesquisar")
            if not termo_busca.strip():
                self.console.print("[yellow]Termo de busca vazio. Tente novamente.[/yellow]")
                return None
            
            with self.console.status(f"[cyan]Pesquisando por '{termo_busca}'...[/cyan]"):
                resultados = data_controller_para_busca.pesquisar_simbolos(termo_busca)
            
            if not resultados:
                self.console.print(f"[yellow]Nenhum resultado encontrado para '{termo_busca}'. Tente digitar o símbolo diretamente.[/yellow]")
                simbolo = Prompt.ask("Digite o símbolo do ativo (ex: PETR4.SA, AAPL)")
                return simbolo.upper().strip() if simbolo.strip() else None

            self.console.print("\n[bold green]Resultados da Busca:[/bold green]")
            for i, res in enumerate(resultados):
                display_text = f"{res.get('symbol', 'N/A')} - {res.get('description', 'N/A')} ({res.get('type', 'N/A')})"
                self.console.print(f"{i+1}. {display_text}")
            self.console.print("0. Digitar símbolo manualmente ou cancelar")

            escolha_num = IntPrompt.ask("Escolha um número da lista ou 0", choices=[str(i) for i in range(len(resultados) + 1)], show_choices=False)
            if escolha_num == 0:
                simbolo = Prompt.ask("Digite o símbolo do ativo (ex: PETR4.SA, AAPL)")
                return simbolo.upper().strip() if simbolo.strip() else None
            elif 1 <= escolha_num <= len(resultados):
                return resultados[escolha_num-1].get('symbol', None)
            else:
                self.console.print("[red]Seleção inválida.[/red]")
                return None
        else:
            simbolo = Prompt.ask("Digite o símbolo do ativo (ex: PETR4.SA, AAPL)")
            return simbolo.upper().strip() if simbolo.strip() else None


    def exibir_menu_analise_ativo(self, simbolo: str):
        self.console.print(Panel(Text(f"Análise do Ativo: {simbolo}", justify="center", style="bold blue"), title="Menu de Análise"))
        self.console.print("1. Ver Gráfico de Preços Históricos")
        self.console.print("2. Calcular e Ver Indicadores Técnicos")
        self.console.print("3. Executar Simulação de Monte Carlo")
        self.console.print("4. Ver Análise Fundamentalista (Relatórios Anuais)")
        self.console.print("5. Backtesting de Estratégia (SmaCross)")
        self.console.print("0. Voltar ao Menu Principal")
        
        escolha = IntPrompt.ask("Escolha uma opção para análise", choices=["0", "1", "2", "3", "4", "5"], show_choices=False)
        return escolha

    def exibir_menu_escolha_modelo_previsao(self, simbolo: str):
        self.console.print(Panel(Text(f"Escolha o Modelo de Previsão para: {simbolo}", justify="center", style="bold yellow"), title="Seleção de Modelo"))
        self.console.print("1. Modelo LSTM")
        self.console.print("2. Modelo Prophet")
        self.console.print("0. Voltar ao Menu Principal")

        escolha = IntPrompt.ask("Escolha um modelo", choices=["0", "1", "2"], show_choices=False)
        return escolha

    def exibir_menu_watchlist_config_ferramentas(self):
        self.console.print(Panel(Text("Watchlist, Configurações e Ferramentas", justify="center", style="bold cyan"), title="Configurações e Ferramentas"))
        self.console.print("1. Gerenciar Watchlist")
        self.console.print("2. Configurar Parâmetros de Visualização")
        self.console.print("3. Configurar Alertas")
        self.console.print("4. Exportar Relatório")
        self.console.print("0. Voltar ao Menu Principal")

        escolha = IntPrompt.ask("Escolha uma opção", choices=[str(i) for i in range(5)], show_choices=False)
        return escolha

    def exibir_submenu_gerenciar_watchlist(self):
        self.console.print(Panel(Text("Gerenciar Watchlist", justify="center", style="bold green"), title="Watchlist"))
        self.console.print("1. Ver Watchlist Atual")
        self.console.print("2. Adicionar Ativo/Índice à Watchlist")
        self.console.print("3. Remover Ativo/Índice da Watchlist")
        self.console.print("0. Voltar")
        escolha = IntPrompt.ask("Escolha uma opção", choices=["0", "1", "2", "3"], show_choices=False)
        return escolha

    def solicitar_tipo_ativo_watchlist(self) -> Optional[str]:
        self.console.print("Qual o tipo do item?")
        self.console.print("1. Ação (ex: AAPL, PETR4.SA)")
        self.console.print("2. Índice (ex: ^GSPC, ^BVSP)")
        escolha_tipo = IntPrompt.ask("Escolha o tipo", choices=["1", "2"], show_choices=False)
        if escolha_tipo == 1:
            return "asset"
        elif escolha_tipo == 2:
            return "index"
        return None

    def solicitar_parametros_visualizacao(self, preferencias_atuais: Dict[str, Any]) -> Dict[str, Any]:
        novas_prefs = preferencias_atuais.copy()
        self.console.print("\n[bold]Configurar Parâmetros de Visualização[/bold]")
        
        novo_periodo = Prompt.ask(f"Período histórico padrão para gráficos (atual: {preferencias_atuais.get('periodo_historico_padrao', 'N/A')}, ex: 1y, 6mo, 250d)", default=str(preferencias_atuais.get('periodo_historico_padrao', '')))
        if novo_periodo.strip():
            novas_prefs['periodo_historico_padrao'] = novo_periodo.strip()

        self.console.print(f"Indicadores técnicos padrão (atuais: {', '.join(preferencias_atuais.get('indicadores_tecnicos_padrao', []))})")
        if Confirm.ask("Deseja alterar os indicadores técnicos padrão?", default=False):
            self.console.print("Digite os novos indicadores separados por vírgula (ex: SMA_9,RSI_14,MACD_Hist). Deixe em branco para manter os atuais se não quiser alterar.")
            novos_indicadores_str = Prompt.ask("Novos indicadores")
            if novos_indicadores_str.strip():
                novas_prefs['indicadores_tecnicos_padrao'] = [ind.strip() for ind in novos_indicadores_str.split(',')]
            elif not novos_indicadores_str and 'indicadores_tecnicos_padrao' in novas_prefs: 
                 del novas_prefs['indicadores_tecnicos_padrao'] 

        return novas_prefs

    def exibir_submenu_configurar_alertas(self):
        self.console.print(Panel(Text("Configurar Alertas", justify="center", style="bold yellow"), title="Alertas"))
        self.console.print("1. Criar Novo Alerta")
        self.console.print("2. Ver Alertas Ativos")
        self.console.print("3. Modificar Alerta")
        self.console.print("4. Remover Alerta")
        self.console.print("0. Voltar")
        escolha = IntPrompt.ask("Escolha uma opção", choices=["0", "1", "2", "3", "4"], show_choices=False)
        return escolha
    
    def solicitar_detalhes_novo_alerta(self, simbolo_padrao: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self.console.print("\n[bold]Criar Novo Alerta[/bold]")
        simbolo = Prompt.ask("Símbolo do ativo para o alerta", default=simbolo_padrao if simbolo_padrao else "")
        if not simbolo.strip():
            self.console.print("[red]Símbolo não pode ser vazio.[/red]")
            return None

        self.console.print("Tipos de Alerta Disponíveis:")
        self.console.print("  1. Preço Acima De")
        self.console.print("  2. Preço Abaixo De")
        self.console.print("  3. RSI Sobrecompra (RSI > X)")
        self.console.print("  4. RSI Sobrevenda (RSI < Y)")
        
        tipo_alerta_escolha = IntPrompt.ask("Escolha o tipo de alerta", choices=["1", "2", "3", "4"], show_choices=False)
        
        condicao = {}
        tipo_alerta_str = ""
        valor_ref = 0.0
        limiar = 0

        if tipo_alerta_escolha == 1:
            tipo_alerta_str = "preco_acima"
            valor_ref = FloatPrompt.ask("Valor de referência para o preço (ex: 150.75)")
            condicao = {'valor_referencia': valor_ref}
        elif tipo_alerta_escolha == 2:
            tipo_alerta_str = "preco_abaixo"
            valor_ref = FloatPrompt.ask("Valor de referência para o preço (ex: 140.50)")
            condicao = {'valor_referencia': valor_ref}
        elif tipo_alerta_escolha == 3:
            tipo_alerta_str = "rsi_sobrecompra"
            limiar = IntPrompt.ask("Limiar de RSI para sobrecompra", default=70)
            condicao = {'limiar_rsi': limiar}
        elif tipo_alerta_escolha == 4:
            tipo_alerta_str = "rsi_sobrevenda"
            limiar = IntPrompt.ask("Limiar de RSI para sobrevenda", default=30)
            condicao = {'limiar_rsi': limiar}
        else:
            return None 

        mensagem = Prompt.ask("Mensagem customizada para o alerta (opcional)")

        return {
            "simbolo": simbolo.upper().strip(),
            "tipo_alerta": tipo_alerta_str,
            "condicao": condicao,
            "ativo": True,
            "mensagem_customizada": mensagem if mensagem.strip() else None
        }

    def solicitar_id_alerta_para_remover(self, alertas: List[Dict[str, Any]]) -> Optional[int]:
        if not alertas:
            self.console.print("[yellow]Não há alertas para remover.[/yellow]")
            return None
        self.console.print("\n[bold]Remover Alerta[/bold]")
        
        from views.terminal.tables import criar_tabela_alertas_configurados 
        self.console.print(criar_tabela_alertas_configurados(alertas))
        
        ids_validos = [str(a['id_alerta']) for a in alertas] + ["0"]
        id_alerta_str = Prompt.ask("Digite o ID do alerta a ser removido (ou 0 para cancelar)", choices=ids_validos, show_choices=False)

        id_alerta = int(id_alerta_str)
        if id_alerta == 0: return None
        return id_alerta


    def exibir_submenu_exportar_relatorio(self):
        self.console.print(Panel(Text("Exportar Relatório", justify="center", style="bold blue"), title="Exportação"))
        self.console.print("1. Exportar Histórico de Preços (CSV)")
        self.console.print("2. Exportar Indicadores Técnicos (CSV)")
        self.console.print("3. Exportar Watchlist (TXT)")
        self.console.print("0. Voltar")
        escolha = IntPrompt.ask("Escolha uma opção de exportação", choices=["0", "1", "2", "3"], show_choices=False)
        return escolha

    def solicitar_detalhes_exportacao(self, tipo_exportacao: int, data_controller_para_busca = None) -> Optional[Dict[str, Any]]:
        detalhes = {"tipo_exportacao": tipo_exportacao}
        if tipo_exportacao == 1 or tipo_exportacao == 2: 
            simbolo = self.solicitar_simbolo_ativo(data_controller_para_busca)
            if not simbolo: return None
            detalhes["simbolo"] = simbolo
            
            periodo = Prompt.ask("Digite o período para os dados (ex: 1y, 6mo, 250d)", default="1y")
            detalhes["periodo"] = periodo
            
            nome_arquivo_base = simbolo.replace(".SA","").replace("^","")
            default_filename = f"{nome_arquivo_base}_{'historico' if tipo_exportacao == 1 else 'indicadores'}"
            nome_arquivo = Prompt.ask(f"Nome do arquivo para salvar (sem extensão)", default=default_filename)
            if not nome_arquivo.strip():
                self.console.print("[red]Nome do arquivo não pode ser vazio.[/red]")
                return None
            detalhes["nome_arquivo"] = nome_arquivo.strip()

        elif tipo_exportacao == 3: 
            nome_arquivo = Prompt.ask("Nome do arquivo para salvar a watchlist (sem extensão)", default="minha_watchlist")
            if not nome_arquivo.strip():
                self.console.print("[red]Nome do arquivo não pode ser vazio.[/red]")
                return None
            detalhes["nome_arquivo"] = nome_arquivo.strip()
        
        return detalhes