import sys
import os
import subprocess
import time

APP_VERSION = "0.9.0" 
WINDOW_TITLE = f"LexiconCLI v{APP_VERSION}"
RELAUNCH_ENV_VAR = "LEXICONCLI_IN_NEW_WINDOW"


def set_windows_console_title(title):
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception as e:
            print(f"Aviso: Não foi possível definir o título do console via ctypes: {e}", file=sys.stderr)

def main_application_logic():
    project_root_for_logic = os.path.dirname(os.path.abspath(__file__))
    if project_root_for_logic not in sys.path:
        sys.path.insert(0, project_root_for_logic)

    from controllers.app_controller import AppController
    from models.database import inicializar_db
    from utils.logger import configurar_logger

    logger = configurar_logger()
    logger.info("Lógica principal da aplicação - LexiconCLI iniciada.")
    
    try:
        inicializar_db()
        logger.info("Banco de dados verificado/inicializado.")
    except Exception as e:
        logger.error(f"Falha ao inicializar o banco de dados: {e}", exc_info=True)
        print(f"Erro crítico ao inicializar o banco de dados: {e}. Verifique os logs.", file=sys.stderr)

    app = AppController()
    app.iniciar()


if __name__ == "__main__":
    if os.name == 'nt': 
        if os.getenv(RELAUNCH_ENV_VAR) != '1':
            try:
                current_env = os.environ.copy()
                current_env[RELAUNCH_ENV_VAR] = '1'
                
                python_executable = sys.executable
                script_path = os.path.abspath(__file__)
                
                working_directory = os.getcwd()
                
                command_to_run_in_new_window = f'"{python_executable}" "{script_path}"'
                full_command = f'start "{WINDOW_TITLE}" /D "{working_directory}" cmd /c "{command_to_run_in_new_window} & pause"'
                
                subprocess.Popen(full_command, shell=True, env=current_env, creationflags=subprocess.CREATE_NO_WINDOW) # CREATE_NO_WINDOW para o cmd intermediário
                
                time.sleep(0.2) 
                sys.exit(0)
            except Exception as e:
                print(f"Erro ao tentar re-lançar em nova janela: {e}", file=sys.stderr)
                set_windows_console_title(WINDOW_TITLE)
                main_application_logic()
        else:
            set_windows_console_title(WINDOW_TITLE)
            main_application_logic()
    else:
        print("Rodando em modo de console padrão (não-Windows ou fallback).", file=sys.stderr)
        if os.name == 'posix':
            sys.stdout.write(f"\x1b]2;{WINDOW_TITLE}\x07")
            sys.stdout.flush()
        main_application_logic()