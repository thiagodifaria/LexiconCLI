import sqlite3
from config.settings import DB_NAME
import os
from utils.logger import logger
from typing import List, Dict, Any, Optional
import json
from models.data_model import AlertaConfigurado


_test_db_connection = None

def set_test_db_connection(conn: Optional[sqlite3.Connection]):
    global _test_db_connection
    _test_db_connection = conn

def criar_diretorio_se_nao_existir(caminho_arquivo):
    diretorio = os.path.dirname(caminho_arquivo)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio)
        logger.info(f"Diretório {diretorio} criado.")

def conectar_db() -> sqlite3.Connection:
    global _test_db_connection
    if _test_db_connection and DB_NAME == ":memory:":
        return _test_db_connection
    
    criar_diretorio_se_nao_existir(DB_NAME)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    return conn

def inicializar_db(conn_externa: Optional[sqlite3.Connection] = None):
    is_external_conn = conn_externa is not None
    conn = conn_externa if is_external_conn else conectar_db()
    
    try:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_api (
            chave TEXT PRIMARY KEY,
            valor TEXT,
            timestamp REAL,
            api_origem TEXT,
            expiracao INTEGER
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dados_historicos_ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simbolo TEXT NOT NULL,
            data TEXT NOT NULL,
            abertura REAL,
            maxima REAL,
            minima REAL,
            fechamento REAL,
            volume INTEGER,
            fonte TEXT,
            UNIQUE(simbolo, data, fonte)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dados_macro_bcb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serie_id INTEGER NOT NULL,
            nome_serie TEXT NOT NULL,
            data TEXT NOT NULL,
            valor REAL,
            UNIQUE(serie_id, data)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simbolo TEXT UNIQUE NOT NULL,
            tipo TEXT NOT NULL CHECK(tipo IN ('asset', 'index'))
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_visual_preferences (
            id_usuario INTEGER PRIMARY KEY,
            periodo_historico_padrao TEXT,
            indicadores_tecnicos_padrao TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_alerts (
            id_alerta INTEGER PRIMARY KEY AUTOINCREMENT,
            simbolo TEXT NOT NULL,
            tipo_alerta TEXT NOT NULL,
            condicao TEXT NOT NULL,
            ativo INTEGER NOT NULL DEFAULT 1,
            mensagem_customizada TEXT
        )
        """)

        conn.commit()
        logger.info("Banco de dados inicializado com sucesso (inicializar_db).")
    except Exception as e:
        logger.error(f"Erro durante inicializar_db: {e}", exc_info=True)
        if not is_external_conn and conn:
            try:
                conn.rollback()
            except Exception as rb_e:
                logger.error(f"Erro durante rollback em inicializar_db: {rb_e}")
        raise 
    finally:
        if not is_external_conn and conn:
            conn.close()


def obter_watchlist_do_db() -> List[Dict[str, Any]]:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT simbolo, tipo FROM user_watchlist ORDER BY tipo, simbolo")
        itens = cursor.fetchall()
        return [dict(row) for row in itens]
    except sqlite3.Error as e:
        logger.error(f"Erro ao obter watchlist do banco de dados: {e}")
        return []
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def adicionar_item_watchlist_db(simbolo: str, tipo: str) -> bool:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_watchlist (simbolo, tipo) VALUES (?, ?)", (simbolo, tipo))
        conn.commit()
        logger.info(f"Item '{simbolo}' ({tipo}) adicionado à watchlist no DB.")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Item '{simbolo}' já existe na watchlist.")
        return False
    except sqlite3.Error as e:
        logger.error(f"Erro ao adicionar item '{simbolo}' à watchlist no DB: {e}")
        return False
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def remover_item_watchlist_db(simbolo: str) -> bool:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_watchlist WHERE simbolo = ?", (simbolo,))
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Item '{simbolo}' removido da watchlist no DB.")
            return True
        else:
            logger.warning(f"Item '{simbolo}' não encontrado na watchlist para remoção.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Erro ao remover item '{simbolo}' da watchlist no DB: {e}")
        return False
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def salvar_preferencias_visualizacao_db(id_usuario: int, periodo_historico: str, indicadores_tecnicos: str) -> bool:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO user_visual_preferences (id_usuario, periodo_historico_padrao, indicadores_tecnicos_padrao)
        VALUES (?, ?, ?)
        """, (id_usuario, periodo_historico, indicadores_tecnicos))
        conn.commit()
        logger.info(f"Preferências de visualização salvas para usuário {id_usuario}.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Erro ao salvar preferências de visualização para usuário {id_usuario}: {e}")
        return False
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def carregar_preferencias_visualizacao_db(id_usuario: int = 0) -> Optional[Dict[str, Any]]:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id_usuario, periodo_historico_padrao, indicadores_tecnicos_padrao FROM user_visual_preferences WHERE id_usuario = ?", (id_usuario,))
        prefs = cursor.fetchone()
        return dict(prefs) if prefs else None
    except sqlite3.Error as e:
        logger.error(f"Erro ao carregar preferences de visualização para usuário {id_usuario}: {e}")
        return None
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def adicionar_alerta_db(alerta: AlertaConfigurado) -> bool:
    conn = conectar_db()
    try:
        condicao_json = json.dumps(alerta.condicao)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO user_alerts (simbolo, tipo_alerta, condicao, ativo, mensagem_customizada)
        VALUES (?, ?, ?, ?, ?)
        """, (alerta.simbolo, alerta.tipo_alerta, condicao_json, 1 if alerta.ativo else 0, alerta.mensagem_customizada))
        conn.commit()
        logger.info(f"Alerta para '{alerta.simbolo}' ({alerta.tipo_alerta}) adicionado ao DB.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Erro ao adicionar alerta para '{alerta.simbolo}' ao DB: {e}")
        return False
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def listar_alertas_db() -> List[Dict[str, Any]]:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id_alerta, simbolo, tipo_alerta, condicao, ativo, mensagem_customizada FROM user_alerts ORDER BY id_alerta")
        alertas_raw = cursor.fetchall()
        alertas_formatados = []
        for row in alertas_raw:
            alerta_dict = dict(row)
            alerta_dict['condicao'] = json.loads(alerta_dict['condicao'])
            alerta_dict['ativo'] = bool(alerta_dict['ativo'])
            alertas_formatados.append(alerta_dict)
        return alertas_formatados
    except sqlite3.Error as e:
        logger.error(f"Erro ao listar alertas do banco de dados: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON da condição de alerta do banco de dados: {e}")
        return []
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def remover_alerta_db(id_alerta: int) -> bool:
    conn = conectar_db()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_alerts WHERE id_alerta = ?", (id_alerta,))
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Alerta ID {id_alerta} removido do DB.")
            return True
        else:
            logger.warning(f"Alerta ID {id_alerta} não encontrado no DB para remoção.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Erro ao remover alerta ID {id_alerta} do DB: {e}")
        return False
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

def atualizar_alerta_db(id_alerta: int, novos_dados: Dict[str, Any]) -> bool:
    conn = conectar_db()
    cursor = conn.cursor()
    try:
        campos_para_atualizar = []
        valores = []
        
        if 'simbolo' in novos_dados:
            campos_para_atualizar.append("simbolo = ?")
            valores.append(novos_dados['simbolo'])
        if 'tipo_alerta' in novos_dados:
            campos_para_atualizar.append("tipo_alerta = ?")
            valores.append(novos_dados['tipo_alerta'])
        if 'condicao' in novos_dados:
            campos_para_atualizar.append("condicao = ?")
            valores.append(json.dumps(novos_dados['condicao']))
        if 'ativo' in novos_dados:
            campos_para_atualizar.append("ativo = ?")
            valores.append(1 if novos_dados['ativo'] else 0)
        if 'mensagem_customizada' in novos_dados:
            campos_para_atualizar.append("mensagem_customizada = ?")
            valores.append(novos_dados['mensagem_customizada'])
        
        if not campos_para_atualizar:
            logger.warning(f"Nenhum campo para atualizar para o alerta ID {id_alerta}.")
            return False
            
        query = f"UPDATE user_alerts SET {', '.join(campos_para_atualizar)} WHERE id_alerta = ?"
        valores.append(id_alerta)
        
        cursor.execute(query, tuple(valores))
        conn.commit()
        
        if cursor.rowcount > 0:
            logger.info(f"Alerta ID {id_alerta} atualizado no DB.")
            return True
        else:
            logger.warning(f"Alerta ID {id_alerta} não encontrado no DB para atualização ou nenhum dado alterado.")
            return False 
            
    except sqlite3.Error as e:
        logger.error(f"Erro ao atualizar alerta ID {id_alerta} no DB: {e}")
        return False
    finally:
        if not (_test_db_connection and DB_NAME == ":memory:"):
            if conn:
                conn.close()

if __name__ == '__main__':
    inicializar_db()