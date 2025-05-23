import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class CotacaoAtivo:
    simbolo: str
    preco_atual: Optional[float] = None
    variacao_percentual: Optional[float] = None
    variacao_absoluta: Optional[float] = None
    preco_abertura: Optional[float] = None
    preco_maximo: Optional[float] = None
    preco_minimo: Optional[float] = None
    preco_fechamento_anterior: Optional[float] = None
    volume: Optional[int] = None
    timestamp_ultima_atualizacao: Optional[int] = None
    nome_empresa: Optional[str] = None 

@dataclass
class IndicadorMacroeconomico: 
    nome: str
    valor: Optional[float] = None
    data_referencia: Optional[str] = None
    unidade: Optional[str] = None

@dataclass
class SerieEconomica: 
    id_serie: str
    nome_serie: str
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    fonte: Optional[str] = None
    unidade: Optional[str] = None
    frequencia: Optional[str] = None
    notas: Optional[str] = None


@dataclass
class DadosHistoricos:
    simbolo: str
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class ExpectativaMercado: 
    indicador: str
    media: Optional[float] = None
    mediana: Optional[float] = None
    data_referencia: Optional[str] = None

@dataclass
class PreferenciasVisualizacao:
    id_usuario: int 
    periodo_historico_padrao: str = "1y"
    indicadores_tecnicos_padrao: List[str] = field(default_factory=lambda: ["SMA_21", "EMA_21", "MACD_Hist", "RSI_14"])

@dataclass
class AlertaConfigurado:
    simbolo: str
    tipo_alerta: str 
    condicao: Dict[str, Any] 
    id_alerta: Optional[int] = None 
    ativo: bool = True
    mensagem_customizada: Optional[str] = None

@dataclass
class RelatorioExportado:
    nome_arquivo: str
    tipo_conteudo: str 
    caminho_salvo: str
    timestamp_geracao: datetime = field(default_factory=datetime.now)