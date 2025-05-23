import pandas as pd
from prophet import Prophet
from utils.logger import logger

class ModeloProphet:
    def __init__(self, config_prophet: dict = None):
        self.model = None
        self.config_prophet = config_prophet if config_prophet else {}
        self.colunas_regressores = []

    def preparar_dados_prophet(self, df_historico: pd.DataFrame, coluna_data: str = 'Date', coluna_target: str = 'Close', colunas_regressores: list = None):
        if df_historico.empty:
            logger.error("DataFrame histórico está vazio para preparação de dados do Prophet.")
            return pd.DataFrame()

        df_prophet = df_historico.copy()

        if coluna_data not in df_prophet.columns and df_prophet.index.name == coluna_data:
            df_prophet.reset_index(inplace=True)
        
        if coluna_data not in df_prophet.columns:
            logger.error(f"Coluna de data '{coluna_data}' não encontrada no DataFrame.")
            return pd.DataFrame()
        if coluna_target not in df_prophet.columns:
            logger.error(f"Coluna target '{coluna_target}' não encontrada no DataFrame.")
            return pd.DataFrame()

        df_prophet.rename(columns={coluna_data: 'ds', coluna_target: 'y'}, inplace=True)
        
        colunas_necessarias = ['ds', 'y']
        self.colunas_regressores = []

        if colunas_regressores:
            for regressor in colunas_regressores:
                if regressor not in df_prophet.columns:
                    logger.warning(f"Coluna de regressor '{regressor}' não encontrada. Será ignorada.")
                else:
                    colunas_necessarias.append(regressor)
                    self.colunas_regressores.append(regressor)
        
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        
        if df_prophet['ds'].dt.tz is not None:
            logger.info(f"Removendo fuso horário da coluna 'ds' para o Prophet. Fuso original: {df_prophet['ds'].dt.tz}")
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        
        df_prophet = df_prophet[colunas_necessarias].dropna(subset=['ds', 'y'])

        if df_prophet.empty:
            logger.error("DataFrame vazio após preparação e tratamento de NaNs para Prophet.")
        
        return df_prophet

    def treinar_modelo(self, df_treino_prophet: pd.DataFrame):
        if df_treino_prophet.empty or 'ds' not in df_treino_prophet.columns or 'y' not in df_treino_prophet.columns:
            logger.error("DataFrame de treino para Prophet está vazio ou não contém colunas 'ds' ou 'y'.")
            self.model = None
            return

        if df_treino_prophet['ds'].dt.tz is not None:
            logger.warning("Coluna 'ds' no DataFrame de treino ainda possui fuso horário. Removendo novamente.")
            df_treino_prophet = df_treino_prophet.copy()
            df_treino_prophet['ds'] = df_treino_prophet['ds'].dt.tz_localize(None)

        self.model = Prophet(**self.config_prophet)

        if self.colunas_regressores:
            for regressor in self.colunas_regressores:
                if regressor in df_treino_prophet.columns:
                    self.model.add_regressor(regressor)
                else:
                    logger.warning(f"Regressor '{regressor}' não encontrado no df_treino_prophet durante add_regressor.")
        
        try:
            self.model.fit(df_treino_prophet)
            logger.info("Modelo Prophet treinado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao treinar modelo Prophet: {e}")
            self.model = None


    def prever_futuro(self, periodos: int, frequencia: str = 'D', df_regressores_futuros: pd.DataFrame = None):
        if self.model is None:
            logger.error("Modelo Prophet não treinado. Chame treinar_modelo() primeiro.")
            return pd.DataFrame()

        future_df = self.model.make_future_dataframe(periods=periodos, freq=frequencia)

        if self.colunas_regressores:
            if df_regressores_futuros is None or df_regressores_futuros.empty:
                logger.error("Regressores foram usados no treino, mas df_regressores_futuros não foi fornecido ou está vazio para previsão.")
                return pd.DataFrame()
            
            if 'ds' not in df_regressores_futuros.columns:
                 logger.error("df_regressores_futuros deve conter a coluna 'ds' com as datas futuras.")
                 return pd.DataFrame()
            
            df_regressores_futuros = df_regressores_futuros.copy()
            df_regressores_futuros['ds'] = pd.to_datetime(df_regressores_futuros['ds'])
            if df_regressores_futuros['ds'].dt.tz is not None:
                df_regressores_futuros['ds'] = df_regressores_futuros['ds'].dt.tz_localize(None)


            missing_regressors = [reg for reg in self.colunas_regressores if reg not in df_regressores_futuros.columns]
            if missing_regressors:
                logger.error(f"Regressores futuros ausentes em df_regressores_futuros: {missing_regressors}")
                return pd.DataFrame()
            
            future_df = pd.merge(future_df, df_regressores_futuros[['ds'] + self.colunas_regressores], on='ds', how='left')
            
            if future_df[self.colunas_regressores].isnull().any().any():
                logger.warning("Valores NaN encontrados nos regressores do DataFrame futuro. Preenchendo com ffill e bfill.")
                for reg_col in self.colunas_regressores:
                    future_df[reg_col] = future_df[reg_col].fillna(method='ffill').fillna(method='bfill')
                if future_df[self.colunas_regressores].isnull().any().any():
                    logger.error("Ainda existem NaNs nos regressores futuros após preenchimento. Não é possível prever.")
                    return pd.DataFrame()

        try:
            forecast_df = self.model.predict(future_df)
            logger.info(f"Previsão futura gerada para {periodos} períodos.")
            return forecast_df
        except Exception as e:
            logger.error(f"Erro ao gerar previsão com Prophet: {e}")
            return pd.DataFrame()