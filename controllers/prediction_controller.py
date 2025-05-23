from models.prediction.lstm_model import ModeloLSTM
from models.prediction.prophet_model import ModeloProphet
from models.prediction.evaluator import ModelEvaluator
import pandas as pd
import numpy as np
from utils.logger import logger
from datetime import datetime

class PredictionController:
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.modelo_lstm_instancia = None
        self.colunas_features_usadas_lstm = None
        self.modelo_prophet_instancia = None

    def _instanciar_modelo_lstm(self):
        return ModeloLSTM(
            lookback_period=self.model_config.get('lookback_period', 60),
            lstm_units_1=self.model_config.get('lstm_units_1', 50),
            lstm_units_2=self.model_config.get('lstm_units_2', 50),
            dense_units=self.model_config.get('dense_units', 25),
            activation_dense=self.model_config.get('activation_dense', 'relu'),
            dropout_rate=self.model_config.get('dropout_rate', 0.2)
        )

    def _instanciar_modelo_prophet(self, config_prophet_especifica: dict = None):
        prophet_configs = self.model_config.get('prophet_configs', {})
        if config_prophet_especifica:
            prophet_configs.update(config_prophet_especifica)
        return ModeloProphet(config_prophet=prophet_configs)

    def treinar_avaliar_modelo_lstm(self, df_dados_completos: pd.DataFrame, coluna_target: str):
        self.modelo_lstm_instancia = self._instanciar_modelo_lstm()
        
        if df_dados_completos.empty:
            logger.error("DataFrame de dados completos está vazio. Não é possível treinar LSTM.")
            return None, None, pd.DataFrame(), {}

        if not isinstance(df_dados_completos.index, pd.DatetimeIndex):
            if 'Date' in df_dados_completos.columns:
                 df_dados_completos['Date'] = pd.to_datetime(df_dados_completos['Date'])
                 df_dados_completos.set_index('Date', inplace=True)
            elif df_dados_completos.index.name == 'Date' or df_dados_completos.index.name == 'index': 
                try:
                    df_dados_completos.index = pd.to_datetime(df_dados_completos.index)
                except Exception as e:
                    logger.error(f"Não foi possível converter índice para DatetimeIndex para LSTM: {e}")
                    return None, None, pd.DataFrame(), {}
            else:
                logger.error("Índice do DataFrame não é DatetimeIndex e não há coluna 'Date' para LSTM.")
                return None, None, pd.DataFrame(), {}

        df_dados_tratados = df_dados_completos.copy()
        for col in df_dados_tratados.columns:
            if col != coluna_target and df_dados_tratados[col].isnull().any():
                df_dados_tratados[col] = df_dados_tratados[col].fillna(method='ffill').fillna(method='bfill')
        
        df_dados_tratados.dropna(subset=[coluna_target] + [col for col in df_dados_tratados.columns if col != coluna_target], inplace=True)


        if len(df_dados_tratados) < self.model_config.get('lookback_period', 60) * 2: 
            logger.error(f"Dados insuficientes após tratamento de NaNs para treinar o modelo LSTM: {len(df_dados_tratados)} linhas.")
            return None, None, pd.DataFrame(), {}


        X_treino, y_treino, X_teste, y_teste, indices_treino, indices_teste, y_original_treino, y_original_teste, scaler_target, self.colunas_features_usadas_lstm = \
            self.modelo_lstm_instancia.preparar_dados_para_treino_teste(
                df_dados_tratados,
                coluna_target=coluna_target,
                train_split_ratio=self.model_config.get('train_test_split', 0.7)
            )

        if X_treino is None or X_teste is None or X_treino.size == 0 or X_teste.size == 0 :
            logger.error("Falha na preparação dos dados para treino/teste LSTM ou resultou em conjuntos vazios.")
            return None, None, pd.DataFrame(), {}
            
        input_shape = (X_treino.shape[1], X_treino.shape[2])
        self.modelo_lstm_instancia.construir_modelo(input_shape=input_shape)

        logger.info(f"Iniciando treinamento LSTM. X_treino: {X_treino.shape}, y_treino: {y_treino.shape}, X_teste: {X_teste.shape}, y_teste: {y_teste.shape}")

        self.modelo_lstm_instancia.treinar_modelo(
            X_treino, y_treino, X_teste, y_teste, 
            epochs=self.model_config.get('epochs', 50),
            batch_size=self.model_config.get('batch_size', 32),
            optimizer=self.model_config.get('optimizer', 'adam'),
            loss=self.model_config.get('loss_function', 'mean_squared_error')
        )

        predicoes_teste_desnormalizadas = self.modelo_lstm_instancia.prever(X_teste)
        
        if predicoes_teste_desnormalizadas is None:
            logger.error("Falha ao obter previsões do modelo LSTM.")
            return self.modelo_lstm_instancia.model, scaler_target, pd.DataFrame(), {}

        df_comparacao = pd.DataFrame({
            'Data': indices_teste, 
            'Real': y_original_teste.values, 
            'Previsto': predicoes_teste_desnormalizadas
        })
        df_comparacao.sort_values(by='Data', inplace=True) 

        metricas = ModelEvaluator.calcular_metricas(df_comparacao['Real'].values, df_comparacao['Previsto'].values)
        
        logger.info(f"Modelo LSTM treinado. Métricas no conjunto de teste: {metricas}")
        
        return self.modelo_lstm_instancia.model, scaler_target, df_comparacao, metricas

    def prever_proximos_passos_lstm(self, df_dados_recentes_com_features: pd.DataFrame, num_passos: int = 1):
        if self.modelo_lstm_instancia is None or self.modelo_lstm_instancia.model is None:
            logger.error("Modelo LSTM não treinado ou não disponível.")
            return None
        if self.colunas_features_usadas_lstm is None:
            logger.error("Nomes das colunas de features LSTM não foram armazenados. Treine o modelo primeiro.")
            return None
        
        df_features_recentes = df_dados_recentes_com_features[self.colunas_features_usadas_lstm].copy()

        df_features_recentes_filled = df_features_recentes.fillna(method='ffill').fillna(method='bfill')
        df_features_recentes_filled.dropna(inplace=True)

        if len(df_features_recentes_filled) < self.modelo_lstm_instancia.lookback_period:
            logger.error(f"Dados recentes insuficientes ({len(df_features_recentes_filled)} linhas) para previsão LSTM com lookback de {self.modelo_lstm_instancia.lookback_period}.")
            return None

        ultimos_dados = df_features_recentes_filled.tail(self.modelo_lstm_instancia.lookback_period)        
        
        ultimos_dados_scaled = self.modelo_lstm_instancia.scaler_features.transform(ultimos_dados)
        
        input_sequence = np.array([ultimos_dados_scaled])

        predicao_scaled = self.modelo_lstm_instancia.model.predict(input_sequence)
        predicao_desnormalizada = self.modelo_lstm_instancia.scaler_target.inverse_transform(predicao_scaled)
        
        logger.info(f"Previsão LSTM para o próximo passo: {predicao_desnormalizada.flatten()[0]}")
        return predicao_desnormalizada.flatten()[0]

    def treinar_avaliar_modelo_prophet(self, df_dados_completos: pd.DataFrame, coluna_data_prophet: str = 'Date', coluna_target_prophet: str = 'Close', colunas_regressores: list = None, periodos_previsao: int = 30, config_prophet: dict = None):
        self.modelo_prophet_instancia = self._instanciar_modelo_prophet(config_prophet_especifica=config_prophet)

        if df_dados_completos.empty:
            logger.error("DataFrame de dados completos está vazio. Não é possível treinar Prophet.")
            return None, pd.DataFrame(), pd.DataFrame()
        
        df_historico_prophet = self.modelo_prophet_instancia.preparar_dados_prophet(
            df_dados_completos,
            coluna_data=coluna_data_prophet,
            coluna_target=coluna_target_prophet,
            colunas_regressores=colunas_regressores
        )

        if df_historico_prophet.empty:
            logger.error("Falha na preparação dos dados para Prophet.")
            return None, pd.DataFrame(), pd.DataFrame()

        tamanho_treino = len(df_historico_prophet) - periodos_previsao
        if tamanho_treino <= 0:
             logger.warning(f"Não há dados suficientes para separar um conjunto de teste para Prophet com {periodos_previsao} períodos de previsão. Usando todo o histórico para treino e prevendo o futuro.")
             df_treino = df_historico_prophet
             df_teste_para_comparacao = pd.DataFrame()
        elif tamanho_treino < 10 :
             logger.warning(f"Poucos dados para treino ({tamanho_treino} pontos) após separar para teste. Usando todo o histórico para treino.")
             df_treino = df_historico_prophet
             df_teste_para_comparacao = pd.DataFrame()
             periodos_previsao_reais = periodos_previsao 
        else:
            df_treino = df_historico_prophet.iloc[:tamanho_treino]
            df_teste_para_comparacao = df_historico_prophet.iloc[tamanho_treino:]
            periodos_previsao_reais = len(df_teste_para_comparacao)


        self.modelo_prophet_instancia.treinar_modelo(df_treino)

        if self.modelo_prophet_instancia.model is None:
            logger.error("Falha ao treinar o modelo Prophet.")
            return None, pd.DataFrame(), pd.DataFrame()

        df_regressores_futuros = None
        if self.modelo_prophet_instancia.colunas_regressores:
            if not df_teste_para_comparacao.empty:
                df_regressores_futuros = df_teste_para_comparacao[['ds'] + self.modelo_prophet_instancia.colunas_regressores]
            else: 
                if not df_historico_prophet.empty:
                    last_date = df_historico_prophet['ds'].max()
                    future_dates = pd.date_range(start=last_date, periods=periodos_previsao + 1, freq='B')[1:] 
                    df_regressores_futuros = pd.DataFrame({'ds': future_dates})
                    
                    for reg_col in self.modelo_prophet_instancia.colunas_regressores:
                        if reg_col in df_dados_completos.columns:
                            last_val = df_dados_completos[reg_col].iloc[-1] if not df_dados_completos[reg_col].dropna().empty else 0
                            df_regressores_futuros[reg_col] = last_val 
                            logger.info(f"Usando último valor conhecido ({last_val}) para regressor futuro '{reg_col}'.")
                        else:
                            logger.warning(f"Não foi possível encontrar dados para o regressor futuro '{reg_col}'. Usando 0.")
                            df_regressores_futuros[reg_col] = 0
                else:
                    logger.error("Não é possível criar regressores futuros sem dados históricos.")
                    return self.modelo_prophet_instancia.model, pd.DataFrame(), df_historico_prophet


        forecast_df = self.modelo_prophet_instancia.prever_futuro(
            periodos=periodos_previsao_reais if not df_teste_para_comparacao.empty else periodos_previsao, 
            frequencia='B', 
            df_regressores_futuros=df_regressores_futuros
        )

        df_comparacao_final = pd.DataFrame()
        if not forecast_df.empty and not df_teste_para_comparacao.empty:
            df_comparacao_final = pd.merge(df_teste_para_comparacao, forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
            df_comparacao_final.rename(columns={'y': 'Real', 'yhat': 'Previsto'}, inplace=True)
        elif not forecast_df.empty: 
            df_comparacao_final = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            df_comparacao_final.rename(columns={'yhat': 'Previsto'}, inplace=True)


        logger.info(f"Modelo Prophet treinado e previsão gerada.")
        return self.modelo_prophet_instancia.model, df_comparacao_final, df_historico_prophet