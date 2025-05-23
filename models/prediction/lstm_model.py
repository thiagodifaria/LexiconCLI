import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from utils.logger import logger

class ModeloLSTM:
    def __init__(self, lookback_period=60, lstm_units_1=50, lstm_units_2=50, dense_units=25, activation_dense='relu', dropout_rate=0.2):
        self.lookback_period = lookback_period
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.activation_dense = activation_dense
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))

    def _preparar_dados(self, df_features: pd.DataFrame, df_target: pd.Series):        
        df_features_filled = df_features.ffill().bfill()
        df_target_filled = df_target.ffill().bfill()
        
        initial_len = len(df_features_filled)
        df_features_filled.dropna(inplace=True)
        df_target_filled = df_target_filled[df_target_filled.index.isin(df_features_filled.index)]
        if len(df_features_filled) < initial_len:
            logger.warning(f"{initial_len - len(df_features_filled)} linhas removidas devido a NaNs persistentes antes da normalização.")

        if df_features_filled.empty or df_target_filled.empty:
            logger.error("DataFrame de features ou target está vazio após tratamento de NaNs.")
            return None, None, None, None
            
        scaled_features = self.scaler_features.fit_transform(df_features_filled)
        scaled_target = self.scaler_target.fit_transform(df_target_filled.values.reshape(-1, 1))

        X, y = [], []
        for i in range(self.lookback_period, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_period:i, :]) 
            y.append(scaled_target[i, 0])
        
        if not X or not y:
            logger.error(f"Não foi possível criar sequências X, y. Comprimento dos dados escalados: {len(scaled_features)}, lookback: {self.lookback_period}")
            return None, None, None, None

        return np.array(X), np.array(y), df_features_filled.index[self.lookback_period:], df_target_filled.iloc[self.lookback_period:]


    def construir_modelo(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=self.lstm_units_1, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(LSTM(units=self.lstm_units_2, return_sequences=False))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(units=self.dense_units, activation=self.activation_dense))
        self.model.add(Dense(units=1)) 
        
        logger.info("Modelo LSTM construído.")


    def treinar_modelo(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, optimizer='adam', loss='mean_squared_error'):
        if self.model is None:
            logger.error("Modelo não construído. Chame construir_modelo() primeiro.")
            return None
        
        self.model.compile(optimizer=optimizer, loss=loss)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1 
        )
        logger.info("Treinamento do modelo LSTM concluído.")
        return history

    def prever(self, X_data):
        if self.model is None:
            logger.error("Modelo não treinado.")
            return None
        
        predicoes_scaled = self.model.predict(X_data)
        predicoes_desnormalizadas = self.scaler_target.inverse_transform(predicoes_scaled)
        return predicoes_desnormalizadas.flatten()

    def preparar_dados_para_treino_teste(self, df_completo: pd.DataFrame, coluna_target: str, train_split_ratio: float = 0.7):
        if coluna_target not in df_completo.columns:
            logger.error(f"Coluna target '{coluna_target}' não encontrada no DataFrame.")
            return None, None, None, None, None, None, None, None, None, None

        df_target_series = df_completo[coluna_target]
        df_features_df = df_completo.drop(columns=[coluna_target])
        
        logger.info(f"Colunas de features utilizadas para o LSTM: {df_features_df.columns.tolist()}")


        X_sequencias, y_sequencias, indices_tempo, y_original_series = self._preparar_dados(df_features_df, df_target_series)

        if X_sequencias is None or y_sequencias is None:
            logger.error("Falha na preparação dos dados em _preparar_dados.")
            return None, None, None, None, None, None, None, None, None, None
        
        if len(X_sequencias) == 0:
            logger.error("Nenhuma sequência gerada para X_treino/X_teste.")
            return None, None, None, None, None, None, None, None, None, None


        split_index = int(len(X_sequencias) * train_split_ratio)

        X_treino, X_teste = X_sequencias[:split_index], X_sequencias[split_index:]
        y_treino, y_teste = y_sequencias[:split_index], y_sequencias[split_index:]
        
        indices_treino, indices_teste = indices_tempo[:split_index], indices_tempo[split_index:]
        y_original_treino, y_original_teste = y_original_series[:split_index], y_original_series[split_index:]


        if X_treino.size == 0 or X_teste.size == 0:
            logger.error(f"Divisão treino/teste resultou em conjunto vazio. X_treino: {X_treino.shape}, X_teste: {X_teste.shape}")
            return None, None, None, None, None, None, None, None, None, None

        logger.info(f"Dados preparados para LSTM: X_treino shape {X_treino.shape}, y_treino shape {y_treino.shape}, X_teste shape {X_teste.shape}, y_teste shape {y_teste.shape}")
        return X_treino, y_treino, X_teste, y_teste, indices_treino, indices_teste, y_original_treino, y_original_teste, self.scaler_target, df_features_df.columns.tolist()