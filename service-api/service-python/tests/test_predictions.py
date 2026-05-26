import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from sklearn.preprocessing import MinMaxScaler
from models.prediction.lstm_model import ModeloLSTM
from models.prediction.prophet_model import ModeloProphet
from models.prediction.evaluator import ModelEvaluator
from controllers.prediction_controller import PredictionController
from config.settings import LSTM_LOOKBACK_PERIOD, LSTM_TRAIN_TEST_SPLIT, LSTM_EPOCHS, LSTM_BATCH_SIZE, PROPHET_FORECAST_PERIODS

@pytest.fixture
def sample_time_series_data_features():
    n_samples = 200
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='B'),
        'Close': np.linspace(100, 150, n_samples) + np.random.normal(0, 5, n_samples),
        'Volume': np.random.randint(1000, 5000, n_samples),
        'SMA_10': np.linspace(98, 148, n_samples) + np.random.normal(0, 3, n_samples),
        'RSI_14': np.random.uniform(30, 70, n_samples)
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

@pytest.fixture
def sample_time_series_data_simple():
    n_samples = 100
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='B'),
        'Close': np.linspace(50, 80, n_samples) + np.random.normal(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    return df 

@pytest.fixture
def model_config_fixture():
    return {
        'lookback_period': LSTM_LOOKBACK_PERIOD,
        'lstm_units_1': 50,
        'lstm_units_2': 50,
        'dense_units': 25,
        'activation_dense': 'relu',
        'dropout_rate': 0.2,
        'train_test_split': LSTM_TRAIN_TEST_SPLIT,
        'epochs': 1, 
        'batch_size': LSTM_BATCH_SIZE,
        'optimizer': 'adam',
        'loss_function': 'mean_squared_error',
        'prophet_configs': {'daily_seasonality': False}
    }

@pytest.fixture
def prediction_controller_fixture(model_config_fixture):
    return PredictionController(model_config=model_config_fixture)


class TestModeloLSTM:
    def test_lstm_preparar_dados_shape(self, sample_time_series_data_features):
        model_lstm = ModeloLSTM(lookback_period=10)
        df_target = sample_time_series_data_features['Close']
        df_features = sample_time_series_data_features.drop(columns=['Close'])
        
        X, y, indices, y_orig = model_lstm._preparar_dados(df_features, df_target)
        
        assert X is not None and y is not None
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == len(sample_time_series_data_features) - 10
        assert X.shape[1] == 10 
        assert X.shape[2] == df_features.shape[1] 
        assert len(indices) == X.shape[0]
        assert len(y_orig) == y.shape[0]

    def test_lstm_preparar_dados_para_treino_teste(self, sample_time_series_data_features):
        model_lstm = ModeloLSTM(lookback_period=10)
        df_completo = sample_time_series_data_features.copy()
        
        results = model_lstm.preparar_dados_para_treino_teste(df_completo, 'Close', train_split_ratio=0.7)
        X_treino, y_treino, X_teste, y_teste, _, _, _, _, scaler_target, feature_cols = results

        assert X_treino is not None and X_teste is not None
        assert X_treino.shape[0] > 0 and X_teste.shape[0] > 0
        assert X_treino.shape[2] == len(feature_cols)
        assert 'Close' not in feature_cols
        assert isinstance(scaler_target, MinMaxScaler)


    @patch('keras.models.Sequential')
    def test_lstm_construir_modelo(self, mock_sequential_class):
        mock_model_instance = MagicMock()
        mock_sequential_class.return_value = mock_model_instance
        
        model_lstm = ModeloLSTM(lstm_units_1=30, lstm_units_2=30, dense_units=15)
        input_shape = (10, 5) 
        model_lstm.construir_modelo(input_shape)
        
        assert model_lstm.model is not None
        assert mock_model_instance.add.call_count >= 4 


    @patch.object(ModeloLSTM, 'construir_modelo') 
    def test_lstm_treinar_modelo_chama_fit(self, mock_construir, sample_time_series_data_features):
        model_lstm = ModeloLSTM(lookback_period=10)
        df_completo = sample_time_series_data_features.copy()
        X_treino, y_treino, X_teste, y_teste, _, _, _, _, _, _ = model_lstm.preparar_dados_para_treino_teste(df_completo, 'Close', 0.7)

        model_lstm.model = MagicMock() 
        mock_construir.return_value = None 
        
        if X_treino.size > 0 and X_teste.size > 0 :
            model_lstm.treinar_modelo(X_treino, y_treino, X_teste, y_teste, epochs=1, batch_size=1)
            model_lstm.model.compile.assert_called_once()
            model_lstm.model.fit.assert_called_once()


class TestModeloProphet:
    def test_prophet_preparar_dados(self, sample_time_series_data_simple):
        model_prophet = ModeloProphet()
        df_historico = sample_time_series_data_simple.copy()
        
        df_prophet = model_prophet.preparar_dados_prophet(df_historico, coluna_data='Date', coluna_target='Close')
        
        assert not df_prophet.empty
        assert 'ds' in df_prophet.columns
        assert 'y' in df_prophet.columns
        assert pd.api.types.is_datetime64_any_dtype(df_prophet['ds'])
        assert df_prophet['ds'].dt.tz is None


    @patch('prophet.Prophet')
    def test_prophet_treinar_modelo_chama_fit(self, mock_prophet_class, sample_time_series_data_simple):
        mock_prophet_instance = MagicMock()
        mock_prophet_class.return_value = mock_prophet_instance
        
        model_prophet = ModeloProphet()
        df_treino = model_prophet.preparar_dados_prophet(sample_time_series_data_simple, 'Date', 'Close')
        
        model_prophet.treinar_modelo(df_treino)
        
        mock_prophet_class.assert_called_once_with(**model_prophet.config_prophet)
        mock_prophet_instance.fit.assert_called_once_with(df_treino)
        assert model_prophet.model is not None


    @patch('prophet.Prophet')
    def test_prophet_prever_futuro_chama_predict(self, mock_prophet_class, sample_time_series_data_simple):
        mock_prophet_instance = MagicMock()
        mock_prophet_instance.make_future_dataframe.return_value = pd.DataFrame({'ds': pd.to_datetime(['2024-01-01', '2024-01-02'])})
        mock_prophet_instance.predict.return_value = pd.DataFrame({'ds': [], 'yhat': []})
        mock_prophet_class.return_value = mock_prophet_instance

        model_prophet = ModeloProphet()
        df_treino = model_prophet.preparar_dados_prophet(sample_time_series_data_simple, 'Date', 'Close')
        model_prophet.treinar_modelo(df_treino) 

        model_prophet.model = mock_prophet_instance 
        forecast_df = model_prophet.prever_futuro(periodos=2, frequencia='D')
        
        mock_prophet_instance.make_future_dataframe.assert_called_once_with(periods=2, freq='D')
        mock_prophet_instance.predict.assert_called_once()


class TestModelEvaluator:
    def test_calcular_metricas_valores_conhecidos(self):
        y_real = np.array([10, 11, 12, 11, 10])
        y_previsto = np.array([10.5, 10.5, 11.5, 11.5, 9.5])
        
        metricas = ModelEvaluator.calcular_metricas(y_real, y_previsto)
        
        assert "mae" in metricas
        assert "rmse" in metricas
        assert "taxa_acerto_direcional" in metricas
        assert metricas["mae"] == pytest.approx(0.5)
        
        
        y_real_dir = np.array([10, 11, 10, 12, 11]) 
        y_prev_dir = np.array([10, 12, 9, 13, 10])  
        metricas_dir = ModelEvaluator.calcular_metricas(y_real_dir, y_prev_dir)
        assert metricas_dir["taxa_acerto_direcional"] == pytest.approx(1.0) 
        assert metricas_dir["num_acertos_direcionais"] == 4
        assert metricas_dir["total_comparacoes_direcionais"] == 4

    def test_calcular_metricas_arrays_vazios(self):
        metricas = ModelEvaluator.calcular_metricas(np.array([]), np.array([]))
        assert metricas == {}

    def test_calcular_metricas_tamanhos_diferentes(self):
        metricas = ModelEvaluator.calcular_metricas(np.array([1,2]), np.array([1]))
        assert metricas == {}


class TestPredictionController:

    @patch('models.prediction.lstm_model.ModeloLSTM')
    def test_pc_treinar_avaliar_lstm_fluxo_sucesso(self, mock_modelo_lstm_class, prediction_controller_fixture, sample_time_series_data_features):
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.preparar_dados_para_treino_teste.return_value = (
            np.random.rand(50, 10, 5), np.random.rand(50), 
            np.random.rand(20, 10, 5), np.random.rand(20),
            pd.to_datetime(pd.Series(['2023-01-01'] * 50)), pd.to_datetime(pd.Series(['2023-01-20'] * 20)),
            pd.Series(np.random.rand(50)), pd.Series(np.random.rand(20)),
            MagicMock(spec=MinMaxScaler), ['feature1', 'feature2']
        )
        mock_lstm_instance.prever.return_value = np.random.rand(20)
        mock_lstm_instance.model = MagicMock() 
        
        mock_modelo_lstm_class.return_value = mock_lstm_instance
        prediction_controller_fixture.modelo_lstm_instancia = mock_lstm_instance 

        df_dados = sample_time_series_data_features.reset_index() 
        
        modelo, scaler, df_comp, metricas = prediction_controller_fixture.treinar_avaliar_modelo_lstm(df_dados, 'Close')

        assert modelo is not None
        assert df_comp is not None
        assert not df_comp.empty
        assert "Real" in df_comp.columns and "Previsto" in df_comp.columns
        assert "mae" in metricas
        mock_lstm_instance.preparar_dados_para_treino_teste.assert_called_once()
        mock_lstm_instance.construir_modelo.assert_called_once()
        mock_lstm_instance.treinar_modelo.assert_called_once()
        mock_lstm_instance.prever.assert_called_once()

    def test_pc_treinar_avaliar_lstm_dados_insuficientes(self, prediction_controller_fixture):
        df_curto = pd.DataFrame({'Date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'Close': [1,2]})
        
        modelo, scaler, df_comp, metricas = prediction_controller_fixture.treinar_avaliar_modelo_lstm(df_curto, 'Close')
        
        assert modelo is None
        assert df_comp.empty
        assert not metricas 

    @patch('models.prediction.prophet_model.ModeloProphet')
    def test_pc_treinar_avaliar_prophet_fluxo_sucesso(self, mock_modelo_prophet_class, prediction_controller_fixture, sample_time_series_data_simple):
        mock_prophet_instance = MagicMock()
        mock_prophet_instance.preparar_dados_prophet.return_value = pd.DataFrame({'ds': sample_time_series_data_simple['Date'], 'y': sample_time_series_data_simple['Close']})
        mock_prophet_instance.prever_futuro.return_value = pd.DataFrame({
            'ds': pd.date_range(sample_time_series_data_simple['Date'].iloc[-1], periods=PROPHET_FORECAST_PERIODS +1, freq='B')[1:],
            'yhat': np.random.rand(PROPHET_FORECAST_PERIODS),
            'yhat_lower': np.random.rand(PROPHET_FORECAST_PERIODS),
            'yhat_upper': np.random.rand(PROPHET_FORECAST_PERIODS)
        })
        mock_prophet_instance.model = MagicMock() 
        mock_prophet_instance.colunas_regressores = [] 

        mock_modelo_prophet_class.return_value = mock_prophet_instance
        prediction_controller_fixture.modelo_prophet_instancia = mock_prophet_instance


        df_dados = sample_time_series_data_simple.copy()
        modelo, df_forecast, df_hist_usado = prediction_controller_fixture.treinar_avaliar_modelo_prophet(
            df_dados, coluna_data_prophet='Date', coluna_target_prophet='Close', periodos_previsao=PROPHET_FORECAST_PERIODS
        )
        
        assert modelo is not None
        assert df_forecast is not None
        assert not df_forecast.empty
        assert 'Previsto' in df_forecast.columns 
        mock_prophet_instance.preparar_dados_prophet.assert_called_once()
        mock_prophet_instance.treinar_modelo.assert_called_once()
        mock_prophet_instance.prever_futuro.assert_called_once()


    def test_pc_prever_proximos_passos_lstm_sem_modelo_treinado(self, prediction_controller_fixture):
        resultado = prediction_controller_fixture.prever_proximos_passos_lstm(pd.DataFrame())
        assert resultado is None