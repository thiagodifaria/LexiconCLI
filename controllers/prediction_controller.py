import os
import joblib
import json
import numpy as np
import pandas as pd
from keras.models import save_model, load_model
from keras.utils import to_categorical
from prophet.serialize import model_to_json, model_from_json
from models.prediction.lstm_model import ModeloLSTM
from models.prediction.prophet_model import ModeloProphet
from models.prediction.evaluator import ModelEvaluator
from models.simulation.monte_carlo_simulator import MonteCarloSimulator
from utils.logger import logger
from datetime import datetime
from config.settings import (
    CLASSIFICATION_THRESHOLDS, 
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_TRAIN_TEST_SPLIT,
    ADVANCED_TRAINING_CONFIG,
    ADVANCED_LOGGING,
    DATA_AUGMENTATION_CONFIG,
    FEATURE_SELECTION_CONFIG,
    OVERFITTING_DETECTION_CONFIG,
    LSTM_DYNAMIC_CONFIG,
    THRESHOLD_OPTIMIZATION_CONFIG,
    CROSS_SYMBOL_VALIDATION_CONFIG,
    HYPERPARAMETER_TUNING_CONFIG,
    BAYESIAN_LSTM_CONFIG,
    MONTE_CARLO_ENSEMBLE_CONFIG,
    UNCERTAINTY_QUANTIFICATION_CONFIG,
    SENTIMENT_ANALYSIS_CONFIG
)

from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

try:
    from models.prediction.advanced.augmentation import FinancialTimeSeriesAugmentation
    from models.prediction.advanced.adaptive_selector import AdaptiveFeatureSelector
    from utils.dataset_analyzer import DatasetAnalyzer
    from utils.complexity_manager import ComplexityManager
    ADVANCED_MODULES_IMPORT_SUCCESS = True
    logger.info("Advanced modules loaded successfully")
except ImportError as e:
    ADVANCED_MODULES_IMPORT_SUCCESS = False
    logger.warning(f"Advanced modules not available: {e}. Using basic functionality")

class PredictionController:
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.modelo_lstm_instancia = None
        self.colunas_features_usadas_lstm = None
        self.modelo_prophet_instancia = None
        self.data_controller = None
        self.dataset_analyzer = None
        self.augmentation_engine = None
        self.feature_selector = None
        self.monte_carlo_simulator = MonteCarloSimulator()
        self.MODEL_SAVE_PATH = "data/trained_models/"
        if not os.path.exists(self.MODEL_SAVE_PATH):
            os.makedirs(self.MODEL_SAVE_PATH)
            logger.info(f"Model directory '{self.MODEL_SAVE_PATH}' created")
        else:
            logger.info(f"Model directory '{self.MODEL_SAVE_PATH}' verified")
        
        self._initialize_advanced_modules()

    def _initialize_advanced_modules(self):
        self.advanced_modules_available = ADVANCED_MODULES_IMPORT_SUCCESS
        if self.advanced_modules_available:
            try:
                self.dataset_analyzer = DatasetAnalyzer()
                self.augmentation_engine = FinancialTimeSeriesAugmentation()
                self.feature_selector = AdaptiveFeatureSelector()
                self.complexity_manager = ComplexityManager()
                logger.info("Advanced modules and Complexity Manager initialized")
            except Exception as e:
                logger.error(f"Error initializing advanced modules: {e}")
                self.advanced_modules_available = False

    def _instanciar_modelo_lstm(self):
        return ModeloLSTM(
            lookback_period=self.model_config.get('lookback_period', 15),
            lstm_units_1=self.model_config.get('lstm_units_1', 8),
            lstm_units_2=self.model_config.get('lstm_units_2', 6),
            dense_units=self.model_config.get('dense_units', 8),
            activation_dense=self.model_config.get('activation_dense', 'relu'),
            dropout_rate=self.model_config.get('dropout_rate', 0.2)
        )

    def _instanciar_modelo_prophet(self, config_prophet_especifica: dict = None):
        prophet_configs = self.model_config.get('prophet_configs', {})
        if config_prophet_especifica:
            prophet_configs.update(config_prophet_especifica)
        return ModeloProphet(config_prophet=prophet_configs)

    def _criar_classes_direcionais(self, df_dados: pd.DataFrame, coluna_preco: str = 'close') -> pd.Series:
        logger.info("Creating directional classes with adaptive thresholds...")
        
        df_dados['target_return'] = df_dados[coluna_preco].pct_change().shift(-1)
        
        if CLASSIFICATION_THRESHOLDS.get('adaptive_mode', False):
            thresholds = self._calculate_adaptive_thresholds(df_dados['target_return'], len(df_dados))
            upper_threshold = thresholds['upper']
            lower_threshold = thresholds['lower']
            logger.info(f"Adaptive thresholds calculated: UPPER={upper_threshold:.4f}, LOWER={lower_threshold:.4f}")
        else:
            upper_threshold = CLASSIFICATION_THRESHOLDS.get('fallback_upper', 0.008)
            lower_threshold = CLASSIFICATION_THRESHOLDS.get('fallback_lower', -0.008)
            logger.info(f"Using fallback thresholds: UPPER={upper_threshold}, LOWER={lower_threshold}")
        
        def classificar_movimento(retorno):
            if pd.isna(retorno):
                return np.nan
            elif retorno > upper_threshold:
                return 2
            elif retorno < lower_threshold:
                return 0
            else:
                return 1
        
        classes = df_dados['target_return'].apply(classificar_movimento)
        
        contagem_classes = classes.value_counts().sort_index()
        logger.info(f"Class distribution: LOW (0): {contagem_classes.get(0, 0)}, "
                   f"NEUTRAL (1): {contagem_classes.get(1, 0)}, HIGH (2): {contagem_classes.get(2, 0)}")
        
        return classes
    
    def _calculate_adaptive_thresholds(self, returns: pd.Series, dataset_size: int) -> dict:
        """Calculate adaptive thresholds based on dataset."""
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 50:
                logger.warning("Too few data points for adaptive calculation, using fallback")
                return {
                    'upper': CLASSIFICATION_THRESHOLDS.get('fallback_upper', 0.008),
                    'lower': CLASSIFICATION_THRESHOLDS.get('fallback_lower', -0.008)
                }
            
            if dataset_size < 1000:
                target_ratio = CLASSIFICATION_THRESHOLDS['target_class_ratio']['small_dataset']
            elif dataset_size < 5000:
                target_ratio = CLASSIFICATION_THRESHOLDS['target_class_ratio']['medium_dataset']
            else:
                target_ratio = CLASSIFICATION_THRESHOLDS['target_class_ratio']['large_dataset']
            
            upper_percentile = 100 - (target_ratio * 50)
            lower_percentile = target_ratio * 50
            
            upper_threshold = np.percentile(returns_clean, upper_percentile)
            lower_threshold = np.percentile(returns_clean, lower_percentile)
            
            upper_threshold = max(0.002, min(0.030, abs(upper_threshold)))
            lower_threshold = min(-0.002, max(-0.030, -abs(lower_threshold)))
            
            return {'upper': upper_threshold, 'lower': lower_threshold}
            
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return {
                'upper': CLASSIFICATION_THRESHOLDS.get('fallback_upper', 0.008),
                'lower': CLASSIFICATION_THRESHOLDS.get('fallback_lower', -0.008)
            }

    def _adicionar_features_tecnicas_completas(self, df_dados: pd.DataFrame, coluna_target: str = 'close') -> pd.DataFrame:
        logger.info("Adding complete technical features for automatic selection...")
        
        df_features = df_dados.copy()
        
        try:
            df_features['rsi'] = RSIIndicator(close=df_features[coluna_target], window=14).rsi()
        except:
            df_features['rsi'] = 50.0
        
        try:
            macd = MACD(close=df_features[coluna_target])
            df_features['macd'] = macd.macd()
            df_features['macd_signal'] = macd.macd_signal()
        except:
            df_features['macd'] = 0.0
            df_features['macd_signal'] = 0.0
        
        try:
            df_features['roc'] = ROCIndicator(close=df_features[coluna_target], window=14).roc()
        except:
            df_features['roc'] = 0.0
        
        try:
            if all(col in df_features.columns for col in ['high', 'low']):
                df_features['williams_r'] = WilliamsRIndicator(
                    high=df_features['high'], 
                    low=df_features['low'], 
                    close=df_features[coluna_target], 
                    lbp=14
                ).williams_r()
                df_features['atr'] = AverageTrueRange(
                    high=df_features['high'], 
                    low=df_features['low'], 
                    close=df_features[coluna_target], 
                    window=14
                ).average_true_range()
            else:
                df_features['williams_r'] = -50.0
                df_features['atr'] = df_features[coluna_target].rolling(14).std()
        except:
            df_features['williams_r'] = -50.0
            df_features['atr'] = df_features[coluna_target].rolling(14).std()
        
        try:
            df_features['volatility_rolling'] = df_features[coluna_target].pct_change().rolling(21).std() * np.sqrt(252)
            df_features['atr_normalized'] = df_features['atr'] / df_features[coluna_target]
        except:
            df_features['volatility_rolling'] = 0.1
            df_features['atr_normalized'] = 0.01
        
        try:
            if 'volume' in df_features.columns and df_features['volume'].sum() > 0:
                df_features['volume_sma'] = df_features['volume'].rolling(21).mean()
                volume_ma = df_features['volume'].rolling(21).mean()
                df_features['volume_ratio'] = df_features['volume'] / volume_ma
            else:
                df_features['volume_sma'] = 1.0
                df_features['volume_ratio'] = 1.0
        except:
            df_features['volume_sma'] = 1.0
            df_features['volume_ratio'] = 1.0
        
        try:
            df_features['price_momentum'] = df_features[coluna_target].pct_change(5)
            df_features['price_acceleration'] = df_features['price_momentum'].diff()
        except:
            df_features['price_momentum'] = 0.0
            df_features['price_acceleration'] = 0.0
        
        try:
            if isinstance(df_features.index, pd.DatetimeIndex):
                df_features['day_of_week'] = df_features.index.dayofweek / 6.0
                df_features['month'] = df_features.index.month / 12.0
            else:
                df_features['day_of_week'] = 0.5
                df_features['month'] = 0.5
        except:
            df_features['day_of_week'] = 0.5
            df_features['month'] = 0.5
        
        df_features['sentiment_score'] = self._get_sentiment_score(coluna_target)
        
        return df_features

    def _get_sentiment_score(self, simbolo):
        """
        PLACEHOLDER: Get sentiment score for symbol.
        Returns neutral (0.0) for now, prepared for future integration.
        """
        return 0.0

    def _perform_dataset_analysis(self, df_dados: pd.DataFrame):
        if not self.advanced_modules_available:
            return {'size': len(df_dados), 'features': df_dados.shape[1]}
        
        try:
            analysis = self.dataset_analyzer.comprehensive_analysis(df_dados)
            
            if ADVANCED_LOGGING.get('log_complexity_analysis', True):
                logger.info("Dataset analysis completed")
                logger.info(f"Dataset size: {analysis.get('size', len(df_dados))}")
                logger.info(f"Number of features: {analysis.get('features', df_dados.shape[1])}")
                logger.info(f"Noise level: {analysis.get('noise_level', 'unknown')}")
                logger.info(f"Stationarity: {analysis.get('stationarity', 'unknown')}")
                logger.info(f"Class balance: {analysis.get('class_balance', 'unknown')}")
            
            return analysis
        except Exception as e:
            logger.error(f"Error in dataset analysis: {e}")
            return {'size': len(df_dados), 'features': df_dados.shape[1]}

    def _perform_feature_selection(self, df_dados: pd.DataFrame, coluna_target: str, dataset_analysis: dict):
        if not self.advanced_modules_available or not FEATURE_SELECTION_CONFIG['enabled']:
            features_basicas = [coluna_target, 'rsi', 'macd', 'atr', 'volume_ratio', 'day_of_week', 'sentiment_score']
            return [f for f in features_basicas if f in df_dados.columns]
        
        try:
            dataset_size = dataset_analysis.get('size', len(df_dados))
            
            if dataset_size < FEATURE_SELECTION_CONFIG['auto_enable_threshold']:
                if ADVANCED_LOGGING.get('log_feature_selection', True):
                    logger.info("Small dataset: using optimized basic features")
                features_basicas = [coluna_target, 'rsi', 'macd', 'atr', 'volume_ratio', 'day_of_week', 'sentiment_score']
                return [f for f in features_basicas if f in df_dados.columns]
            
            available_features = [col for col in df_dados.columns if col != coluna_target]
            selected_features = self.feature_selector.select_features(
                df_dados, available_features, coluna_target, dataset_analysis
            )
            
            selected_features_with_target = [coluna_target] + selected_features
            
            if ADVANCED_LOGGING.get('log_feature_selection', True):
                logger.info("Automatic feature selection completed")
                logger.info(f"Available features: {len(available_features)}")
                logger.info(f"Selected features: {len(selected_features)}")
                logger.info(f"Final features: {selected_features_with_target}")
            
            return selected_features_with_target
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            features_basicas = [coluna_target, 'rsi', 'macd', 'atr', 'volume_ratio', 'day_of_week', 'sentiment_score']
            return [f for f in features_basicas if f in df_dados.columns]

    def _perform_data_augmentation(self, df_dados: pd.DataFrame, dataset_analysis: dict):
        if not self.advanced_modules_available or not DATA_AUGMENTATION_CONFIG['enabled']:
            return df_dados
        
        try:
            dataset_size = dataset_analysis.get('size', len(df_dados))
            
            if dataset_size >= DATA_AUGMENTATION_CONFIG['auto_enable_threshold']:
                if ADVANCED_LOGGING.get('log_augmentation_stats', True):
                    logger.info("Large dataset: augmentation disabled")
                return df_dados
            
            augmented_data = self.augmentation_engine.apply_augmentation(df_dados, dataset_analysis)
            
            if ADVANCED_LOGGING.get('log_augmentation_stats', True):
                original_size = len(df_dados)
                augmented_size = len(augmented_data)
                multiplier = augmented_size / original_size if original_size > 0 else 1
                
                logger.info("Data augmentation applied")
                logger.info(f"Original dataset: {original_size} samples")
                logger.info(f"Augmented dataset: {augmented_size} samples")
                logger.info(f"Multiplier: {multiplier:.1f}x")
            
            return augmented_data
            
        except Exception as e:
            logger.error(f"Error in data augmentation: {e}")
            return df_dados

    def _realize_analise_complexidade(self, df_dados: pd.DataFrame, num_features: int):
        if not ADVANCED_LOGGING.get('log_complexity_analysis', True):
            return
            
        dataset_size = len(df_dados)
        lookback_period = self.model_config.get('lookback_period', 15)
        effective_samples = dataset_size - lookback_period
        
        lstm_units_1 = self.model_config.get('lstm_units_1', 8)
        lstm_units_2 = self.model_config.get('lstm_units_2', 6) 
        dense_units = self.model_config.get('dense_units', 8)
        
        params_lstm1 = 2 * 4 * lstm_units_1 * (num_features + lstm_units_1 + 1)
        params_lstm2 = 2 * 4 * lstm_units_2 * (lstm_units_1 * 2 + lstm_units_2 + 1)
        params_dense1 = (lstm_units_2 * 2) * dense_units + dense_units
        params_dense2 = dense_units * 3 + 3
        
        total_params = params_lstm1 + params_lstm2 + params_dense1 + params_dense2
        
        samples_per_param = effective_samples / total_params if total_params > 0 else 0
        
        logger.info("Complexity analysis completed")
        logger.info(f"Dataset size: {dataset_size} samples")
        logger.info(f"Effective samples: {effective_samples}")
        logger.info(f"Features used: {num_features}")
        logger.info(f"Estimated parameters: {total_params:,}")
        logger.info(f"Samples/parameters ratio: {samples_per_param:.2f}")
        
        if samples_per_param < 1:
            logger.error("EXTREME RISK: More parameters than samples!")
        elif samples_per_param < 5:
            logger.warning("HIGH RISK: Few data points for parameters")
        elif samples_per_param < 10:
            logger.info("MODERATE RISK: Acceptable ratio")
        else:
            logger.info("LOW RISK: Adequate ratio")

    def treinar_avaliar_modelo_lstm_bayesian(self, df_dados_completos: pd.DataFrame, coluna_target: str, simbolo: str):
        """Train standard LSTM model. Maintained for compatibility."""
        df_dados_completos.columns = [col.lower() for col in df_dados_completos.columns]
        self.modelo_lstm_instancia = self._instanciar_modelo_lstm()
        
        if df_dados_completos.empty:
            logger.error("Complete data DataFrame is empty. Cannot train LSTM")
            return None, None, pd.DataFrame(), {}

        if not isinstance(df_dados_completos.index, pd.DatetimeIndex):
            if 'date' in df_dados_completos.columns:
                 df_dados_completos['date'] = pd.to_datetime(df_dados_completos['date'])
                 df_dados_completos.set_index('date', inplace=True)
            elif df_dados_completos.index.name == 'date' or df_dados_completos.index.name == 'index': 
                try:
                    df_dados_completos.index = pd.to_datetime(df_dados_completos.index)
                except Exception as e:
                    logger.error(f"Could not convert index to DatetimeIndex for LSTM: {e}")
                    return None, None, pd.DataFrame(), {}
            else:
                logger.error("DataFrame index is not DatetimeIndex and no 'date' column for LSTM")
                return None, None, pd.DataFrame(), {}

        logger.info("Starting intelligent pipeline")
        
        df_dados_completos = self._adicionar_features_tecnicas_completas(df_dados_completos, coluna_target)
        
        dataset_analysis = self._perform_dataset_analysis(df_dados_completos)
        
        selected_features = self._perform_feature_selection(df_dados_completos, coluna_target, dataset_analysis)
        
        logger.info("Creating target variable as directional classes...")
        classes_direcionais = self._criar_classes_direcionais(df_dados_completos, coluna_target)
        df_dados_completos['target_classes'] = classes_direcionais

        linhas_antes = len(df_dados_completos)
        df_dados_completos.dropna(inplace=True)
        logger.info(f"Removed {linhas_antes - len(df_dados_completos)} rows with NaNs")

        df_dados_tratados = df_dados_completos.copy()
        for col in df_dados_tratados.columns:
            if col not in ['target_classes', 'target_return'] and df_dados_tratados[col].isnull().any():
                df_dados_tratados[col] = df_dados_tratados[col].fillna(method='ffill').fillna(method='bfill')
        
        df_dados_tratados.dropna(subset=['target_classes'] + selected_features, inplace=True)

        if len(df_dados_tratados) < self.model_config.get('lookback_period', 15) * 2: 
            logger.error(f"Insufficient data: {len(df_dados_tratados)} rows")
            return None, None, pd.DataFrame(), {}

        df_augmented = self._perform_data_augmentation(df_dados_tratados, dataset_analysis)
        
        self._realize_analise_complexidade(df_augmented, len(selected_features))

        X_treino, y_treino, X_teste, y_teste, indices_treino, indices_teste, y_original_treino, y_original_teste, _, self.colunas_features_usadas_lstm = \
            self.modelo_lstm_instancia.preparar_dados_para_treino_teste_classificacao(
                df_augmented,
                selected_features,
                coluna_target_classes='target_classes',
                train_split_ratio=LSTM_TRAIN_TEST_SPLIT
            )

        if X_treino is None or X_teste is None or X_treino.size == 0 or X_teste.size == 0:
            logger.error("Failed to prepare data for LSTM train/test")
            return None, None, pd.DataFrame(), {}
            
        input_shape = (X_treino.shape[1], X_treino.shape[2])
        self.modelo_lstm_instancia.construir_modelo_classificacao(input_shape=input_shape, num_classes=3)

        logger.info(f"Starting intelligent LSTM training")
        logger.info(f"X_train: {X_treino.shape}, y_train: {y_treino.shape}")
        logger.info(f"X_test: {X_teste.shape}, y_test: {y_teste.shape}")
        logger.info(f"Selected features: {len(selected_features)}")

        self.modelo_lstm_instancia.treinar_modelo_classificacao(
            X_treino, y_treino, X_teste, y_teste, 
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            optimizer='adam'
        )

        predicoes_teste_probabilities = self.modelo_lstm_instancia.prever_classificacao(X_teste)
        
        if predicoes_teste_probabilities is None:
            logger.error("Failed to get predictions from LSTM classification model")
            return self.modelo_lstm_instancia.model, None, pd.DataFrame(), {}

        predicoes_classes = np.argmax(predicoes_teste_probabilities, axis=1)
        classes_reais = np.argmax(y_teste, axis=1)

        df_comparacao = pd.DataFrame({
            'Data': indices_teste, 
            'Classe_Real': classes_reais, 
            'Classe_Prevista': predicoes_classes,
            'Prob_Baixa': predicoes_teste_probabilities[:, 0],
            'Prob_Neutro': predicoes_teste_probabilities[:, 1], 
            'Prob_Alta': predicoes_teste_probabilities[:, 2]
        })
        df_comparacao.sort_values(by='Data', inplace=True)

        metricas = ModelEvaluator.calcular_metricas_classificacao_avancadas(
            y_teste, predicoes_teste_probabilities, 
            self.modelo_lstm_instancia.training_history if hasattr(self.modelo_lstm_instancia, 'training_history') else None
        )
        
        logger.info(f"Intelligent LSTM model trained")
        logger.info(f"Selected features: {len(selected_features)}")
        logger.info(f"Accuracy: {metricas.get('accuracy', 'N/A'):.4f}")
        logger.info(f"F1-Score: {metricas.get('f1_score_weighted', 'N/A'):.4f}")
        logger.info("Intelligent pipeline completed")
        
        if self.modelo_lstm_instancia and self.modelo_lstm_instancia.model:
            try:
                path_modelo = os.path.join(self.MODEL_SAVE_PATH, f"lstm_classification_model_{simbolo}.h5")
                path_scaler = os.path.join(self.MODEL_SAVE_PATH, f"lstm_classification_scaler_{simbolo}.joblib")
                
                save_model(self.modelo_lstm_instancia.model, path_modelo)
                joblib.dump(self.modelo_lstm_instancia.scaler_features, path_scaler) 
                
                logger.info(f"Intelligent LSTM model for {simbolo} saved")
            except Exception as e:
                logger.error(f"Error saving intelligent LSTM model for {simbolo}: {e}")
        
        return self.modelo_lstm_instancia.model, None, df_comparacao, metricas

    def treinar_avaliar_modelo_lstm_bayesian(self, df_dados_completos: pd.DataFrame, coluna_target: str, simbolo: str):
        """
        Train Bayesian LSTM model with uncertainty quantification.
        Uses same intelligent pipeline but with Bayesian architecture.
        """
        df_dados_completos.columns = [col.lower() for col in df_dados_completos.columns]
        self.modelo_lstm_instancia = self._instanciar_modelo_lstm()
        
        if df_dados_completos.empty:
            logger.error("Complete data DataFrame is empty. Cannot train Bayesian LSTM")
            return None, None, pd.DataFrame(), {}
    
        if not isinstance(df_dados_completos.index, pd.DatetimeIndex):
            if 'date' in df_dados_completos.columns:
                 df_dados_completos['date'] = pd.to_datetime(df_dados_completos['date'])
                 df_dados_completos.set_index('date', inplace=True)
            elif df_dados_completos.index.name == 'date' or df_dados_completos.index.name == 'index': 
                try:
                    df_dados_completos.index = pd.to_datetime(df_dados_completos.index)
                except Exception as e:
                    logger.error(f"Could not convert index to DatetimeIndex for Bayesian LSTM: {e}")
                    return None, None, pd.DataFrame(), {}
            else:
                logger.error("DataFrame index is not DatetimeIndex and no 'date' column for Bayesian LSTM")
                return None, None, pd.DataFrame(), {}
    
        logger.info("Starting Bayesian LSTM pipeline with uncertainty quantification")
        
        df_dados_completos = self._adicionar_features_tecnicas_completas(df_dados_completos, coluna_target)
        dataset_analysis = self._perform_dataset_analysis(df_dados_completos)
        selected_features = self._perform_feature_selection(df_dados_completos, coluna_target, dataset_analysis)
        
        logger.info("Creating target variable as directional classes...")
        classes_direcionais = self._criar_classes_direcionais(df_dados_completos, coluna_target)
        df_dados_completos['target_classes'] = classes_direcionais
    
        linhas_antes = len(df_dados_completos)
        df_dados_completos.dropna(inplace=True)
        logger.info(f"Removed {linhas_antes - len(df_dados_completos)} rows with NaNs")
    
        df_dados_tratados = df_dados_completos.copy()
        for col in df_dados_tratados.columns:
            if col not in ['target_classes', 'target_return'] and df_dados_tratados[col].isnull().any():
                df_dados_tratados[col] = df_dados_tratados[col].fillna(method='ffill').fillna(method='bfill')
        
        df_dados_tratados.dropna(subset=['target_classes'] + selected_features, inplace=True)
    
        if len(df_dados_tratados) < self.model_config.get('lookback_period', 15) * 2: 
            logger.error(f"Insufficient data: {len(df_dados_tratados)} rows")
            return None, None, pd.DataFrame(), {}
    
        df_augmented = self._perform_data_augmentation(df_dados_tratados, dataset_analysis)
        self._realize_analise_complexidade(df_augmented, len(selected_features))
    
        X_treino, y_treino, X_teste, y_teste, indices_treino, indices_teste, y_original_treino, y_original_teste, _, self.colunas_features_usadas_lstm = \
            self.modelo_lstm_instancia.preparar_dados_para_treino_teste_classificacao(
                df_augmented,
                selected_features,
                coluna_target_classes='target_classes',
                train_split_ratio=LSTM_TRAIN_TEST_SPLIT
            )
    
        if X_treino is None or X_teste is None or X_treino.size == 0 or X_teste.size == 0:
            logger.error("Failed to prepare data for Bayesian LSTM train/test")
            return None, None, pd.DataFrame(), {}
            
        input_shape = (X_treino.shape[1], X_treino.shape[2])
        
        self.modelo_lstm_instancia.construir_modelo_classificacao_bayesian(input_shape=input_shape, num_classes=3)
    
        logger.info(f"Starting Bayesian LSTM training with Monte Carlo Dropout")
        logger.info(f"X_train: {X_treino.shape}, y_train: {y_treino.shape}")
        logger.info(f"X_test: {X_teste.shape}, y_test: {y_teste.shape}")
        logger.info(f"Selected features: {len(selected_features)}")
    
        self.modelo_lstm_instancia.bayesian_model.compile(
            optimizer='adam',
            loss=self.modelo_lstm_instancia._preparar_focal_loss_e_class_weights(np.argmax(y_treino, axis=1))[0],
            metrics=['accuracy']
        )
        
        callbacks = self.modelo_lstm_instancia._configurar_callbacks_avancados()
        
        history = self.modelo_lstm_instancia.bayesian_model.fit(
            X_treino, y_treino,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_data=(X_teste, y_teste),
            callbacks=callbacks,
            verbose=ADVANCED_TRAINING_CONFIG['verbose_training'],
            class_weight=self.modelo_lstm_instancia.class_weights if self.modelo_lstm_instancia.class_weights else None
        )
        
        self.modelo_lstm_instancia.training_history = history
    
        uncertainty_results = self.modelo_lstm_instancia.prever_classificacao_com_incerteza(X_teste)
        
        if uncertainty_results is None:
            logger.error("Failed to get uncertainty predictions from Bayesian LSTM")
            return self.modelo_lstm_instancia.bayesian_model, None, pd.DataFrame(), {}
    
        predicoes_teste_probabilities = uncertainty_results['predictions']
        
        logger.info("Otimizando thresholds para dados desbalanceados...")
        optimal_thresholds, best_f1 = self._optimize_classification_thresholds(y_teste, predicoes_teste_probabilities)
        logger.info(f"Thresholds otimizados: BAIXA={optimal_thresholds[0]:.3f}, NEUTRO={optimal_thresholds[1]:.3f}, ALTA={optimal_thresholds[2]:.3f}")
        logger.info(f"F1-score otimizado: {best_f1:.4f}")
        
        predicoes_classes = self._apply_custom_thresholds(predicoes_teste_probabilities, optimal_thresholds)
        self.optimal_thresholds = optimal_thresholds
        
        classes_reais = np.argmax(y_teste, axis=1)
    
        df_comparacao = pd.DataFrame({
            'Data': indices_teste, 
            'Classe_Real': classes_reais, 
            'Classe_Prevista': predicoes_classes,
            'Prob_Baixa': predicoes_teste_probabilities[:, 0],
            'Prob_Neutro': predicoes_teste_probabilities[:, 1], 
            'Prob_Alta': predicoes_teste_probabilities[:, 2],
            'Epistemic_Uncertainty': uncertainty_results['total_uncertainty'],
            'Confidence_Score': uncertainty_results['confidence_scores']
        })
        df_comparacao.sort_values(by='Data', inplace=True)
    
        metricas = ModelEvaluator.calcular_metricas_classificacao_avancadas(
            y_teste, predicoes_teste_probabilities, 
            self.modelo_lstm_instancia.training_history
        )
        
        metricas['uncertainty_metrics'] = {
            'mean_epistemic_uncertainty': float(np.mean(uncertainty_results['epistemic_uncertainty'])),
            'mean_confidence': float(np.mean(uncertainty_results['confidence_scores'])),
            'low_confidence_ratio': float(np.mean(uncertainty_results['confidence_scores'] < 0.5))
        }
        
        metricas['threshold_optimization'] = {
            'optimal_thresholds': optimal_thresholds,
            'optimized_f1_score': best_f1,
            'optimization_applied': True
        }
        
        logger.info(f"Bayesian LSTM model trained with uncertainty quantification")
        logger.info(f"Selected features: {len(selected_features)}")
        logger.info(f"Accuracy: {metricas.get('accuracy', 'N/A'):.4f}")
        logger.info(f"F1-Score: {metricas.get('f1_score_weighted', 'N/A'):.4f}")
        logger.info(f"Mean confidence: {metricas['uncertainty_metrics']['mean_confidence']:.4f}")
        logger.info("Bayesian pipeline completed")
        
        if self.modelo_lstm_instancia and self.modelo_lstm_instancia.bayesian_model:
            try:
                path_modelo = os.path.join(self.MODEL_SAVE_PATH, f"bayesian_lstm_model_{simbolo}.h5")
                path_scaler = os.path.join(self.MODEL_SAVE_PATH, f"bayesian_lstm_scaler_{simbolo}.joblib")
                
                save_model(self.modelo_lstm_instancia.bayesian_model, path_modelo)
                joblib.dump(self.modelo_lstm_instancia.scaler_features, path_scaler)
                
                if hasattr(self.modelo_lstm_instancia, 'calibration_data'):
                    path_calibration = os.path.join(self.MODEL_SAVE_PATH, f"bayesian_lstm_calibration_{simbolo}.joblib")
                    joblib.dump(self.modelo_lstm_instancia.calibration_data, path_calibration)
                
                if hasattr(self, 'optimal_thresholds'):
                    path_thresholds = os.path.join(self.MODEL_SAVE_PATH, f"optimal_thresholds_{simbolo}.joblib")
                    joblib.dump(self.optimal_thresholds, path_thresholds)
                    logger.info(f"Thresholds otimizados salvos para {simbolo}")
                
                logger.info(f"Bayesian LSTM model for {simbolo} saved")
            except Exception as e:
                logger.error(f"Error saving Bayesian LSTM model for {simbolo}: {e}")
        
        return self.modelo_lstm_instancia.bayesian_model, None, df_comparacao, metricas

    def _criar_ensemble_monte_carlo_lstm(self, df_dados_recentes: pd.DataFrame, simbolo: str):
        """
        Create ensemble combining LSTM predictions with Monte Carlo simulations.
        
        Args:
            df_dados_recentes: Recent data with features
            simbolo: Symbol being predicted
            
        Returns:
            dict: Ensemble predictions with weighted voting
        """
        if not MONTE_CARLO_ENSEMBLE_CONFIG.get('enabled', True):
            logger.info("Monte Carlo ensemble integration disabled")
            return None
        
        try:
            lstm_result = self.prever_proximos_passos_lstm(df_dados_recentes, num_passos=1)
            
            if lstm_result is None:
                logger.error("Failed to get LSTM predictions for ensemble")
                return None
            
            num_simulations = MONTE_CARLO_ENSEMBLE_CONFIG.get('num_simulations', 1000)
            dias_simulacao = 5
            
            mc_results = self.monte_carlo_simulator.run_simulation(
                df_dados_recentes, 
                dias_simulacao, 
                num_simulations
            )
            
            if mc_results is None:
                logger.error("Failed to run Monte Carlo simulation")
                return None
            
            mc_stats = self.monte_carlo_simulator.calcular_estatisticas_simulacao(mc_results)
            
            current_price = df_dados_recentes['close'].iloc[-1]
            final_prices = mc_results[-1, :]
            
            returns = (final_prices - current_price) / current_price
            
            thresholds = self._calculate_adaptive_thresholds(pd.Series(returns), len(df_dados_recentes))
            upper_threshold = thresholds['upper']
            lower_threshold = thresholds['lower']
            
            low_count = np.sum(returns < lower_threshold)
            high_count = np.sum(returns > upper_threshold)
            neutral_count = num_simulations - low_count - high_count
            
            mc_probabilities = {
                'BAIXA': low_count / num_simulations,
                'NEUTRO': neutral_count / num_simulations,
                'ALTA': high_count / num_simulations
            }
            
            lstm_weight = MONTE_CARLO_ENSEMBLE_CONFIG.get('lstm_weight', 0.7)
            mc_weight = MONTE_CARLO_ENSEMBLE_CONFIG.get('monte_carlo_weight', 0.3)
            
            ensemble_probabilities = {
                'BAIXA': lstm_weight * lstm_result['probabilidades']['BAIXA'] + mc_weight * mc_probabilities['BAIXA'],
                'NEUTRO': lstm_weight * lstm_result['probabilidades']['NEUTRO'] + mc_weight * mc_probabilities['NEUTRO'],
                'ALTA': lstm_weight * lstm_result['probabilidades']['ALTA'] + mc_weight * mc_probabilities['ALTA']
            }
            
            final_class = max(ensemble_probabilities, key=ensemble_probabilities.get)
            
            confidence_threshold = MONTE_CARLO_ENSEMBLE_CONFIG.get('confidence_threshold', 0.6)
            max_prob = max(ensemble_probabilities.values())
            
            ensemble_result = {
                'classe_predita': final_class,
                'probabilidades_ensemble': ensemble_probabilities,
                'probabilidades_lstm': lstm_result['probabilidades'],
                'probabilidades_monte_carlo': mc_probabilities,
                'confianca': max_prob,
                'confianca_suficiente': max_prob >= confidence_threshold,
                'metodo_integracao': MONTE_CARLO_ENSEMBLE_CONFIG.get('integration_method', 'weighted_average'),
                'pesos': {
                    'lstm': lstm_weight,
                    'monte_carlo': mc_weight
                },
                'estatisticas_monte_carlo': mc_stats
            }
            
            logger.info(f"Monte Carlo ensemble created: {final_class} (confidence: {max_prob:.3f})")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error creating Monte Carlo LSTM ensemble: {e}")
            return None

    def prever_com_ensemble_completo(self, df_dados_recentes: pd.DataFrame, simbolo: str, num_passos: int = 1):
        """
        Make predictions using complete ensemble: Bayesian LSTM + Monte Carlo Simulator.
        Includes full uncertainty quantification and confidence intervals.
        
        Args:
            df_dados_recentes: Recent data with features
            simbolo: Symbol being predicted
            num_passos: Number of steps ahead (default 1)
            
        Returns:
            dict: Complete predictions with uncertainty and ensemble results
        """
        try:
            if self.colunas_features_usadas_lstm is None:
                logger.error("LSTM feature columns not stored")
                return None
            
            df_features_recentes = df_dados_recentes[self.colunas_features_usadas_lstm].copy()
            df_features_recentes_filled = df_features_recentes.fillna(method='ffill').fillna(method='bfill')
            df_features_recentes_filled.dropna(inplace=True)

            if len(df_features_recentes_filled) < self.modelo_lstm_instancia.lookback_period:
                logger.error(f"Insufficient recent data ({len(df_features_recentes_filled)} rows)")
                return None

            ultimos_dados = df_features_recentes_filled.tail(self.modelo_lstm_instancia.lookback_period)        
            ultimos_dados_scaled = self.modelo_lstm_instancia.scaler_features.transform(ultimos_dados)
            input_sequence = np.array([ultimos_dados_scaled])

            if BAYESIAN_LSTM_CONFIG.get('enabled', True) and hasattr(self.modelo_lstm_instancia, 'prever_classificacao_com_incerteza'):
                bayesian_result = self.modelo_lstm_instancia.prever_classificacao_com_incerteza(input_sequence)
                
                if bayesian_result is None:
                    logger.warning("Bayesian prediction failed, falling back to standard")
                    predicao_probabilities = self.modelo_lstm_instancia.prever_classificacao(input_sequence)
                    bayesian_result = {
                        'predictions': predicao_probabilities,
                        'epistemic_uncertainty': np.zeros_like(predicao_probabilities),
                        'confidence_scores': np.ones(len(predicao_probabilities)),
                        'method': 'standard_fallback'
                    }
            else:
                predicao_probabilities = self.modelo_lstm_instancia.prever_classificacao(input_sequence)
                bayesian_result = {
                    'predictions': predicao_probabilities,
                    'epistemic_uncertainty': np.zeros_like(predicao_probabilities),
                    'confidence_scores': np.ones(len(predicao_probabilities)),
                    'method': 'standard'
                }
            
            ensemble_result = self._criar_ensemble_monte_carlo_lstm(df_dados_recentes, simbolo)
            
            final_probabilities = bayesian_result['predictions'][0]
            classe_predita = np.argmax(final_probabilities)
            mapeamento_classes = {0: "BAIXA", 1: "NEUTRO", 2: "ALTA"}
            
            complete_result = {
                'classe_predita': mapeamento_classes[classe_predita],
                'probabilidades': {
                    'BAIXA': float(final_probabilities[0]),
                    'NEUTRO': float(final_probabilities[1]), 
                    'ALTA': float(final_probabilities[2])
                },
                'uncertainty_quantification': {
                    'epistemic_uncertainty': float(bayesian_result.get('epistemic_uncertainty', [0])[0]),
                    'confidence_score': float(bayesian_result.get('confidence_scores', [1])[0]),
                    'prediction_intervals': bayesian_result.get('prediction_intervals', None),
                    'method': bayesian_result.get('method', 'unknown')
                }
            }
            
            if ensemble_result is not None:
                complete_result['ensemble'] = ensemble_result
                
                if ensemble_result.get('confianca_suficiente', False):
                    complete_result['classe_predita'] = ensemble_result['classe_predita']
                    complete_result['probabilidades'] = ensemble_result['probabilidades_ensemble']
                    complete_result['metodo_final'] = 'ensemble'
                else:
                    complete_result['metodo_final'] = 'bayesian_lstm'
            else:
                complete_result['metodo_final'] = 'bayesian_lstm_only'
            
            complete_result['sentiment'] = {
                'score': self._get_sentiment_score(simbolo),
                'impact': 0.0,
                'source': 'placeholder'
            }
            
            logger.info(f"Complete ensemble prediction: {complete_result['classe_predita']} "
                       f"(confidence: {complete_result['uncertainty_quantification']['confidence_score']:.3f})")
            
            return complete_result
            
        except Exception as e:
            logger.error(f"Error in complete ensemble prediction: {e}")
            return None

    def prever_proximos_passos_lstm(self, df_dados_recentes_com_features: pd.DataFrame, num_passos: int = 1):
        """
        Predict next steps using LSTM. Enhanced with Bayesian capabilities if available.
        """
        if self.modelo_lstm_instancia is None or (self.modelo_lstm_instancia.model is None and self.modelo_lstm_instancia.bayesian_model is None):
            logger.error("No LSTM model trained or available")
            return None
        if self.colunas_features_usadas_lstm is None:
            logger.error("LSTM feature column names not stored")
            return None
        
        df_features_recentes = df_dados_recentes_com_features[self.colunas_features_usadas_lstm].copy()
        df_features_recentes_filled = df_features_recentes.fillna(method='ffill').fillna(method='bfill')
        df_features_recentes_filled.dropna(inplace=True)
    
        if len(df_features_recentes_filled) < self.modelo_lstm_instancia.lookback_period:
            logger.error(f"Insufficient recent data ({len(df_features_recentes_filled)} rows)")
            return None
    
        ultimos_dados = df_features_recentes_filled.tail(self.modelo_lstm_instancia.lookback_period)        
        ultimos_dados_scaled = self.modelo_lstm_instancia.scaler_features.transform(ultimos_dados)
        input_sequence = np.array([ultimos_dados_scaled])
    
        if (BAYESIAN_LSTM_CONFIG.get('enabled', True) and 
            hasattr(self.modelo_lstm_instancia, 'prever_classificacao_com_incerteza') and
            self.modelo_lstm_instancia.bayesian_model is not None):
            
            uncertainty_result = self.modelo_lstm_instancia.prever_classificacao_com_incerteza(input_sequence)
            if uncertainty_result is not None:
                predicao_probabilities = uncertainty_result['predictions']
                confidence = uncertainty_result['confidence_scores'][0]
                
                if hasattr(self, 'optimal_thresholds'):
                    classe_predita = self._apply_custom_thresholds([predicao_probabilities[0]], self.optimal_thresholds)[0]
                    logger.info(f"Predição com thresholds otimizados: {classe_predita}")
                else:
                    classe_predita = np.argmax(predicao_probabilities[0])
                    logger.info("Usando threshold padrão - thresholds não otimizados")
                
                mapeamento_classes = {0: "BAIXA", 1: "NEUTRO", 2: "ALTA"}
                
                resultado = {
                    'classe_predita': mapeamento_classes[classe_predita],
                    'probabilidades': {
                        'BAIXA': float(predicao_probabilities[0][0]),
                        'NEUTRO': float(predicao_probabilities[0][1]), 
                        'ALTA': float(predicao_probabilities[0][2])
                    },
                    'confianca': float(confidence),
                    'metodo': 'bayesian_lstm',
                    'thresholds_otimizados': hasattr(self, 'optimal_thresholds')
                }
                
                logger.info(f"Bayesian LSTM prediction: {resultado}")
                return resultado
        
        predicao_probabilities = self.modelo_lstm_instancia.prever_classificacao(input_sequence)
        
        if hasattr(self, 'optimal_thresholds'):
            classe_predita = self._apply_custom_thresholds([predicao_probabilities[0]], self.optimal_thresholds)[0]
            logger.info(f"Predição standard com thresholds otimizados: {classe_predita}")
        else:
            classe_predita = np.argmax(predicao_probabilities[0])
            logger.info("Usando threshold padrão - thresholds não otimizados")
        
        mapeamento_classes = {0: "BAIXA", 1: "NEUTRO", 2: "ALTA"}
        resultado = {
            'classe_predita': mapeamento_classes[classe_predita],
            'probabilidades': {
                'BAIXA': float(predicao_probabilities[0][0]),
                'NEUTRO': float(predicao_probabilities[0][1]), 
                'ALTA': float(predicao_probabilities[0][2])
            },
            'confianca': float(np.max(predicao_probabilities[0])),
            'metodo': 'standard_lstm',
            'thresholds_otimizados': hasattr(self, 'optimal_thresholds')
        }
        
        logger.info(f"LSTM prediction: {resultado}")
        return resultado
    
    def carregar_modelo_lstm(self, simbolo: str) -> bool:
        """Load LSTM model. Attempts to load Bayesian first, then standard."""
        path_bayesian_modelo = os.path.join(self.MODEL_SAVE_PATH, f"bayesian_lstm_model_{simbolo}.h5")
        path_bayesian_scaler = os.path.join(self.MODEL_SAVE_PATH, f"bayesian_lstm_scaler_{simbolo}.joblib")
        
        if os.path.exists(path_bayesian_modelo) and os.path.exists(path_bayesian_scaler):
            try:
                logger.info(f"Loading Bayesian LSTM model for {simbolo}")
                if not self.modelo_lstm_instancia:
                    self.modelo_lstm_instancia = self._instanciar_modelo_lstm()
    
                modelo_carregado = load_model(path_bayesian_modelo)
                scaler_carregado = joblib.load(path_bayesian_scaler)
                
                self.modelo_lstm_instancia.bayesian_model = modelo_carregado
                self.modelo_lstm_instancia.scaler_features = scaler_carregado
                
                path_calibration = os.path.join(self.MODEL_SAVE_PATH, f"bayesian_lstm_calibration_{simbolo}.joblib")
                if os.path.exists(path_calibration):
                    self.modelo_lstm_instancia.calibration_data = joblib.load(path_calibration)
                
                path_thresholds = os.path.join(self.MODEL_SAVE_PATH, f"optimal_thresholds_{simbolo}.joblib")
                if os.path.exists(path_thresholds):
                    self.optimal_thresholds = joblib.load(path_thresholds)
                    logger.info(f"Thresholds otimizados carregados para {simbolo}: {self.optimal_thresholds}")
                else:
                    logger.info(f"Thresholds otimizados não encontrados para {simbolo}")
                
                logger.info(f"Bayesian LSTM model for {simbolo} loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading Bayesian LSTM model for {simbolo}: {e}")
        
        path_modelo = os.path.join(self.MODEL_SAVE_PATH, f"lstm_classification_model_{simbolo}.h5")
        path_scaler = os.path.join(self.MODEL_SAVE_PATH, f"lstm_classification_scaler_{simbolo}.joblib")
    
        if os.path.exists(path_modelo) and os.path.exists(path_scaler):
            try:
                logger.info(f"Loading standard LSTM model for {simbolo}")
                if not self.modelo_lstm_instancia:
                    self.modelo_lstm_instancia = self._instanciar_modelo_lstm()
    
                modelo_carregado = load_model(path_modelo)
                scaler_carregado = joblib.load(path_scaler)
                
                self.modelo_lstm_instancia.model = modelo_carregado
                self.modelo_lstm_instancia.scaler_features = scaler_carregado
                
                path_thresholds = os.path.join(self.MODEL_SAVE_PATH, f"optimal_thresholds_{simbolo}.joblib")
                if os.path.exists(path_thresholds):
                    self.optimal_thresholds = joblib.load(path_thresholds)
                    logger.info(f"Thresholds otimizados carregados para {simbolo}: {self.optimal_thresholds}")
                else:
                    logger.info(f"Thresholds otimizados não encontrados para {simbolo}")
                
                logger.info(f"Standard LSTM model for {simbolo} loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading standard LSTM model for {simbolo}: {e}")
                return False
        
        logger.warning(f"No LSTM model found for {simbolo}")
        return False

    def carregar_modelo_prophet(self, simbolo: str) -> bool:
        path_modelo_json = os.path.join(self.MODEL_SAVE_PATH, f"prophet_model_{simbolo}.json")

        if os.path.exists(path_modelo_json):
            try:
                logger.info(f"Loading Prophet model for {simbolo}")
                if not self.modelo_prophet_instancia:
                    self.modelo_prophet_instancia = self._instanciar_modelo_prophet()

                with open(path_modelo_json, 'r') as fin:
                    self.modelo_prophet_instancia.model = model_from_json(fin.read())
                
                logger.info(f"Prophet model for {simbolo} loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading Prophet model for {simbolo}: {e}")
                return False
        logger.warning(f"No Prophet model found for {simbolo}")
        return False

    def treinar_avaliar_modelo_prophet(self, df_dados_completos: pd.DataFrame, simbolo: str, coluna_data_prophet: str = 'Date', coluna_target_prophet: str = 'Close', colunas_regressores: list = None, periodos_previsao: int = 30, config_prophet: dict = None):
        df_dados_completos.columns = [col.lower() for col in df_dados_completos.columns]
        self.modelo_prophet_instancia = self._instanciar_modelo_prophet(config_prophet_especifica=config_prophet)

        if df_dados_completos.empty:
            logger.error("Empty DataFrame for Prophet")
            return None, pd.DataFrame(), pd.DataFrame()

        if 'rsi' not in df_dados_completos.columns:
            logger.info("Adding technical indicators for Prophet...")
            df_dados_completos['rsi'] = RSIIndicator(close=df_dados_completos['close'], window=14).rsi()
            macd = MACD(close=df_dados_completos['close'])
            df_dados_completos['macd'] = macd.macd()
            df_dados_completos['macd_signal'] = macd.macd_signal()
            df_dados_completos['atr'] = AverageTrueRange(high=df_dados_completos['high'], low=df_dados_completos['low'], close=df_dados_completos['close'], window=14).average_true_range()

            try:
                if hasattr(self, 'data_controller') and self.data_controller:
                    df_ibov = self.data_controller.buscar_dados_historicos('^BVSP', periodo="5y").dataframe
                    df_dados_completos['ibov_return'] = df_ibov['close'].pct_change()
                else:
                    df_dados_completos['ibov_return'] = 0
            except Exception as e:
                logger.warning(f"Error fetching IBOVESPA: {e}")
                df_dados_completos['ibov_return'] = 0

            df_dados_completos.dropna(inplace=True)

        colunas_regressores_prophet = ['rsi', 'macd', 'macd_signal', 'atr', 'ibov_return']
        colunas_regressores_prophet = [col for col in colunas_regressores_prophet if col in df_dados_completos.columns]
        
        df_historico_prophet = self.modelo_prophet_instancia.preparar_dados_prophet(
            df_dados_completos,
            coluna_data=coluna_data_prophet,
            coluna_target=coluna_target_prophet,
            colunas_regressores=colunas_regressores_prophet
        )

        if df_historico_prophet.empty:
            logger.error("Failed to prepare data for Prophet")
            return None, pd.DataFrame(), pd.DataFrame()

        tamanho_treino = len(df_historico_prophet) - periodos_previsao
        if tamanho_treino <= 0:
             logger.warning(f"Insufficient data for Prophet test")
             df_treino = df_historico_prophet
             df_teste_para_comparacao = pd.DataFrame()
        elif tamanho_treino < 10 :
             logger.warning(f"Few training data points ({tamanho_treino})")
             df_treino = df_historico_prophet
             df_teste_para_comparacao = pd.DataFrame()
             periodos_previsao_reais = periodos_previsao 
        else:
            df_treino = df_historico_prophet.iloc[:tamanho_treino]
            df_teste_para_comparacao = df_historico_prophet.iloc[tamanho_treino:]
            periodos_previsao_reais = len(df_teste_para_comparacao)

        self.modelo_prophet_instancia.treinar_modelo(df_treino)

        if self.modelo_prophet_instancia.model is None:
            logger.error("Failed to train Prophet")
            return None, pd.DataFrame(), pd.DataFrame()

        df_regressores_futuros = None
        if self.modelo_prophet_instancia.colunas_regressores:
            if not df_teste_para_comparacao.empty:
                df_regressores_futuros = df_teste_para_comparacao[['ds'] + self.modelo_prophet_instancia.colunas_regressores]
            else: 
                last_date = df_historico_prophet['ds'].max()
                future_dates = pd.date_range(start=last_date, periods=periodos_previsao + 1, freq='B')[1:]
                df_regressores_futuros = pd.DataFrame({'ds': future_dates})
                
                ultimos_valores = df_dados_completos[self.modelo_prophet_instancia.colunas_regressores].iloc[-1]
                
                for regressor, valor in ultimos_valores.items():
                    df_regressores_futuros[regressor] = valor
                
                logger.info(f"Future regressors DataFrame created: {ultimos_valores.to_dict()}")

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

        logger.info(f"Prophet model trained")
        
        if self.modelo_prophet_instancia and self.modelo_prophet_instancia.model:
            try:
                path_modelo_json = os.path.join(self.MODEL_SAVE_PATH, f"prophet_model_{simbolo}.json")
                with open(path_modelo_json, 'w') as fout:
                    fout.write(model_to_json(self.modelo_prophet_instancia.model))
                logger.info(f"Prophet model for {simbolo} saved")
            except Exception as e:
                logger.error(f"Error saving Prophet for {simbolo}: {e}")
        
        return self.modelo_prophet_instancia.model, df_comparacao_final, df_historico_prophet

    def _optimize_classification_thresholds(self, y_true, y_pred_probs):
        """Otimiza thresholds para classificação multiclasse desbalanceada."""
        from sklearn.metrics import f1_score
        
        best_f1 = 0
        best_thresholds = [0.33, 0.33, 0.33]
        
        for t_baixa in [0.4, 0.5, 0.6, 0.7]:
            for t_neutro in [0.3, 0.4, 0.5, 0.6]:
                for t_alta in [0.4, 0.5, 0.6, 0.7]:
                    thresholds = [t_baixa, t_neutro, t_alta]
                    y_pred_custom = self._apply_custom_thresholds(y_pred_probs, thresholds)
                    f1 = f1_score(np.argmax(y_true, axis=1), y_pred_custom, average='weighted')
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresholds = thresholds
        
        return best_thresholds, best_f1
    
    def _apply_custom_thresholds(self, probabilities, thresholds):
        """Aplica thresholds customizados para cada classe."""
        predictions = []
        for probs in probabilities:
            if probs[0] > thresholds[0]:
                predictions.append(0)
            elif probs[2] > thresholds[2]:
                predictions.append(2)
            else:
                predictions.append(1)
        return np.array(predictions)