import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Layer
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow as tf
from utils.logger import logger
from config.settings import (
    FOCAL_LOSS_CONFIG, 
    CLASS_WEIGHTING_CONFIG, 
    ADVANCED_TRAINING_CONFIG,
    ADVANCED_LOGGING,
    LSTM_LOOKBACK_PERIOD,
    LSTM_DROPOUT_RATE,
    LSTM_L2_REGULARIZATION,
    LSTM_ACTIVATION_DENSE,
    LSTM_DYNAMIC_CONFIG,
    HYPERPARAMETER_TUNING_CONFIG,
    BAYESIAN_LSTM_CONFIG,
    UNCERTAINTY_QUANTIFICATION_CONFIG
)

try:
    from models.prediction.advanced.losses import (
        CategoricalFocalLoss,
        categorical_focal_loss,
        calculate_class_weights_smart,
        get_recommended_focal_parameters
    )
    ADVANCED_STRATEGIES_AVAILABLE = True
    logger.info("Advanced loss strategies loaded successfully")
except ImportError as e:
    ADVANCED_STRATEGIES_AVAILABLE = False
    logger.warning(f"Advanced strategies not available: {e}. Using standard training")


class MCDropout(Layer):
    """Monte Carlo Dropout layer that stays active during inference."""
    def __init__(self, rate, **kwargs):
        super(MCDropout, self).__init__(**kwargs)
        self.rate = rate
        
    def call(self, inputs, training=None):
        if training is None:
            training = True
        return tf.nn.dropout(inputs, rate=self.rate) if training else inputs
    
    def get_config(self):
        config = super(MCDropout, self).get_config()
        config.update({'rate': self.rate})
        return config


class ModeloLSTM:
    def __init__(self, lookback_period=LSTM_LOOKBACK_PERIOD, lstm_units_1=8, lstm_units_2=6, 
                 dense_units=8, activation_dense=LSTM_ACTIVATION_DENSE, dropout_rate=LSTM_DROPOUT_RATE,
                 dataset_size=None, num_features=None):
        """
        Initialize LSTM model with dynamic architecture support.
        
        Args:
            lookback_period: Number of time steps to look back
            lstm_units_1: Units in first LSTM layer
            lstm_units_2: Units in second LSTM layer
            dense_units: Units in dense layer
            activation_dense: Activation for dense layer
            dropout_rate: Dropout rate
            dataset_size: Size of dataset for dynamic configuration
            num_features: Number of features for dynamic configuration
        """
        self.dataset_size = dataset_size
        self.num_features = num_features
        self.lookback_period = lookback_period
        self.model = None
        self.bayesian_model = None
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        
        if LSTM_DYNAMIC_CONFIG.get('enabled', False) and dataset_size and num_features:
            optimal_config = self._calculate_optimal_architecture(dataset_size, num_features)
            self.lstm_units_1 = optimal_config['lstm_units_1']
            self.lstm_units_2 = optimal_config['lstm_units_2']
            self.dense_units = optimal_config['dense_units']
            self.dropout_rate = optimal_config['dropout_rate']
            logger.info("Dynamic architecture calculated")
            logger.info(f"Dataset: {dataset_size} samples, {num_features} features")
            logger.info(f"Architecture: {self.lstm_units_1}→{self.lstm_units_2}→{self.dense_units}")
        else:
            self.lstm_units_1 = lstm_units_1
            self.lstm_units_2 = lstm_units_2
            self.dense_units = dense_units
            self.dropout_rate = dropout_rate
            
        self.activation_dense = activation_dense
        self.focal_loss_instance = None
        self.class_weights = None
        self.training_history = None
        
        self.bayesian_config = BAYESIAN_LSTM_CONFIG
        self.uncertainty_config = UNCERTAINTY_QUANTIFICATION_CONFIG
        
        if ADVANCED_LOGGING.get('log_complexity_analysis', True):
            logger.info(f"LSTM configured: {self.lstm_units_1}→{self.lstm_units_2}→{self.dense_units}, dropout={self.dropout_rate}")

    def _preparar_dados(self, df_features: pd.DataFrame, df_target: pd.Series):        
        df_features_filled = df_features.ffill().bfill()
        df_target_filled = df_target.ffill().bfill()
        
        initial_len = len(df_features_filled)
        df_features_filled.dropna(inplace=True)
        df_target_filled = df_target_filled[df_target_filled.index.isin(df_features_filled.index)]
        if len(df_features_filled) < initial_len:
            logger.warning(f"{initial_len - len(df_features_filled)} rows removed due to persistent NaNs before normalization")

        if df_features_filled.empty or df_target_filled.empty:
            logger.error("Feature or target DataFrame is empty after NaN treatment")
            return None, None, None, None
            
        scaled_features = self.scaler_features.fit_transform(df_features_filled)
        scaled_target = self.scaler_target.fit_transform(df_target_filled.values.reshape(-1, 1))

        X, y = [], []
        for i in range(self.lookback_period, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_period:i, :]) 
            y.append(scaled_target[i, 0])
        
        if not X or not y:
            logger.error(f"Could not create sequences X, y. Scaled data length: {len(scaled_features)}, lookback: {self.lookback_period}")
            return None, None, None, None

        return np.array(X), np.array(y), df_features_filled.index[self.lookback_period:], df_target_filled.iloc[self.lookback_period:]

    def _preparar_dados_classificacao(self, df_features: pd.DataFrame, target_classes: pd.Series):
        """
        Prepare data specifically for classification problem.
        """        
        df_features_filled = df_features.ffill().bfill()
        target_classes_filled = target_classes.ffill().bfill()
        
        initial_len = len(df_features_filled)
        df_features_filled.dropna(inplace=True)
        target_classes_filled = target_classes_filled[target_classes_filled.index.isin(df_features_filled.index)]
        
        if len(df_features_filled) < initial_len:
            logger.warning(f"{initial_len - len(df_features_filled)} rows removed due to persistent NaNs (classification)")

        if df_features_filled.empty or target_classes_filled.empty:
            logger.error("Feature or target_classes DataFrame is empty after NaN treatment")
            return None, None, None, None
            
        if ADVANCED_LOGGING['log_class_distribution']:
            from collections import Counter
            class_distribution = Counter(target_classes_filled)
            logger.info(f"Class distribution in dataset: {dict(class_distribution)}")
            
            total = len(target_classes_filled)
            percentuais = {k: f"{(v/total)*100:.1f}%" for k, v in class_distribution.items()}
            logger.info(f"Class percentages: {percentuais}")
            
        scaled_features = self.scaler_features.fit_transform(df_features_filled)

        X, y = [], []
        for i in range(self.lookback_period, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_period:i, :]) 
            y.append(target_classes_filled.iloc[i])
        
        if not X or not y:
            logger.error(f"Could not create sequences X, y for classification. Data length: {len(scaled_features)}, lookback: {self.lookback_period}")
            return None, None, None, None

        X = np.array(X)
        y = np.array(y, dtype=int)
        
        y_onehot = to_categorical(y, num_classes=3)
        
        return X, y_onehot, df_features_filled.index[self.lookback_period:], target_classes_filled.iloc[self.lookback_period:]

    def construir_modelo(self, input_shape):
        """
        Build a Stacked Bidirectional LSTM model for multivariate data (REGRESSION).
        Maintained for compatibility with existing tests.
        """
        num_features = input_shape[1]
        
        self.model = Sequential()
        
        self.model.add(Bidirectional(
            LSTM(units=self.lstm_units_1, return_sequences=True), 
            input_shape=(self.lookback_period, num_features)
        ))
        self.model.add(Dropout(0.3))
        
        self.model.add(Bidirectional(
            LSTM(units=self.lstm_units_2)
        ))
        self.model.add(Dropout(0.3))
        
        self.model.add(Dense(units=1))
        
        logger.info(f"Stacked Bidirectional LSTM model built with {num_features} input features")
        logger.info(f"Architecture: BiLSTM({self.lstm_units_1}) → BiLSTM({self.lstm_units_2}) → Dense(1)")
        
    def _calculate_optimal_architecture(self, dataset_size: int, num_features: int) -> dict:
        """Calculate optimal architecture based on dataset."""
        try:
            target_ratio = LSTM_DYNAMIC_CONFIG.get('complexity_target_ratio', 5.0)
            effective_samples = dataset_size - self.lookback_period
            
            if dataset_size < 1000:
                base_config = LSTM_DYNAMIC_CONFIG['base_units']['small_dataset']
            elif dataset_size < 5000:
                base_config = LSTM_DYNAMIC_CONFIG['base_units']['medium_dataset']
            else:
                base_config = LSTM_DYNAMIC_CONFIG['base_units']['large_dataset']
            
            lstm_units_1, lstm_units_2, dense_units = base_config
            
            params = self._estimate_parameters(num_features, lstm_units_1, lstm_units_2, dense_units)
            current_ratio = effective_samples / params if params > 0 else 0
            
            if current_ratio < LSTM_DYNAMIC_CONFIG.get('min_ratio', 2.0):
                scale_factor = 0.7
                lstm_units_1 = max(4, int(lstm_units_1 * scale_factor))
                lstm_units_2 = max(3, int(lstm_units_2 * scale_factor))
                dense_units = max(4, int(dense_units * scale_factor))
                logger.info(f"Architecture reduced - low ratio: {current_ratio:.2f}")
                
            elif current_ratio > LSTM_DYNAMIC_CONFIG.get('max_ratio', 15.0):
                scale_factor = 1.3
                lstm_units_1 = min(20, int(lstm_units_1 * scale_factor))
                lstm_units_2 = min(16, int(lstm_units_2 * scale_factor))
                dense_units = min(16, int(dense_units * scale_factor))
                logger.info(f"Architecture expanded - high ratio: {current_ratio:.2f}")
            
            if dataset_size < 500:
                dropout_rate = 0.3
            elif dataset_size < 2000:
                dropout_rate = 0.2
            else:
                dropout_rate = 0.15
            
            return {
                'lstm_units_1': lstm_units_1,
                'lstm_units_2': lstm_units_2, 
                'dense_units': dense_units,
                'dropout_rate': dropout_rate
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal architecture: {e}")
            return {
                'lstm_units_1': 8,
                'lstm_units_2': 6,
                'dense_units': 8,
                'dropout_rate': 0.2
            }

    def _estimate_parameters(self, num_features: int, lstm1: int, lstm2: int, dense: int) -> int:
        """Estimate number of parameters in the architecture."""
        params_lstm1 = 2 * 4 * lstm1 * (num_features + lstm1 + 1)
        params_lstm2 = 2 * 4 * lstm2 * (lstm1 * 2 + lstm2 + 1)
        params_dense1 = (lstm2 * 2) * dense + dense
        params_output = dense * 3 + 3
        
        return params_lstm1 + params_lstm2 + params_dense1 + params_output

    def construir_modelo_classificacao(self, input_shape, num_classes=3):
        """
        Build standard classification LSTM model.
        """
        num_features = input_shape[1]
        
        self.model = Sequential()
        
        self.model.add(Bidirectional(
            LSTM(units=self.lstm_units_1, return_sequences=True, 
                 kernel_regularizer=l2(LSTM_L2_REGULARIZATION)), 
            input_shape=(self.lookback_period, num_features)
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        self.model.add(Bidirectional(
            LSTM(units=self.lstm_units_2, 
                 kernel_regularizer=l2(LSTM_L2_REGULARIZATION))
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        self.model.add(Dense(units=self.dense_units, activation=self.activation_dense))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(units=num_classes, activation='softmax'))
        
        if ADVANCED_LOGGING.get('log_complexity_analysis', True):
            logger.info("Classification LSTM model constructed")
            logger.info(f"Input shape: {input_shape} ({num_features} features)")
            logger.info(f"Architecture:")
            logger.info(f"  • Input: ({self.lookback_period}, {num_features})")
            logger.info(f"  • BiLSTM Layer 1: {self.lstm_units_1} units")
            logger.info(f"  • Dropout: {self.dropout_rate}")
            logger.info(f"  • BiLSTM Layer 2: {self.lstm_units_2} units")
            logger.info(f"  • Dense Layer: {self.dense_units} units")
            logger.info(f"  • Output: {num_classes} classes (softmax)")
            logger.info(f"Regularization: L2={LSTM_L2_REGULARIZATION}, Dropout={self.dropout_rate}")

    def construir_modelo_classificacao_bayesian(self, input_shape, num_classes=3):
        """
        Build Bayesian LSTM model with Monte Carlo Dropout for uncertainty quantification.
        Uses MCDropout layer that remains active during inference.
        """
        num_features = input_shape[1]
        
        lstm_units = min(16, self.lstm_units_1)
        dropout_rate = max(0.4, self.dropout_rate)
        
        self.bayesian_model = Sequential()
        
        self.bayesian_model.add(LSTM(
            units=lstm_units,
            return_sequences=False,
            kernel_regularizer=l2(LSTM_L2_REGULARIZATION),
            input_shape=(self.lookback_period, num_features)
        ))
        self.bayesian_model.add(MCDropout(dropout_rate))
        
        self.bayesian_model.add(Dense(units=num_classes, activation='softmax'))
        
        logger.info(f"Bayesian LSTM: LSTM({lstm_units}) → MCDropout({dropout_rate}) → Softmax({num_classes})")

    def _preparar_focal_loss_e_class_weights(self, y_train_classes_raw):
        """
        Prepare Focal Loss and Class Weights based on configuration.
        """
        if not ADVANCED_STRATEGIES_AVAILABLE:
            logger.warning("Advanced strategies not available. Using categorical_crossentropy without class weights")
            return 'categorical_crossentropy', None
        
        loss_function = 'categorical_crossentropy'
        class_weights_dict = None
        
        if FOCAL_LOSS_CONFIG['enabled']:
            alpha = FOCAL_LOSS_CONFIG['alpha']
            gamma = FOCAL_LOSS_CONFIG['gamma']
            
            if FOCAL_LOSS_CONFIG['auto_adjust']:
                from collections import Counter
                class_dist = Counter(y_train_classes_raw)
                alpha_auto, gamma_auto = get_recommended_focal_parameters(class_dist)
                alpha, gamma = alpha_auto, gamma_auto
                
                if ADVANCED_LOGGING['log_focal_parameters']:
                    logger.info(f"Focal Loss parameters auto-adjusted: alpha={alpha}, gamma={gamma}")
            
            self.focal_loss_instance = CategoricalFocalLoss(alpha=alpha, gamma=gamma)
            loss_function = self.focal_loss_instance
            
            if ADVANCED_LOGGING['log_focal_parameters']:
                logger.info(f"Focal Loss activated: alpha={alpha}, gamma={gamma}")
        
        if CLASS_WEIGHTING_CONFIG['enabled']:
            if CLASS_WEIGHTING_CONFIG['auto_calculate']:
                strategy = CLASS_WEIGHTING_CONFIG['strategy']
                smoothing = CLASS_WEIGHTING_CONFIG['smoothing_factor']
                
                class_weights_dict = calculate_class_weights_smart(
                    y_train_classes_raw, 
                    strategy=strategy, 
                    smoothing=smoothing
                )
                
                if ADVANCED_LOGGING['log_class_weights']:
                    logger.info(f"Class weights calculated automatically ({strategy}): {class_weights_dict}")
            else:
                class_weights_dict = CLASS_WEIGHTING_CONFIG['manual_weights'].copy()
                
                if ADVANCED_LOGGING['log_class_weights']:
                    logger.info(f"Manual class weights used: {class_weights_dict}")
        
        self.class_weights = class_weights_dict
        return loss_function, class_weights_dict

    def treinar_modelo(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, optimizer='adam', loss='mean_squared_error'):
        """
        Train regression model. Maintained for compatibility.
        """
        if self.model is None:
            logger.error("Model not built. Call construir_modelo() first")
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
        logger.info("LSTM model training completed")
        return history

    def treinar_modelo_classificacao(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16, optimizer='adam'):
        """
        Train classification model with advanced strategies.
        """
        if self.model is None:
            logger.error("Classification model not built. Call construir_modelo_classificacao() first")
            return None
        
        y_train_classes_raw = np.argmax(y_train, axis=1)
        
        loss_function, class_weights_dict = self._preparar_focal_loss_e_class_weights(y_train_classes_raw)
        
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_function,
            metrics=['accuracy']
        )
        
        callbacks = self._configurar_callbacks_avancados()
        
        if ADVANCED_LOGGING['log_training_metrics']:
            logger.info("Starting LSTM training")
            logger.info(f"Configuration:")
            logger.info(f"  • Epochs: {epochs}")
            logger.info(f"  • Batch size: {batch_size}")
            logger.info(f"  • Optimizer: {optimizer}")
            logger.info(f"  • Lookback period: {self.lookback_period}")
            logger.info(f"  • LSTM units: {self.lstm_units_1}×{self.lstm_units_2}")
            logger.info(f"  • Dense units: {self.dense_units}")
            logger.info(f"  • Dropout rate: {self.dropout_rate}")
            logger.info(f"Advanced strategies:")
            logger.info(f"  • Loss function: {type(loss_function).__name__ if hasattr(loss_function, '__name__') else str(loss_function)}")
            logger.info(f"  • Class weights: {'Enabled' if class_weights_dict else 'Disabled'}")
            logger.info(f"  • Focal Loss: {'Enabled' if FOCAL_LOSS_CONFIG['enabled'] and ADVANCED_STRATEGIES_AVAILABLE else 'Disabled'}")
        
        fit_params = {
            'x': X_train,
            'y': y_train,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_data': (X_val, y_val),
            'callbacks': callbacks,
            'verbose': ADVANCED_TRAINING_CONFIG['verbose_training']
        }
        
        if class_weights_dict:
            fit_params['class_weight'] = class_weights_dict
        
        history = self.model.fit(**fit_params)
        
        if ADVANCED_LOGGING['log_training_metrics'] and history:
            final_accuracy = history.history.get('val_accuracy', [0])[-1] if 'val_accuracy' in history.history else 'N/A'
            final_loss = history.history.get('val_loss', [0])[-1] if 'val_loss' in history.history else 'N/A'
            
            logger.info("LSTM training completed")
            logger.info(f"Final results:")
            logger.info(f"  • Validation accuracy: {final_accuracy}")
            logger.info(f"  • Validation loss: {final_loss}")
        
        self.training_history = history
        return history

    def _configurar_callbacks_avancados(self):
        """
        Configure advanced callbacks based on settings.
        """
        callbacks = []
        
        early_stopping = EarlyStopping(
            monitor=ADVANCED_TRAINING_CONFIG['early_stopping_monitor'],
            patience=ADVANCED_TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            mode='max' if 'accuracy' in ADVANCED_TRAINING_CONFIG['early_stopping_monitor'] else 'min',
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = ReduceLROnPlateau(
            monitor=ADVANCED_TRAINING_CONFIG['early_stopping_monitor'],
            patience=ADVANCED_TRAINING_CONFIG['reduce_lr_patience'],
            factor=ADVANCED_TRAINING_CONFIG['reduce_lr_factor'],
            min_lr=ADVANCED_TRAINING_CONFIG['min_lr'],
            mode='max' if 'accuracy' in ADVANCED_TRAINING_CONFIG['early_stopping_monitor'] else 'min',
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks

    def prever(self, X_data):
        """
        Make regression predictions. Maintained for compatibility.
        """
        if self.model is None:
            logger.error("Model not trained")
            return None
        
        predicoes_scaled = self.model.predict(X_data)
        predicoes_desnormalizadas = self.scaler_target.inverse_transform(predicoes_scaled)
        return predicoes_desnormalizadas.flatten()

    def prever_classificacao(self, X_data):
        """
        Make classification predictions returning class probabilities.
        """
        model_to_use = self._get_prediction_model()
        
        if model_to_use is None:
            logger.error("No trained model available for prediction")
            return None
        
        probabilidades = model_to_use.predict(X_data)
        logger.info(f"Classification prediction made for {len(X_data)} samples")
        return probabilidades

    def prever_classificacao_com_incerteza(self, X_data, num_samples=None):
        """
        Make predictions with uncertainty quantification using Monte Carlo Dropout.
        
        Args:
            X_data: Input data
            num_samples: Number of Monte Carlo samples (default from config)
            
        Returns:
            dict: Contains predictions, uncertainties, and confidence intervals
        """
        if num_samples is None:
            num_samples = self.bayesian_config.get('monte_carlo_samples', 100)
        
        model_to_use = self._get_prediction_model()
        
        if model_to_use is None:
            logger.error("No trained model available for uncertainty prediction")
            return None
        
        is_bayesian = (self.bayesian_model is not None and 
                      self.bayesian_config.get('enabled', True) and
                      self.bayesian_config.get('dropout_inference', True))
        
        if is_bayesian:
            logger.info(f"Using Bayesian LSTM with {num_samples} Monte Carlo samples")
            predictions_list = []
            
            for i in range(num_samples):
                pred = model_to_use.predict(X_data, verbose=0)
                predictions_list.append(pred)
            
            predictions_array = np.array(predictions_list)
            
            mean_predictions = np.mean(predictions_array, axis=0)
            
            uncertainty_metrics = self._calculate_uncertainty_metrics(predictions_array)
            
            confidence_levels = self.bayesian_config.get('confidence_intervals', [0.05, 0.95])
            prediction_intervals = np.percentile(predictions_array, 
                                               [confidence_levels[0]*100, confidence_levels[1]*100], 
                                               axis=0)
            
            result = {
                'predictions': mean_predictions,
                'epistemic_uncertainty': uncertainty_metrics['epistemic_uncertainty'],
                'total_uncertainty': uncertainty_metrics['total_uncertainty'],
                'confidence_scores': uncertainty_metrics['confidence_scores'],
                'prediction_intervals': {
                    'lower': prediction_intervals[0],
                    'upper': prediction_intervals[1]
                },
                'num_samples': num_samples,
                'method': 'monte_carlo_dropout'
            }
            
            logger.info("Uncertainty quantification completed")
            return result
        else:
            logger.info("Using standard LSTM (no uncertainty quantification)")
            predictions = model_to_use.predict(X_data)
            
            return {
                'predictions': predictions,
                'epistemic_uncertainty': np.zeros_like(predictions),
                'total_uncertainty': np.zeros_like(predictions),
                'confidence_scores': np.ones(len(predictions)),
                'prediction_intervals': {
                    'lower': predictions,
                    'upper': predictions
                },
                'num_samples': 1,
                'method': 'standard'
            }

    def _calculate_uncertainty_metrics(self, predictions_array):
        """
        Calculate detailed uncertainty metrics from Monte Carlo samples.
        
        Args:
            predictions_array: Array of shape (num_samples, num_instances, num_classes)
            
        Returns:
            dict: Uncertainty metrics including epistemic and total uncertainty
        """
        epistemic_uncertainty = np.var(predictions_array, axis=0)
        
        total_uncertainty = np.mean(epistemic_uncertainty, axis=1)
        
        confidence_scores = 1 - np.clip(total_uncertainty, 0, 1)
        
        mean_predictions = np.mean(predictions_array, axis=0)
        predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8), axis=1)
        
        return {
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence_scores': confidence_scores,
            'predictive_entropy': predictive_entropy
        }

    def _ensemble_predictions(self, predictions_list):
        """
        Combine multiple predictions using ensemble strategies.
        
        Args:
            predictions_list: List of prediction arrays
            
        Returns:
            dict: Ensemble predictions with voting results
        """
        predictions_array = np.array(predictions_list)
        
        mean_predictions = np.mean(predictions_array, axis=0)
        
        if hasattr(self, 'prediction_confidences'):
            weights = np.array(self.prediction_confidences)
            weighted_predictions = np.average(predictions_array, axis=0, weights=weights)
        else:
            weighted_predictions = mean_predictions
        
        class_predictions = np.argmax(predictions_array, axis=2)
        majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=class_predictions
        )
        
        variances = np.var(predictions_array, axis=0)
        variance_weights = 1 / (1 + variances)
        variance_weighted = np.average(predictions_array, axis=0, weights=variance_weights.mean(axis=1))
        
        return {
            'mean_ensemble': mean_predictions,
            'weighted_ensemble': weighted_predictions,
            'majority_vote': majority_vote,
            'variance_weighted': variance_weighted,
            'ensemble_variance': variances
        }

    def _get_prediction_model(self):
        """
        Intelligently select which model to use for predictions.
        
        Returns:
            Model instance or None
        """
        if (self.bayesian_model is not None and 
            self.bayesian_config.get('enabled', True)):
            logger.info("Using Bayesian model for predictions")
            return self.bayesian_model
        elif self.model is not None:
            logger.info("Using standard model for predictions")
            return self.model
        else:
            logger.error("No model available for predictions")
            return None

    def calibrate_uncertainty(self, y_true, y_pred_probs, uncertainties):
        """
        Calibrate uncertainty estimates for better reliability.
        
        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities
            uncertainties: Uncertainty estimates
            
        Returns:
            dict: Calibration metrics and adjusted uncertainties
        """
        if not self.uncertainty_config.get('calibration_enabled', True):
            return {'calibrated': False, 'uncertainties': uncertainties}
        
        try:
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true_classes = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
            
            sorted_indices = np.argsort(uncertainties)
            
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_accuracies = []
            bin_confidences = []
            
            for i in range(n_bins):
                bin_mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
                if np.sum(bin_mask) > 0:
                    bin_accuracy = np.mean(y_pred[bin_mask] == y_true_classes[bin_mask])
                    bin_confidence = np.mean(1 - uncertainties[bin_mask])
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_confidence)
            
            ece = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
            
            temperature = 1.0
            if ece > 0.1:
                temperature = 1.5
            
            calibrated_probs = y_pred_probs ** (1/temperature)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            calibrated_uncertainties = 1 - np.max(calibrated_probs, axis=1)
            
            logger.info(f"Uncertainty calibration completed. ECE: {ece:.4f}")
            
            return {
                'calibrated': True,
                'uncertainties': calibrated_uncertainties,
                'original_uncertainties': uncertainties,
                'ece': ece,
                'temperature': temperature,
                'calibration_curve': {
                    'accuracies': bin_accuracies,
                    'confidences': bin_confidences
                }
            }
            
        except Exception as e:
            logger.error(f"Error in uncertainty calibration: {e}")
            return {'calibrated': False, 'uncertainties': uncertainties}

    def preparar_dados_para_treino_teste(self, df_completo: pd.DataFrame, coluna_target: str, train_split_ratio: float = 0.7):
        """
        Prepare data for train/test split. Maintained for compatibility.
        """
        if coluna_target not in df_completo.columns:
            logger.error(f"Target column '{coluna_target}' not found in DataFrame")
            return None, None, None, None, None, None, None, None, None, None

        df_target_series = df_completo[coluna_target]
        df_features_df = df_completo.drop(columns=[coluna_target])
        
        logger.info(f"Feature columns used for LSTM: {df_features_df.columns.tolist()}")

        X_sequencias, y_sequencias, indices_tempo, y_original_series = self._preparar_dados(df_features_df, df_target_series)

        if X_sequencias is None or y_sequencias is None:
            logger.error("Failed to prepare data in _preparar_dados")
            return None, None, None, None, None, None, None, None, None, None
        
        if len(X_sequencias) == 0:
            logger.error("No sequences generated for X_treino/X_teste")
            return None, None, None, None, None, None, None, None, None, None

        split_index = int(len(X_sequencias) * train_split_ratio)

        X_treino, X_teste = X_sequencias[:split_index], X_sequencias[split_index:]
        y_treino, y_teste = y_sequencias[:split_index], y_sequencias[split_index:]
        
        indices_treino, indices_teste = indices_tempo[:split_index], indices_tempo[split_index:]
        y_original_treino, y_original_teste = y_original_series[:split_index], y_original_series[split_index:]

        if X_treino.size == 0 or X_teste.size == 0:
            logger.error(f"Train/test split resulted in empty set. X_treino: {X_treino.shape}, X_teste: {X_teste.shape}")
            return None, None, None, None, None, None, None, None, None, None

        logger.info(f"Data prepared for LSTM: X_treino shape {X_treino.shape}, y_treino shape {y_treino.shape}, X_teste shape {X_teste.shape}, y_teste shape {y_teste.shape}")
        return X_treino, y_treino, X_teste, y_teste, indices_treino, indices_teste, y_original_treino, y_original_teste, self.scaler_target, df_features_df.columns.tolist()

    def preparar_dados_para_treino_teste_classificacao(self, df_completo: pd.DataFrame, colunas_features: list, coluna_target_classes: str, train_split_ratio: float = 0.7):
        """
        Prepare data specifically for classification train/test.
        """
        if coluna_target_classes not in df_completo.columns:
            logger.error(f"Target class column '{coluna_target_classes}' not found in DataFrame")
            return None, None, None, None, None, None, None, None, None, None

        target_classes_series = df_completo[coluna_target_classes]
        df_features_df = df_completo[colunas_features]
        
        logger.info(f"Feature columns used for LSTM classification: {df_features_df.columns.tolist()}")

        X_sequencias, y_sequencias_onehot, indices_tempo, y_original_classes = self._preparar_dados_classificacao(df_features_df, target_classes_series)

        if X_sequencias is None or y_sequencias_onehot is None:
            logger.error("Failed to prepare classification data")
            return None, None, None, None, None, None, None, None, None, None
        
        if len(X_sequencias) == 0:
            logger.error("No sequences generated for classification")
            return None, None, None, None, None, None, None, None, None, None

        split_index = int(len(X_sequencias) * train_split_ratio)

        X_treino, X_teste = X_sequencias[:split_index], X_sequencias[split_index:]
        y_treino, y_teste = y_sequencias_onehot[:split_index], y_sequencias_onehot[split_index:]
        
        indices_treino, indices_teste = indices_tempo[:split_index], indices_tempo[split_index:]
        y_original_treino, y_original_teste = y_original_classes[:split_index], y_original_classes[split_index:]

        if X_treino.size == 0 or X_teste.size == 0:
            logger.error(f"Classification train/test split resulted in empty set. X_treino: {X_treino.shape}, X_teste: {X_teste.shape}")
            return None, None, None, None, None, None, None, None, None, None

        logger.info(f"Classification data prepared: X_treino {X_treino.shape}, y_treino {y_treino.shape} (one-hot), X_teste {X_teste.shape}, y_teste {y_teste.shape} (one-hot)")
        return X_treino, y_treino, X_teste, y_teste, indices_treino, indices_teste, y_original_treino, y_original_teste, None, df_features_df.columns.tolist()