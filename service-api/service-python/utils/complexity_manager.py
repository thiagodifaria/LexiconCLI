import numpy as np
import pandas as pd
from utils.logger import logger
from config.adaptive_configs import AdaptiveConfigs

class ComplexityManager:
    """Gerencia complexidade de modelos automaticamente baseado no dataset."""
    
    def __init__(self):
        self.adaptive_configs = AdaptiveConfigs()
        self.complexity_cache = {}
        
        self.target_ratios = {
            'conservative': 15.0,
            'balanced': 12.0,
            'aggressive': 8.0
        }
        
        self.min_acceptable_ratio = 2.0
        self.max_reasonable_ratio = 20.0

    def analyze_dataset_complexity(self, df_data: pd.DataFrame, target_column: str = None) -> dict:
        """Analisa complexidade do dataset e retorna métricas relevantes."""
        try:
            analysis = {
                'size': len(df_data),
                'features': df_data.shape[1],
                'missing_ratio': df_data.isnull().sum().sum() / df_data.size,
                'complexity_level': 'medium'
            }
            
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                volatilities = []
                for col in numeric_cols[:5]:
                    if len(df_data[col].dropna()) > 10:
                        volatility = df_data[col].pct_change().std()
                        if not np.isnan(volatility):
                            volatilities.append(volatility)
                
                if volatilities:
                    avg_volatility = np.mean(volatilities)
                    if avg_volatility > 0.1:
                        analysis['noise_level'] = 'high'
                    elif avg_volatility > 0.05:
                        analysis['noise_level'] = 'medium'
                    else:
                        analysis['noise_level'] = 'low'
                else:
                    analysis['noise_level'] = 'medium'
            else:
                analysis['noise_level'] = 'medium'
            
            if target_column and target_column in df_data.columns:
                class_counts = df_data[target_column].value_counts()
                if len(class_counts) > 1:
                    imbalance_ratio = class_counts.max() / class_counts.min()
                    analysis['class_imbalance_ratio'] = imbalance_ratio
                    analysis['is_balanced'] = imbalance_ratio < 3.0
                else:
                    analysis['is_balanced'] = True
                    analysis['class_imbalance_ratio'] = 1.0
            
            complexity_factors = 0
            if analysis['size'] < 500:
                complexity_factors += 1
            if analysis['missing_ratio'] > 0.1:
                complexity_factors += 1
            if analysis['noise_level'] == 'high':
                complexity_factors += 1
            if analysis.get('class_imbalance_ratio', 1) > 5:
                complexity_factors += 1
            
            if complexity_factors >= 3:
                analysis['complexity_level'] = 'high'
            elif complexity_factors <= 1:
                analysis['complexity_level'] = 'low'
            else:
                analysis['complexity_level'] = 'medium'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de complexidade do dataset: {e}")
            return {
                'size': len(df_data) if not df_data.empty else 0,
                'features': df_data.shape[1] if not df_data.empty else 0,
                'complexity_level': 'medium',
                'noise_level': 'medium'
            }

    def calculate_optimal_architecture(self, dataset_analysis: dict, 
                                     lookback_period: int = 15) -> dict:
        """Calcula arquitetura LSTM ótima baseada na análise do dataset."""
        try:
            dataset_size = dataset_analysis['size']
            num_features = dataset_analysis['features']
            complexity_level = dataset_analysis.get('complexity_level', 'medium')
            noise_level = dataset_analysis.get('noise_level', 'medium')
            
            cache_key = f"{dataset_size}_{num_features}_{complexity_level}_{noise_level}"
            if cache_key in self.complexity_cache:
                return self.complexity_cache[cache_key]
            
            base_config = self.adaptive_configs.get_optimal_config(
                dataset_size, num_features, noise_level
            )
            
            effective_samples = max(50, dataset_size - lookback_period)
            
            if complexity_level == 'high':
                target_ratio = self.target_ratios['conservative']
            elif complexity_level == 'low':
                target_ratio = self.target_ratios['aggressive']
            else:
                target_ratio = self.target_ratios['balanced']
            
            optimal_units = self._optimize_architecture_units(
                effective_samples, num_features, target_ratio, base_config['lstm_units']
            )
            
            result = {
                'lstm_units_1': optimal_units[0],
                'lstm_units_2': optimal_units[1],
                'dense_units': optimal_units[2],
                'dropout_rate': base_config['dropout_rate'],
                'l2_regularization': base_config['l2_reg'],
                'batch_size': min(base_config['batch_size'], max(8, effective_samples // 10)),
                'epochs': base_config['epochs'],
                'estimated_parameters': self._estimate_total_parameters(num_features, optimal_units),
                'samples_per_parameter': effective_samples / self._estimate_total_parameters(num_features, optimal_units),
                'complexity_assessment': self._assess_complexity_level(
                    effective_samples / self._estimate_total_parameters(num_features, optimal_units)
                )
            }
            
            result = self.adaptive_configs.validate_config(result, dataset_size, num_features)
            
            self.complexity_cache[cache_key] = result
            
            logger.info(f"Arquitetura otimizada: {optimal_units}, "
                       f"ratio={result['samples_per_parameter']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no cálculo de arquitetura ótima: {e}")
            return {
                'lstm_units_1': 6,
                'lstm_units_2': 4,
                'dense_units': 6,
                'dropout_rate': 0.3,
                'l2_regularization': 0.0005,
                'batch_size': 16,
                'epochs': 40,
                'estimated_parameters': 1000,
                'samples_per_parameter': dataset_analysis['size'] / 1000,
                'complexity_assessment': 'fallback'
            }

    def _optimize_architecture_units(self, effective_samples: int, num_features: int, 
                                   target_ratio: float, base_units: list) -> list:
        """Otimiza número de unidades em cada camada para atingir target ratio."""
        try:
            current_units = base_units.copy()
            current_params = self._estimate_total_parameters(num_features, current_units)
            current_ratio = effective_samples / current_params if current_params > 0 else 0
            
            if target_ratio * 0.8 <= current_ratio <= target_ratio * 1.2:
                return current_units
            
            if current_ratio < target_ratio * 0.8:
                scale_factor = min(0.7, np.sqrt(current_ratio / target_ratio))
            else:
                scale_factor = min(1.5, np.sqrt(current_ratio / target_ratio))
            
            optimized_units = [
                max(3, min(20, int(current_units[0] * scale_factor))),
                max(3, min(16, int(current_units[1] * scale_factor))),
                max(4, min(16, int(current_units[2] * scale_factor)))
            ]
            
            final_params = self._estimate_total_parameters(num_features, optimized_units)
            final_ratio = effective_samples / final_params if final_params > 0 else 0
            
            if final_ratio < self.min_acceptable_ratio:
                optimized_units = [
                    max(3, optimized_units[0] // 2),
                    max(3, optimized_units[1] // 2),
                    max(4, optimized_units[2] // 2)
                ]
            
            return optimized_units
            
        except Exception as e:
            logger.error(f"Erro na otimização de unidades: {e}")
            return [6, 4, 6]

    def _estimate_total_parameters(self, num_features: int, units: list) -> int:
        """Estima número total de parâmetros da arquitetura."""
        if len(units) < 3:
            return 1000
        
        lstm1, lstm2, dense = units[0], units[1], units[2]
        
        params_lstm1 = 2 * 4 * lstm1 * (num_features + lstm1 + 1)
        
        params_lstm2 = 2 * 4 * lstm2 * (lstm1 * 2 + lstm2 + 1)
        
        params_dense = (lstm2 * 2) * dense + dense
        
        params_output = dense * 3 + 3
        
        return params_lstm1 + params_lstm2 + params_dense + params_output

    def _assess_complexity_level(self, ratio: float) -> str:
        """Avalia nível de complexidade baseado no ratio samples/parameters."""
        if ratio < 2.0:
            return 'muito_alto'
        elif ratio < 5.0:
            return 'alto'
        elif ratio < 10.0:
            return 'moderado'
        elif ratio < 20.0:
            return 'baixo'
        else:
            return 'muito_baixo'

    def recommend_training_strategy(self, complexity_assessment: str, 
                                  dataset_size: int) -> dict:
        """Recomenda estratégia de treinamento baseada na complexidade."""
        strategies = {
            'muito_alto': {
                'early_stopping_patience': 8,
                'reduce_lr_patience': 5,
                'validation_split': 0.25,
                'data_augmentation_multiplier': 3.0,
                'extra_regularization': True
            },
            'alto': {
                'early_stopping_patience': 10,
                'reduce_lr_patience': 6,
                'validation_split': 0.2,
                'data_augmentation_multiplier': 2.0,
                'extra_regularization': True
            },
            'moderado': {
                'early_stopping_patience': 12,
                'reduce_lr_patience': 8,
                'validation_split': 0.2,
                'data_augmentation_multiplier': 1.5,
                'extra_regularization': False
            },
            'baixo': {
                'early_stopping_patience': 15,
                'reduce_lr_patience': 10,
                'validation_split': 0.15,
                'data_augmentation_multiplier': 1.0,
                'extra_regularization': False
            },
            'muito_baixo': {
                'early_stopping_patience': 20,
                'reduce_lr_patience': 12,
                'validation_split': 0.15,
                'data_augmentation_multiplier': 1.0,
                'extra_regularization': False
            }
        }
        
        return strategies.get(complexity_assessment, strategies['moderado'])

    def monitor_training_complexity(self, training_history, validation_threshold: float = 0.15) -> dict:
        """Monitora complexidade durante o treinamento."""
        try:
            if not training_history or not hasattr(training_history, 'history'):
                return {'monitoring_failed': True}
            
            history = training_history.history
            
            train_loss = history.get('loss', [])
            val_loss = history.get('val_loss', [])
            train_acc = history.get('accuracy', [])
            val_acc = history.get('val_accuracy', [])
            
            analysis = {}
            
            if len(train_loss) > 5 and len(val_loss) > 5:
                final_train_loss = np.mean(train_loss[-3:])
                final_val_loss = np.mean(val_loss[-3:])
                loss_gap = final_val_loss - final_train_loss
                
                analysis['overfitting_detected'] = loss_gap > validation_threshold
                analysis['loss_gap'] = loss_gap
                
                if len(val_loss) > 10:
                    recent_val_trend = np.polyfit(range(5), val_loss[-5:], 1)[0]
                    analysis['overfitting_trend'] = recent_val_trend > 0.01
            
            if len(train_loss) > 10:
                recent_improvement = train_loss[-10] - train_loss[-1]
                analysis['converged'] = recent_improvement < 0.001
                analysis['improvement_rate'] = recent_improvement / 10
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro no monitoramento de complexidade: {e}")
            return {'monitoring_error': str(e)}