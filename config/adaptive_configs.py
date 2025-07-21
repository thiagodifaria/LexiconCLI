import numpy as np
from utils.logger import logger

class AdaptiveConfigs:
    """Gerencia configurações adaptativas baseadas nas características do dataset."""
    
    def __init__(self):
        self.dataset_categories = {
            'micro': {'max_size': 250, 'name': 'micro'},
            'small': {'max_size': 1000, 'name': 'small'},
            'medium': {'max_size': 5000, 'name': 'medium'},
            'large': {'max_size': float('inf'), 'name': 'large'}
        }
        
        self.complexity_profiles = {
            'micro': {
                'lstm_units': [4, 3],
                'dropout_rate': 0.5,
                'l2_reg': 0.001,
                'batch_size': 8,
                'epochs': 30,
                'target_ratio': 12.0
            },
            'small': {
                'lstm_units': [8, 4],
                'dropout_rate': 0.4,
                'l2_reg': 0.0008,
                'batch_size': 16,
                'epochs': 40,
                'target_ratio': 10.0
            },
            'medium': {
                'lstm_units': [12, 6],
                'dropout_rate': 0.3,
                'l2_reg': 0.0005,
                'batch_size': 24,
                'epochs': 50,
                'target_ratio': 8.0
            },
            'large': {
                'lstm_units': [16, 8],
                'dropout_rate': 0.25,
                'l2_reg': 0.0003,
                'batch_size': 32,
                'epochs': 60,
                'target_ratio': 6.0
            }
        }
        
        self.threshold_strategies = {
            'micro': {'method': 'percentile', 'ratio': 0.5},
            'small': {'method': 'percentile', 'ratio': 0.4},
            'medium': {'method': 'percentile', 'ratio': 0.35},
            'large': {'method': 'percentile', 'ratio': 0.3}
        }

    def get_dataset_category(self, dataset_size: int) -> str:
        """Determina a categoria do dataset baseada no tamanho."""
        for category, config in self.dataset_categories.items():
            if dataset_size <= config['max_size']:
                return category
        return 'large'

    def get_optimal_config(self, dataset_size: int, num_features: int, 
                          noise_level: str = 'medium') -> dict:
        """Retorna configuração ótima para o dataset específico."""
        try:
            category = self.get_dataset_category(dataset_size)
            base_config = self.complexity_profiles[category].copy()
            
            if num_features > 15:
                base_config['dropout_rate'] = min(0.5, base_config['dropout_rate'] * 1.2)
                base_config['l2_reg'] = min(0.002, base_config['l2_reg'] * 1.5)
            elif num_features < 5:
                base_config['dropout_rate'] = max(0.1, base_config['dropout_rate'] * 0.8)
                base_config['l2_reg'] = max(0.00005, base_config['l2_reg'] * 0.7)
            
            if noise_level == 'high':
                base_config['dropout_rate'] = min(0.5, base_config['dropout_rate'] * 1.3)
                base_config['epochs'] = max(20, base_config['epochs'] - 10)
            elif noise_level == 'low':
                base_config['dropout_rate'] = max(0.1, base_config['dropout_rate'] * 0.7)
                base_config['epochs'] = min(100, base_config['epochs'] + 15)
            
            base_config['dataset_category'] = category
            
            logger.info(f"Configuração adaptativa: categoria={category}, "
                       f"features={num_features}, ruído={noise_level}")
            
            return base_config
            
        except Exception as e:
            logger.error(f"Erro ao obter configuração adaptativa: {e}")
            return self.complexity_profiles['small']

    def get_threshold_config(self, dataset_size: int, returns_data: np.ndarray = None) -> dict:
        """Calcula configuração de thresholds adaptativos."""
        try:
            category = self.get_dataset_category(dataset_size)
            strategy = self.threshold_strategies[category]
            
            if returns_data is not None and len(returns_data) > 50:
                returns_clean = returns_data[~np.isnan(returns_data)]
                
                if strategy['method'] == 'percentile':
                    ratio = strategy['ratio']
                    upper_pct = 100 - (ratio * 50)
                    lower_pct = ratio * 50
                    
                    upper_threshold = np.percentile(returns_clean, upper_pct)
                    lower_threshold = np.percentile(returns_clean, lower_pct)
                    
                    upper_threshold = max(0.003, min(0.025, abs(upper_threshold)))
                    lower_threshold = min(-0.003, max(-0.025, -abs(lower_threshold)))
                    
                    return {
                        'upper': upper_threshold,
                        'lower': lower_threshold,
                        'method': 'adaptive_percentile',
                        'category': category
                    }
            
            fallback_thresholds = {
                'micro': {'upper': 0.015, 'lower': -0.015},
                'small': {'upper': 0.010, 'lower': -0.010},
                'medium': {'upper': 0.008, 'lower': -0.008},
                'large': {'upper': 0.006, 'lower': -0.006}
            }
            
            thresholds = fallback_thresholds[category]
            thresholds['method'] = 'fallback'
            thresholds['category'] = category
            
            return thresholds
            
        except Exception as e:
            logger.error(f"Erro ao calcular thresholds adaptativos: {e}")
            return {'upper': 0.008, 'lower': -0.008, 'method': 'error_fallback'}

    def get_augmentation_config(self, dataset_size: int, class_balance: dict = None) -> dict:
        """Retorna configuração de data augmentation baseada no dataset."""
        if dataset_size >= 2000:
            return {'enabled': False, 'reason': 'dataset_large_enough'}
        
        base_multiplier = max(1.5, min(4.0, 1000 / dataset_size))
        
        if class_balance:
            min_class_ratio = min(class_balance.values()) / sum(class_balance.values())
            if min_class_ratio < 0.15:
                base_multiplier *= 1.3
        
        return {
            'enabled': True,
            'multiplier': round(base_multiplier, 1),
            'techniques': {
                'jittering': {'enabled': True, 'noise_factor': 0.006},
                'time_warping': {'enabled': dataset_size > 100, 'sigma': 0.12},
                'magnitude_warping': {'enabled': dataset_size > 150, 'sigma': 0.12}
            },
            'safety_params': {
                'max_iterations': min(500, dataset_size * 2),
                'timeout_seconds': min(8, max(3, dataset_size // 100))
            }
        }

    def validate_config(self, config: dict, dataset_size: int, num_features: int) -> dict:
        """Valida e ajusta configuração se necessário."""
        try:
            if config.get('batch_size', 16) > dataset_size * 0.7:
                config['batch_size'] = max(4, int(dataset_size * 0.1))
                logger.warning(f"Batch size ajustado para {config['batch_size']}")
            
            if dataset_size < 200 and config.get('epochs', 50) > 40:
                config['epochs'] = 30
                logger.warning("Épocas reduzidas para dataset pequeno")
            
            lstm_units = config.get('lstm_units', [8, 6, 8])
            estimated_params = self._estimate_params(num_features, lstm_units)
            ratio = (dataset_size * 0.7) / estimated_params if estimated_params > 0 else 0
            
            if ratio < 2.0:
                scale = max(0.5, ratio / 3.0)
                config['lstm_units'] = [max(3, int(u * scale)) for u in lstm_units]
                logger.warning(f"Arquitetura reduzida - ratio baixo: {ratio:.2f}")
            
            return config
            
        except Exception as e:
            logger.error(f"Erro na validação de configuração: {e}")
            return config

    def _estimate_params(self, num_features: int, lstm_units: list) -> int:
        """Estima número de parâmetros da arquitetura."""
        if len(lstm_units) < 3:
            return 1000
        
        lstm1, lstm2, dense = lstm_units[0], lstm_units[1], lstm_units[2]
        
        params_lstm1 = 2 * 4 * lstm1 * (num_features + lstm1 + 1)
        params_lstm2 = 2 * 4 * lstm2 * (lstm1 * 2 + lstm2 + 1)
        params_dense = (lstm2 * 2) * dense + dense
        params_output = dense * 3 + 3
        
        return params_lstm1 + params_lstm2 + params_dense + params_output