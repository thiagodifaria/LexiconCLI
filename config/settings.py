import os
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY") 
NASDAQ_DATA_LINK_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

DB_NAME = "data/market_data.db"

LOG_FILE = "logs/financial_system.log"
LOG_LEVEL = "INFO"

DEFAULT_ASSETS_MONITOR = ["AAPL", "MSFT", "GOOGL", "NVDA", "PETR4.SA", "VALE3.SA", "ITUB4.SA"]
DEFAULT_INDICES_MONITOR = ["^BVSP", "^GSPC", "^IXIC"]

CACHE_EXPIRATION_SGS = 24 * 60 * 60  
CACHE_EXPIRATION_PTAX = 24 * 60 * 60  
CACHE_EXPIRATION_FINNHUB_QUOTE = 15 * 60 
CACHE_EXPIRATION_YFINANCE_HISTORY = 4 * 60 * 60 
CACHE_EXPIRATION_ALPHA_VANTAGE_TIMESERIES = 24 * 60 * 60 
CACHE_EXPIRATION_INVESTPY = 4 * 60 * 60 
CACHE_EXPIRATION_FRED = 24 * 60 * 60  
CACHE_EXPIRATION_NASDAQ = 24 * 60 * 60
CACHE_EXPIRATION_NEWS_SENTIMENT = 6 * 60 * 60

LSTM_LOOKBACK_PERIOD = 15
LSTM_TRAIN_TEST_SPLIT = 0.7
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 16
LSTM_DROPOUT_RATE = 0.2
LSTM_L2_REGULARIZATION = 0.0002
LSTM_ACTIVATION_DENSE = 'relu'

LSTM_DYNAMIC_CONFIG = {
    'enabled': True,
    'complexity_target_ratio': 5.0,
    'min_ratio': 2.0,
    'max_ratio': 15.0,
    'auto_architecture': True,
    'base_units': {
        'small_dataset': [6, 4, 6],
        'medium_dataset': [8, 6, 8],
        'large_dataset': [12, 10, 10]
    }
}

BAYESIAN_LSTM_CONFIG = {
    'enabled': True,
    'monte_carlo_samples': 5,
    'dropout_inference': True,
    'uncertainty_threshold': 0.1,
    'ensemble_voting': True,
    'fallback_to_standard': True,
    'confidence_intervals': [0.05, 0.95],
    'min_samples_for_uncertainty': 50
}

MONTE_CARLO_ENSEMBLE_CONFIG = {
    'enabled': True,
    'num_simulations': 1000,
    'integration_method': 'weighted_average',
    'lstm_weight': 0.7,
    'monte_carlo_weight': 0.3,
    'confidence_threshold': 0.6,
    'ensemble_voting_threshold': 0.5
}

UNCERTAINTY_QUANTIFICATION_CONFIG = {
    'enabled': True,
    'methods': ['monte_carlo_dropout', 'ensemble_variance'],
    'calibration_enabled': True,
    'prediction_intervals': True,
    'uncertainty_threshold_action': 'flag_low_confidence'
}

SENTIMENT_ANALYSIS_CONFIG = {
    'enabled': True,
    'cache_hours': 6,
    'weight_in_prediction': 0.15,
    'apis': {
        'finnhub': {
            'enabled': True,
            'priority': 1,
            'max_articles_per_symbol': 10,
            'days_lookback': 3
        },
        'alpha_vantage': {
            'enabled': True,
            'priority': 2,
            'function': 'NEWS_SENTIMENT',
            'max_articles_per_symbol': 20
        },
        'newsapi': {
            'enabled': True,
            'priority': 3,
            'requests_per_month': 900,
            'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com'
        },
        'yfinance': {
            'enabled': True,
            'priority': 4,
            'max_articles_per_symbol': 5
        }
    },
    'processing': {
        'model': 'vader',
        'finbert_fallback': True,
        'sentiment_aggregation': 'weighted_average',
        'time_decay_factor': 0.1
    },
    'fallback_chain': ['finnhub', 'alpha_vantage', 'newsapi', 'yfinance', 'cache']
}

CLASSIFICATION_THRESHOLDS = {
    'adaptive_mode': True,
    'fallback_upper': 0.012,
    'fallback_lower': -0.012,
    'optimization_metric': 'f1_weighted',
    'target_class_ratio': {
        'small_dataset': 0.12,
        'medium_dataset': 0.10,
        'large_dataset': 0.08
    }
}

FOCAL_LOSS_CONFIG = {
    'enabled': True,
    'alpha': 0.25,
    'gamma': 2.0,
    'auto_adjust': True
}

CLASS_WEIGHTING_CONFIG = {
    'enabled': True,
    'strategy': 'balanced_smooth',
    'smoothing_factor': 0.5,
    'auto_calculate': True,
    'manual_weights': {
        0: 3.0,
        1: 0.6,
        2: 3.0
    }
}

ADVANCED_TRAINING_CONFIG = {
    'early_stopping_patience': 15,
    'early_stopping_monitor': 'val_accuracy',
    'reduce_lr_patience': 8,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6,
    'verbose_training': 1
}

ADVANCED_LOGGING = {
    'log_class_distribution': True,
    'log_focal_parameters': True,
    'log_class_weights': True,
    'log_training_metrics': True,
    'log_complexity_analysis': True,
    'log_augmentation_stats': True,
    'log_feature_selection': True,
    'log_overfitting_detection': True,
    'log_uncertainty_metrics': True,
    'log_sentiment_analysis': True,
    'log_bayesian_performance': True
}

DATA_AUGMENTATION_CONFIG = {
    'enabled': True,
    'auto_enable_threshold': 1000,
    'techniques': {
        'jittering': {
            'enabled': True,
            'noise_factor': 0.008,
            'probability': 1.0
        },
        'time_warping': {
            'enabled': True,
            'sigma': 0.15,
            'knot': 3,
            'probability': 0.7
        },
        'magnitude_warping': {
            'enabled': True,
            'sigma': 0.15,
            'knot': 3,
            'probability': 0.5
        }
    },
    'multiplier_strategy': 'adaptive',
    'multipliers': {
        'small_dataset': 3,
        'medium_dataset': 2,
        'large_dataset': 1
    },
    'dataset_size_thresholds': {
        'small': 1000,
        'medium': 5000
    },
    'safety_params': {
        'max_iterations': 1000,
        'timeout_seconds': 10,
        'validation_enabled': True,
        'fallback_to_jittering': True
    }
}

FEATURE_SELECTION_CONFIG = {
    'enabled': True,
    'auto_enable_threshold': 10,
    'selection_methods': {
        'correlation_filter': {
            'enabled': True,
            'threshold': 0.85,
            'priority': 1
        },
        'variance_filter': {
            'enabled': True,
            'threshold': 0.01,
            'priority': 2
        },
        'importance_ranking': {
            'enabled': True,
            'method': 'mutual_info',
            'top_k_ratio': 0.8,
            'priority': 3
        }
    },
    'feature_limits': {
        'min_features': 4,
        'max_features_small_dataset': 8,
        'max_features_medium_dataset': 12,
        'max_features_large_dataset': 20
    }
}

OVERFITTING_DETECTION_CONFIG = {
    'enabled': True,
    'metrics': {
        'validation_gap_threshold': 0.15,
        'train_val_loss_ratio_threshold': 1.5,
        'accuracy_plateau_patience': 5
    },
    'actions': {
        'reduce_complexity': True,
        'increase_regularization': True,
        'early_stopping_aggressive': True
    }
}

PROPHET_FORECAST_PERIODS = 30 
PROPHET_DEFAULT_CONFIGS = {
    'daily_seasonality': False,
    'weekly_seasonality': True,
    'yearly_seasonality': True,
    'seasonality_mode': 'additive',
}

BCB_SERIES_IPCA = 433
BCB_SERIES_SELIC = 432
BCB_SERIES_USD_BRL_EXCHANGE = 1

INVESTPY_COUNTRY_MAP = {
    ".SA": "brazil",
    "": "united states" 
}

DEFAULT_HISTORICAL_PERIOD = "1y"
DEFAULT_INDICATORS_VIEW = ["SMA_21", "EMA_21", "MACD_Hist", "RSI_14"]
ALERT_CHECK_INTERVAL_SECONDS = 60 * 5 
DEFAULT_EXPORT_PATH = "exports/"
DEFAULT_USER_ID_PREFERENCES = 0

THRESHOLD_OPTIMIZATION_CONFIG = {
    'enabled': True,
    'methods': ['f1_weighted', 'gmean', 'kappa'],
    'search_range': (0.001, 0.050),
    'search_steps': 1000,
    'cross_validation_folds': 3,
    'min_samples_per_class': 5,
    'optimization_tolerance': 1e-4
}

CROSS_SYMBOL_VALIDATION_CONFIG = {
    'enabled': True,
    'test_symbols': ['AAPL', 'AMZN', 'MSFT', 'GOOGL'],
    'performance_threshold': 0.55,
    'consistency_check': True
}

HYPERPARAMETER_TUNING_CONFIG = {
    'enabled': True,
    'method': 'random_search',
    'max_trials': 20,
    'optimization_metric': 'val_f1_weighted',
    'param_space': {
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'l2_reg': [0.0001, 0.0002, 0.0005, 0.001],
        'batch_size': [8, 16, 24, 32]
    }
}