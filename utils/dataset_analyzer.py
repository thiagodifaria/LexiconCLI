import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import Counter
from utils.logger import logger

class DatasetAnalyzer:
    def __init__(self):
        self.analysis_cache = {}
        
    def comprehensive_analysis(self, df_data: pd.DataFrame) -> dict:
        try:
            data_hash = str(hash(str(df_data.shape) + str(df_data.columns.tolist())))
            
            if data_hash in self.analysis_cache:
                return self.analysis_cache[data_hash]
            
            analysis = {
                'size': len(df_data),
                'features': df_data.shape[1],
                'missing_data_ratio': self._calculate_missing_data_ratio(df_data),
                'noise_level': self._estimate_noise_level(df_data),
                'stationarity': self._test_stationarity(df_data),
                'class_balance': self._analyze_class_distribution(df_data),
                'temporal_patterns': self._analyze_temporal_patterns(df_data),
                'feature_types': self._classify_feature_types(df_data),
                'data_quality': self._assess_data_quality(df_data),
                'complexity_indicators': self._calculate_complexity_indicators(df_data),
                'statistical_properties': self._calculate_statistical_properties(df_data)
            }
            
            self.analysis_cache[data_hash] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise abrangente do dataset: {e}")
            return {
                'size': len(df_data) if not df_data.empty else 0,
                'features': df_data.shape[1] if not df_data.empty else 0,
                'error': str(e)
            }
    
    def _calculate_missing_data_ratio(self, df_data: pd.DataFrame) -> float:
        if df_data.empty:
            return 1.0
        
        total_cells = df_data.size
        missing_cells = df_data.isna().sum().sum()
        
        return missing_cells / total_cells if total_cells > 0 else 1.0
    
    def _estimate_noise_level(self, df_data: pd.DataFrame) -> str:
        try:
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return 'unknown'
            
            noise_indicators = []
            
            for col in numeric_columns:
                series = df_data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                series_diff = np.diff(series)
                
                if len(series_diff) > 0:
                    noise_ratio = np.std(series_diff) / (np.std(series) + 1e-8)
                    noise_indicators.append(noise_ratio)
            
            if not noise_indicators:
                return 'unknown'
            
            avg_noise = np.mean(noise_indicators)
            
            if avg_noise < 0.1:
                return 'low'
            elif avg_noise < 0.3:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            logger.error(f"Erro na estimativa de ruído: {e}")
            return 'unknown'
    
    def _test_stationarity(self, df_data: pd.DataFrame) -> dict:
        try:
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return {'overall': 'unknown', 'details': {}}
            
            stationarity_results = {}
            
            for col in numeric_columns:
                series = df_data[col].dropna()
                
                if len(series) < 20:
                    stationarity_results[col] = 'insufficient_data'
                    continue
                
                try:
                    mean_stability = self._test_mean_stability(series)
                    variance_stability = self._test_variance_stability(series)
                    
                    if mean_stability and variance_stability:
                        stationarity_results[col] = 'stationary'
                    elif mean_stability or variance_stability:
                        stationarity_results[col] = 'weak_stationary'
                    else:
                        stationarity_results[col] = 'non_stationary'
                        
                except Exception:
                    stationarity_results[col] = 'test_failed'
            
            stationary_count = sum(1 for result in stationarity_results.values() if result == 'stationary')
            total_tested = len([r for r in stationarity_results.values() if r != 'insufficient_data'])
            
            if total_tested == 0:
                overall_status = 'unknown'
            elif stationary_count / total_tested > 0.7:
                overall_status = 'mostly_stationary'
            elif stationary_count / total_tested > 0.3:
                overall_status = 'mixed'
            else:
                overall_status = 'mostly_non_stationary'
            
            return {
                'overall': overall_status,
                'details': stationarity_results
            }
            
        except Exception as e:
            logger.error(f"Erro no teste de estacionariedade: {e}")
            return {'overall': 'unknown', 'details': {}}
    
    def _test_mean_stability(self, series: pd.Series, window_size: int = None) -> bool:
        if window_size is None:
            window_size = max(10, len(series) // 4)
        
        if len(series) < window_size * 2:
            return True
        
        rolling_means = series.rolling(window=window_size).mean().dropna()
        
        if len(rolling_means) < 2:
            return True
        
        mean_trend = np.polyfit(range(len(rolling_means)), rolling_means, 1)[0]
        mean_std = np.std(rolling_means)
        
        return abs(mean_trend) < mean_std * 0.1
    
    def _test_variance_stability(self, series: pd.Series, window_size: int = None) -> bool:
        if window_size is None:
            window_size = max(10, len(series) // 4)
        
        if len(series) < window_size * 2:
            return True
        
        rolling_vars = series.rolling(window=window_size).var().dropna()
        
        if len(rolling_vars) < 2:
            return True
        
        var_ratio = rolling_vars.max() / (rolling_vars.min() + 1e-8)
        
        return var_ratio < 4.0
    
    def _analyze_class_distribution(self, df_data: pd.DataFrame) -> dict:
        try:
            class_columns = [col for col in df_data.columns if 'class' in col.lower() or 'target' in col.lower()]
            
            if not class_columns:
                return {'status': 'no_target_found', 'details': {}}
            
            results = {}
            
            for col in class_columns:
                if col in df_data.columns:
                    value_counts = df_data[col].value_counts()
                    
                    if len(value_counts) == 0:
                        continue
                    
                    total_samples = len(df_data[col].dropna())
                    
                    class_distribution = {}
                    for class_value, count in value_counts.items():
                        class_distribution[str(class_value)] = {
                            'count': int(count),
                            'percentage': float(count / total_samples * 100)
                        }
                    
                    imbalance_ratio = value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf')
                    
                    if imbalance_ratio < 1.5:
                        balance_status = 'balanced'
                    elif imbalance_ratio < 3.0:
                        balance_status = 'slightly_imbalanced'
                    elif imbalance_ratio < 10.0:
                        balance_status = 'imbalanced'
                    else:
                        balance_status = 'severely_imbalanced'
                    
                    results[col] = {
                        'distribution': class_distribution,
                        'balance_status': balance_status,
                        'imbalance_ratio': float(imbalance_ratio),
                        'num_classes': len(value_counts)
                    }
            
            if results:
                main_target = list(results.keys())[0]
                return {
                    'status': 'analyzed',
                    'main_target': main_target,
                    'details': results[main_target],
                    'all_targets': results
                }
            else:
                return {'status': 'no_valid_targets', 'details': {}}
                
        except Exception as e:
            logger.error(f"Erro na análise de distribuição de classes: {e}")
            return {'status': 'error', 'details': {'error': str(e)}}
    
    def _analyze_temporal_patterns(self, df_data: pd.DataFrame) -> dict:
        try:
            if not isinstance(df_data.index, pd.DatetimeIndex):
                return {'status': 'no_datetime_index', 'patterns': {}}
            
            patterns = {}
            
            if len(df_data) > 7:
                patterns['weekday_effect'] = self._detect_weekday_effect(df_data)
            
            if len(df_data) > 30:
                patterns['monthly_seasonality'] = self._detect_monthly_seasonality(df_data)
            
            if len(df_data) > 100:
                patterns['trend_analysis'] = self._analyze_trends(df_data)
            
            patterns['temporal_gaps'] = self._detect_temporal_gaps(df_data)
            
            return {
                'status': 'analyzed',
                'patterns': patterns
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de padrões temporais: {e}")
            return {'status': 'error', 'patterns': {}}
    
    def _detect_weekday_effect(self, df_data: pd.DataFrame) -> dict:
        try:
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return {'detected': False}
            
            df_data['weekday'] = df_data.index.dayofweek
            
            weekday_effects = {}
            
            for col in numeric_columns[:3]:
                weekday_means = df_data.groupby('weekday')[col].mean()
                
                if len(weekday_means) > 1:
                    weekday_std = weekday_means.std()
                    overall_std = df_data[col].std()
                    
                    effect_strength = weekday_std / (overall_std + 1e-8)
                    weekday_effects[col] = {
                        'effect_strength': float(effect_strength),
                        'significant': effect_strength > 0.1
                    }
            
            significant_effects = sum(1 for effect in weekday_effects.values() if effect['significant'])
            
            return {
                'detected': significant_effects > 0,
                'strength': 'strong' if significant_effects > len(weekday_effects) * 0.5 else 'weak',
                'details': weekday_effects
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de efeito de dia da semana: {e}")
            return {'detected': False}
    
    def _detect_monthly_seasonality(self, df_data: pd.DataFrame) -> dict:
        try:
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return {'detected': False}
            
            df_data['month'] = df_data.index.month
            
            monthly_effects = {}
            
            for col in numeric_columns[:3]:
                monthly_means = df_data.groupby('month')[col].mean()
                
                if len(monthly_means) > 1:
                    monthly_std = monthly_means.std()
                    overall_std = df_data[col].std()
                    
                    effect_strength = monthly_std / (overall_std + 1e-8)
                    monthly_effects[col] = {
                        'effect_strength': float(effect_strength),
                        'significant': effect_strength > 0.05
                    }
            
            significant_effects = sum(1 for effect in monthly_effects.values() if effect['significant'])
            
            return {
                'detected': significant_effects > 0,
                'strength': 'strong' if significant_effects > len(monthly_effects) * 0.5 else 'weak',
                'details': monthly_effects
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de sazonalidade mensal: {e}")
            return {'detected': False}
    
    def _analyze_trends(self, df_data: pd.DataFrame) -> dict:
        try:
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return {'trends_detected': False}
            
            trends = {}
            
            for col in numeric_columns[:5]:
                series = df_data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                
                trend_strength = abs(r_value)
                
                if trend_strength > 0.3 and p_value < 0.05:
                    trend_direction = 'upward' if slope > 0 else 'downward'
                    trend_significance = 'strong' if trend_strength > 0.7 else 'moderate'
                else:
                    trend_direction = 'none'
                    trend_significance = 'weak'
                
                trends[col] = {
                    'direction': trend_direction,
                    'strength': float(trend_strength),
                    'significance': trend_significance,
                    'slope': float(slope),
                    'p_value': float(p_value)
                }
            
            return {
                'trends_detected': any(t['direction'] != 'none' for t in trends.values()),
                'details': trends
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de tendências: {e}")
            return {'trends_detected': False}
    
    def _detect_temporal_gaps(self, df_data: pd.DataFrame) -> dict:
        try:
            if len(df_data) < 2:
                return {'gaps_detected': False}
            
            time_diffs = pd.Series(df_data.index).diff().dropna()
            
            median_diff = time_diffs.median()
            
            large_gaps = time_diffs[time_diffs > median_diff * 3]
            
            return {
                'gaps_detected': len(large_gaps) > 0,
                'num_gaps': len(large_gaps),
                'largest_gap': str(large_gaps.max()) if len(large_gaps) > 0 else None,
                'median_interval': str(median_diff)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de gaps temporais: {e}")
            return {'gaps_detected': False}
    
    def _classify_feature_types(self, df_data: pd.DataFrame) -> dict:
        try:
            feature_types = {
                'numeric': [],
                'categorical': [],
                'datetime': [],
                'binary': [],
                'high_cardinality': []
            }
            
            for col in df_data.columns:
                if pd.api.types.is_numeric_dtype(df_data[col]):
                    unique_values = df_data[col].nunique()
                    
                    if unique_values == 2:
                        feature_types['binary'].append(col)
                    else:
                        feature_types['numeric'].append(col)
                        
                elif pd.api.types.is_datetime64_any_dtype(df_data[col]):
                    feature_types['datetime'].append(col)
                    
                else:
                    unique_values = df_data[col].nunique()
                    total_values = len(df_data[col].dropna())
                    
                    if unique_values > total_values * 0.5:
                        feature_types['high_cardinality'].append(col)
                    else:
                        feature_types['categorical'].append(col)
            
            return feature_types
            
        except Exception as e:
            logger.error(f"Erro na classificação de tipos de features: {e}")
            return {}
    
    def _assess_data_quality(self, df_data: pd.DataFrame) -> dict:
        try:
            quality_metrics = {}
            
            missing_ratio = self._calculate_missing_data_ratio(df_data)
            quality_metrics['missing_data_ratio'] = missing_ratio
            
            if missing_ratio < 0.05:
                quality_metrics['missing_data_status'] = 'excellent'
            elif missing_ratio < 0.15:
                quality_metrics['missing_data_status'] = 'good'
            elif missing_ratio < 0.30:
                quality_metrics['missing_data_status'] = 'acceptable'
            else:
                quality_metrics['missing_data_status'] = 'poor'
            
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                outlier_ratios = []
                
                for col in numeric_columns:
                    series = df_data[col].dropna()
                    
                    if len(series) > 10:
                        Q1 = series.quantile(0.25)
                        Q3 = series.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = series[(series < lower_bound) | (series > upper_bound)]
                        outlier_ratio = len(outliers) / len(series)
                        outlier_ratios.append(outlier_ratio)
                
                if outlier_ratios:
                    avg_outlier_ratio = np.mean(outlier_ratios)
                    quality_metrics['outlier_ratio'] = avg_outlier_ratio
                    
                    if avg_outlier_ratio < 0.05:
                        quality_metrics['outlier_status'] = 'low'
                    elif avg_outlier_ratio < 0.15:
                        quality_metrics['outlier_status'] = 'moderate'
                    else:
                        quality_metrics['outlier_status'] = 'high'
            
            if len(df_data) < 100:
                quality_metrics['size_adequacy'] = 'insufficient'
            elif len(df_data) < 1000:
                quality_metrics['size_adequacy'] = 'small'
            elif len(df_data) < 10000:
                quality_metrics['size_adequacy'] = 'adequate'
            else:
                quality_metrics['size_adequacy'] = 'large'
            
            quality_scores = []
            
            if quality_metrics.get('missing_data_status') == 'excellent':
                quality_scores.append(1.0)
            elif quality_metrics.get('missing_data_status') == 'good':
                quality_scores.append(0.8)
            elif quality_metrics.get('missing_data_status') == 'acceptable':
                quality_scores.append(0.6)
            else:
                quality_scores.append(0.3)
            
            if quality_metrics.get('outlier_status') == 'low':
                quality_scores.append(1.0)
            elif quality_metrics.get('outlier_status') == 'moderate':
                quality_scores.append(0.7)
            else:
                quality_scores.append(0.4)
            
            if quality_metrics.get('size_adequacy') == 'large':
                quality_scores.append(1.0)
            elif quality_metrics.get('size_adequacy') == 'adequate':
                quality_scores.append(0.8)
            elif quality_metrics.get('size_adequacy') == 'small':
                quality_scores.append(0.5)
            else:
                quality_scores.append(0.2)
            
            overall_quality = np.mean(quality_scores) if quality_scores else 0.5
            quality_metrics['overall_quality_score'] = overall_quality
            
            if overall_quality > 0.8:
                quality_metrics['overall_quality'] = 'high'
            elif overall_quality > 0.6:
                quality_metrics['overall_quality'] = 'medium'
            else:
                quality_metrics['overall_quality'] = 'low'
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação de qualidade dos dados: {e}")
            return {'overall_quality': 'unknown'}
    
    def _calculate_complexity_indicators(self, df_data: pd.DataFrame) -> dict:
        try:
            indicators = {}
            
            indicators['dataset_size'] = len(df_data)
            indicators['feature_count'] = df_data.shape[1]
            indicators['feature_to_sample_ratio'] = df_data.shape[1] / len(df_data) if len(df_data) > 0 else float('inf')
            
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 1:
                correlation_matrix = df_data[numeric_columns].corr().abs()
                
                upper_triangle = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                
                high_correlations = (upper_triangle > 0.8).sum().sum()
                total_pairs = len(numeric_columns) * (len(numeric_columns) - 1) / 2
                
                indicators['high_correlation_ratio'] = high_correlations / total_pairs if total_pairs > 0 else 0
                indicators['feature_redundancy'] = 'high' if indicators['high_correlation_ratio'] > 0.3 else 'low'
            
            categorical_columns = df_data.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                cardinalities = [df_data[col].nunique() for col in categorical_columns]
                indicators['avg_categorical_cardinality'] = np.mean(cardinalities)
                indicators['max_categorical_cardinality'] = max(cardinalities)
            
            missing_ratios = df_data.isna().mean()
            indicators['features_with_missing'] = (missing_ratios > 0).sum()
            indicators['avg_missing_ratio'] = missing_ratios.mean()
            
            if indicators['feature_to_sample_ratio'] > 0.1:
                indicators['complexity_level'] = 'high'
            elif indicators['feature_to_sample_ratio'] > 0.05:
                indicators['complexity_level'] = 'medium'
            else:
                indicators['complexity_level'] = 'low'
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro no cálculo de indicadores de complexidade: {e}")
            return {}
    
    def _calculate_statistical_properties(self, df_data: pd.DataFrame) -> dict:
        try:
            properties = {}
            
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return properties
            
            for col in numeric_columns[:5]:
                series = df_data[col].dropna()
                
                if len(series) < 3:
                    continue
                
                col_properties = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'range': float(series.max() - series.min()),
                    'coefficient_of_variation': float(series.std() / abs(series.mean())) if series.mean() != 0 else float('inf')
                }
                
                if abs(col_properties['skewness']) > 2:
                    col_properties['skewness_level'] = 'high'
                elif abs(col_properties['skewness']) > 0.5:
                    col_properties['skewness_level'] = 'moderate'
                else:
                    col_properties['skewness_level'] = 'low'
                
                if col_properties['kurtosis'] > 3:
                    col_properties['kurtosis_level'] = 'high'
                elif col_properties['kurtosis'] > 0:
                    col_properties['kurtosis_level'] = 'moderate'
                else:
                    col_properties['kurtosis_level'] = 'low'
                
                properties[col] = col_properties
            
            return properties
            
        except Exception as e:
            logger.error(f"Erro no cálculo de propriedades estatísticas: {e}")
            return {}

    def detect_regime_changes(self, price_series: pd.Series) -> dict:
        try:
            if len(price_series) < 50:
                return {'regime_changes_detected': False, 'reason': 'insufficient_data'}
            
            returns = price_series.pct_change().dropna()
            
            if len(returns) < 30:
                return {'regime_changes_detected': False, 'reason': 'insufficient_returns'}
            
            window_size = min(20, len(returns) // 3)
            
            rolling_vol = returns.rolling(window=window_size).std()
            rolling_mean = returns.rolling(window=window_size).mean()
            
            vol_changes = []
            mean_changes = []
            
            for i in range(window_size, len(rolling_vol) - window_size):
                before_vol = rolling_vol.iloc[i-window_size:i].mean()
                after_vol = rolling_vol.iloc[i:i+window_size].mean()
                
                before_mean = rolling_mean.iloc[i-window_size:i].mean()
                after_mean = rolling_mean.iloc[i:i+window_size].mean()
                
                if before_vol > 0:
                    vol_change = abs(after_vol - before_vol) / before_vol
                    if vol_change > 0.5:
                        vol_changes.append((i, vol_change))
                
                if abs(before_mean) > 1e-6:
                    mean_change = abs(after_mean - before_mean) / abs(before_mean)
                    if mean_change > 0.3:
                        mean_changes.append((i, mean_change))
            
            significant_changes = len(vol_changes) + len(mean_changes)
            
            return {
                'regime_changes_detected': significant_changes > 0,
                'volatility_regime_changes': len(vol_changes),
                'mean_regime_changes': len(mean_changes),
                'total_changes': significant_changes,
                'change_frequency': significant_changes / len(returns) * 100
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de mudanças de regime: {e}")
            return {'regime_changes_detected': False, 'reason': 'analysis_error'}