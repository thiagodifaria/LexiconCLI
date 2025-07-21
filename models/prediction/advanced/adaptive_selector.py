import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr
from utils.logger import logger
from config.settings import FEATURE_SELECTION_CONFIG, ADVANCED_LOGGING

class AdaptiveFeatureSelector:
    def __init__(self):
        self.config = FEATURE_SELECTION_CONFIG
        self.selection_methods = self.config.get('selection_methods', {})
        self.feature_limits = self.config.get('feature_limits', {})
        
    def select_features(self, df_data: pd.DataFrame, available_features: list, target_column: str, dataset_analysis: dict) -> list:
        if not self.config.get('enabled', True):
            return available_features[:6]
            
        dataset_size = dataset_analysis.get('size', len(df_data))
        
        if len(available_features) < self.config.get('auto_enable_threshold', 10):
            if ADVANCED_LOGGING.get('log_feature_selection', True):
                logger.info("Poucas features disponíveis: usando todas")
            return available_features
        
        max_features = self._determine_max_features(dataset_size)
        
        if len(available_features) <= max_features:
            return available_features
        
        selected_features = available_features.copy()
        
        if self.selection_methods.get('correlation_filter', {}).get('enabled', True):
            selected_features = self._apply_correlation_filter(df_data, selected_features, target_column)
        
        if self.selection_methods.get('variance_filter', {}).get('enabled', True):
            selected_features = self._apply_variance_filter(df_data, selected_features)
        
        if self.selection_methods.get('importance_ranking', {}).get('enabled', True):
            selected_features = self._apply_importance_ranking(df_data, selected_features, target_column, max_features)
        
        final_features = self._ensure_minimum_features(selected_features)
        
        if ADVANCED_LOGGING.get('log_feature_selection', True):
            logger.info(f"Feature selection: {len(available_features)} -> {len(final_features)} features")
        
        return final_features
    
    def _determine_max_features(self, dataset_size: int) -> int:
        if dataset_size < 1000:
            return self.feature_limits.get('max_features_small_dataset', 8)
        elif dataset_size < 5000:
            return self.feature_limits.get('max_features_medium_dataset', 12)
        else:
            return self.feature_limits.get('max_features_large_dataset', 20)
    
    def _apply_correlation_filter(self, df_data: pd.DataFrame, features: list, target_column: str) -> list:
        try:
            correlation_config = self.selection_methods.get('correlation_filter', {})
            threshold = correlation_config.get('threshold', 0.85)
            
            numeric_features = [f for f in features if f in df_data.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_features) < 2:
                return features
            
            feature_data = df_data[numeric_features].dropna()
            
            if feature_data.empty:
                return features
            
            correlation_matrix = feature_data.corr().abs()
            
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            highly_correlated_pairs = []
            for col in upper_triangle.columns:
                for idx in upper_triangle.index:
                    if upper_triangle.loc[idx, col] > threshold:
                        highly_correlated_pairs.append((idx, col, upper_triangle.loc[idx, col]))
            
            features_to_remove = set()
            
            target_correlations = {}
            if target_column in df_data.columns:
                for feat in numeric_features:
                    try:
                        target_corr = abs(df_data[feat].corr(df_data[target_column]))
                        target_correlations[feat] = target_corr if not np.isnan(target_corr) else 0
                    except:
                        target_correlations[feat] = 0
            
            for feat1, feat2, corr_value in highly_correlated_pairs:
                if feat1 not in features_to_remove and feat2 not in features_to_remove:
                    target_corr1 = target_correlations.get(feat1, 0)
                    target_corr2 = target_correlations.get(feat2, 0)
                    
                    if target_corr1 < target_corr2:
                        features_to_remove.add(feat1)
                    else:
                        features_to_remove.add(feat2)
            
            filtered_features = [f for f in features if f not in features_to_remove]
            
            if ADVANCED_LOGGING.get('log_feature_selection', True) and features_to_remove:
                logger.info(f"Removidas por alta correlação: {len(features_to_remove)} features")
            
            return filtered_features if filtered_features else features
            
        except Exception as e:
            logger.error(f"Erro no filtro de correlação: {e}")
            return features
    
    def _apply_variance_filter(self, df_data: pd.DataFrame, features: list) -> list:
        try:
            variance_config = self.selection_methods.get('variance_filter', {})
            threshold = variance_config.get('threshold', 0.01)
            
            numeric_features = [f for f in features if f in df_data.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_features) < 2:
                return features
            
            feature_data = df_data[numeric_features].dropna()
            
            if feature_data.empty:
                return features
            
            variances = feature_data.var()
            low_variance_features = variances[variances < threshold].index.tolist()
            
            filtered_features = [f for f in features if f not in low_variance_features]
            
            if ADVANCED_LOGGING.get('log_feature_selection', True) and low_variance_features:
                logger.info(f"Removidas por baixa variância: {len(low_variance_features)} features")
            
            return filtered_features if filtered_features else features
            
        except Exception as e:
            logger.error(f"Erro no filtro de variância: {e}")
            return features
    
    def _apply_importance_ranking(self, df_data: pd.DataFrame, features: list, target_column: str, max_features: int) -> list:
        try:
            importance_config = self.selection_methods.get('importance_ranking', {})
            method = importance_config.get('method', 'mutual_info')
            top_k_ratio = importance_config.get('top_k_ratio', 0.8)
            
            if target_column not in df_data.columns:
                return features[:max_features]
            
            feature_data = df_data[features + [target_column]].dropna()
            
            if feature_data.empty or len(feature_data) < 10:
                return features[:max_features]
            
            target_data = feature_data[target_column]
            
            if target_data.dtype == 'object' or target_data.nunique() < 10:
                if target_data.dtype == 'object':
                    le = LabelEncoder()
                    target_encoded = le.fit_transform(target_data.astype(str))
                else:
                    target_encoded = target_data.values
                    
                importance_method = self._calculate_mutual_info_importance
            else:
                target_encoded = target_data.values
                importance_method = self._calculate_correlation_importance
            
            feature_importance = {}
            
            for feature in features:
                if feature in feature_data.columns:
                    feature_values = feature_data[feature].values
                    
                    if len(feature_values) > 0:
                        importance = importance_method(feature_values, target_encoded)
                        feature_importance[feature] = importance
            
            if not feature_importance:
                return features[:max_features]
            
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            num_features_to_select = min(max_features, int(len(sorted_features) * top_k_ratio))
            num_features_to_select = max(num_features_to_select, self.feature_limits.get('min_features', 4))
            
            selected_features = [feat for feat, _ in sorted_features[:num_features_to_select]]
            
            if ADVANCED_LOGGING.get('log_feature_selection', True):
                logger.info(f"Ranking aplicado: top {num_features_to_select} features selecionadas")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Erro no ranking de importância: {e}")
            return features[:max_features]
    
    def _calculate_mutual_info_importance(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        try:
            if len(np.unique(feature_values)) < 2:
                return 0.0
            
            feature_values_reshaped = feature_values.reshape(-1, 1)
            mi_scores = mutual_info_classif(feature_values_reshaped, target_values, random_state=42)
            
            return mi_scores[0]
            
        except Exception as e:
            logger.error(f"Erro no cálculo de mutual information: {e}")
            return 0.0
    
    def _calculate_correlation_importance(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        try:
            if len(np.unique(feature_values)) < 2 or len(np.unique(target_values)) < 2:
                return 0.0
            
            pearson_corr, _ = pearsonr(feature_values, target_values)
            
            if np.isnan(pearson_corr):
                spearman_corr, _ = spearmanr(feature_values, target_values)
                return abs(spearman_corr) if not np.isnan(spearman_corr) else 0.0
            
            return abs(pearson_corr)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de correlação: {e}")
            return 0.0
    
    def _ensure_minimum_features(self, features: list) -> list:
        min_features = self.feature_limits.get('min_features', 4)
        
        if len(features) < min_features:
            logger.warning(f"Número de features abaixo do mínimo ({min_features}). Mantendo features essenciais.")
            
            essential_features = ['rsi', 'macd', 'atr', 'volume_ratio', 'day_of_week', 'close']
            
            for essential in essential_features:
                if essential not in features and len(features) < min_features:
                    features.append(essential)
        
        return features
    
    def analyze_feature_relationships(self, df_data: pd.DataFrame, features: list, target_column: str) -> dict:
        try:
            analysis = {
                'correlation_matrix': {},
                'target_correlations': {},
                'feature_stats': {},
                'redundancy_groups': []
            }
            
            numeric_features = [f for f in features if f in df_data.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_features) < 2:
                return analysis
            
            feature_data = df_data[numeric_features].dropna()
            
            if feature_data.empty:
                return analysis
            
            correlation_matrix = feature_data.corr()
            analysis['correlation_matrix'] = correlation_matrix.to_dict()
            
            if target_column in df_data.columns:
                target_correlations = {}
                for feature in numeric_features:
                    if feature != target_column:
                        try:
                            corr = df_data[feature].corr(df_data[target_column])
                            target_correlations[feature] = corr if not np.isnan(corr) else 0.0
                        except:
                            target_correlations[feature] = 0.0
                analysis['target_correlations'] = target_correlations
            
            for feature in numeric_features:
                try:
                    stats = {
                        'mean': float(feature_data[feature].mean()),
                        'std': float(feature_data[feature].std()),
                        'min': float(feature_data[feature].min()),
                        'max': float(feature_data[feature].max()),
                        'missing_ratio': float(df_data[feature].isna().mean())
                    }
                    analysis['feature_stats'][feature] = stats
                except:
                    continue
            
            redundancy_groups = []
            processed = set()
            threshold = self.selection_methods.get('correlation_filter', {}).get('threshold', 0.85)
            
            for i, feat1 in enumerate(numeric_features):
                if feat1 in processed:
                    continue
                    
                group = [feat1]
                processed.add(feat1)
                
                for j, feat2 in enumerate(numeric_features[i+1:], i+1):
                    if feat2 in processed:
                        continue
                        
                    try:
                        corr = abs(correlation_matrix.loc[feat1, feat2])
                        if corr > threshold:
                            group.append(feat2)
                            processed.add(feat2)
                    except:
                        continue
                
                if len(group) > 1:
                    redundancy_groups.append(group)
            
            analysis['redundancy_groups'] = redundancy_groups
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de relacionamentos: {e}")
            return {}