import numpy as np
import pandas as pd
import signal
import time
from scipy.interpolate import CubicSpline
from utils.logger import logger
from config.settings import DATA_AUGMENTATION_CONFIG, ADVANCED_LOGGING

class TimeoutException(Exception):
    """Exception para timeout de operações."""
    pass

def timeout_handler(signum, frame):
    """Handler para timeout."""
    raise TimeoutException("Operação de augmentation excedeu tempo limite")

class FinancialTimeSeriesAugmentation:
    def __init__(self):
        self.config = DATA_AUGMENTATION_CONFIG
        self.techniques = self.config.get('techniques', {})
        self.safety_params = self.config.get('safety_params', {})
        self.max_iterations = self.safety_params.get('max_iterations', 1000)
        self.timeout_seconds = self.safety_params.get('timeout_seconds', 10)
        self.validation_enabled = self.safety_params.get('validation_enabled', True)
        self.fallback_to_jittering = self.safety_params.get('fallback_to_jittering', True)
        
    def apply_augmentation(self, df_data: pd.DataFrame, dataset_analysis: dict) -> pd.DataFrame:
        if not self.config.get('enabled', True):
            return df_data
            
        try:
            if not self._validate_input_data(df_data):
                logger.warning("Dados de entrada inválidos para augmentation - retornando dados originais")
                return df_data
            
            dataset_size = dataset_analysis.get('size', len(df_data))
            multiplier = self._determine_multiplier(dataset_size)
            
            if multiplier <= 1:
                return df_data
            
            if self.timeout_seconds > 0:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
            
            start_time = time.time()
            augmented_samples = []
            original_data = df_data.copy()
            
            samples_to_generate = int(len(df_data) * (multiplier - 1))
            samples_to_generate = min(samples_to_generate, self.max_iterations)
            
            success_count = 0
            error_count = 0
            
            for i in range(samples_to_generate):
                try:
                    if time.time() - start_time > self.timeout_seconds * 0.8:
                        logger.warning(f"Aproximando timeout - parando augmentation em {i} amostras")
                        break
                    
                    technique = self._select_random_technique()
                    augmented_sample = self._apply_technique_safely(original_data, technique)
                    
                    if augmented_sample is not None and not augmented_sample.empty:
                        if self._validate_augmented_sample(augmented_sample, original_data):
                            augmented_samples.append(augmented_sample)
                            success_count += 1
                        else:
                            error_count += 1
                            if error_count > 5:
                                logger.warning("Muitos erros de validação - parando augmentation")
                                break
                    else:
                        error_count += 1
                        
                except TimeoutException:
                    logger.error("Timeout durante augmentation - parando processo")
                    break
                except Exception as e:
                    logger.error(f"Erro na iteração {i} do augmentation: {e}")
                    error_count += 1
                    if error_count > 10:
                        logger.error("Muitos erros - abortando augmentation")
                        break
            
            if self.timeout_seconds > 0:
                signal.alarm(0)
            
            if augmented_samples:
                combined_data = pd.concat([original_data] + augmented_samples, ignore_index=True)
                combined_data = self._preserve_temporal_order(combined_data, original_data)
                
                if ADVANCED_LOGGING.get('log_augmentation_stats', True):
                    logger.info(f"Augmentation concluído: {len(combined_data)} amostras totais")
                
                return combined_data
            else:
                logger.warning("Nenhuma amostra válida gerada - retornando dados originais")
                return original_data
                
        except TimeoutException:
            logger.error(f"Timeout de augmentation ({self.timeout_seconds}s) - retornando dados originais")
            return df_data
        except Exception as e:
            logger.error(f"Erro crítico no augmentation: {e}")
            return df_data
        finally:
            try:
                signal.alarm(0)
            except:
                pass
    
    def _validate_input_data(self, df_data: pd.DataFrame) -> bool:
        if df_data is None or df_data.empty:
            return False
            
        if len(df_data) < 10:
            logger.warning(f"Dataset muito pequeno para augmentation: {len(df_data)} amostras")
            return False
            
        numeric_columns = df_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            logger.warning("Insuficientes colunas numéricas para augmentation")
            return False
            
        if df_data.isnull().all().all():
            logger.warning("Dataset contém apenas valores nulos")
            return False
            
        return True
    
    def _validate_augmented_sample(self, augmented_sample: pd.DataFrame, original_data: pd.DataFrame) -> bool:
        if augmented_sample is None or augmented_sample.empty:
            return False
            
        if len(augmented_sample) != len(original_data):
            return False
            
        numeric_columns = augmented_sample.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if augmented_sample[col].isnull().sum() > len(augmented_sample) * 0.1:
                return False
            if np.isinf(augmented_sample[col]).any():
                return False
                
        return True
    
    def _apply_technique_safely(self, original_data: pd.DataFrame, technique: str) -> pd.DataFrame:
        try:
            if technique == 'jittering':
                return self._apply_jittering(original_data)
            elif technique == 'time_warping':
                return self._apply_time_warping_safe(original_data)
            elif technique == 'magnitude_warping':
                return self._apply_magnitude_warping_safe(original_data)
            else:
                return self._apply_jittering(original_data)
                
        except Exception as e:
            logger.warning(f"Erro na técnica {technique}: {e} - usando fallback jittering")
            if self.fallback_to_jittering:
                try:
                    return self._apply_jittering(original_data)
                except:
                    return None
            return None
    
    def _determine_multiplier(self, dataset_size: int) -> int:
        thresholds = self.config.get('dataset_size_thresholds', {})
        multipliers = self.config.get('multipliers', {})
        
        if dataset_size < thresholds.get('small', 1000):
            return multipliers.get('small_dataset', 3)
        elif dataset_size < thresholds.get('medium', 5000):
            return multipliers.get('medium_dataset', 2)
        else:
            return multipliers.get('large_dataset', 1)
    
    def _select_random_technique(self) -> str:
        available_techniques = []
        
        for technique_name, technique_config in self.techniques.items():
            if technique_config.get('enabled', True):
                probability = technique_config.get('probability', 1.0)
                if np.random.random() < probability:
                    available_techniques.append(technique_name)
        
        if available_techniques:
            return np.random.choice(available_techniques)
        else:
            return 'jittering'
    
    def _apply_jittering(self, df_data: pd.DataFrame) -> pd.DataFrame:
        try:
            jitter_config = self.techniques.get('jittering', {})
            noise_factor = jitter_config.get('noise_factor', 0.008)
            
            augmented_data = df_data.copy()
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['target_classes', 'target_return']:
                    continue
                    
                original_values = df_data[col].values
                
                if len(original_values) == 0 or np.all(np.isnan(original_values)):
                    continue
                
                std_dev = np.nanstd(original_values) * noise_factor
                if std_dev == 0 or np.isnan(std_dev):
                    continue
                    
                noise = np.random.normal(0, std_dev, len(original_values))
                augmented_data[col] = original_values + noise
                
                if col == 'volume' and (augmented_data[col] < 0).any():
                    augmented_data[col] = np.abs(augmented_data[col])
            
            return augmented_data
            
        except Exception as e:
            logger.error(f"Erro no jittering: {e}")
            return None
    
    def _apply_time_warping_safe(self, df_data: pd.DataFrame) -> pd.DataFrame:
        try:
            warp_config = self.techniques.get('time_warping', {})
            sigma = warp_config.get('sigma', 0.15)
            knot = warp_config.get('knot', 3)
            
            augmented_data = df_data.copy()
            data_length = len(df_data)
            
            if data_length < max(10, knot * 3):
                logger.debug(f"Dataset muito pequeno para time_warping: {data_length} < {knot * 3}")
                return self._apply_jittering(df_data)
            
            if knot < 2:
                logger.debug("Knots insuficientes para CubicSpline")
                return self._apply_jittering(df_data)
            
            try:
                orig_steps = np.arange(data_length)
                
                if sigma <= 0 or sigma > 1.0:
                    sigma = 0.15
                    logger.debug(f"Sigma ajustado para valor seguro: {sigma}")
                
                random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
                random_warps = np.clip(random_warps, 0.5, 2.0)
                
                warp_steps = (np.ones((knot + 2,)) * (data_length - 1) / (knot + 1)) * np.arange(knot + 2)
                
                if len(warp_steps) != len(random_warps):
                    logger.debug("Incompatibilidade de dimensões para CubicSpline")
                    return self._apply_jittering(df_data)
                
                if np.any(np.isnan(warp_steps)) or np.any(np.isnan(random_warps)):
                    logger.debug("Valores NaN detectados para CubicSpline")
                    return self._apply_jittering(df_data)
                
                start_spline_time = time.time()
                warper = CubicSpline(warp_steps, warp_steps * random_warps)
                
                if time.time() - start_spline_time > 1.0:
                    logger.warning("CubicSpline demorou muito - usando fallback")
                    return self._apply_jittering(df_data)
                
                new_steps = warper(orig_steps)
                new_steps = np.clip(new_steps, 0, data_length - 1)
                
                if np.any(np.isnan(new_steps)) or np.any(np.isinf(new_steps)):
                    logger.debug("CubicSpline gerou valores inválidos")
                    return self._apply_jittering(df_data)
                
            except Exception as spline_error:
                logger.debug(f"Erro na criação/uso de CubicSpline: {spline_error}")
                return self._apply_jittering(df_data)
            
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['target_classes', 'target_return']:
                    continue
                    
                original_values = df_data[col].values
                
                if len(original_values) > 1 and not np.all(np.isnan(original_values)):
                    try:
                        warped_values = np.interp(new_steps, orig_steps, original_values)
                        
                        if not np.any(np.isnan(warped_values)) and not np.any(np.isinf(warped_values)):
                            augmented_data[col] = warped_values
                        
                    except Exception as interp_error:
                        logger.debug(f"Erro na interpolação da coluna {col}: {interp_error}")
                        continue
            
            return augmented_data
            
        except Exception as e:
            logger.debug(f"Erro geral no time warping: {e} - usando fallback")
            return self._apply_jittering(df_data)
    
    def _apply_magnitude_warping_safe(self, df_data: pd.DataFrame) -> pd.DataFrame:
        try:
            mag_config = self.techniques.get('magnitude_warping', {})
            sigma = mag_config.get('sigma', 0.15)
            knot = mag_config.get('knot', 3)
            
            augmented_data = df_data.copy()
            data_length = len(df_data)
            
            if data_length < max(10, knot * 3):
                logger.debug(f"Dataset muito pequeno para magnitude_warping: {data_length}")
                return self._apply_jittering(df_data)
            
            if knot < 2:
                logger.debug("Knots insuficientes para magnitude_warping")
                return self._apply_jittering(df_data)
            
            try:
                orig_steps = np.arange(data_length)
                
                if sigma <= 0 or sigma > 1.0:
                    sigma = 0.15
                
                random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
                random_warps = np.clip(random_warps, 0.5, 2.0)
                
                warp_steps = (np.ones((knot + 2,)) * (data_length - 1) / (knot + 1)) * np.arange(knot + 2)
                
                if len(warp_steps) != len(random_warps) or np.any(np.isnan(warp_steps)) or np.any(np.isnan(random_warps)):
                    return self._apply_jittering(df_data)
                
                start_spline_time = time.time()
                warper = CubicSpline(warp_steps, random_warps)
                
                if time.time() - start_spline_time > 1.0:
                    return self._apply_jittering(df_data)
                
                multipliers = warper(orig_steps)
                multipliers = np.clip(multipliers, 0.5, 2.0)
                
                if np.any(np.isnan(multipliers)) or np.any(np.isinf(multipliers)):
                    return self._apply_jittering(df_data)
                
            except Exception as spline_error:
                logger.debug(f"Erro na magnitude warping CubicSpline: {spline_error}")
                return self._apply_jittering(df_data)
            
            numeric_columns = df_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['target_classes', 'target_return', 'day_of_week', 'month', 'volume']:
                    continue
                    
                original_values = df_data[col].values
                if not np.all(np.isnan(original_values)):
                    warped_values = original_values * multipliers
                    
                    if not np.any(np.isnan(warped_values)) and not np.any(np.isinf(warped_values)):
                        augmented_data[col] = warped_values
            
            return augmented_data
            
        except Exception as e:
            logger.debug(f"Erro geral no magnitude warping: {e} - usando fallback")
            return self._apply_jittering(df_data)
    
    def _preserve_temporal_order(self, combined_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'target_classes' in combined_data.columns:
                for i, (idx, row) in enumerate(combined_data.iterrows()):
                    if i < len(original_data):
                        combined_data.loc[idx, 'target_classes'] = original_data.iloc[i]['target_classes']
                    else:
                        similar_idx = np.random.choice(len(original_data))
                        combined_data.loc[idx, 'target_classes'] = original_data.iloc[similar_idx]['target_classes']
            
            if 'target_return' in combined_data.columns:
                for i, (idx, row) in enumerate(combined_data.iterrows()):
                    if i < len(original_data):
                        combined_data.loc[idx, 'target_return'] = original_data.iloc[i]['target_return']
                    else:
                        similar_idx = np.random.choice(len(original_data))
                        combined_data.loc[idx, 'target_return'] = original_data.iloc[similar_idx]['target_return']
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Erro ao preservar ordem temporal: {e}")
            return combined_data
    
    def validate_augmented_data(self, original_data: pd.DataFrame, augmented_data: pd.DataFrame) -> bool:
        try:
            if len(augmented_data) <= len(original_data):
                return False
            
            numeric_columns = original_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['target_classes', 'target_return']:
                    continue
                    
                orig_mean = np.nanmean(original_data[col])
                orig_std = np.nanstd(original_data[col])
                
                aug_mean = np.nanmean(augmented_data[col])
                aug_std = np.nanstd(augmented_data[col])
                
                if abs(aug_mean - orig_mean) > orig_std * 0.3:
                    logger.debug(f"Média muito diferente para {col}: orig={orig_mean:.3f}, aug={aug_mean:.3f}")
                    return False
                
                if abs(aug_std - orig_std) > orig_std * 0.3:
                    logger.debug(f"Desvio muito diferente para {col}: orig={orig_std:.3f}, aug={aug_std:.3f}")
                    return False
                
                if np.any(np.isinf(augmented_data[col])) or np.any(np.isnan(augmented_data[col])):
                    logger.debug(f"Valores inválidos detectados em {col}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação dos dados aumentados: {e}")
            return False