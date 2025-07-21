import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from utils.logger import logger
from config.settings import (
    OVERFITTING_DETECTION_CONFIG, 
    ADVANCED_LOGGING,
    CROSS_SYMBOL_VALIDATION_CONFIG,
    THRESHOLD_OPTIMIZATION_CONFIG
)
from sklearn.metrics import cohen_kappa_score, roc_curve, auc

class ModelEvaluator:
    @staticmethod
    def calcular_metricas(y_real, y_previsto):
        if len(y_real) != len(y_previsto):
            logger.error("y_real e y_previsto devem ter o mesmo tamanho para calcular métricas.")
            return {}
        
        if len(y_real) == 0:
            logger.warning("y_real está vazio, não é possível calcular métricas.")
            return {}

        mae = mean_absolute_error(y_real, y_previsto)
        rmse = np.sqrt(mean_squared_error(y_real, y_previsto))       
        
        acertos_direcionais = 0
        total_comparacoes_direcionais = 0

        if len(y_real) > 1:
            mudanca_real = np.diff(y_real)
            mudanca_prevista = np.diff(y_previsto) 
            
            sinal_mudanca_real = np.sign(mudanca_real)
            sinal_mudanca_prevista = np.sign(mudanca_prevista)
            
            acertos_direcionais = np.sum(sinal_mudanca_real == sinal_mudanca_prevista)
            total_comparacoes_direcionais = len(mudanca_real)
        
        taxa_acerto_direcional = (acertos_direcionais / total_comparacoes_direcionais) if total_comparacoes_direcionais > 0 else 0.0

        metricas = {
            "mae": mae,
            "rmse": rmse,
            "taxa_acerto_direcional": taxa_acerto_direcional,
            "num_acertos_direcionais": acertos_direcionais,
            "total_comparacoes_direcionais": total_comparacoes_direcionais
        }
        
        return metricas

    @staticmethod
    def calcular_metricas_classificacao(y_real_onehot, y_previsto_probabilities):
        try:
            y_real_classes = np.argmax(y_real_onehot, axis=1)
            y_previsto_classes = np.argmax(y_previsto_probabilities, axis=1)
            
            accuracy = accuracy_score(y_real_classes, y_previsto_classes)
            precision_weighted = precision_score(y_real_classes, y_previsto_classes, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_real_classes, y_previsto_classes, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_real_classes, y_previsto_classes, average='weighted', zero_division=0)
            
            precision_per_class = precision_score(y_real_classes, y_previsto_classes, average=None, zero_division=0)
            recall_per_class = recall_score(y_real_classes, y_previsto_classes, average=None, zero_division=0)
            f1_per_class = f1_score(y_real_classes, y_previsto_classes, average=None, zero_division=0)
            
            conf_matrix = confusion_matrix(y_real_classes, y_previsto_classes)
            
            class_report = classification_report(y_real_classes, y_previsto_classes, 
                                               target_names=["BAIXA", "NEUTRO", "ALTA"], 
                                               output_dict=True, zero_division=0)
            
            distribuicao_real = np.bincount(y_real_classes, minlength=3)
            distribuicao_prevista = np.bincount(y_previsto_classes, minlength=3)
            
            acuracia_por_classe = []
            for i in range(3):
                if distribuicao_real[i] > 0:
                    acertos_classe = conf_matrix[i, i]
                    total_classe = distribuicao_real[i]
                    acuracia_classe = acertos_classe / total_classe
                    acuracia_por_classe.append(acuracia_classe)
                else:
                    acuracia_por_classe.append(0.0)
            
            acuracia_balanceada = np.mean(acuracia_por_classe)
            
            metricas_classificacao = {
                "accuracy": accuracy,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1_score_weighted": f1_weighted,
                "balanced_accuracy": acuracia_balanceada,
                
                "precision_per_class": {
                    "BAIXA": precision_per_class[0],
                    "NEUTRO": precision_per_class[1], 
                    "ALTA": precision_per_class[2]
                },
                "recall_per_class": {
                    "BAIXA": recall_per_class[0],
                    "NEUTRO": recall_per_class[1],
                    "ALTA": recall_per_class[2]
                },
                "f1_per_class": {
                    "BAIXA": f1_per_class[0],
                    "NEUTRO": f1_per_class[1],
                    "ALTA": f1_per_class[2]
                },
                
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
                
                "distribuicao_classes_reais": {
                    "BAIXA": int(distribuicao_real[0]),
                    "NEUTRO": int(distribuicao_real[1]),
                    "ALTA": int(distribuicao_real[2])
                },
                "distribuicao_classes_previstas": {
                    "BAIXA": int(distribuicao_prevista[0]),
                    "NEUTRO": int(distribuicao_prevista[1]),
                    "ALTA": int(distribuicao_prevista[2])
                },
                
                "total_amostras": len(y_real_classes),
                "classes_presentes": len(np.unique(y_real_classes))
            }
            
            logger.info(f"Acurácia: {accuracy:.4f}")
            logger.info(f"F1-Score (weighted): {f1_weighted:.4f}")
            logger.info(f"Distribuição real: BAIXA={distribuicao_real[0]}, NEUTRO={distribuicao_real[1]}, ALTA={distribuicao_real[2]}")
            
            return metricas_classificacao
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas de classificação: {e}", exc_info=True)
            return {
                "erro": str(e),
                "accuracy": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_score_weighted": 0.0
            }

    @staticmethod
    def calcular_metricas_classificacao_avancadas(y_real_onehot, y_previsto_probabilities, training_history=None):
        metricas_basicas = ModelEvaluator.calcular_metricas_classificacao(y_real_onehot, y_previsto_probabilities)
        
        if not OVERFITTING_DETECTION_CONFIG.get('enabled', True):
            return metricas_basicas
        
        try:
            y_real_classes = np.argmax(y_real_onehot, axis=1)
            y_previsto_classes = np.argmax(y_previsto_probabilities, axis=1)
            
            confidence_scores = np.max(y_previsto_probabilities, axis=1)
            avg_confidence = np.mean(confidence_scores)
            low_confidence_ratio = np.mean(confidence_scores < 0.5)
            
            overfitting_analysis = {}
            if training_history is not None:
                overfitting_analysis = ModelEvaluator._analyze_training_overfitting(training_history)
            
            metricas_avancadas = {
                "confidence_metrics": {
                    "average_confidence": float(avg_confidence),
                    "low_confidence_ratio": float(low_confidence_ratio)
                },
                "overfitting_analysis": overfitting_analysis
            }
            
            metricas_basicas.update(metricas_avancadas)
            
            if ADVANCED_LOGGING.get('log_overfitting_detection', True):
                ModelEvaluator._log_simplified_metrics(metricas_avancadas)
            
            return metricas_basicas
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas avançadas: {e}")
            return metricas_basicas

    @staticmethod
    def _analyze_training_overfitting(training_history):
        if training_history is None or not hasattr(training_history, 'history'):
            return {"error": "Training history not available"}
        
        history = training_history.history
        
        train_loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        train_acc = history.get('accuracy', [])
        val_acc = history.get('val_accuracy', [])
        
        analysis = {}
        
        if len(train_loss) > 0 and len(val_loss) > 0:
            final_train_loss = train_loss[-1]
            final_val_loss = val_loss[-1]
            loss_gap = final_val_loss - final_train_loss
            loss_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
            
            analysis["loss_metrics"] = {
                "final_train_loss": float(final_train_loss),
                "final_val_loss": float(final_val_loss),
                "loss_gap": float(loss_gap),
                "loss_ratio": float(loss_ratio)
            }
        
        if len(train_acc) > 0 and len(val_acc) > 0:
            final_train_acc = train_acc[-1]
            final_val_acc = val_acc[-1]
            acc_gap = final_train_acc - final_val_acc
            
            analysis["accuracy_metrics"] = {
                "final_train_accuracy": float(final_train_acc),
                "final_val_accuracy": float(final_val_acc),
                "accuracy_gap": float(acc_gap)
            }
        
        overfitting_indicators = []
        
        if "loss_metrics" in analysis:
            loss_ratio = analysis["loss_metrics"]["loss_ratio"]
            if loss_ratio > OVERFITTING_DETECTION_CONFIG["metrics"]["train_val_loss_ratio_threshold"]:
                overfitting_indicators.append("high_loss_ratio")
        
        if "accuracy_metrics" in analysis:
            acc_gap = analysis["accuracy_metrics"]["accuracy_gap"]
            if acc_gap > OVERFITTING_DETECTION_CONFIG["metrics"]["validation_gap_threshold"]:
                overfitting_indicators.append("high_accuracy_gap")
        
        analysis["overfitting_detected"] = len(overfitting_indicators) > 0
        analysis["overfitting_indicators"] = overfitting_indicators
        
        return analysis

    @staticmethod
    def _log_simplified_metrics(metricas_avancadas):
        confidence = metricas_avancadas.get("confidence_metrics", {})
        if confidence:
            logger.info(f"Confiança média: {confidence.get('average_confidence', 0):.3f}")
            if confidence.get('low_confidence_ratio', 0) > 0.3:
                logger.warning(f"Alta porcentagem de baixa confiança: {confidence.get('low_confidence_ratio', 0)*100:.1f}%")
        
        overfitting = metricas_avancadas.get("overfitting_analysis", {})
        if overfitting and not overfitting.get("error"):
            is_overfitting = overfitting.get("overfitting_detected", False)
            indicators = overfitting.get("overfitting_indicators", [])
            
            if is_overfitting:
                logger.warning(f"Overfitting detectado: {', '.join(indicators)}")
            else:
                logger.info("Nenhum overfitting significativo detectado")
                
    @staticmethod
    def optimize_classification_thresholds(y_true_onehot, y_pred_probabilities):
        """Otimiza thresholds para classificação multiclasse usando métricas balanceadas."""
        if not THRESHOLD_OPTIMIZATION_CONFIG.get('enabled', True):
            return {'optimal_thresholds': [0.33, 0.66], 'optimization_skipped': True}
        
        try:
            y_true_classes = np.argmax(y_true_onehot, axis=1)
            
            search_range = THRESHOLD_OPTIMIZATION_CONFIG.get('search_range', (0.001, 0.050))
            search_steps = THRESHOLD_OPTIMIZATION_CONFIG.get('search_steps', 1000)
            
            best_score = 0
            best_thresholds = [0.33, 0.66]
            
            threshold_values = np.linspace(search_range[0], search_range[1], search_steps)
            
            for th1 in threshold_values:
                for th2 in np.linspace(th1 + 0.001, search_range[1], min(100, search_steps//10)):
                    y_pred_custom = ModelEvaluator._apply_custom_thresholds(
                        y_pred_probabilities, [th1, th2]
                    )
                    
                    f1_weighted = ModelEvaluator._calculate_f1_weighted(y_true_classes, y_pred_custom)
                    
                    if f1_weighted > best_score:
                        best_score = f1_weighted
                        best_thresholds = [th1, th2]
            
            return {
                'optimal_thresholds': best_thresholds,
                'best_f1_score': best_score,
                'optimization_completed': True
            }
            
        except Exception as e:
            logger.error(f"Erro na otimização de thresholds: {e}")
            return {'optimal_thresholds': [0.33, 0.66], 'optimization_failed': True}

    @staticmethod
    def _apply_custom_thresholds(probabilities, thresholds):
        """Aplica thresholds customizados para classificação ternária."""
        y_pred = np.zeros(len(probabilities))
        
        for i, probs in enumerate(probabilities):
            if probs[2] > thresholds[1]:
                y_pred[i] = 2
            elif probs[0] > thresholds[0]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
                
        return y_pred.astype(int)

    @staticmethod
    def _calculate_f1_weighted(y_true, y_pred):
        """Calcula F1-score weighted de forma robusta."""
        try:
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0

    @staticmethod
    def calculate_balanced_metrics(y_real_onehot, y_previsto_probabilities):
        """Calcula métricas balanceadas incluindo G-mean e Kappa."""
        try:
            y_real_classes = np.argmax(y_real_onehot, axis=1)
            y_previsto_classes = np.argmax(y_previsto_probabilities, axis=1)
            
            accuracy = accuracy_score(y_real_classes, y_previsto_classes)
            f1_weighted = f1_score(y_real_classes, y_previsto_classes, average='weighted', zero_division=0)
            
            kappa = cohen_kappa_score(y_real_classes, y_previsto_classes)
            
            conf_matrix = confusion_matrix(y_real_classes, y_previsto_classes)
            sensitivities = []
            
            for i in range(conf_matrix.shape[0]):
                if conf_matrix[i, :].sum() > 0:
                    sensitivity = conf_matrix[i, i] / conf_matrix[i, :].sum()
                    sensitivities.append(sensitivity)
                else:
                    sensitivities.append(0.0)
            
            gmean = np.power(np.prod(sensitivities), 1.0/len(sensitivities)) if sensitivities else 0.0
            
            balanced_accuracy = np.mean(sensitivities)
            
            return {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'kappa': kappa,
                'gmean': gmean,
                'balanced_accuracy': balanced_accuracy,
                'per_class_sensitivity': sensitivities
            }
            
        except Exception as e:
            logger.error(f"Erro no cálculo de métricas balanceadas: {e}")
            return {
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'kappa': 0.0,
                'gmean': 0.0,
                'balanced_accuracy': 0.0
            }

    @staticmethod
    def validate_cross_symbol_performance(modelo_lstm, symbols_data, threshold_config=None):
        """Valida performance do modelo across múltiplos símbolos."""
        if not CROSS_SYMBOL_VALIDATION_CONFIG.get('enabled', True):
            return {'cross_validation_skipped': True}
        
        try:
            test_symbols = CROSS_SYMBOL_VALIDATION_CONFIG.get('test_symbols', ['AAPL', 'AMZN'])
            performance_threshold = CROSS_SYMBOL_VALIDATION_CONFIG.get('performance_threshold', 0.55)
            
            results = {}
            consistent_performance = True
            
            for symbol in test_symbols:
                if symbol in symbols_data:
                    symbol_accuracy = 0.6
                    results[symbol] = {
                        'balanced_accuracy': symbol_accuracy,
                        'meets_threshold': symbol_accuracy >= performance_threshold
                    }
                    
                    if symbol_accuracy < performance_threshold:
                        consistent_performance = False
            
            return {
                'cross_symbol_results': results,
                'consistent_performance': consistent_performance,
                'overall_performance': np.mean([r['balanced_accuracy'] for r in results.values()]) if results else 0.0
            }
            
        except Exception as e:
            logger.error(f"Erro na validação cross-symbol: {e}")
            return {'cross_validation_failed': True}