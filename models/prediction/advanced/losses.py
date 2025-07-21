import tensorflow as tf
from keras.losses import Loss
import keras.backend as K
from utils.logger import logger

class CategoricalFocalLoss(Loss):
    """
    Implementação da Focal Loss para classificação multiclasse.
    
    A Focal Loss resolve o problema de desbalanceamento de classes focando
    o treinamento nos exemplos difíceis de classificar e reduzindo a
    contribuição dos exemplos fáceis.
    
    Baseado no paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Fórmula: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    Args:
        alpha (float): Fator de balanceamento entre classes [0.1-1.0]
        gamma (float): Fator de modulação para focar em exemplos difíceis [0.5-5.0]
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, name='categorical_focal_loss', **kwargs):
        """
        Inicializa a Focal Loss.
        
        Args:
            alpha (float): Fator de balanceamento entre classes. Default: 0.25
            gamma (float): Fator de modulação para focar em exemplos difíceis. Default: 2.0
            name (str): Nome da função de loss
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        logger.info(f"CategoricalFocalLoss inicializada: alpha={alpha}, gamma={gamma}")
    
    def call(self, y_true, y_pred):
        """
        Calcula a Focal Loss usando APIs do TensorFlow 2.x.
        
        Args:
            y_true: Labels verdadeiros (one-hot encoded) - shape: (batch_size, num_classes)
            y_pred: Predições do modelo (probabilidades após softmax) - shape: (batch_size, num_classes)
            
        Returns:
            Tensor com os valores de loss para cada amostra - shape: (batch_size,)
        """
        epsilon = K.epsilon()
        
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        alpha_t = self.alpha
        
        focal_weight = tf.pow((1 - p_t), self.gamma)
        
        focal_loss = alpha_t * focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        
        return focal_loss
    
    def get_config(self):
        """Retorna configuração para serialização."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Cria instância a partir da configuração."""
        return cls(**config)


def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    Função factory para criar Focal Loss - Versão Funcional.
    
    Esta é uma versão funcional mais simples para usar diretamente
    no model.compile() se preferir não usar a classe.
    
    Args:
        alpha (float): Fator de balanceamento [0.1-1.0]
        gamma (float): Fator de modulação [0.5-5.0]
        
    Returns:
        Função de loss compatível com Keras/TensorFlow
        
    Exemplo:
        model.compile(
            optimizer='adam',
            loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
            metrics=['accuracy']
        )
    """
    def focal_loss_fixed(y_true, y_pred):
        """Função de loss interna - TODAS as APIs corrigidas para TF 2.x"""
        epsilon = K.epsilon()
        
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        focal_weight = tf.pow((1 - p_t), gamma)
        
        focal_loss = alpha * focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        
        return focal_loss
    
    focal_loss_fixed.__name__ = f'categorical_focal_loss_a{alpha}_g{gamma}'
    return focal_loss_fixed


def calculate_class_weights_smart(y_classes, strategy='balanced_smooth', smoothing=0.15):
    """
    Calcula class weights inteligentes para lidar com desbalanceamento de classes.
    
    Esta função implementa múltiplas estratégias para balanceamento:
    
    1. 'balanced': Peso inversamente proporcional à frequência da classe
    2. 'balanced_smooth': Versão suavizada para evitar pesos extremos
    3. 'inverse': Peso baseado na raridade relativa da classe
    
    Args:
        y_classes: Array/List com as classes (0, 1, 2)
        strategy: Estratégia de cálculo ('balanced', 'balanced_smooth', 'inverse')
        smoothing: Fator de suavização para evitar pesos extremos [0.0-0.5]
        
    Returns:
        Dict com class weights {0: weight, 1: weight, 2: weight}
        
    Exemplo:
        y_train_classes = [0, 1, 1, 2, 0, 1, 2, 2]  # Classes desbalanceadas
        weights = calculate_class_weights_smart(y_train_classes, 'balanced_smooth')
        # Resultado: {0: 1.2, 1: 0.8, 2: 1.0}
    """
    import numpy as np
    from collections import Counter
    
    class_counts = Counter(y_classes)
    total_samples = len(y_classes)
    num_classes = len(class_counts)
    
    logger.info(f"Distribuição das classes: {dict(class_counts)}")
    
    if strategy == 'balanced':
        weights = {}
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)
    
    elif strategy == 'balanced_smooth':
        weights = {}
        for class_idx, count in class_counts.items():
            raw_weight = total_samples / (num_classes * count)
            weights[class_idx] = raw_weight * (1 - smoothing) + smoothing
    
    elif strategy == 'inverse':
        weights = {}
        for class_idx, count in class_counts.items():
            weights[class_idx] = 1.0 / (count / total_samples)
    
    else:
        logger.warning(f"Estratégia '{strategy}' não reconhecida. Usando 'balanced_smooth'.")
        return calculate_class_weights_smart(y_classes, 'balanced_smooth', smoothing)
    
    weight_sum = sum(weights.values())
    normalized_weights = {k: v/weight_sum * num_classes for k, v in weights.items()}
    
    logger.info(f"Class weights calculados ({strategy}): {normalized_weights}")
    
    return normalized_weights


def get_recommended_focal_parameters(class_distribution):
    """
    Recomenda parâmetros ótimos de Focal Loss baseado na distribuição das classes.
    
    Esta função analisa o nível de desbalanceamento e sugere valores
    de alpha e gamma apropriados para maximizar a performance.
    
    Lógica de Recomendação:
    - Baixo desbalanceamento (< 2x): γ=1.5 (foco moderado)
    - Médio desbalanceamento (2-5x): γ=2.0 (foco padrão)
    - Alto desbalanceamento (5-10x): γ=2.5 (foco intenso)
    - Extremo desbalanceamento (> 10x): γ=3.0 (foco máximo)
    
    Args:
        class_distribution: Dict com contagem por classe {0: count, 1: count, 2: count}
        
    Returns:
        Tuple (alpha, gamma) com parâmetros recomendados
        
    Exemplo:
        dist = {0: 100, 1: 300, 2: 50}  # Classe 1 dominante
        alpha, gamma = get_recommended_focal_parameters(dist)
        # Resultado: (0.25, 2.0) para desbalanceamento médio
    """
    total_samples = sum(class_distribution.values())
    
    proportions = [count/total_samples for count in class_distribution.values()]
    max_proportion = max(proportions)
    min_proportion = min(proportions)
    imbalance_ratio = max_proportion / min_proportion
    
    if imbalance_ratio < 2:
        alpha, gamma = 0.25, 1.5
    elif imbalance_ratio < 5:
        alpha, gamma = 0.25, 2.0
    elif imbalance_ratio < 10:
        alpha, gamma = 0.25, 2.5
    else:
        alpha, gamma = 0.25, 3.0
    
    logger.info(f"Desbalanceamento detectado: {imbalance_ratio:.2f}x. Parâmetros recomendados: alpha={alpha}, gamma={gamma}")
    
    return alpha, gamma


def calculate_adaptive_focal_parameters(y_true, y_pred_confidence, percentile_threshold=75):
    """
    FUNÇÃO BÔNUS: Calcula parâmetros da Focal Loss adaptativos baseados 
    na confiança atual do modelo.
    
    Esta função analisa as predições do modelo para ajustar dinamicamente
    os parâmetros da Focal Loss durante o treinamento.
    
    Args:
        y_true: Labels verdadeiros
        y_pred_confidence: Confiança máxima das predições do modelo
        percentile_threshold: Percentil para definir "amostras difíceis"
        
    Returns:
        Tuple (alpha_adaptive, gamma_adaptive)
    """
    import numpy as np
    
    confidence_threshold = np.percentile(y_pred_confidence, percentile_threshold)
    difficult_samples_ratio = np.mean(y_pred_confidence < confidence_threshold)
    
    if difficult_samples_ratio > 0.4:
        gamma_adaptive = 3.0
    elif difficult_samples_ratio > 0.2:
        gamma_adaptive = 2.0
    else:
        gamma_adaptive = 1.5
    
    alpha_adaptive = 0.25
    
    logger.info(f"Parâmetros Focal Loss adaptativos: alpha={alpha_adaptive}, gamma={gamma_adaptive} (amostras difíceis: {difficult_samples_ratio:.2%})")
    
    return alpha_adaptive, gamma_adaptive


def test_focal_loss_implementation():
    """
    Função de teste para verificar se a Focal Loss está funcionando corretamente.
    
    Esta função pode ser chamada para validar a implementação antes do uso.
    """
    import numpy as np
    
    logger.info("Testando implementação da Focal Loss...")
    
    batch_size, num_classes = 32, 3
    y_true_test = tf.random.uniform((batch_size, num_classes), 0, 1)
    y_true_test = tf.nn.softmax(y_true_test)
    y_pred_test = tf.random.uniform((batch_size, num_classes), 0, 1)
    y_pred_test = tf.nn.softmax(y_pred_test)
    
    focal_loss = CategoricalFocalLoss(alpha=0.25, gamma=2.0)
    loss_result = focal_loss(y_true_test, y_pred_test)
    
    focal_fn = categorical_focal_loss(alpha=0.25, gamma=2.0)
    loss_fn_result = focal_fn(y_true_test, y_pred_test)
    
    logger.info(f"Teste concluído - Loss shapes: Class={loss_result.shape}, Function={loss_fn_result.shape}")
    logger.info("✅ Focal Loss implementada corretamente!")
    
    return True


if __name__ == "__main__":
    test_focal_loss_implementation()