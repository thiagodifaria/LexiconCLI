# pyi_rth_tensorflow.py - Runtime hook para TensorFlow
import os
import sys

# Configurar variáveis de ambiente para TensorFlow antes da importação
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silenciar warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar oneDNN se causar problemas

# Tentar configurar paths do TensorFlow se necessário
try:
    import tensorflow
    # Verificar se o TensorFlow carregou corretamente
    print(f"TensorFlow loaded successfully: {tensorflow.__version__}")
except ImportError as e:
    print(f"Warning: TensorFlow import failed: {e}")
except Exception as e:
    print(f"Warning: TensorFlow initialization error: {e}")

# Configurações específicas para PyInstaller
if getattr(sys, 'frozen', False):
    # Estamos rodando em um executável PyInstaller
    os.environ['TF_DISABLE_INTERACTIVE_MODE'] = '1'
    
    # Tentar importar módulos problemáticos silenciosamente
    try:
        from tensorflow.lite.python.metrics import _pywrap_tensorflow_lite_metrics_wrapper
    except ImportError:
        # Se falhar, criar um mock para evitar erros
        class MockMetricsWrapper:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        # Tentar injetar o mock
        try:
            import tensorflow.lite.python.metrics
            tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper = MockMetricsWrapper()
        except:
            pass