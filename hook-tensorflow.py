# hook-tensorflow.py
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files, collect_dynamic_libs
import os

# Coletar TUDO automaticamente
datas, binaries, hiddenimports = collect_all('tensorflow')

# Forçar coleta de submódulos específicos problemáticos
problematic_modules = [
    'tensorflow.lite',
    'tensorflow.lite.python',
    'tensorflow.lite.python.convert',
    'tensorflow.lite.python.util',
    'tensorflow.lite.python.metrics',
    'tensorflow.lite.python.metrics.wrapper',
    'tensorflow.lite.python.authoring',
    'tensorflow.python.debug',
    'tensorflow.python.framework',
    'tensorflow.python.platform',
    'tensorflow.python.util',
    'tensorflow.python.ops',
    'tensorflow.python.eager',
    'tensorflow.python.client',
    'tensorflow.python.training',
    'tensorflow.python.saved_model',
    'tensorflow.python.distribute',
    'tensorflow.python.keras'
]

for module in problematic_modules:
    try:
        hiddenimports += collect_submodules(module)
    except:
        pass

# Tentar coletar extensões compiladas (_pywrap)
try:
    # Procurar por arquivos _pywrap no TensorFlow
    import tensorflow as tf
    tf_path = os.path.dirname(tf.__file__)
    
    # Adicionar paths comuns onde estão as extensões compiladas
    for root, dirs, files in os.walk(tf_path):
        for file in files:
            if file.startswith('_pywrap') and (file.endswith('.so') or file.endswith('.pyd') or file.endswith('.dll')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, tf_path)
                binaries.append((full_path, os.path.dirname(rel_path)))
except Exception as e:
    print(f"Warning: Could not collect _pywrap extensions: {e}")

# Adicionar imports específicos que estão falhando
hiddenimports += [
    'tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper',
    'tensorflow.python.framework._pywrap_python_api_dispatcher',
    'tensorflow.python.framework._pywrap_python_op_gen',
    'tensorflow.python.client._pywrap_tf_session',
    'tensorflow.python.client._pywrap_device_lib',
    'tensorflow.python.grappler._pywrap_tf_optimizer',
    'tensorflow.python.util._pywrap_utils',
    'tensorflow.python.ops._pywrap_py_func',
    'tensorflow.python.eager._pywrap_tfe',
    'tensorflow.python.training._pywrap_checkpoint_reader',
    'tensorflow.python.saved_model._pywrap_saved_model',
    'tensorflow.python.distribute._pywrap_collective_ops',
    'tensorflow.python.lib.core._pywrap_py_exception_registry',
    'tensorflow.python.lib.core._pywrap_bfloat16',
    'tensorflow.python.lib.core._pywrap_float8',
]

# Forçar coleta de todos os _pywrap conhecidos
pywrap_modules = [
    '_pywrap_tensorflow_internal',
    '_pywrap_tfe', 
    '_pywrap_utils',
    '_pywrap_py_func',
    '_pywrap_checkpoint_reader',
    '_pywrap_saved_model',
    '_pywrap_collective_ops',
    '_pywrap_py_exception_registry',
    '_pywrap_bfloat16',
    '_pywrap_float8',
    '_pywrap_tensorflow_lite_metrics_wrapper'
]

for module in pywrap_modules:
    try:
        hiddenimports.append(f'tensorflow.python.{module}')
        hiddenimports.append(f'tensorflow.lite.python.metrics.{module}')
    except:
        pass

print(f"TensorFlow hook: collected {len(hiddenimports)} hidden imports")
print(f"TensorFlow hook: collected {len(binaries)} binaries")
print(f"TensorFlow hook: collected {len(datas)} data files")