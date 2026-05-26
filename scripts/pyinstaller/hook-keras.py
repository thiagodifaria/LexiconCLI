# hook-keras.py
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Coletar tudo do Keras
datas, binaries, hiddenimports = collect_all('keras')

# Adicionar submódulos específicos
hiddenimports += collect_submodules('keras')
hiddenimports += collect_submodules('keras.src')
hiddenimports += collect_submodules('keras.activations')
hiddenimports += collect_submodules('keras.layers')
hiddenimports += collect_submodules('keras.models')
hiddenimports += collect_submodules('keras.utils')
hiddenimports += collect_submodules('keras.callbacks')
hiddenimports += collect_submodules('keras.optimizers')
hiddenimports += collect_submodules('keras.losses')
hiddenimports += collect_submodules('keras.metrics')

# Keras TensorFlow backend
hiddenimports += [
    'keras._tf_keras',
    'keras._tf_keras.keras',
    'keras.src.backend',
    'keras.src.backend.common',
    'keras.src.backend.common.dtypes',
    'keras.src.backend.common.variables',
    'keras.src.utils',
    'keras.src.utils.audio_dataset_utils',
    'keras.src.utils.dataset_utils',
    'keras.src.tree',
    'keras.src.tree.tree_api',
    'keras.src.tree.optree_impl'
]

print(f"Keras hook: collected {len(hiddenimports)} hidden imports")