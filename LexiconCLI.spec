# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None

# Coletar TUDO automaticamente
tf_datas, tf_binaries, tf_hiddenimports = collect_all('tensorflow')
keras_datas, keras_binaries, keras_hiddenimports = collect_all('keras')
prophet_datas, prophet_binaries, prophet_hiddenimports = collect_all('prophet')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
backtesting_datas, backtesting_binaries, backtesting_hiddenimports = collect_all('backtesting')

# Adicionar arquivos específicos do Prophet que costumam faltar
try:
    import prophet
    prophet_path = os.path.dirname(prophet.__file__)
    prophet_version_file = os.path.join(prophet_path, '__version__.py')
    if os.path.exists(prophet_version_file):
        prophet_datas.append((prophet_version_file, 'prophet'))
except:
    pass

# Imports ocultos necessários
hidden_imports = [
    # Core do projeto
    'controllers.app_controller',
    'controllers.data_controller', 
    'controllers.analysis_controller',
    'controllers.prediction_controller',
    'controllers.backtesting_controller',
    'controllers.simulation_controller',
    'models.database',
    'models.data_model',
    'models.indicators',
    'models.market_data',
    'models.prediction.lstm_model',
    'models.prediction.prophet_model',
    'models.prediction.evaluator',
    'models.prediction.advanced.augmentation',
    'models.prediction.advanced.adaptive_selector',
    'models.prediction.advanced.ensemble_model',
    'models.prediction.advanced.losses',
    'models.simulation.monte_carlo_simulator',
    'utils.logger',
    'utils.complexity_manager',
    'utils.dataset_analyzer',
    'utils.formatters',
    'utils.notifications',
    'utils.validators',
    'views.terminal.main_view',
    'views.terminal.menu',
    'views.terminal.charts',
    'views.terminal.tables',
    'config.settings',
    'config.adaptive_configs',
    'config.api_keys',
    
    # Dependências financeiras
    'yfinance',
    'finnhub',
    'alpha_vantage',
    'fredapi',
    'python_bcb',
    'investpy',
    'nasdaq_data_link',
    
    # Prophet e dependências
    'prophet',
    'prophet.serialize',
    'cmdstanpy',
    'pystan',
    
    # Indicadores técnicos
    'ta',
    'ta.momentum',
    'ta.trend', 
    'ta.volatility',
    'ta.volume',
    
    # Interface
    'rich',
    'rich.console',
    'rich.panel',
    'rich.table',
    'rich.text',
    'rich.progress',
    'blessed',
    'tabulate',
    
    # Visualização
    'plotext',
    'matplotlib',
    'matplotlib.pyplot',
    'plotly',
    'plotly.graph_objects',
    'plotly.subplots',
    
    # Backtesting
    'backtesting',
    'backtesting.lib',
    
    # Utilitários
    'pandas',
    'numpy',
    'scipy',
    'statsmodels',
    'requests',
    'dotenv',
    'plyer',
    'sqlite3',
    'datetime',
    'json',
    'os',
    'sys',
    're',
    
    # Dependências adicionais TensorFlow
    'flatbuffers',
    'absl',
    'google.protobuf',
    'protobuf',
    'sklearn.preprocessing',
    'sklearn.metrics',
    'joblib'
]

# Adicionar todos os imports automáticos
hidden_imports.extend(tf_hiddenimports)
hidden_imports.extend(keras_hiddenimports)
hidden_imports.extend(prophet_hiddenimports)
hidden_imports.extend(scipy_hiddenimports)
hidden_imports.extend(sklearn_hiddenimports)
hidden_imports.extend(backtesting_hiddenimports)

# Dados a incluir (configurações e templates)
datas = [
    ('config', 'config'),
    ('.env', '.') if os.path.exists('.env') else None
]
datas = [d for d in datas if d is not None]

# Adicionar dados automáticos
datas.extend(tf_datas)
datas.extend(keras_datas)
datas.extend(prophet_datas)
datas.extend(scipy_datas)
datas.extend(sklearn_datas)
datas.extend(backtesting_datas)

# Binários (extensões compiladas)
binaries = []
binaries.extend(tf_binaries)
binaries.extend(keras_binaries)
binaries.extend(prophet_binaries)
binaries.extend(scipy_binaries)
binaries.extend(sklearn_binaries)
binaries.extend(backtesting_binaries)

# Tentar coletar bibliotecas dinâmicas do TensorFlow especificamente
try:
    tf_libs = collect_dynamic_libs('tensorflow')
    binaries.extend(tf_libs)
except:
    pass

# Exclusões para reduzir tamanho (REMOVIDO TensorFlow Lite para teste)
excludes = [
    'matplotlib.tests',
    'IPython',
    'jupyter',
    'notebook',
    'tkinter',
    'PyQt5',
    'PySide2',
    'cv2',
    'PIL.ImageTk',  # Pode reduzir tamanho
    'PIL.ImageWin'  # Windows específico
]

a = Analysis(
    ['main.py'],
    pathex=[os.getcwd()],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=['pyi_rth_tensorflow.py'],  # Runtime hook para TensorFlow
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode={
        'tensorflow': 'pyz+py',
        'keras': 'pyz+py',
        'tensorflow.lite': 'pyz+py',
        'prophet': 'pyz+py',
        'scipy': 'pyz+py',
        'sklearn': 'pyz+py',
        'backtesting': 'pyz+py'
    }
)

# Filtrar duplicatas em hiddenimports
a.hiddenimports = list(set(a.hiddenimports))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Executável único (--onefile)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles, 
    a.datas,
    [],
    name='LexiconCLI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)