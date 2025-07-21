from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

datas, binaries, hiddenimports = collect_all('backtesting')

try:
    import backtesting
    backtesting_path = os.path.dirname(backtesting.__file__)
    
    assets_to_include = [
        'autoscale_cb.js',
        '_plotting.js', 
        '*.js',
        '*.css',
        '*.html',
        '*.json'
    ]
    
    for root, dirs, files in os.walk(backtesting_path):
        for file in files:
            if any(file.endswith(ext) for ext in ['.js', '.css', '.html', '.json']):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, backtesting_path)
                if rel_path == '.':
                    dest_path = 'backtesting'
                else:
                    dest_path = os.path.join('backtesting', rel_path)
                datas.append((full_path, dest_path))
                
except Exception as e:
    print(f"Warning: Could not collect backtesting assets: {e}")

hiddenimports += [
    'backtesting.lib',
    'backtesting._plotting',
    'backtesting._util',
    'backtesting.test'
]

print(f"Backtesting hook: collected {len(hiddenimports)} hidden imports")
print(f"Backtesting hook: collected {len(datas)} data files")