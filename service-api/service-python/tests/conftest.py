import sys
import os

SERVICE_PYTHON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVICE_PYTHON_ROOT, '..', '..'))
CLIENT_CLI_ROOT = os.path.join(PROJECT_ROOT, 'client-cli')

for path in (SERVICE_PYTHON_ROOT, CLIENT_CLI_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
