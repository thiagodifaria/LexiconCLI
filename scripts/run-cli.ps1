$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$root\service-api\service-python;$root\client-cli;$env:PYTHONPATH"
python "$root\client-cli\main.py"
