$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$root\service-api\service-python;$root\client-cli;$env:PYTHONPATH"
python "$root\service-api\service-python\ops.py" stream @args
