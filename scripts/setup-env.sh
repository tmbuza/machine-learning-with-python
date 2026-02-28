#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-python3}"

$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user --name cdi-ml-py --display-name "CDI ML (Python)"

echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
