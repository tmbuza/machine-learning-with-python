#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found." >&2
  echo "Install Python 3.12 or set PYTHON_BIN to a suitable interpreter." >&2
  echo "Example: PYTHON_BIN=python3 bash scripts/bash/00-create-venv.sh" >&2
  exit 1
fi

if [[ ! -f requirements.txt ]]; then
  echo "Error: requirements.txt was not found in ${REPO_ROOT}." >&2
  exit 1
fi

if [[ -d .venv ]]; then
  echo "Error: .venv already exists." >&2
  echo "Remove or rename it before requesting a fresh environment." >&2
  exit 1
fi

echo "Creating .venv with ${PYTHON_BIN}..."

"${PYTHON_BIN}" -m venv .venv

echo "Upgrading pip, setuptools, and wheel..."

.venv/bin/python -m pip install --upgrade pip setuptools wheel

echo "Installing project dependencies..."

.venv/bin/python -m pip install -r requirements.txt

echo "Verifying the environment..."

.venv/bin/python - <<'PY'
import sys

import joblib
import matplotlib
import numpy
import pandas
import scipy
import sklearn

print(f"Python:       {sys.version.split()[0]}")
print(f"Interpreter:  {sys.executable}")
print(f"pandas:       {pandas.__version__}")
print(f"NumPy:        {numpy.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"joblib:       {joblib.__version__}")`
PY

echo
echo "Environment created successfully."
echo
echo "Activate it with:"
echo "  source .venv/bin/activate"
echo
echo "In VS Code, select:"
echo "  ${REPO_ROOT}/.venv/bin/python"
echo
echo "Command Palette:"
echo "  Python: Select Interpreter"