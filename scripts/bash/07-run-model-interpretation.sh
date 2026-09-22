#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

cd "${REPO_ROOT}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: ${VENV_PYTHON} was not found or is not executable." >&2
  echo "Create the repository environment with scripts/bash/00-create-venv.sh, then try again." >&2
  exit 1
fi

"${VENV_PYTHON}" scripts/python/07-run-model-interpretation.py
