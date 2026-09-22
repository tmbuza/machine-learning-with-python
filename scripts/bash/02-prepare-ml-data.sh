#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

cd "${REPO_ROOT}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: ${VENV_PYTHON} was not found." >&2
  echo "Create and select the repository environment as described in 00-preface.qmd." >&2
  exit 1
fi

"${VENV_PYTHON}" scripts/python/02-prepare-ml-data.py
