#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f .githooks/pre-commit ]]; then
  echo "Error: .githooks/pre-commit was not found." >&2
  exit 1
fi

chmod +x .githooks/pre-commit
git config --local core.hooksPath .githooks

CONFIGURED_PATH="$(git config --local --get core.hooksPath)"

if [[ "${CONFIGURED_PATH}" != ".githooks" ]]; then
  echo "Error: Git hooks path was not configured correctly." >&2
  exit 1
fi

echo "Git hooks configured successfully."
echo "Hooks path: ${CONFIGURED_PATH}"