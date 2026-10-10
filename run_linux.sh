#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python3 ]]; then
  echo "Primero cree el entorno: python3 -m venv .venv && .venv/bin/python3 -m pip install -r requirements.txt" >&2
  exit 1
fi
exec .venv/bin/python3 -m streamlit run app.py --server.address 127.0.0.1
