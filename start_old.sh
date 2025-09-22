#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

uvicorn old:app --host 0.0.0.0 --port 8001 --workers 1 --loop uvloop --http httptools

