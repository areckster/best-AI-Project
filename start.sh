#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Start the application without auto-reload; single worker to free CPU for the model
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools
