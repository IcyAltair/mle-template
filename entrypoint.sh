#!/usr/bin/env sh
set -e

mkdir -p /app/logs

echo "Running tests..."
python -m pytest -q --disable-warnings --maxfail=1 2>&1 | tee /app/logs/pytest.log

echo "Tests passed. Starting API..."
exec uvicorn src.api:app --host 0.0.0.0 --port 8000