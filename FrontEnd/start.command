#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v npm >/dev/null 2>&1; then
  echo "[ERROR] npm not found. Please install Node.js first."
  exit 1
fi

echo "[INFO] Starting dev server..."
echo "[INFO] Browser will open automatically after startup."

npm run dev -- --open
