#!/bin/bash
set -e

# Set your API key in the environment before running, e.g.
#   export OPENAI_API_KEY=sk-...
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========== BBH EXPERIMENTS =========="
"$PYTHON_BIN" experiments/bbh/run.py

echo ""
echo "========== SYNTHETIC EXPERIMENTS =========="
"$PYTHON_BIN" experiments/synthetic/run.py

echo ""
echo "========== INSURANCE EXPERIMENTS =========="
"$PYTHON_BIN" experiments/insurance/run.py

echo ""
echo "========== REVIEW EXPERIMENTS =========="
"$PYTHON_BIN" experiments/review/run.py

echo ""
echo "========== GENERATING PLOTS =========="
"$PYTHON_BIN" experiments/plot_results.py

echo ""
echo "All experiments complete!"
