#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${SKILLEM_PYTHON:-python3}
CONFIG=${1:-configs/search_mcmc.json}
TRAIN=${2:-runs/collection/successes.jsonl}
OUTPUT=${3:-runs/search-mcmc}
"$PYTHON" -m skillem.cli train --config "$CONFIG" --input "$TRAIN" --output "$OUTPUT" "${@:4}"
