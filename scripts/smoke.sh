#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${SKILLEM_PYTHON:-python3}
smoke_root=${SKILLEM_SMOKE_ROOT:-runs/smoke}
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
"$PYTHON" -m skillem.cli finite-demo
"$PYTHON" scripts/make_smoke.py
for method in mcmc variational snis prior_uniform sft static_skill is_em is_em_mixture is_em_learned; do
  "$PYTHON" -m skillem.cli train --config "$smoke_root/$method.json" \
    --input "$smoke_root/successes.jsonl" --output "$smoke_root/$method"
done
for method in mcmc is_em is_em_mixture is_em_learned; do
  "$PYTHON" -m skillem.cli train --config "$smoke_root/$method.json" \
    --input "$smoke_root/successes.jsonl" --output "$smoke_root/$method" --resume
done
