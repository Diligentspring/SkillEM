#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE=materials/SkillRL-4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d
ASSETS=${SKILLEM_RETRIEVAL_DIR:-materials/retrieval}
PYTHON=${SKILLEM_RETRIEVER_PYTHON:-python3}
if [[ ! -f "$ASSETS/e5_Flat.index" || ! -f "$ASSETS/wiki-18.jsonl" ]]; then
  echo 'Assemble the downloaded index and decompress the corpus with scripts/prepare_retrieval.py.' >&2
  exit 1
fi
"$PYTHON" "$SOURCE/examples/search/retriever/retrieval_server.py" \
  --index_path "$ASSETS/e5_Flat.index" --corpus_path "$ASSETS/wiki-18.jsonl" \
  --retriever_name e5 --retriever_model materials/models/e5-base-v2 \
  --topk 3 --port 8030 "$@"
