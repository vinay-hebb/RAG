#!/usr/bin/env bash
set -euo pipefail

# Run LangChain retrieval evaluation for baseline, reranker, and agentic RAG.
# Defaults mirror README values; override via environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-squad}"
SPLIT="${SPLIT:-train[:2000]}"
SAMPLE_SIZE="${SAMPLE_SIZE:-800}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-100}"
TOP_K="${TOP_K:-1,3,5,10}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
CACHE_FOLDER="${CACHE_FOLDER:-artifacts}"
INDEX_PATH="${INDEX_PATH:-artifacts/langchain_index}"

RERANKER_MODEL="${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
RERANK_CANDIDATES="${RERANK_CANDIDATES:-20}"

BASE_JSON="${BASE_JSON:-artifacts/lc_metrics.json}"
BASE_PLOT="${BASE_PLOT:-artifacts/lc_metrics.png}"
RERANK_JSON="${RERANK_JSON:-artifacts/lc_metrics_reranker.json}"
RERANK_PLOT="${RERANK_PLOT:-artifacts/lc_metrics_reranker.png}"
AGENTIC_JSON="${AGENTIC_JSON:-artifacts/lc_metrics_agentic.json}"
AGENTIC_PLOT="${AGENTIC_PLOT:-artifacts/lc_metrics_agentic.png}"

AGENTIC_MODEL="${AGENTIC_MODEL:-gemma3:1b}"
AGENTIC_PROVIDER="${AGENTIC_PROVIDER:-ollama}"
AGENTIC_REWRITES="${AGENTIC_REWRITES:-3}"
AGENTIC_TEMPERATURE="${AGENTIC_TEMPERATURE:-0.2}"
AGENTIC_USE_RERANKER="${AGENTIC_USE_RERANKER:-true}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
START_OLLAMA="${START_OLLAMA:-true}"

COMMON_FLAGS=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --sample-size "$SAMPLE_SIZE"
  --chunk-size "$CHUNK_SIZE"
  --chunk-overlap "$CHUNK_OVERLAP"
  --top-k "$TOP_K"
  --embedding-model "$EMBEDDING_MODEL"
  --index-path "$INDEX_PATH"
  --cache-folder "$CACHE_FOLDER"
)

print_cmd() {
  printf 'Command:'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

start_ollama_if_needed() {
  if [[ "$AGENTIC_PROVIDER" != "ollama" ]]; then
    return
  fi
  if [[ ! "$START_OLLAMA" =~ ^(1|true|yes|on)$ ]]; then
    return
  fi
  if pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "Ollama already running."
    return
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama CLI not found in PATH."
    exit 1
  fi
  echo "Starting ollama serve..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  local pid=$!
  sleep 2
  echo "Ollama started (pid $pid). Logs: /tmp/ollama.log"
}

ensure_agentic_model_exists() {
  if [[ "$AGENTIC_PROVIDER" != "ollama" ]]; then
    return
  fi
  case "$OLLAMA_BASE_URL" in
    http://localhost:11434/*|http://127.0.0.1:11434/*) ;;
    *) return ;;
  esac
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama CLI not found; cannot verify model '$AGENTIC_MODEL'." >&2
    return
  fi
  if ! ollama show "$AGENTIC_MODEL" >/dev/null 2>&1; then
    echo "Ollama model '$AGENTIC_MODEL' not found. Pull it with: ollama pull $AGENTIC_MODEL" >&2
    exit 1
  fi
}

echo "============================================================"
echo "Stage 1/3: Baseline retrieval (no reranker)"
echo "============================================================"
echo "Running baseline evaluation (no reranker)..."
print_cmd "$PYTHON_BIN" langchain_dataset_metrics.py \
  "${COMMON_FLAGS[@]}" \
  --mode baseline \
  --metrics-json "$BASE_JSON" \
  --metrics-plot "$BASE_PLOT"
"$PYTHON_BIN" langchain_dataset_metrics.py \
  "${COMMON_FLAGS[@]}" \
  --mode baseline \
  --metrics-json "$BASE_JSON" \
  --metrics-plot "$BASE_PLOT"

echo
echo "============================================================"
echo "Stage 2/3: Reranker retrieval"
echo "============================================================"
echo "Running reranker evaluation..."
print_cmd "$PYTHON_BIN" langchain_dataset_metrics.py \
  "${COMMON_FLAGS[@]}" \
  --mode reranker \
  --use-reranker \
  --reranker-model "$RERANKER_MODEL" \
  --rerank-candidates "$RERANK_CANDIDATES" \
  --metrics-json "$RERANK_JSON" \
  --metrics-plot "$RERANK_PLOT"
"$PYTHON_BIN" langchain_dataset_metrics.py \
  "${COMMON_FLAGS[@]}" \
  --mode reranker \
  --use-reranker \
  --reranker-model "$RERANKER_MODEL" \
  --rerank-candidates "$RERANK_CANDIDATES" \
  --metrics-json "$RERANK_JSON" \
  --metrics-plot "$RERANK_PLOT"

echo
echo "============================================================"
echo "Stage 3/3: Agentic multi-query retrieval"
echo "============================================================"
echo "Running agentic RAG evaluation..."
start_ollama_if_needed
ensure_agentic_model_exists
agentic_flags=(
  "${COMMON_FLAGS[@]}"
  --mode agentic
  --rewrites "$AGENTIC_REWRITES"
  --model "$AGENTIC_MODEL"
  --provider "$AGENTIC_PROVIDER"
  --temperature "$AGENTIC_TEMPERATURE"
  --ollama-base-url "$OLLAMA_BASE_URL"
  --metrics-json "$AGENTIC_JSON"
  --metrics-plot "$AGENTIC_PLOT"
)
if [[ "$AGENTIC_USE_RERANKER" =~ ^(1|true|yes|on)$ ]]; then
  agentic_flags+=(--use-reranker --reranker-model "$RERANKER_MODEL" --rerank-candidates "$RERANK_CANDIDATES")
fi
print_cmd "$PYTHON_BIN" langchain_dataset_metrics.py "${agentic_flags[@]}"
"$PYTHON_BIN" langchain_dataset_metrics.py "${agentic_flags[@]}"

echo
echo "Done. Metrics written to:"
echo "  Baseline: $BASE_JSON"
echo "  Reranker: $RERANK_JSON"
echo "  Agentic:  $AGENTIC_JSON"
echo "Plots:"
echo "  Baseline: $BASE_PLOT"
echo "  Reranker: $RERANK_PLOT"
echo "  Agentic:  $AGENTIC_PLOT"
