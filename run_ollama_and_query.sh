#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-gemma3:1b}"
INDEX_PATH="${INDEX_PATH:-artifacts/langchain_index}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
DEFAULT_QUESTION="What is retrieval-augmented generation?"
QUESTION="${QUESTION:-${1:-$DEFAULT_QUESTION}}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STARTUP_DELAY="${OLLAMA_STARTUP_DELAY:-2}"

start_ollama_if_needed() {
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
  sleep "$STARTUP_DELAY"
  echo "Ollama started (pid $pid). Logs: /tmp/ollama.log"
}

start_ollama_if_needed

echo "Running query with model '$MODEL'..."
"$PYTHON_BIN" langchain_rag.py query \
  --index-path "$INDEX_PATH" \
  --question "$QUESTION" \
  --provider ollama \
  --model "$MODEL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --ollama-base-url "$OLLAMA_BASE_URL"
