# Illustration of Retrieval-Augmented Generation (RAG) with reranker

This repo illustrates how RAG can be used and how reranker improves metrics. It can ingest some `.txt`/`.md` files, index with FAISS, and query with OpenAI or Ollama. 

## Evaluation results
- Current sample run (`squad` split `train[:2000]`, 800 examples, `all-MiniLM-L6-v2` embeddings):
![LangChain dataset metrics plot](artifacts/comparison.png)

## Quickstart

1) Install deps (Python 3.10+):
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Put some text under `data/` (any `.txt`/`.md` files). A sample file is included.

3) Build the LangChain index:
```
python langchain_rag.py ingest --data-dir data --index-path artifacts/langchain_index --embedding-model all-MiniLM-L6-v2
```

4) Query:
```
export OPENAI_API_KEY=sk-...
python langchain_rag.py query --index-path artifacts/langchain_index --question "What is retrieval-augmented generation?" --model gpt-4o-mini
python langchain_rag.py query --index-path artifacts/langchain_index --question "What is retrieval-augmented generation?" --provider ollama --model gemma3:1b
```
Helper script that starts `ollama serve` (if needed) and runs the Ollama query:
```
./run_ollama_and_query.sh "What is retrieval-augmented generation?"
```
The CLI reads files, chunks, embeds, saves to `artifacts/langchain_index/`, retrieves, and answers with citations.
Model downloads (embeddings/rerankers) default to `artifacts/`; override with `--cache-folder` or `CACHE_FOLDER`.

### Agentic RAG on SQuAD
- Build the SQuAD index (defaults: `squad`, split `validation[:2000]`, 1 context/question, 500-char chunks):
```
python agentic_rag.py ingest --index-path artifacts/squad_index --sample-size 2000 --contexts-per-question 1
```
- Ask with multi-query retrieval + optional reranker + reflection (uses query rewrites and cites context indices):
```
python agentic_rag.py ask --question "Who wrote the novel Pride and Prejudice?" --index-path artifacts/squad_index --provider ollama --model gemma3:1b --use-reranker --reflect
```
- One-shot runner that auto-starts Ollama, then calls the agentic pipeline (defaults shown above):
```
./run_agentic_squad.sh "Who wrote the novel Pride and Prejudice?"
```
Tune `--rewrites`, `--top-k`, or `--rerank-candidates` (or env vars like `MODEL`, `RERANK_CANDIDATES`, `REFLECT`) to adjust behavior. Set `OPENAI_MODEL`/`OPENAI_API_KEY` to use OpenAI instead of Ollama.

## Evaluate with a Hugging Face dataset (LangChain)
- Quick retrieval check (uses `RecursiveCharacterTextSplitter`, LangChain FAISS, and your chosen embedding model):
```
python langchain_dataset_metrics.py --dataset squad --split "train[:2000]" --sample-size 800 --embedding-model all-MiniLM-L6-v2 --metrics-json artifacts/lc_metrics.json --metrics-plot artifacts/lc_metrics.png
```
- Flags are the usual suspects: tweak `--chunk-size`, `--chunk-overlap`, `--top-k`, and `--embedding-model`. Prints metrics; optionally writes JSON/PNG.
- Compare baseline, reranker, and agentic RAG (runs all three and writes comparison JSON for the Dash UI):
```
./run_rag_comparison.sh
```
Outputs land in `artifacts/` (`lc_metrics*.json/.png`, `lc_metrics_reranker.json`, and `lc_metrics_agentic.json`).

## Web app (questions + metrics)
- Start the Dash UI (uses the LangChain FAISS index at `artifacts/langchain_index`, auto-starts `ollama serve`, and shows metrics inline if found):
```
python app.py
```
- Pick a sample question or type your own. Context shows with scores; answers include citations. Metrics come from `artifacts/lc_metrics.json` (baseline) and `artifacts/lc_metrics_reranker.json` (if present) to render the baseline vs reranker chart.
- Override metric paths with `METRICS_PATH` (baseline) and `RERANKER_METRICS_PATH` (reranker) if you store results elsewhere.
- Defaults: provider=`ollama`, model=`gemma3:1b` (auto-pulled if missing), embedding model=`all-MiniLM-L6-v2`. Override with `WEB_APP_PROVIDER`, `WEB_APP_MODEL`, `WEB_APP_EMBED_MODEL`, `WEB_APP_TOP_K`, or `OLLAMA_BASE_URL`.

## Useful flags
```
python langchain_rag.py ingest --help
python langchain_rag.py query --help
```
Key flags: `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--top-k`, `--index-path`, `--data-dir`, `--provider`, `--ollama-base-url`.
Model caching: `--cache-folder` (or `CACHE_FOLDER`) controls where embeddings/reranker weights land (default `artifacts/`).

## Project layout
- `langchain_rag.py` — ingestion, indexing, retrieval, and query CLI (LangChain-based).
- `langchain_dataset_metrics.py` — HF QA evaluator (hit-rate/MRR + optional plot).
- `app.py` — Dash UI to ask questions and view metrics.
- `run_ollama_and_query.sh` — helper to start Ollama (if needed) and run a query with `langchain_rag.py`.
- `requirements.txt` — Python dependencies.
- `data/` — put your source documents here (text/markdown).
- `artifacts/` — (created at runtime) stores the serialized FAISS index and metrics.
