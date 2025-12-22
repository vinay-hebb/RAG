"""Quick-and-dirty retrieval check for the LangChain RAG setup (baseline, reranker, and agentic)."""

from __future__ import annotations
import os
# os.environ['OLLAMA_NO_GPU']="1"
# os.environ['CUDA_VISIBLE_DEVICES']=""
# os.environ['USE_TORCH_DEVICE']='cpu'

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set
import asyncio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_rag import DEFAULT_CACHE_FOLDER, build_embeddings, build_llm, build_reranker, retrieve_and_rerank, _infer_provider
from langchain_rag import generate_rewrites, multi_query_retrieve, ensure_ollama_model_available

# Import agentic RAG components for batching
from agentic_rag import run_langgraph_agent, GraphState


DEFAULT_TOP_K = (1, 3, 5, 10)


@dataclass(frozen=True)
class QAExample:
    question: str
    doc_ids: Set[str]


@dataclass(frozen=True)
class Corpus:
    chunks: List[Document]
    examples: List[QAExample]


def parse_top_k(raw: str) -> List[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one top-k value is required.")
    return sorted(set(values))


def load_corpus(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    sample_size: int,
    seed: int,
    chunk_size: int,
    chunk_overlap: int,
    cache_dir: str | None,
) -> Corpus:
    def resolve_dataset_config() -> str | None:
        if dataset_config:
            return dataset_config
        try:
            configs = get_dataset_config_names(dataset_name, cache_dir=cache_dir)
        except Exception:
            return None
        if not configs:
            return None
        if len(configs) == 1:
            print(f"Using only available dataset config '{configs[0]}' for {dataset_name}")
            return configs[0]
        preferred_defaults = {
            "hotpotqa/hotpot_qa": "distractor",
            "hotpot_qa": "distractor",
        }
        default_config = preferred_defaults.get(dataset_name)
        if default_config and default_config in configs:
            print(f"Defaulting to dataset config '{default_config}' for {dataset_name}")
            return default_config
        available = ", ".join(configs)
        raise ValueError(
            f"Dataset '{dataset_name}' requires a config (e.g., 'distractor' for HotpotQA). "
            f"Set --dataset-config to pick one. Available configs: {available}"
        )

    resolved_config = resolve_dataset_config()
    try:
        dataset = load_dataset(dataset_name, name=resolved_config, split=split, cache_dir=cache_dir)
    except ValueError as exc:
        if "Config name is missing" in str(exc):
            raise ValueError(
                f"Dataset '{dataset_name}' requires a config (e.g., 'distractor' for HotpotQA). "
                "Set --dataset-config to pick one."
            ) from exc
        raise
    dataset = dataset.shuffle(seed=seed)
    if sample_size > 0:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.select(range(sample_size))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: List[Document] = []
    examples: List[QAExample] = []
    context_lookup: Dict[str, str] = {}

    for row in tqdm(dataset, desc="Preparing dataset"):
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        if not question or not context:
            continue

        doc_ids: Set[str] = set()

        # HotpotQA supplies a list of (title, sentences) pairs; fall back to single-string contexts otherwise.
        contexts = row.get("context")
        if isinstance(contexts, list):
            for idx, pair in enumerate(contexts):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                title, sentences = pair[0], pair[1]
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                doc_id = context_lookup.get(text)
                if doc_id is None:
                    doc_id = f"{dataset_name}-doc-{len(context_lookup)}"
                    context_lookup[text] = doc_id
                    doc = Document(
                        page_content=text,
                        metadata={"doc_id": doc_id, "dataset": dataset_name, "split": split, "source": str(title)},
                    )
                    chunks.extend(splitter.split_documents([doc]))
                doc_ids.add(doc_id)
        else:
            doc_id = context_lookup.get(context)
            if doc_id is None:
                doc_id = f"{dataset_name}-doc-{len(context_lookup)}"
                context_lookup[context] = doc_id
                doc = Document(
                    page_content=context,
                    metadata={"doc_id": doc_id, "dataset": dataset_name, "split": split, "source": doc_id},
                )
                chunks.extend(splitter.split_documents([doc]))
            doc_ids.add(doc_id)

        if not doc_ids:
            continue
        examples.append(QAExample(question=question, doc_ids=doc_ids))

    if not examples:
        raise ValueError(f"No valid examples found in {dataset_name}:{split}")

    return Corpus(chunks=chunks, examples=examples)


def build_store(corpus: Corpus, embedding_model: str, cache_folder: Path) -> FAISS:
    embeddings = build_embeddings(embedding_model, cache_folder=cache_folder)
    print('Building index...')
    return FAISS.from_documents(corpus.chunks, embeddings)


def save_index(store: FAISS, path: Path) -> None:
    """Persist the FAISS store so the web app can reuse the index."""
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def _extract_doc(obj) -> Document:
    if isinstance(obj, Document):
        return obj
    if hasattr(obj, "doc"):
        return obj.doc  # RetrievedDoc from agentic_rag
    if isinstance(obj, (list, tuple)) and obj:
        candidate = obj[0]
        if isinstance(candidate, Document):
            return candidate
    raise TypeError(f"Unsupported retrieval result type: {type(obj)}")


def make_standard_retriever(store: FAISS, reranker, candidate_k: int) -> Callable[[List[str], int], List[Sequence[Document]]]:
    async def retrieve_batch(questions: List[str], top_k: int) -> List[Sequence[Document]]:
        all_results = []
        for question in questions:
            results = await retrieve_and_rerank(
                store,
                question,
                top_k=top_k,
                reranker=reranker,
                candidate_k=max(candidate_k, top_k),
            )
            all_results.append([doc for doc, _ in results])
        return all_results
    return retrieve_batch


async def make_agentic_retriever(
    store: FAISS,
    llm,
    rewrites: int,
    reranker,
    candidate_k: int,
    sequential_retrieval: bool,
    reflect: bool,
) -> Callable[[List[str], int], List[Sequence[Document]]]:
    async def retrieve_batch(questions: List[str], top_k: int) -> List[Sequence[Document]]:
        # The run_langgraph_agent expects a config dict
        config = {
            "top_k": top_k, # Use the top_k passed to this inner function
            "rewrites": rewrites,
            "rerank_candidates": candidate_k,
            "use_reranker": reranker is not None,
            "reflect": reflect,
            "sequential_retrieval": sequential_retrieval,
        }
        
        results_for_all_questions = await run_langgraph_agent(
            questions=questions,
            llm=llm,
            store=store,
            reranker=reranker,
            config=config,
        )
        
        all_retrieved_docs: List[Sequence[Document]] = []
        for result in results_for_all_questions:
            # run_langgraph_agent returns a list of RetrievedDoc objects
            # Need to extract the Document from each RetrievedDoc
            all_retrieved_docs.append([_extract_doc(item) for item in result["documents"]])
        return all_retrieved_docs

    return retrieve_batch


async def evaluate_retrieval(
    corpus: Corpus,
    top_k_values: Sequence[int],
    retrieve_fn: Callable[[List[str], int], List[Sequence[Document]]], # Changed signature
    batch_size: int = 1, # New parameter
) -> Dict[str, float]:
    max_k = max(top_k_values)
    hits_by_k = {k: 0 for k in top_k_values}
    ndcg_by_k = {k: 0.0 for k in top_k_values}
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []

    # Process examples in batches
    for i in tqdm(range(0, len(corpus.examples), batch_size), desc="Evaluating batches"):
        batch_examples = corpus.examples[i : i + batch_size]
        batch_questions = [ex.question for ex in batch_examples]
        
        batch_docs_results: List[Sequence[Document]] = await retrieve_fn(batch_questions, max_k)

        for j, example in enumerate(batch_examples):
            docs = batch_docs_results[j]
            doc_hits = [doc.metadata.get("doc_id") for doc in docs]
            rank = None
            for idx, doc_id in enumerate(doc_hits):
                if doc_id and doc_id in example.doc_ids:
                    rank = idx
                    break
            if rank is None:
                reciprocal_ranks.append(0.0)
                ranks.append(max_k + 1)
            else:
                reciprocal_ranks.append(1.0 / float(rank + 1))
                ranks.append(rank + 1)
                for k in top_k_values:
                    if rank < k:
                        hits_by_k[k] += 1
                        ndcg_by_k[k] += 1.0 / math.log2(float(rank + 2))

    total = len(corpus.examples)
    hit_rates = {f"hit_rate@{k}": hits_by_k[k] / float(total) for k in top_k_values}
    ndcgs = {f"ndcg@{k}": ndcg_by_k[k] / float(total) for k in top_k_values}

    metrics: Dict[str, float] = {
        "total_queries": float(total),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    metrics.update(hit_rates)
    metrics.update(ndcgs)
    return metrics


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def plot_hit_rates(path: Path, metrics: Dict[str, float], top_k_values: Sequence[int]) -> None:
    hit_rates = [metrics.get(f"hit_rate@{k}", 0.0) for k in top_k_values]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(k) for k in top_k_values], hit_rates, color="#4a90e2")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k")
    ax.set_ylabel("Hit-rate")
    ax.set_title("LangChain retrieval hit-rate@k")
    for bar, hr in zip(bars, hit_rates):
        ax.annotate(
            f"{hr:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, hr),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small harness to measure retrieval quality on a QA set.")
    parser.add_argument("--dataset", type=str, default="squad", help="Hugging Face dataset name")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name for multi-config datasets (auto-picks sole config or defaults to 'distractor' for HotpotQA)",
    )
    parser.add_argument("--split", type=str, default="train[:2000]", help="Dataset split")
    parser.add_argument("--sample-size", type=int, default=800, help="How many examples to sample (0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--top-k", type=str, default="1,3,5,10", help="Comma-separated list of k values")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--hf-cache-dir", type=str, default=None, help="Cache directory for Hugging Face datasets")
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=str(DEFAULT_CACHE_FOLDER),
        help="Directory to store downloaded model weights and tokenizer files",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="artifacts/langchain_index",
        help="Where to persist the FAISS index for the web app",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking when evaluating retrieval",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="How many candidates to retrieve from the vector store before reranking",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "reranker", "agentic"],
        default="baseline",
        help="Retrieval strategy to evaluate",
    )
    parser.add_argument("--rewrites", type=int, default=3, help="Number of rewrites to generate in agentic mode")
    parser.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", "gemma:2b"), help="Chat model for agentic mode")
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the Ollama API",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider for agentic mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM in agentic mode",
    )
    parser.add_argument(
        "--sequential-retrieval",
        action="store_true",
        help="Force multi-query retrieval to run sequentially instead of in parallel in agentic mode",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="Run a second-pass reflection step to verify the draft answer in agentic mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of questions to process in a single batch for evaluation (>=1)",
    )
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path to save metrics as JSON")
    parser.add_argument("--metrics-plot", type=str, default=None, help="Optional path to save hit-rate@k bar plot (PNG)")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    import os
    print(f"OLLAMA_NO_GPU={os.getenv('OLLAMA_NO_GPU')}")
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"USE_TORCH_DEVICE={os.getenv('USE_TORCH_DEVICE')}")
    top_k_values = parse_top_k(args.top_k)
    cache_folder = Path(args.cache_folder)
    mode = args.mode
    use_reranker = args.use_reranker or mode == "reranker"
    reranker = build_reranker(args.reranker_model, cache_folder=cache_folder) if use_reranker else None
    corpus = load_corpus(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        cache_dir=args.hf_cache_dir,
    )
    store = build_store(corpus, args.embedding_model, cache_folder=cache_folder)
    candidate_k = max(args.rerank_candidates if use_reranker else max(top_k_values), max(top_k_values))
    if mode == "agentic":
        resolved_provider = args.provider or _infer_provider(args.model)
        if resolved_provider == "ollama":
            ensure_ollama_model_available(args.model, args.ollama_base_url)
        llm = build_llm(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            ollama_base_url=args.ollama_base_url,
        )
        retrieve_fn = await make_agentic_retriever(
            store,
            llm=llm,
            rewrites=args.rewrites,
            reranker=reranker,
            candidate_k=candidate_k,
            sequential_retrieval=args.sequential_retrieval,
            reflect=args.reflect, # Pass the reflect argument
        )
    else:
        retrieve_fn = make_standard_retriever(store, reranker=reranker, candidate_k=candidate_k)

metrics = await evaluate_retrieval(corpus, top_k_values=top_k_values, retrieve_fn=retrieve_fn, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2))
    index_path = Path(args.index_path)
    save_index(store, index_path)
    print(f"Saved FAISS index to {index_path}")
    if args.metrics_json:
        save_metrics(Path(args.metrics_json), metrics)
        print(f"Saved metrics to {args.metrics_json}")
    if args.metrics_plot:
        plot_hit_rates(Path(args.metrics_plot), metrics, top_k_values)
        print(f"Saved plot to {args.metrics_plot}")


if __name__ == "__main__":
    import sys
    # --subset distractor
    # sys.argv += "--mode agentic --dataset hotpotqa/hotpot_qa --rewrites 3 --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --cache-folder /vol01/vinay/data/ --split train[:20]".split()
    asyncio.run(main(parse_args()))
