"""Quick-and-dirty retrieval check for the LangChain RAG setup (baseline, reranker, and agentic)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_rag import DEFAULT_CACHE_FOLDER, build_embeddings, build_llm, build_reranker, retrieve_and_rerank
from agentic_rag import generate_rewrites, multi_query_retrieve


DEFAULT_TOP_K = (1, 3, 5, 10)


@dataclass(frozen=True)
class QAExample:
    question: str
    doc_id: str


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
    split: str,
    sample_size: int,
    seed: int,
    chunk_size: int,
    chunk_overlap: int,
) -> Corpus:
    dataset = load_dataset(dataset_name, split=split)
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

        doc_id = context_lookup.get(context)
        if doc_id is None:
            doc_id = f"{dataset_name}-doc-{len(context_lookup)}"
            context_lookup[context] = doc_id
            doc = Document(
                page_content=context,
                metadata={"doc_id": doc_id, "dataset": dataset_name, "split": split, "source": doc_id},
            )
            chunks.extend(splitter.split_documents([doc]))

        examples.append(QAExample(question=question, doc_id=doc_id))

    if not examples:
        raise ValueError(f"No valid examples found in {dataset_name}:{split}")

    return Corpus(chunks=chunks, examples=examples)


def build_store(corpus: Corpus, embedding_model: str, cache_folder: Path) -> FAISS:
    embeddings = build_embeddings(embedding_model, cache_folder=cache_folder)
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


def make_standard_retriever(store: FAISS, reranker, candidate_k: int) -> Callable[[str, int], Sequence[Document]]:
    def retrieve(question: str, top_k: int) -> Sequence[Document]:
        results = retrieve_and_rerank(
            store,
            question,
            top_k=top_k,
            reranker=reranker,
            candidate_k=max(candidate_k, top_k),
        )
        return [doc for doc, _ in results]

    return retrieve


def make_agentic_retriever(
    store: FAISS,
    llm,
    rewrites: int,
    reranker,
    candidate_k: int,
) -> Callable[[str, int], Sequence[Document]]:
    def retrieve(question: str, top_k: int) -> Sequence[Document]:
        queries = generate_rewrites(llm, question, rewrites)
        results = multi_query_retrieve(
            store,
            queries=queries,
            top_k=top_k,
            candidate_k=max(candidate_k, top_k),
            reranker=reranker,
        )
        return [_extract_doc(item) for item in results]

    return retrieve


def evaluate_retrieval(
    corpus: Corpus,
    top_k_values: Sequence[int],
    retrieve_fn: Callable[[str, int], Sequence[Document]],
) -> Dict[str, float]:
    max_k = max(top_k_values)
    hits_by_k = {k: 0 for k in top_k_values}
    ndcg_by_k = {k: 0.0 for k in top_k_values}
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []

    for example in tqdm(corpus.examples, desc="Evaluating queries"):
        docs = retrieve_fn(example.question, max_k)
        doc_hits = [doc.metadata.get("doc_id") for doc in docs]
        rank = None
        for i, doc_id in enumerate(doc_hits):
            if doc_id == example.doc_id:
                rank = i
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
    parser.add_argument("--split", type=str, default="train[:2000]", help="Dataset split")
    parser.add_argument("--sample-size", type=int, default=800, help="How many examples to sample (0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--top-k", type=str, default="1,3,5,10", help="Comma-separated list of k values")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
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
    parser.add_argument("--model", type=str, default="gemma3:1b", help="Chat model for agentic mode")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider for agentic mode",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for the Ollama API",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for agentic mode")
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path to save metrics as JSON")
    parser.add_argument("--metrics-plot", type=str, default=None, help="Optional path to save hit-rate@k bar plot (PNG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = parse_top_k(args.top_k)
    cache_folder = Path(args.cache_folder)
    mode = args.mode
    use_reranker = args.use_reranker or mode == "reranker"
    reranker = build_reranker(args.reranker_model, cache_folder=cache_folder) if use_reranker else None
    corpus = load_corpus(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store = build_store(corpus, args.embedding_model, cache_folder=cache_folder)
    candidate_k = max(args.rerank_candidates if use_reranker else max(top_k_values), max(top_k_values))
    if mode == "agentic":
        llm = build_llm(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            ollama_base_url=args.ollama_base_url,
        )
        retrieve_fn = make_agentic_retriever(
            store,
            llm=llm,
            rewrites=args.rewrites,
            reranker=reranker,
            candidate_k=candidate_k,
        )
    else:
        retrieve_fn = make_standard_retriever(store, reranker=reranker, candidate_k=candidate_k)

    metrics = evaluate_retrieval(corpus, top_k_values=top_k_values, retrieve_fn=retrieve_fn)
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
    main()
