"""Quick-and-dirty retrieval check for the LangChain RAG setup."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_rag import build_embeddings


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


def build_store(corpus: Corpus, embedding_model: str) -> FAISS:
    embeddings = build_embeddings(embedding_model)
    return FAISS.from_documents(corpus.chunks, embeddings)


def save_index(store: FAISS, path: Path) -> None:
    """Persist the FAISS store so the web app can reuse the index."""
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def evaluate_retrieval(
    store: FAISS,
    corpus: Corpus,
    top_k_values: Sequence[int],
) -> Dict[str, float]:
    max_k = max(top_k_values)
    hits_by_k = {k: 0 for k in top_k_values}
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []

    for example in tqdm(corpus.examples, desc="Evaluating queries"):
        results = store.similarity_search_with_score(example.question, k=max_k)
        doc_hits = [doc.metadata.get("doc_id") for doc, _ in results]
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

    total = len(corpus.examples)
    hit_rates = {f"hit_rate@{k}": hits_by_k[k] / float(total) for k in top_k_values}

    metrics: Dict[str, float] = {
        "total_queries": float(total),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    metrics.update(hit_rates)
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
        "--index-path",
        type=str,
        default="artifacts/langchain_index",
        help="Where to persist the FAISS index for the web app",
    )
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path to save metrics as JSON")
    parser.add_argument("--metrics-plot", type=str, default=None, help="Optional path to save hit-rate@k bar plot (PNG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = parse_top_k(args.top_k)
    corpus = load_corpus(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store = build_store(corpus, args.embedding_model)
    metrics = evaluate_retrieval(store, corpus, top_k_values=top_k_values)
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
