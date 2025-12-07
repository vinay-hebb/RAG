"""Agentic RAG flow tuned for SQuAD with multi-query retrieval and reflection."""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_rag import (
    DEFAULT_CACHE_FOLDER,
    DEFAULT_RERANKER_MODEL,
    _infer_provider,
    build_embeddings,
    build_llm,
    build_reranker,
    retrieve_and_rerank,
    split_documents,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_DATASET = "squad"
DEFAULT_SUBSET: str | None = None
DEFAULT_SPLIT = "validation[:2000]"
DEFAULT_INDEX_PATH = Path("artifacts/squad_index")
DEFAULT_TOP_K = 6
DEFAULT_REWRITES = 3
DEFAULT_CONTEXTS_PER_QUESTION = 1
LOCAL_OLLAMA_URLS = {
    "http://localhost:11434",
    "http://localhost:11434/v1",
    "http://127.0.0.1:11434",
    "http://127.0.0.1:11434/v1",
}


@dataclass(frozen=True)
class RetrievedDoc:
    doc: Document
    score: float
    query: str


def _clean_line(line: str) -> str:
    stripped = line.strip()
    while stripped and stripped[0] in "-•0123456789. ":
        stripped = stripped[1:].strip()
    return stripped


def _add_doc(
    docs: List[Document],
    text: str,
    source: str,
    kind: str,
    question_id: str,
) -> None:
    content = text.strip()
    if not content:
        return
    doc_id = f"{kind}:{source}:{abs(hash(content)) % 1_000_000_000}"
    metadata = {"source": source, "kind": kind, "doc_id": doc_id}
    if question_id:
        metadata["question_id"] = question_id
    docs.append(Document(page_content=content, metadata=metadata))


def extract_squad_docs(example: Dict[str, object]) -> List[Document]:
    docs: List[Document] = []
    question_id = str(example.get("id") or "")
    title = str(example.get("title") or "context")
    context = str(example.get("context") or "")
    _add_doc(docs, context, title, "context", question_id)
    return docs


def load_squad_contexts(
    dataset: str,
    subset: str | None,
    split: str,
    sample_size: int,
    seed: int,
    contexts_per_question: int,
) -> List[Document]:
    if subset:
        dataset_split = load_dataset(dataset, subset, split=split)
    else:
        dataset_split = load_dataset(dataset, split=split)
    dataset_split = dataset_split.shuffle(seed=seed)
    if sample_size > 0:
        sample_size = min(sample_size, len(dataset_split))
        dataset_split = dataset_split.select(range(sample_size))

    all_docs: List[Document] = []
    seen_texts: set[str] = set()

    for example in dataset_split:
        docs = extract_squad_docs(example)
        random.shuffle(docs)
        if contexts_per_question > 0:
            docs = docs[:contexts_per_question]

        for doc in docs:
            content = doc.page_content.strip()
            if not content or content in seen_texts:
                continue
            seen_texts.add(content)
            all_docs.append(doc)

    if not all_docs:
        raise ValueError("No contexts extracted from the SQuAD split.")
    logger.info("Loaded %d unique context documents from SQuAD.", len(all_docs))
    return all_docs


def build_index(args: argparse.Namespace) -> None:
    docs = load_squad_contexts(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        contexts_per_question=args.contexts_per_question,
    )
    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embeddings = build_embeddings(args.embedding_model, cache_folder=Path(args.cache_folder))
    store = FAISS.from_documents(chunks, embeddings)
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    logger.info(
        "Saved SQuAD index with %d chunks to %s",
        len(chunks),
        index_path,
    )


def generate_rewrites(llm, question: str, rewrite_count: int) -> List[str]:
    if rewrite_count <= 0:
        return [question]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You rewrite questions to broaden retrieval coverage. "
                    "Return distinct, concise rewrites that emphasize different key entities or facts. "
                    "Avoid numbering or quotes."
                ),
            ),
            ("user", "Original question: {question}\nNumber of rewrites: {count}"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    raw = chain.invoke({"question": question, "count": rewrite_count})
    rewrites: List[str] = [question]
    seen_lower = {question.lower()}
    for line in raw.splitlines():
        candidate = _clean_line(line)
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen_lower:
            continue
        rewrites.append(candidate)
        seen_lower.add(lowered)
        if len(rewrites) >= rewrite_count + 1:
            break
    return rewrites


def multi_query_retrieve(
    store: FAISS,
    queries: Sequence[str],
    top_k: int,
    candidate_k: int,
    reranker=None,
) -> List[RetrievedDoc]:
    collected: Dict[str, RetrievedDoc] = {}
    search_k = max(top_k, candidate_k)
    for query in queries:
        results = retrieve_and_rerank(
            store,
            query,
            top_k=top_k,
            reranker=reranker,
            candidate_k=search_k,
        )
        for doc, score in results:
            key = doc.page_content
            existing = collected.get(key)
            if existing is None or score > existing.score:
                collected[key] = RetrievedDoc(doc=doc, score=score, query=query)
    ranked = sorted(collected.values(), key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def ensure_ollama_model_available(model: str, base_url: str) -> None:
    """Fail fast with a clear error if the Ollama model is missing locally."""
    normalized_url = base_url.rstrip("/")
    if normalized_url not in LOCAL_OLLAMA_URLS:
        return
    if shutil.which("ollama") is None:
        logger.warning("Ollama provider selected but 'ollama' CLI not found; skipping local model check.")
        return
    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Ollama model '{model}' not found. Pull it with `ollama pull {model}` "
            f"or choose a different --model/--provider. Base URL: {base_url}"
        )


def format_context(docs: Sequence[RetrievedDoc]) -> str:
    if not docs:
        return "No supporting context retrieved."
    formatted: List[str] = []
    for idx, item in enumerate(docs):
        source = item.doc.metadata.get("source", "unknown")
        formatted.append(
            f"[{idx}] (via '{item.query}' | source={source} | score={item.score:.3f})\n{item.doc.page_content}"
        )
    return "\n\n".join(formatted)


def synthesize_answer(llm, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Answer with the given context. Keep it concise. "
                    "Cite supporting snippets with bracketed indices like [0]. "
                    "If the context does not support an answer, say you do not know."
                ),
            ),
            ("user", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"context": context, "question": question})


def reflect_answer(llm, question: str, draft: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You double-check answers against evidence. "
                    "If the draft lacks support, rewrite a short, supported answer with citations or say you do not know."
                ),
            ),
            (
                "user",
                "Question: {question}\nDraft answer: {draft}\nContext:\n{context}\n\nImproved answer:",
            ),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"question": question, "draft": draft, "context": context})


def run_agent(args: argparse.Namespace) -> None:
    embeddings = build_embeddings(args.embedding_model, cache_folder=Path(args.cache_folder))
    store = FAISS.load_local(
        str(Path(args.index_path)),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    resolved_provider = args.provider or _infer_provider(args.model)
    if resolved_provider == "ollama":
        try:
            ensure_ollama_model_available(args.model, args.ollama_base_url)
        except RuntimeError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
    llm = build_llm(
        model=args.model,
        provider=resolved_provider,
        temperature=args.temperature,
        ollama_base_url=args.ollama_base_url,
    )
    reranker = build_reranker(args.reranker_model, cache_folder=Path(args.cache_folder)) if args.use_reranker else None
    rewrites = generate_rewrites(llm, args.question, args.rewrites)
    logger.info("Query rewrites: %s", rewrites)

    results = multi_query_retrieve(
        store,
        queries=rewrites,
        top_k=args.top_k,
        candidate_k=args.rerank_candidates,
        reranker=reranker,
    )

    print("\nRetrieved context:")
    for idx, item in enumerate(results):
        source = item.doc.metadata.get("source", "unknown")
        print(f"[{idx}] score={item.score:.3f} source={source} via='{item.query}'")
        print(item.doc.page_content[:500].strip())
        print("---")

    context_text = format_context(results)
    draft = synthesize_answer(llm, args.question, context_text)
    final = reflect_answer(llm, args.question, draft, context_text) if args.reflect else draft

    print("\nDraft answer:\n")
    print(draft)
    if args.reflect:
        print("\nRefined answer:\n")
        print(final)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic RAG pipeline on SQuAD.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Build a FAISS index from SQuAD contexts.")
    ingest.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Hugging Face dataset name")
    ingest.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help="Dataset subset (optional for SQuAD)",
    )
    ingest.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Dataset split to sample")
    ingest.add_argument("--sample-size", type=int, default=2000, help="How many SQuAD rows to use (0 for all)")
    ingest.add_argument("--seed", type=int, default=13, help="Shuffle seed for sampling")
    ingest.add_argument(
        "--contexts-per-question",
        type=int,
        default=DEFAULT_CONTEXTS_PER_QUESTION,
        help="Max number of contexts to keep per question (0 for unlimited)",
    )
    ingest.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk for splitting")
    ingest.add_argument("--chunk-overlap", type=int, default=120, help="Overlap between chunks")
    ingest.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    ingest.add_argument(
        "--cache-folder",
        type=str,
        default=str(DEFAULT_CACHE_FOLDER),
        help="Directory to store downloaded model weights and tokenizer files",
    )
    ingest.add_argument(
        "--index-path",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help="Where to persist the FAISS index",
    )
    ingest.set_defaults(func=build_index)

    ask = subparsers.add_parser("ask", help="Run the agentic RAG flow over a built index.")
    ask.add_argument("--index-path", type=str, default=str(DEFAULT_INDEX_PATH), help="Folder containing index files")
    ask.add_argument("--question", type=str, required=True, help="User question to answer")
    ask.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to keep after merging rewrites")
    ask.add_argument("--rewrites", type=int, default=DEFAULT_REWRITES, help="How many rewrites to request")
    ask.add_argument("--rerank-candidates", type=int, default=18, help="How many candidates to pull before reranking")
    ask.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking on retrieved candidates",
    )
    ask.add_argument(
        "--reranker-model",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model name for reranking",
    )
    ask.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Chat model name (OpenAI or Ollama)",
    )
    ask.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default=None,
        help="LLM provider (defaults to auto-detect from model name)",
    )
    ask.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the Ollama API",
    )
    ask.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the chat model")
    ask.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model used when building the index",
    )
    ask.add_argument(
        "--cache-folder",
        type=str,
        default=str(DEFAULT_CACHE_FOLDER),
        help="Directory to store downloaded model weights and tokenizer files",
    )
    ask.add_argument(
        "--reflect",
        action="store_true",
        help="Run a second-pass reflection step to verify the draft answer",
    )
    ask.set_defaults(func=run_agent)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
