"""Tiny LangChain RAG CLI that keeps things simple (FAISS + OpenAI/Ollama)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _is_openai_model(name: str) -> bool:
    if name.startswith(("openai/", "openai:", "text-embedding-", "gpt-", "o1-", "o3-")):
        return True
    return name in {"gpt-4o", "gpt-4o-mini", "o3-mini"}


def _strip_prefix(name: str, prefix: str) -> str:
    if name.startswith(prefix):
        return name.split(prefix, 1)[1]
    return name


def _normalize_openai_name(name: str) -> str:
    return _strip_prefix(_strip_prefix(name, "openai/"), "openai:")


def _normalize_ollama_name(name: str) -> str:
    return _strip_prefix(_strip_prefix(name, "ollama/"), "ollama:")


def _infer_provider(name: str) -> str:
    if name.startswith(("ollama/", "ollama:")) or (":" in name and not name.startswith("openai:")):
        return "ollama"
    return "openai"


def build_embeddings(model_name: str) -> Embeddings:
    if _is_openai_model(model_name):
        normalized = _normalize_openai_name(model_name)
        return OpenAIEmbeddings(model=normalized)
    return HuggingFaceEmbeddings(model_name=model_name)


def load_documents(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": str(path)}))
    if not docs:
        raise FileNotFoundError(f"No text/markdown files found under {data_dir}")
    return docs


def split_documents(documents: Sequence[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


def ingest_corpus(data_dir: Path, index_path: Path, embedding_model: str, chunk_size: int, chunk_overlap: int) -> None:
    documents = load_documents(data_dir)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = build_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    print(f"Ingested {len(documents)} documents into {len(chunks)} chunks.")
    print(f"Saved FAISS index to {index_path}")


def build_llm(
    model: str,
    provider: Optional[str],
    temperature: float,
    ollama_base_url: str,
) -> BaseChatModel:
    resolved = provider or _infer_provider(model)
    if resolved == "ollama":
        normalized = _normalize_ollama_name(model)
        return ChatOllama(model=normalized, base_url=ollama_base_url, temperature=temperature)
    normalized = _normalize_openai_name(model)
    return ChatOpenAI(model=normalized, temperature=temperature)


def _format_docs(docs: Sequence[Document]) -> str:
    if not docs:
        return "No context retrieved."
    formatted: List[str] = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{idx}] {doc.page_content}\nSource: {source}")
    return "\n\n".join(formatted)


def build_chain(store: FAISS, llm: BaseChatModel, top_k: int):
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Keep replies short. Stick to the provided context. "
                    "Cite sources in square brackets like [0]. If you cannot find it, say you do not know."
                ),
            ),
            ("user", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
        ]
    )
    return (
        {"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def run_ingest(args: argparse.Namespace) -> None:
    ingest_corpus(
        data_dir=Path(args.data_dir),
        index_path=Path(args.index_path),
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def run_query(args: argparse.Namespace) -> None:
    embeddings = build_embeddings(args.embedding_model)
    store = FAISS.load_local(
        str(Path(args.index_path)),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = store.as_retriever(search_kwargs={"k": args.top_k})
    docs = retriever.invoke(args.question)
    print("\nRetrieved context:")
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        print(f"[{idx}] source={source}")
        print(doc.page_content[:400].strip())
        print("---")

    llm = build_llm(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        ollama_base_url=args.ollama_base_url,
    )
    chain = build_chain(store, llm, args.top_k)
    answer = chain.invoke(args.question)
    print("\nAnswer:\n")
    print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain RAG workflow using FAISS.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build and persist a FAISS index with LangChain components.")
    ingest_parser.add_argument("--data-dir", type=str, default="data", help="Directory containing .txt/.md files")
    ingest_parser.add_argument("--index-path", type=str, default="artifacts/langchain_index", help="Folder to store index files")
    ingest_parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name (OpenAI or HuggingFace)")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk for splitting")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between chunks")
    ingest_parser.set_defaults(func=run_ingest)

    query_parser = subparsers.add_parser("query", help="Query an existing FAISS index")
    query_parser.add_argument("--index-path", type=str, default="artifacts/langchain_index", help="Folder containing index files")
    query_parser.add_argument("--question", type=str, required=True, help="User question")
    query_parser.add_argument("--top-k", type=int, default=4, help="How many chunks to retrieve")
    query_parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Chat model name (OpenAI or Ollama)")
    query_parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model to use for retrieval",
    )
    query_parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default=None,
        help="LLM provider (defaults to auto-detect from model name)",
    )
    query_parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        help="Base URL for the Ollama API",
    )
    query_parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the chat model")
    query_parser.set_defaults(func=run_query)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
