"""Agentic RAG flow tuned for SQuAD with multi-query retrieval and reflection."""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import List, Sequence
import asyncio

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_rag import (
    DEFAULT_CACHE_FOLDER,
    DEFAULT_RERANKER_MODEL,
    _infer_provider,
    build_embeddings,
    build_llm,
    build_reranker,
    retrieve_and_rerank,
    rerank_results,
    split_documents,

    # Moved from agentic_rag.py
    DEFAULT_DATASET,
    DEFAULT_SUBSET,
    DEFAULT_SPLIT,
    DEFAULT_CONTEXTS_PER_QUESTION,
    LOCAL_OLLAMA_URLS,
    RetrievedDoc,
    load_squad_contexts,
    generate_rewrites,
    multi_query_retrieve,
    ensure_ollama_model_available,
    format_context,
    synthesize_answer,
    reflect_answer,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_INDEX_PATH = Path("artifacts/squad_index")
DEFAULT_TOP_K = 6
DEFAULT_REWRITES = 3

from typing import List, TypedDict, Union, Literal, Optional # Added for LangGraphState
from langchain_core.language_models.chat_models import BaseChatModel # Added for LangGraphState
from langchain_core.runnables import RunnableConfig # Added for LangGraphState
from langgraph.graph import StateGraph, START # Added for LangGraph

# --- Start LangGraph integration ---

# Define the Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        questions: The user's initial questions (List of strings).
        rewritten_queries: List of lists of rewritten queries for retrieval (one list per question).
        documents: List of lists of retrieved documents (one list of RetrievedDoc objects per question).
        draft_answers: The initial draft answers from synthesis (List of strings).
        final_answers: The final answers after reflection (if enabled) (List of strings).
        llm: The LLM instance used for generations.
        store: The FAISS vector store.
        reranker: The CrossEncoder reranker instance (optional).
        config: Configuration for the run (e.g., top_k, rerank_candidates).
    """
    questions: List[str]
    rewritten_queries: List[List[str]]
    documents: List[List[RetrievedDoc]]
    draft_answers: List[str]
    final_answers: List[str]
    llm: BaseChatModel
    store: FAISS
    reranker: Optional[object] # CrossEncoder type
    config: dict # To pass args like top_k, rerank_candidates, use_reranker, reflect


# Node Functions
async def rewrite_node(state: GraphState) -> dict:
    """Rewrites each question to broaden retrieval coverage."""
    logger.info("---REWRITE QUESTIONS---")
    questions = state["questions"]
    llm = state["llm"]
    config = state["config"]
    rewrite_count = config.get("rewrites", 3)

    all_rewritten_queries = []
    for question in questions:
        rewrites = await generate_rewrites(llm, question, rewrite_count)
        all_rewritten_queries.append(rewrites)
        logger.info("Rewritten queries for '%s': %s", question, rewrites)
    return {"rewritten_queries": all_rewritten_queries}


async def retrieve_node(state: GraphState) -> dict:
    """Performs multi-query retrieval and reranking for each question."""
    logger.info("---RETRIEVE DOCUMENTS FOR ALL QUESTIONS---")
    questions = state["questions"]
    all_rewritten_queries = state["rewritten_queries"]
    store = state["store"]
    reranker = state["reranker"]
    config = state["config"]
    top_k = config.get("top_k", 6)
    rerank_candidates = config.get("rerank_candidates", 18)
    sequential_retrieval = config.get("sequential_retrieval", False)

    all_retrieved_documents = []
    # Use asyncio.gather to run retrieval for all questions in parallel
    retrieval_tasks = []
    for i, question in enumerate(questions):
        rewritten_queries = all_rewritten_queries[i]
        retrieval_tasks.append(
            multi_query_retrieve(
                store,
                queries=rewritten_queries,
                base_question=question,
                top_k=top_k,
                candidate_k=rerank_candidates,
                reranker=reranker,
                run_sequentially=sequential_retrieval,
            )
        )
    
    all_retrieved_documents = await asyncio.gather(*retrieval_tasks)

    for i, docs_for_q in enumerate(all_retrieved_documents):
        logger.info("Retrieved %d documents for question '%s'.", len(docs_for_q), questions[i])
    
    return {"documents": all_retrieved_documents}


async def generate_node(state: GraphState) -> dict:
    """Synthesizes draft answers from retrieved context for each question."""
    logger.info("---GENERATE DRAFT ANSWERS---")
    questions = state["questions"]
    all_documents = state["documents"]
    llm = state["llm"]

    draft_answers = []
    # Use asyncio.gather to run generation for all questions in parallel
    generation_tasks = []
    for i, question in enumerate(questions):
        documents_for_question = all_documents[i]
        context_text = format_context(documents_for_question)
        generation_tasks.append(synthesize_answer(llm, question, context_text))
    
    draft_answers = await asyncio.gather(*generation_tasks)

    for i, draft in enumerate(draft_answers):
        logger.info("Generated draft answer for question '%s'.", questions[i])
    
    return {"draft_answers": draft_answers}


async def reflect_node(state: GraphState) -> dict:
    """Reflects on the draft answers and refines them for each question."""
    logger.info("---REFLECT ON ANSWERS---")
    questions = state["questions"]
    draft_answers = state["draft_answers"]
    all_documents = state["documents"]
    llm = state["llm"]

    final_answers = []
    # Use asyncio.gather to run reflection for all questions in parallel
    reflection_tasks = []
    for i, question in enumerate(questions):
        draft = draft_answers[i]
        documents_for_question = all_documents[i]
        context_text = format_context(documents_for_question)
        reflection_tasks.append(reflect_answer(llm, question, draft, context_text))
    
    final_answers = await asyncio.gather(*reflection_tasks)

    for i, final in enumerate(final_answers):
        logger.info("Refined answer through reflection for question '%s'.", questions[i])
    
    return {"final_answers": final_answers}

def should_reflect(state: GraphState) -> Literal["reflect", "__end__"]:
    """Determines whether to proceed to reflection or end."""
    config = state["config"]
    if config.get("reflect"):
        return "reflect"
    return "__end__"


# Build the LangGraph
def build_agent_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("reflect", reflect_node)

    # Set up edges
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate")

    # Conditional edge for reflection
    workflow.add_conditional_edges(
        "generate",
        should_reflect,
        {"reflect": "reflect", "__end__": "__end__"},
    )
    workflow.add_edge("reflect", "__end__") # Reflection is the last step if taken

    return workflow.compile()

async def run_langgraph_agent(
    questions: List[str],
    llm: BaseChatModel,
    store: FAISS,
    reranker: Optional[object],
    config: dict,
) -> List[dict]:
    """
    Runs the agentic RAG flow using LangGraph for a batch of questions.
    Accepts pre-initialized components.
    Returns a list of dictionaries, one for each question's final state.
    """
    # Prepare initial state and config
    initial_state = GraphState(
        questions=questions, # Now a list of questions
        rewritten_queries=[[] for _ in questions], # Initialize list of lists
        documents=[[] for _ in questions],         # Initialize list of lists
        draft_answers=["" for _ in questions],      # Initialize list of strings
        final_answers=["" for _ in questions],      # Initialize list of strings
        llm=llm,
        store=store,
        reranker=reranker,
        config=config,
    )

    app = build_agent_graph()
    final_state = await app.ainvoke(initial_state)

    results_for_all_questions = []
    for i, question in enumerate(final_state["questions"]):
        print(f"\n--- LangGraph Agent Results for Question {i+1} ---")
        print("\nInitial Question:")
        print(question)
        print("\nRewritten Queries:")
        for q in final_state["rewritten_queries"][i]:
            print(f"- {q}")

        print("\nRetrieved Context:")
        if final_state["documents"] and len(final_state["documents"]) > i:
            for idx, item in enumerate(final_state["documents"][i]):
                source = item.doc.metadata.get("source", "unknown")
                print(f"[{idx}] score={item.score:.3f} source={source} via='{item.query}'")
                print(item.doc.page_content[:500].strip())
                print("---")
        else:
            print("No documents retrieved.")

        print("\nDraft Answer:")
        print(final_state["draft_answers"][i])

        if final_state["config"].get("reflect"):
            print("\nFinal Answer (after reflection):")
            print(final_state["final_answers"][i])
        else:
            print("\nReflection was not enabled.")
        
        results_for_all_questions.append({
            "question": question,
            "rewritten_queries": final_state["rewritten_queries"][i],
            "documents": final_state["documents"][i],
            "draft_answer": final_state["draft_answers"][i],
            "final_answer": final_state["final_answers"][i] if final_state["config"].get("reflect") else final_state["draft_answers"][i],
        })
    
    return results_for_all_questions

# --- End LangGraph integration ---


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
    ask.add_argument("--questions", type=str, nargs='+', required=True, help="User questions to answer (space-separated)")
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
        default=os.getenv("OLLAMA_MODEL", "gemma:2b"),
        help="Chat model name (Ollama or OpenAI)",
    )
    ask.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider (defaults to auto-detect from model name)",
    )
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
    ask.add_argument(
        "--sequential-retrieval",
        action="store_true",
        help="Force multi-query retrieval to run sequentially instead of in parallel",
    )
    ask.set_defaults(func=run_langgraph_agent)

    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.command == "ingest":
        args.func(args)
    elif args.command == "ask":
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

        config = {
            "top_k": args.top_k,
            "rewrites": args.rewrites,
            "rerank_candidates": args.rerank_candidates,
            "use_reranker": args.use_reranker,
            "reflect": args.reflect,
            "sequential_retrieval": args.sequential_retrieval,
        }
        await run_langgraph_agent(
            questions=args.questions,
            llm=llm,
            store=store,
            reranker=reranker,
            config=config,
        )


if __name__ == "__main__":
    asyncio.run(main())
