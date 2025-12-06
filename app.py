"""Small Dash UI to poke at the LangChain RAG index and peek at metrics."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dash
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from datasets import load_dataset
import plotly.graph_objects as go
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_rag import build_chain, build_embeddings, build_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

INDEX_PATH = Path(os.getenv("INDEX_PATH", "artifacts/langchain_index"))
EMBEDDING_MODEL = os.getenv("WEB_APP_EMBED_MODEL", "all-MiniLM-L6-v2")
METRICS_PATH = Path(os.getenv("METRICS_PATH", "artifacts/lc_metrics.json"))
DEFAULT_TOP_K = int(os.getenv("WEB_APP_TOP_K", "4"))
DEFAULT_MODEL = os.getenv("WEB_APP_MODEL", os.getenv("OPENAI_MODEL", "gemma3:1b"))
DEFAULT_PROVIDER = os.getenv("WEB_APP_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
DEFAULT_SAMPLE_DATASET = os.getenv("SAMPLE_QUESTION_DATASET", "squad")
DEFAULT_SAMPLE_SPLIT = os.getenv("SAMPLE_QUESTION_SPLIT", "train[:200]")
SAMPLE_QUESTION_LIMIT = int(os.getenv("SAMPLE_QUESTION_LIMIT", "4"))

FALLBACK_SAMPLE_QUESTIONS: List[str] = [
    "What is retrieval-augmented generation?",
    "How does chunk size affect retrieval quality?",
    "Which components make up this RAG pipeline?",
    "How are citations handled in the answers?",
]

store: Optional[FAISS] = None


def load_sample_questions() -> List[str]:
    """Pull a handful of questions from HF; fall back to static ones on any hiccup."""
    try:
        dataset = load_dataset(DEFAULT_SAMPLE_DATASET, split=DEFAULT_SAMPLE_SPLIT)
    except Exception as exc:
        logger.warning("Could not fetch sample questions (%s); using built-ins.", exc)
        return FALLBACK_SAMPLE_QUESTIONS

    seen: List[str] = []
    for row in dataset:
        question = str(row.get("question", "")).strip()
        if question and question not in seen:
            seen.append(question)
    if not seen:
        return FALLBACK_SAMPLE_QUESTIONS
    choices = seen if len(seen) <= SAMPLE_QUESTION_LIMIT else random.sample(seen, SAMPLE_QUESTION_LIMIT)
    random.shuffle(choices)
    return choices[:SAMPLE_QUESTION_LIMIT]


SAMPLE_QUESTIONS: List[str] = load_sample_questions()
DEFAULT_QUESTION = SAMPLE_QUESTIONS[0] if SAMPLE_QUESTIONS else FALLBACK_SAMPLE_QUESTIONS[0]


def load_store(path: Path, embedding_model: str) -> Tuple[Optional[FAISS], Optional[object]]:
    if not path.exists():
        return None, None
    embeddings = build_embeddings(embedding_model)
    loaded_store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    return loaded_store, embeddings


def load_metrics(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Metrics file looked broken; ignoring it.")
        return None


def extract_hit_rates(metrics: Dict[str, float]) -> List[Tuple[int, float]]:
    hits: List[Tuple[int, float]] = []
    for key, value in metrics.items():
        if not key.startswith("hit_rate@"):
            continue
        try:
            k = int(key.split("@", 1)[1])
            hits.append((k, float(value)))
        except (ValueError, TypeError):
            continue
    hits.sort(key=lambda pair: pair[0])
    return hits


def build_hit_rate_figure(metrics: Optional[Dict[str, float]]) -> go.Figure:
    if not metrics:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(t=60, l=60, r=40, b=60),
            paper_bgcolor="#0b1222",
            plot_bgcolor="#0b1222",
            font=dict(color="#e2e8f0"),
            title="No metrics yet — run langchain_dataset_metrics.py and refresh.",
        )
        return fig

    hit_rates = extract_hit_rates(metrics)
    if not hit_rates:
        return build_hit_rate_figure(None)

    labels = [f"@{k}" for k, _ in hit_rates]
    values = [v for _, v in hit_rates]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#38bdf8",
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(t=60, l=80, r=40, b=60),
        paper_bgcolor="#0b1222",
        plot_bgcolor="#0b1222",
        font=dict(color="#e2e8f0"),
        title="Hit-rate@k",
        xaxis_title="k (top-k)",
        yaxis_title="Hit rate",
    )
    fig.update_yaxes(range=[0, 1.05], dtick=0.25, gridcolor="#1f2937", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def retrieve_with_scores(active_store: FAISS, question: str, k: int) -> List[Tuple[Document, float]]:
    return active_store.similarity_search_with_score(question, k=k)


def is_ollama_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def start_ollama_if_needed(delay_seconds: float = 2.0) -> None:
    if DEFAULT_PROVIDER != "ollama":
        return
    if is_ollama_running():
        return
    log_path = Path("/tmp/ollama_web_app.log")
    with log_path.open("ab") as log:
        process = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log)
    time.sleep(delay_seconds)
    logger.info("Started ollama serve (pid=%s); logs at %s", process.pid, log_path)


def ensure_ollama_model(model: str) -> None:
    if DEFAULT_PROVIDER != "ollama":
        return
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return
        logger.info("Pulling Ollama model '%s'...", model)
        subprocess.run(["ollama", "pull", model], check=True)
    except FileNotFoundError:
        logger.warning("ollama CLI not found; install Ollama to use provider=ollama.")


def answer_question(question: str) -> Tuple[str, List[Tuple[Document, float]]]:
    if store is None:
        raise RuntimeError("Index not found. Run ingest to create artifacts/langchain_index.")
    start_ollama_if_needed()
    ensure_ollama_model(DEFAULT_MODEL)
    docs_with_scores = retrieve_with_scores(store, question, k=DEFAULT_TOP_K)
    llm = build_llm(
        model=DEFAULT_MODEL,
        provider=DEFAULT_PROVIDER,
        temperature=float(os.getenv("WEB_APP_TEMPERATURE", "0.2")),
        ollama_base_url=OLLAMA_BASE_URL,
    )
    chain = build_chain(store, llm, top_k=DEFAULT_TOP_K)
    answer = chain.invoke(question)
    return answer, docs_with_scores


store, _embeddings = load_store(INDEX_PATH, EMBEDDING_MODEL)
metrics_cache = load_metrics(METRICS_PATH)
hit_rate_fig = build_hit_rate_figure(metrics_cache)

app = Dash(__name__, title="LangChain RAG Playground")
server = app.server


def card(children: Sequence[html.Div]) -> html.Div:
    style = {
        "background": "#111827",
        "border": "1px solid #1f2937",
        "borderRadius": "10px",
        "padding": "16px 20px",
        "marginBottom": "20px",
        "boxShadow": "0 6px 18px rgba(0,0,0,0.25)",
    }
    return html.Div(children, style=style)


app.layout = html.Div(
    style={"background": "#0f172a", "color": "#e2e8f0", "minHeight": "100vh", "margin": 0, "padding": 0},
    children=[
        html.Header(
            style={
                "padding": "16px 24px",
                "background": "linear-gradient(135deg, #1e293b, #0f172a)",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
            },
            children=html.H2("LangChain RAG Playground", style={"margin": 0}),
        ),
        html.Main(
            style={"padding": "24px", "maxWidth": "960px", "margin": "0 auto"},
            children=[
                card([
                    html.Label("Ask a question", style={"display": "block", "marginBottom": "8px", "fontWeight": 600, "color": "#93c5fd"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        q,
                                        id={"type": "sample-button", "index": idx},
                                        n_clicks=0,
                                        className="pill-button",
                                        style={
                                            "background": "#0b1222",
                                            "border": "1px solid #334155",
                                            "color": "#e2e8f0",
                                            "padding": "8px 12px",
                                            "borderRadius": "999px",
                                            "cursor": "pointer",
                                            "width": "auto",
                                        },
                                    )
                                    for idx, q in enumerate(SAMPLE_QUESTIONS)
                                ],
                                style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "12px"},
                            ),
                            dcc.Input(
                                id="custom-question",
                                type="text",
                                placeholder="Type your question",
                                value=DEFAULT_QUESTION,
                                style={
                                    "padding": "10px 12px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #334155",
                                    "background": "#0b1222",
                                    "color": "#e2e8f0",
                                    "width": "100%",
                                    "boxSizing": "border-box",
                                },
                            ),
                            html.Div(
                                html.Button(
                                    "Get Answer",
                                    id="ask-button",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "12px",
                                        "padding": "10px 12px",
                                        "borderRadius": "8px",
                                        "border": "none",
                                        "background": "linear-gradient(135deg, #2563eb, #38bdf8)",
                                        "color": "#e2e8f0",
                                        "fontWeight": 700,
                                        "cursor": "pointer",
                                        "width": "100%",
                                    },
                                )
                            ),
                        ]
                    ),
                    html.Div(id="error-text", style={"color": "#f87171", "fontWeight": 700, "marginTop": "10px"}),
                ]),
                card([
                    html.Div("Answer", style={"fontSize": "18px", "marginBottom": "8px"}),
                    html.Div(id="answer-text", style={"whiteSpace": "pre-wrap", "lineHeight": 1.5}),
                ]),
                card([
                    html.Div("Retrieved Context", style={"fontSize": "18px", "marginBottom": "8px"}),
                    html.Div(id="retrieved-context"),
                ]),
                card([
                    html.Div("Evaluation metrics on squad", style={"fontSize": "18px", "marginBottom": "8px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "alignItems": "stretch"},
                        children=[
                            dcc.Graph(
                                id="metrics-graph",
                                figure=hit_rate_fig,
                                config={"displayModeBar": False},
                                style={"width": "100%"},
                            ),
                            html.Div(
                                style={
                                    "background": "#0b1222",
                                    "border": "1px solid #1f2937",
                                    "borderRadius": "10px",
                                    "padding": "12px",
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr",
                                    "rowGap": "12px",
                                    "alignContent": "start",
                                },
                                children=[
                                    html.Div(
                                        [
                                            html.Div("MRR", style={"fontSize": "13px", "color": "#93c5fd", "marginBottom": "4px"}),
                                            html.Div(f"{metrics_cache.get('mrr', 0.0):.3f}" if metrics_cache else "—", style={"fontSize": "20px", "fontWeight": 700}),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Mean Rank", style={"fontSize": "13px", "color": "#93c5fd", "marginBottom": "4px"}),
                                            html.Div(f"{metrics_cache.get('mean_rank', 0.0):.2f}" if metrics_cache else "—", style={"fontSize": "20px", "fontWeight": 700}),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Number of samples", style={"fontSize": "13px", "color": "#93c5fd", "marginBottom": "4px"}),
                                            html.Div(f"{int(metrics_cache.get('total_queries', 0.0)):,}" if metrics_cache else "—", style={"fontSize": "20px", "fontWeight": 700}),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                ]),
            ],
        ),
    ],
)


def format_retrieved(docs_with_scores: List[Tuple[Document, float]]) -> List[html.Div]:
    items: List[html.Div] = []
    for doc, score in docs_with_scores:
        items.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(f"score={score:.4f}", className="badge", style={"display": "inline-block", "padding": "4px 8px", "background": "#1f2937", "borderRadius": "6px", "marginRight": "8px"}),
                            html.Span(f"source={doc.metadata.get('source', 'unknown')}", className="badge", style={"display": "inline-block", "padding": "4px 8px", "background": "#1f2937", "borderRadius": "6px"}),
                        ]
                    ),
                    html.Div(doc.page_content, style={"marginTop": "8px"}),
                ],
                style={"background": "#0b1222", "borderRadius": "8px", "padding": "12px", "marginTop": "8px", "border": "1px solid #1f2937"},
            )
        )
    return items


@app.callback(
    Output("answer-text", "children"),
    Output("retrieved-context", "children"),
    Output("error-text", "children"),
    Input({"type": "sample-button", "index": ALL}, "n_clicks"),
    Input("ask-button", "n_clicks"),
    State("custom-question", "value"),
    prevent_initial_call=False,
)
def handle_question(sample_clicks: List[int], ask_clicks: int, custom_question: Optional[str]):
    question = DEFAULT_QUESTION
    triggered = ctx.triggered_id

    if triggered == "ask-button":
        question = (custom_question or "").strip() or DEFAULT_QUESTION
    elif isinstance(triggered, dict) and triggered.get("type") == "sample-button":
        try:
            idx = int(triggered.get("index", 0))
            if 0 <= idx < len(SAMPLE_QUESTIONS):
                question = SAMPLE_QUESTIONS[idx]
        except (ValueError, TypeError):
            question = DEFAULT_QUESTION

    try:
        answer, retrieved = answer_question(question)
    except Exception as exc:
        msg = str(exc) or "Something went wrong while answering."
        if "artifacts/langchain_index" in msg or "Index not found" in msg:
            msg += " Build the index first (langchain_dataset_metrics.py or ingest CLI)."
        return "", [], msg

    return answer, format_retrieved(retrieved), ""


def run() -> None:
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)


if __name__ == "__main__":
    run()
