"""LangGraph orchestration for the arXiv Paper Digest & QA Agent.

Nodes call existing services; this module does not reimplement business logic.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.services.arxiv import ArxivService, ArxivServiceError
from app.services.chunker import Chunker
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService, LLMServiceError
from app.services.pdf import PDFService, PDFServiceError
from app.services.pdf_parser import PDFParseError, PDFParser
from app.services.vector_store import VectorStore, VectorStoreError
from app.state import AgentState, DocumentChunk, PaperMetadata, ParsedDocument

# New-style IDs: 2401.12345 / 2401.12345v3
# Also abs/pdf URLs and legacy archive/number forms.
_ARXIV_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/([^\s?#]+)",
    re.IGNORECASE,
)
_ARXIV_NEWID_RE = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
_ARXIV_LEGACY_RE = re.compile(
    r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b",
    re.IGNORECASE,
)

_BRIEFING_CONTEXT_CHAR_LIMIT = 14_000
_TOPIC_SEARCH_LIMIT = 10
_TOPIC_SELECT_TOP_K = 3
_QA_TOP_K = 3

# Lazily constructed so importing the graph does not always load the embedder.
_arxiv_service: ArxivService | None = None
_pdf_service: PDFService | None = None
_pdf_parser: PDFParser | None = None
_chunker: Chunker | None = None
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStore | None = None
_llm_service: LLMService | None = None


def _arxiv() -> ArxivService:
    global _arxiv_service
    if _arxiv_service is None:
        _arxiv_service = ArxivService()
    return _arxiv_service


def _pdf() -> PDFService:
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service


def _parser() -> PDFParser:
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = PDFParser()
    return _pdf_parser


def _chunks() -> Chunker:
    global _chunker
    if _chunker is None:
        _chunker = Chunker()
    return _chunker


def _embeddings() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def _vectors() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _llm() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def _halted(state: AgentState) -> bool:
    return bool(state.get("error"))


def _extract_arxiv_id(query: str) -> str | None:
    """Extract a normalized arXiv ID from a free-text query, if present."""
    text = query.strip()
    url_match = _ARXIV_URL_RE.search(text)
    if url_match:
        return url_match.group(1).removesuffix(".pdf").strip("/")

    new_match = _ARXIV_NEWID_RE.search(text)
    if new_match:
        return new_match.group(1)

    legacy_match = _ARXIV_LEGACY_RE.search(text)
    if legacy_match:
        return legacy_match.group(1)

    # Bare ID as the entire query.
    compact = text.strip().removesuffix(".pdf")
    if _ARXIV_NEWID_RE.fullmatch(compact) or _ARXIV_LEGACY_RE.fullmatch(compact):
        return compact
    return None


def collection_name_for(arxiv_id: str) -> str:
    """Build a deterministic Chroma collection name from an arXiv ID."""
    safe = arxiv_id.strip().replace("/", "_").replace(".", "_")
    return f"paper_{safe}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    # Embeddings are L2-normalized by EmbeddingService; keep a safe fallback.
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _briefing_context(document: ParsedDocument) -> str:
    """Build bounded briefing context from parsed sections (not the whole PDF)."""
    preferred = {
        "abstract",
        "introduction",
        "method",
        "methods",
        "methodology",
        "approach",
        "experiments",
        "results",
        "experimental results",
        "discussion",
        "conclusion",
        "conclusions",
        "limitations",
    }

    parts: list[str] = []
    for section in document.sections:
        if section.title.lower() in preferred or section.title == "Preamble":
            text = section.text.strip()
            if text:
                parts.append(f"## {section.title}\n{text}")

    if not parts:
        parts = [
            f"## {section.title}\n{section.text.strip()}"
            for section in document.sections
            if section.text.strip()
        ]

    if not parts:
        return document.full_text[:_BRIEFING_CONTEXT_CHAR_LIMIT]

    context = "\n\n".join(parts)
    if len(context) <= _BRIEFING_CONTEXT_CHAR_LIMIT:
        return context

    # Truncate at a paragraph/sentence boundary to avoid mid-word cuts that
    # can confuse briefing generation (no silent mid-string display truncation).
    cut = context[:_BRIEFING_CONTEXT_CHAR_LIMIT]
    boundary = max(cut.rfind("\n\n"), cut.rfind("\n"), cut.rfind(". "))
    if boundary > _BRIEFING_CONTEXT_CHAR_LIMIT // 2:
        return cut[: boundary + 1].rstrip()
    return cut.rstrip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def understand_query(state: AgentState) -> dict[str, Any]:
    """Classify the user query as a specific paper or a topic (no LLM)."""
    if _halted(state):
        return {}

    query = (state.get("user_query") or "").strip()
    if not query:
        return {"error": "user_query is empty", "query_type": "topic"}

    arxiv_id = _extract_arxiv_id(query)
    if arxiv_id:
        return {"query_type": "paper", "user_query": arxiv_id, "error": None}

    return {"query_type": "topic", "user_query": query, "error": None}


def retrieve_arxiv(state: AgentState) -> dict[str, Any]:
    """Fetch paper metadata from arXiv via ArxivService."""
    if _halted(state):
        return {}

    query_type = state.get("query_type")
    query = (state.get("user_query") or "").strip()

    try:
        if query_type == "paper":
            paper = _arxiv().get_paper(query)
            return {
                "paper_metadata": paper,
                "papers": [paper],
                "selected_papers": [paper],
                "error": None,
            }

        papers = _arxiv().search_papers(query, max_results=_TOPIC_SEARCH_LIMIT)
        if not papers:
            return {
                "papers": [],
                "error": f"No arXiv papers found for topic query: {query!r}",
            }
        return {"papers": papers, "error": None}
    except ArxivServiceError as exc:
        return {"error": f"arXiv retrieval failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected arXiv retrieval error: {exc}"}


def select_papers(state: AgentState) -> dict[str, Any]:
    """Rank topic candidates with embeddings; keep the top 3 (no LLM)."""
    if _halted(state):
        return {}

    papers = list(state.get("papers") or [])
    query = (state.get("user_query") or "").strip()
    if not papers:
        return {"error": "No candidate papers available to select from"}

    try:
        embedder = _embeddings()
        query_vec = embedder.embed_query(query)
        docs = [f"{p.title}\n{p.abstract}" for p in papers]
        doc_vecs = embedder.embed_texts(docs)

        scored: list[tuple[float, PaperMetadata]] = [
            (_cosine_similarity(query_vec, vec), paper)
            for paper, vec in zip(papers, doc_vecs, strict=False)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [paper for _, paper in scored[:_TOPIC_SELECT_TOP_K]]
        return {"selected_papers": selected, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Paper selection failed: {exc}"}


def fetch_pdf(state: AgentState) -> dict[str, Any]:
    """Download/cache the focused paper PDF (first selected for topic searches)."""
    if _halted(state):
        return {}

    paper = state.get("paper_metadata")
    if paper is None:
        selected = list(state.get("selected_papers") or [])
        if not selected:
            return {"error": "No paper selected for PDF download"}
        paper = selected[0]

    try:
        pdf_path = _pdf().download_pdf(paper)
        return {
            "paper_metadata": paper,
            "pdf_path": str(pdf_path),
            "error": None,
        }
    except PDFServiceError as exc:
        return {
            "error": (
                f"Paper metadata was found, but the PDF could not be downloaded "
                f"for '{paper.arxiv_id}': {exc}"
            )
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected PDF download error: {exc}"}


def parse_pdf(state: AgentState) -> dict[str, Any]:
    """Parse the local PDF into a ParsedDocument."""
    if _halted(state):
        return {}

    pdf_path = state.get("pdf_path")
    if not pdf_path:
        return {"error": "pdf_path is missing; cannot parse PDF"}

    try:
        document = _parser().parse(Path(pdf_path))
        return {"parsed_document": document, "error": None}
    except PDFParseError as exc:
        return {
            "error": (
                f"PDF is unreadable or contains insufficient extractable text "
                f"({pdf_path}): {exc}"
            )
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected PDF parse error: {exc}"}


def chunk_and_embed(state: AgentState) -> dict[str, Any]:
    """Chunk the parsed document, embed chunks, and upsert into ChromaDB."""
    if _halted(state):
        return {}

    document = state.get("parsed_document")
    paper = state.get("paper_metadata")
    if document is None:
        return {"error": "parsed_document is missing; cannot chunk"}
    if paper is None:
        return {"error": "paper_metadata is missing; cannot chunk"}

    try:
        chunks = _chunks().chunk(document, paper_id=paper.arxiv_id)
        if not chunks:
            return {"error": "Chunking produced no usable text chunks"}

        embeddings = _embeddings().embed_texts([chunk.text for chunk in chunks])
        collection = collection_name_for(paper.arxiv_id)
        _vectors().add_chunks(collection, chunks, embeddings)

        return {
            "chunks": chunks,
            "vector_collection": collection,
            "error": None,
        }
    except VectorStoreError as exc:
        return {"error": f"Vector store failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Chunk/embed pipeline failed: {exc}"}


def generate_briefing(state: AgentState) -> dict[str, Any]:
    """Generate an executive briefing from parsed paper content via LLMService."""
    if _halted(state):
        return {}

    paper = state.get("paper_metadata")
    document = state.get("parsed_document")
    if paper is None:
        return {"error": "paper_metadata is missing; cannot generate briefing"}
    if document is None:
        return {"error": "parsed_document is missing; cannot generate briefing"}

    try:
        context = _briefing_context(document)
        briefing = _llm().generate_briefing(paper=paper, context=context)
        return {"briefing": briefing, "error": None}
    except LLMServiceError as exc:
        return {"error": f"Briefing generation failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected briefing error: {exc}"}


def retrieve_context(state: AgentState) -> dict[str, Any]:
    """Embed the question and retrieve top chunks from the paper collection."""
    if _halted(state):
        return {}

    question = (state.get("question") or "").strip()
    collection = state.get("vector_collection")
    if not question:
        return {"error": "question is empty", "retrieved_chunks": []}
    if not collection:
        return {
            "error": "vector_collection is missing; run the digest pipeline first",
            "retrieved_chunks": [],
        }

    try:
        query_embedding = _embeddings().embed_query(question)
        chunks = _vectors().search(
            collection_name=collection,
            query_embedding=query_embedding,
            top_k=_QA_TOP_K,
        )
        return {"retrieved_chunks": chunks, "error": None}
    except VectorStoreError as exc:
        return {"error": f"Context retrieval failed: {exc}", "retrieved_chunks": []}
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Unexpected context retrieval error: {exc}",
            "retrieved_chunks": [],
        }


def answer_question(state: AgentState) -> dict[str, Any]:
    """Grounded QA over retrieved chunks only."""
    if _halted(state):
        return {}

    question = (state.get("question") or "").strip()
    chunks = list(state.get("retrieved_chunks") or [])
    if not question:
        return {"error": "question is empty"}

    try:
        response = _llm().answer_question(question=question, context=chunks)
        history = list(state.get("conversation_history") or [])
        history.append(response)
        return {
            "qa_response": response,
            "conversation_history": history,
            "error": None,
        }
    except LLMServiceError as exc:
        return {"error": f"QA generation failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected QA error: {exc}"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_retrieval(
    state: AgentState,
) -> Literal["fetch_pdf", "select_papers", "__end__"]:
    """Route paper queries directly to fetch_pdf; topics go through selection."""
    if state.get("error"):
        return "__end__"
    if state.get("query_type") == "paper":
        return "fetch_pdf"
    return "select_papers"


def route_after_node(state: AgentState) -> Literal["continue", "__end__"]:
    """Stop the digest pipeline early when a node recorded an error."""
    if state.get("error"):
        return "__end__"
    return "continue"


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_graph():
    """Build the paper digest pipeline (through briefing)."""
    builder = StateGraph(AgentState)

    builder.add_node("understand_query", understand_query)
    builder.add_node("retrieve_arxiv", retrieve_arxiv)
    builder.add_node("select_papers", select_papers)
    builder.add_node("fetch_pdf", fetch_pdf)
    builder.add_node("parse_pdf", parse_pdf)
    builder.add_node("chunk_and_embed", chunk_and_embed)
    builder.add_node("generate_briefing", generate_briefing)

    builder.add_edge(START, "understand_query")
    builder.add_edge("understand_query", "retrieve_arxiv")
    builder.add_conditional_edges(
        "retrieve_arxiv",
        route_after_retrieval,
        {
            "fetch_pdf": "fetch_pdf",
            "select_papers": "select_papers",
            "__end__": END,
        },
    )
    builder.add_edge("select_papers", "fetch_pdf")

    # Linear digest path with early-exit on error between stages.
    for source, dest in (
        ("fetch_pdf", "parse_pdf"),
        ("parse_pdf", "chunk_and_embed"),
        ("chunk_and_embed", "generate_briefing"),
    ):
        builder.add_conditional_edges(
            source,
            route_after_node,
            {"continue": dest, "__end__": END},
        )

    builder.add_edge("generate_briefing", END)
    return builder.compile()


def build_qa_graph():
    """Build the optional QA subgraph (no infinite loop)."""
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("answer_question", answer_question)

    builder.add_edge(START, "retrieve_context")
    builder.add_conditional_edges(
        "retrieve_context",
        route_after_node,
        {"continue": "answer_question", "__end__": END},
    )
    builder.add_edge("answer_question", END)
    return builder.compile()


graph = build_graph()
qa_graph = build_qa_graph()
