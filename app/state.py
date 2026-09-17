from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata retrieved from arXiv for a research paper."""

    title: str
    authors: list[str]
    arxiv_id: str
    abstract: str
    published_date: str
    updated_date: str | None = None
    categories: list[str]
    pdf_url: str
    abs_url: str


class DocumentSection(BaseModel):
    """A logical section extracted from a paper."""

    title: str
    text: str
    page_start: int
    page_end: int


class ParsedDocument(BaseModel):
    """Structured representation of a parsed PDF."""

    full_text: str
    sections: list[DocumentSection]
    page_count: int


class DocumentChunk(BaseModel):
    """A searchable chunk of paper content."""

    chunk_id: str
    paper_id: str
    text: str
    section: str
    page_start: int
    page_end: int


class Briefing(BaseModel):
    """Structured executive briefing generated from a paper."""

    summary: str
    problem_statement: str
    method: list[str]
    key_results: list[str]
    limitations: list[str]
    follow_up_questions: list[str]


class QASource(BaseModel):
    """Source information supporting a QA answer."""

    chunk_id: str
    section: str
    page_start: int
    page_end: int


class QAResponse(BaseModel):
    """Grounded answer to a user's question about a paper."""

    question: str
    answer: str
    sources: list[QASource]
    grounded: bool


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # User input
    user_query: str

    # Query understanding
    query_type: Literal["paper", "topic"]

    # arXiv retrieval
    papers: list[PaperMetadata]
    selected_papers: list[PaperMetadata]

    # Current paper
    paper_metadata: PaperMetadata
    pdf_path: str

    # PDF processing
    parsed_document: ParsedDocument
    chunks: list[DocumentChunk]

    # Vector database
    vector_collection: str

    # Generated briefing
    briefing: Briefing

    # QA
    question: str
    retrieved_chunks: list[DocumentChunk]
    qa_response: QAResponse
    conversation_history: list[QAResponse]

    # Error handling
    error: str | None