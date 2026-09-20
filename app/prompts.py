"""Prompt templates for OpenAI briefing and grounded QA."""

from __future__ import annotations

from app.state import DocumentChunk, PaperMetadata

BRIEFING_SYSTEM_PROMPT = """\
You are an academic research assistant that writes concise executive briefings.

Hard rules:
- Use ONLY the supplied paper metadata and paper content.
- Do NOT invent facts, results, methods, or limitations.
- Do NOT use outside knowledge or prior training about the paper.
- If something is not supported by the supplied content, write that it is \
not specified or not supported by the provided paper content.
- Keep claims tightly grounded in the supplied text.
- follow_up_questions must be complete, standalone questions derived only \
from the supplied paper content (gaps, unclear claims, or next analyses \
suggested by the paper itself).
- Do NOT invent generic follow-up questions from outside knowledge.
- Do NOT include carriage returns or numbering inside each follow-up string; \
each list item is one full question.
"""

QA_SYSTEM_PROMPT = """\
You are a grounded question-answering assistant for scientific papers.

Hard rules:
- Use only the supplied paper excerpts.
- If the excerpts do not contain enough information to answer the question, \
say so. Do not use outside knowledge.
- Answer ONLY from the supplied context chunks.
- Do NOT infer unsupported facts.
- If the context does not contain enough information, say so clearly and set \
grounded to false.
- When the answer is supported, set grounded to true and cite the chunk_id \
values you relied on.
- Never cite a chunk_id that was not provided in the context.
- Do not invent citations.
"""


def build_briefing_prompt(paper: PaperMetadata, context: str) -> str:
    """Build the user prompt for executive briefing generation."""
    authors = ", ".join(paper.authors) if paper.authors else "Not specified"
    categories = ", ".join(paper.categories) if paper.categories else "Not specified"

    return f"""\
Create an executive briefing for the following paper using ONLY the content below.

Paper metadata:
- Title: {paper.title}
- Authors: {authors}
- arXiv ID: {paper.arxiv_id}
- Published: {paper.published_date}
- Updated: {paper.updated_date or "Not specified"}
- Categories: {categories}
- Abstract: {paper.abstract}

Paper content:
{context}

Populate these fields:
- summary: short overview
- problem_statement: problem the paper addresses
- method: list of method / approach points
- key_results: list of key results
- limitations: list of limitations
- follow_up_questions: list of complete, paper-grounded follow-up questions

Each follow_up_questions item must be one full question sentence (no leading \
numbers, no truncated fragments).

If a field is not supported by the supplied content, say that explicitly \
instead of inventing details.
"""


def build_qa_prompt(question: str, context: list[DocumentChunk]) -> str:
    """Build the user prompt for grounded QA over retrieved chunks."""
    if not context:
        context_block = "(No context chunks were supplied.)"
    else:
        parts: list[str] = []
        for chunk in context:
            parts.append(
                f"[chunk_id={chunk.chunk_id} | section={chunk.section} | "
                f"pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}"
            )
        context_block = "\n\n---\n\n".join(parts)

    return f"""\
Use only the supplied paper excerpts. If the excerpts do not contain enough \
information to answer the question, say so. Do not use outside knowledge.

Question:
{question}

Context chunks:
{context_block}

Return:
- answer: the grounded answer text
- grounded: true if the context is sufficient; false otherwise
- cited_chunk_ids: list of chunk_id values used for the answer \
(empty when grounded is false)

Use grounded=false and cited_chunk_ids=[] when the context is insufficient.
Do not invent citations.
"""
