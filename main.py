"""Interactive CLI for the Autonomous arXiv Paper Digest & QA Agent."""

from __future__ import annotations

import re
import sys
from typing import Sequence

from dotenv import load_dotenv

from app.graph import graph, qa_graph
from app.state import Briefing, PaperMetadata, QAResponse, QASource


def _display_text(value: str) -> str:
    """Normalize text for terminal display without truncating content."""
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _format_follow_up_question(question: str) -> str:
    """Return one complete follow-up question string (no leading numbers)."""
    text = _display_text(question)
    text = re.sub(r"^\d+[\.)]\s*", "", text)
    return " ".join(text.split())


def print_header() -> None:
    """Print the application banner."""
    print("=" * 50)
    print(" Autonomous arXiv Paper Digest & QA Agent")
    print("=" * 50)
    print()


def print_pipeline_error(message: str) -> None:
    """Display a user-facing pipeline failure."""
    print()
    print("-" * 50)
    print("Pipeline Error")
    print("-" * 50)
    print()
    print(message)
    print()


def print_digest_progress(state: dict) -> None:
    """Print checkmarks only for steps that actually completed."""
    print()
    if state.get("paper_metadata") is not None:
        print("[OK] Paper retrieved")
    if state.get("pdf_path"):
        print("[OK] PDF downloaded")
    if state.get("parsed_document") is not None:
        print("[OK] PDF parsed")
    if state.get("chunks") and state.get("vector_collection"):
        print("[OK] Document chunked and indexed")
    if state.get("briefing") is not None:
        print("[OK] Executive briefing generated")
    print()


def print_paper_metadata(paper: PaperMetadata) -> None:
    """Display metadata for the processed paper."""
    authors = ", ".join(paper.authors) if paper.authors else "Not specified"
    print("=" * 50)
    print(" PAPER")
    print("=" * 50)
    print()
    print("Title:")
    print(paper.title)
    print()
    print("Authors:")
    print(authors)
    print()
    print("arXiv ID:")
    print(paper.arxiv_id)
    print()
    print("Published:")
    print(paper.published_date)
    print()
    print("Link:")
    print(paper.abs_url)
    print()


def _print_bullet_list(items: Sequence[str]) -> None:
    if not items:
        print("• Not specified in the provided paper content.")
        return
    for item in items:
        text = _display_text(item)
        if text:
            print(f"• {text}")


def print_briefing(briefing: Briefing) -> None:
    """Display the executive briefing sections."""
    print("=" * 50)
    print(" EXECUTIVE BRIEFING")
    print("=" * 50)
    print()
    print("Why this paper matters")
    print("-" * 22)
    print(_display_text(briefing.summary))
    print()
    print("Problem")
    print("-" * 7)
    print(_display_text(briefing.problem_statement))
    print()
    print("Method / Approach")
    print("-" * 17)
    _print_bullet_list(briefing.method)
    print()
    print("Key Results / Claims")
    print("-" * 20)
    _print_bullet_list(briefing.key_results)
    print()
    print("Limitations")
    print("-" * 11)
    _print_bullet_list(briefing.limitations)
    print()
    print("Suggested Follow-up Questions")
    print("-" * 29)
    follow_ups = [
        _format_follow_up_question(question)
        for question in briefing.follow_up_questions
        if _format_follow_up_question(question)
    ]
    if not follow_ups:
        print("1. Not specified in the provided paper content.")
    else:
        for index, question in enumerate(follow_ups, start=1):
            # Print as a single logical line so numbering cannot be overwritten
            # by stray carriage returns inside the model text.
            print(f"{index}. {question}")
    print()


def format_source_line(source: QASource) -> str:
    """Format a QA source for display without relying on chunk IDs alone."""
    section = _display_text(source.section) or "Unknown section"
    if source.page_start == source.page_end:
        return f"• {section} — page {source.page_start}"
    return f"• {section} — pages {source.page_start}-{source.page_end}"


def print_qa_response(response: QAResponse) -> None:
    """Display a grounded QA answer and its sources."""
    print()
    print("Assistant:")
    print(_display_text(response.answer))
    print()
    print("Grounded:")
    print("Yes" if response.grounded else "No")
    print()
    print("Sources:")
    if not response.grounded or not response.sources:
        print("None")
    else:
        for source in response.sources:
            print(format_source_line(source))
    print()


def _is_exit_command(text: str) -> bool:
    return text.strip().lower() in {"exit", "quit", "q"}


def run_qa_loop(vector_collection: str) -> None:
    """Interactive grounded Q&A against the indexed paper collection."""
    print("=" * 50)
    print(" PAPER Q&A")
    print("=" * 50)
    print()
    print("Ask questions about this paper.")
    print("Type 'exit' or 'quit' to finish.")
    print()

    conversation_history: list[QAResponse] = []

    while True:
        try:
            question = input("You:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye!")
            return

        if _is_exit_command(question):
            print()
            print("Goodbye!")
            return

        if not question:
            print()
            print("Please enter a question.")
            print()
            continue

        try:
            qa_state = qa_graph.invoke(
                {
                    "question": question,
                    "vector_collection": vector_collection,
                    "conversation_history": conversation_history,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 — keep CLI errors concise
            print_pipeline_error(f"QA failed: {exc}")
            continue

        if qa_state.get("error"):
            print_pipeline_error(str(qa_state["error"]))
            continue

        response = qa_state.get("qa_response")
        if response is None:
            print_pipeline_error("QA completed without a response.")
            continue

        conversation_history = list(qa_state.get("conversation_history") or [])
        if response not in conversation_history:
            conversation_history.append(response)

        print_qa_response(response)


def run_cli() -> int:
    """Run the interactive digest + QA CLI. Returns a process exit code."""
    load_dotenv()
    # Avoid Windows cp1252 crashes on Unicode from model output / formatting.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    print_header()
    print("Enter a research topic or arXiv ID:")

    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        print("Goodbye!")
        return 0

    if not user_input:
        print_pipeline_error("Please enter a research topic or arXiv ID.")
        return 1

    if _is_exit_command(user_input):
        print()
        print("Goodbye!")
        return 0

    print()
    print("Processing research query...")

    try:
        state = graph.invoke(
            {
                "user_query": user_input,
                "conversation_history": [],
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        print_pipeline_error(f"Unexpected failure while running the pipeline: {exc}")
        return 1

    if state.get("error"):
        print_digest_progress(state)
        print_pipeline_error(str(state["error"]))
        return 1

    print_digest_progress(state)

    paper = state.get("paper_metadata")
    briefing = state.get("briefing")
    vector_collection = state.get("vector_collection")

    if paper is None or briefing is None or not vector_collection:
        print_pipeline_error(
            "Pipeline finished without complete paper, briefing, or vector index."
        )
        return 1

    print_paper_metadata(paper)
    print_briefing(briefing)
    run_qa_loop(vector_collection)
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
