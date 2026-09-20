"""OpenAI-backed LLM helpers for briefings and grounded QA."""

from __future__ import annotations

import os
import re

from openai import APIError, OpenAI, OpenAIError
from pydantic import BaseModel, Field

from app.prompts import (
    BRIEFING_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    build_briefing_prompt,
    build_qa_prompt,
)
from app.state import (
    Briefing,
    DocumentChunk,
    PaperMetadata,
    QAResponse,
    QASource,
)

_DEFAULT_MODEL = "gpt-5-mini"
_ENV_API_KEY = "OPENAI_API_KEY"
_ENV_MODEL = "OPENAI_MODEL"


class LLMServiceError(Exception):
    """Raised when the LLM cannot be configured or a generation call fails."""


class _QAModelOutput(BaseModel):
    """Structured model response for grounded QA (before source attachment)."""

    answer: str
    grounded: bool
    cited_chunk_ids: list[str] = Field(default_factory=list)


class LLMService:
    """Generate briefings and grounded answers with the OpenAI Responses API.

    Requires ``OPENAI_API_KEY`` in the environment. The model name may be set
    via ``OPENAI_MODEL`` (default: ``gpt-5-mini``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: Optional override for ``OPENAI_API_KEY``.
            model: Optional override for ``OPENAI_MODEL``.

        Raises:
            LLMServiceError: If no API key is available.
        """
        self._load_dotenv_if_available()

        resolved_key = api_key or os.getenv(_ENV_API_KEY)
        if not resolved_key or not resolved_key.strip():
            raise LLMServiceError(
                f"{_ENV_API_KEY} is missing. Set it in the environment "
                "or a local .env file."
            )
        if resolved_key.strip() == "your_api_key_here":
            raise LLMServiceError(
                f"{_ENV_API_KEY} is still set to the placeholder value. "
                "Replace it with a real OpenAI API key."
            )

        self._model = (model or os.getenv(_ENV_MODEL) or _DEFAULT_MODEL).strip()
        self._client = OpenAI(api_key=resolved_key.strip())

    @property
    def model(self) -> str:
        """Configured OpenAI model name."""
        return self._model

    def generate_briefing(
        self,
        paper: PaperMetadata,
        context: str,
    ) -> Briefing:
        """Generate an executive briefing grounded in supplied paper text.

        Args:
            paper: Paper metadata (title, authors, abstract, etc.).
            context: Relevant paper text used as the only content source.

        Returns:
            A structured ``Briefing``.

        Raises:
            LLMServiceError: On API or parsing failures.
        """
        prompt = build_briefing_prompt(paper, context)
        raw = self._generate_structured(
            system_prompt=BRIEFING_SYSTEM_PROMPT,
            user_prompt=prompt,
            text_format=Briefing,
        )
        try:
            if isinstance(raw, Briefing):
                briefing = raw
            else:
                briefing = Briefing.model_validate(raw)
            return self._sanitize_briefing(briefing)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(
                f"Failed to parse briefing response: {exc}"
            ) from exc

    def answer_question(
        self,
        question: str,
        context: list[DocumentChunk],
    ) -> QAResponse:
        """Answer a question using only the supplied context chunks.

        Args:
            question: User question.
            context: Retrieved chunks (retrieval is done by the caller).

        Returns:
            A ``QAResponse`` with answer, sources, and grounding flag.

        Raises:
            LLMServiceError: On API or parsing failures.
        """
        if not context:
            return QAResponse(
                question=question,
                answer=(
                    "The supplied context does not contain enough information "
                    "to answer this question."
                ),
                sources=[],
                grounded=False,
            )

        prompt = build_qa_prompt(question, context)
        raw = self._generate_structured(
            system_prompt=QA_SYSTEM_PROMPT,
            user_prompt=prompt,
            text_format=_QAModelOutput,
        )

        try:
            if isinstance(raw, _QAModelOutput):
                parsed = raw
            else:
                parsed = _QAModelOutput.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(
                f"Failed to parse QA response: {exc}"
            ) from exc

        chunk_by_id = {chunk.chunk_id: chunk for chunk in context}
        sources = self._sources_from_citations(parsed.cited_chunk_ids, chunk_by_id)

        grounded = bool(parsed.grounded)
        if not grounded:
            sources = []
        elif not sources:
            # Model claimed grounding but omitted citations; attach all supplied
            # chunks so the caller still has provenance metadata.
            sources = [
                QASource(
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
                for chunk in context
            ]

        return QAResponse(
            question=question,
            answer=parsed.answer.strip(),
            sources=sources,
            grounded=grounded,
        )

    def _generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        text_format: type[BaseModel],
    ) -> BaseModel:
        """Call OpenAI Responses structured outputs and return a Pydantic model."""
        kwargs: dict = {
            "model": self._model,
            "input": user_prompt,
            "instructions": system_prompt,
            "text_format": text_format,
        }
        if self._supports_temperature():
            kwargs["temperature"] = 0.2

        try:
            response = self._client.responses.parse(**kwargs)
        except (APIError, OpenAIError) as exc:
            # Some GPT-5 configurations reject custom temperature; retry once.
            message = str(exc)
            if "temperature" in message.lower() and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                try:
                    response = self._client.responses.parse(**kwargs)
                except (APIError, OpenAIError) as retry_exc:
                    raise LLMServiceError(
                        f"OpenAI API request failed: {retry_exc}"
                    ) from retry_exc
            else:
                raise LLMServiceError(
                    f"OpenAI API request failed: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"OpenAI API request failed: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise LLMServiceError(
                "OpenAI returned no structured output for the requested schema"
            )
        return parsed

    def _supports_temperature(self) -> bool:
        """Return whether the configured model typically accepts temperature."""
        # GPT-5 family often rejects non-default temperature values.
        return not self._model.startswith("gpt-5")

    @staticmethod
    def _sanitize_text(value: str) -> str:
        """Normalize model text for safe display (no silent truncation)."""
        cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def _sanitize_briefing(cls, briefing: Briefing) -> Briefing:
        """Clean Briefing fields and keep each follow-up as one full question."""
        follow_ups: list[str] = []
        for item in briefing.follow_up_questions:
            question = cls._sanitize_text(item)
            question = re.sub(r"^\d+[\.)]\s*", "", question)
            question = " ".join(question.split())
            if question:
                follow_ups.append(question)

        return Briefing(
            summary=cls._sanitize_text(briefing.summary),
            problem_statement=cls._sanitize_text(briefing.problem_statement),
            method=[cls._sanitize_text(item) for item in briefing.method if item.strip()],
            key_results=[
                cls._sanitize_text(item) for item in briefing.key_results if item.strip()
            ],
            limitations=[
                cls._sanitize_text(item) for item in briefing.limitations if item.strip()
            ],
            follow_up_questions=follow_ups,
        )

    @staticmethod
    def _sources_from_citations(
        cited_chunk_ids: list[str],
        chunk_by_id: dict[str, DocumentChunk],
    ) -> list[QASource]:
        """Map model-cited chunk IDs to ``QASource`` metadata from context."""
        sources: list[QASource] = []
        seen: set[str] = set()

        for chunk_id in cited_chunk_ids:
            if chunk_id in seen or chunk_id not in chunk_by_id:
                continue
            seen.add(chunk_id)
            chunk = chunk_by_id[chunk_id]
            sources.append(
                QASource(
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )
        return sources

    @staticmethod
    def _load_dotenv_if_available() -> None:
        """Load a local ``.env`` file when python-dotenv is installed."""
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv()
