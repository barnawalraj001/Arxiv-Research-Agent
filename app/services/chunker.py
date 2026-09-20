"""Section-aware text chunking for parsed arXiv papers."""

from __future__ import annotations

from app.state import DocumentChunk, DocumentSection, ParsedDocument

# ~800 tokens at ~4 characters/token.
_DEFAULT_CHUNK_SIZE = 3200
_DEFAULT_OVERLAP = 400


class Chunker:
    """Split a ``ParsedDocument`` into overlapping ``DocumentChunk`` units.

    Chunking is strictly section-local: text from different sections is never
    combined. Size is character-based (no tokenizer dependency), targeting
    roughly 800 tokens via a 4-characters-per-token approximation.
    """

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> None:
        """Initialize chunking parameters.

        Args:
            chunk_size: Target characters per chunk (default 3200 ≈ 800 tokens).
            overlap: Character overlap between consecutive chunks in the same
                section (default 400 ≈ 100 tokens).

        Raises:
            ValueError: If sizes are invalid.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._step = chunk_size - overlap

    def chunk(self, document: ParsedDocument, paper_id: str) -> list[DocumentChunk]:
        """Chunk a parsed document section by section.

        Args:
            document: Parsed PDF content with sections.
            paper_id: Stable paper identifier stored on every chunk and used
                in deterministic chunk IDs.

        Returns:
            Ordered list of non-empty ``DocumentChunk`` objects.
        """
        chunks: list[DocumentChunk] = []
        chunk_index = 1

        for section in document.sections:
            section_chunks = self._chunk_section(section, paper_id, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _chunk_section(
        self,
        section: DocumentSection,
        paper_id: str,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Create one or more chunks for a single section."""
        text = section.text.strip()
        if not text:
            return []

        # Short sections become a single chunk.
        if len(text) <= self._chunk_size:
            return [
                DocumentChunk(
                    chunk_id=self._make_chunk_id(paper_id, start_index),
                    paper_id=paper_id,
                    text=text,
                    section=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            ]

        chunks: list[DocumentChunk] = []
        start = 0
        local_index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self._chunk_size, text_length)
            piece = text[start:end].strip()

            if piece:
                page_start, page_end = self._infer_page_range(
                    section=section,
                    char_start=start,
                    char_end=end,
                    text_length=text_length,
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._make_chunk_id(
                            paper_id, start_index + local_index
                        ),
                        paper_id=paper_id,
                        text=piece,
                        section=section.title,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
                local_index += 1

            if end >= text_length:
                break
            start += self._step

        return chunks

    @staticmethod
    def _make_chunk_id(paper_id: str, index: int) -> str:
        """Build a deterministic chunk ID such as ``2401.12345v3_chunk_0001``."""
        return f"{paper_id}_chunk_{index:04d}"

    @staticmethod
    def _infer_page_range(
        section: DocumentSection,
        char_start: int,
        char_end: int,
        text_length: int,
    ) -> tuple[int, int]:
        """Approximate chunk page bounds from the section's page span.

        Character offsets within a section are mapped proportionally onto
        ``[page_start, page_end]``. This is a coarse heuristic because the
        parser does not expose per-character page boundaries.
        """
        if (
            text_length <= 0
            or section.page_start == section.page_end
            or char_end <= char_start
        ):
            return section.page_start, section.page_end

        page_span = section.page_end - section.page_start
        # Use the midpoint of the covered character range for stability.
        rel_start = char_start / text_length
        rel_end = max(char_start, char_end - 1) / text_length

        page_start = section.page_start + int(rel_start * page_span)
        page_end = section.page_start + int(rel_end * page_span)

        page_start = min(max(page_start, section.page_start), section.page_end)
        page_end = min(max(page_end, page_start), section.page_end)
        return page_start, page_end
