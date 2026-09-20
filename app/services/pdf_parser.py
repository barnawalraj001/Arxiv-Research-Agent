"""PDF text extraction and simple academic section detection."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from app.state import DocumentSection, ParsedDocument

# Longer titles first so "Experimental Results" wins over "Results", etc.
_SECTION_TITLES: tuple[str, ...] = (
    "Experimental Results",
    "Related Work",
    "Methodology",
    "Limitations",
    "Conclusions",
    "Conclusion",
    "Discussion",
    "Experiments",
    "Introduction",
    "Background",
    "References",
    "Abstract",
    "Approach",
    "Methods",
    "Method",
    "Results",
)

_SECTION_ALTERNATION = "|".join(re.escape(title) for title in _SECTION_TITLES)

# Optional numbering: "1.", "1.2", "I.", "IV", etc.
_NUMBERING = r"(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?)\s+"

# Full-line heading, e.g. "I. INTRODUCTION" or "3 Methods"
_HEADING_LINE = re.compile(
    rf"^\s*(?:{_NUMBERING})?({_SECTION_ALTERNATION})\s*$",
    re.IGNORECASE,
)

# Same-line heading body uses an em/en dash only (IEEE "Abstract—...").
# Colon/period are intentionally excluded to avoid mid-sentence false positives
# such as "methods. Since..." or "experiments: 1)...".
_INLINE_HEADING = re.compile(
    rf"^\s*(?:{_NUMBERING})?({_SECTION_ALTERNATION})\s*[—–]\s*(.*)$",
    re.IGNORECASE,
)

# Minimum non-whitespace characters required to treat extraction as successful.
_MIN_MEANINGFUL_CHARS = 40


class PDFParseError(Exception):
    """Raised when a PDF cannot be opened, read, or yields no usable text."""


class PDFParser:
    """Extract text and coarse section structure from a local PDF.

    Uses PyMuPDF for page-ordered text extraction and simple regex-based
    detection of common academic headings. Does not perform OCR.
    """

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse a local PDF into a ``ParsedDocument``.

        Args:
            pdf_path: Path to a PDF file on disk.

        Returns:
            Structured document with full text, sections, and page count.

        Raises:
            PDFParseError: If the file is missing, unreadable, empty, or
                contains no extractable text (e.g. a scanned/image PDF).
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise PDFParseError(f"PDF not found: {path}")

        try:
            document = pymupdf.open(path)
        except Exception as exc:  # noqa: BLE001 — PyMuPDF raises varied errors
            raise PDFParseError(f"Failed to open PDF '{path}': {exc}") from exc

        try:
            page_count = document.page_count
            if page_count <= 0:
                raise PDFParseError(f"PDF has no pages: {path}")

            page_texts: list[str] = []
            numbered_lines: list[tuple[int, str]] = []

            for page_index in range(page_count):
                page_number = page_index + 1
                try:
                    page_text = document.load_page(page_index).get_text("text")
                except Exception as exc:  # noqa: BLE001
                    raise PDFParseError(
                        f"Failed to extract text from page {page_number} "
                        f"of '{path}': {exc}"
                    ) from exc

                # Section detection uses raw lines; prose normalization is applied
                # after sections are assembled so headings stay line-accurate.
                page_texts.append(page_text)
                for line in page_text.splitlines():
                    numbered_lines.append((page_number, line))

            sections = self._detect_sections(numbered_lines, page_count)
            sections = [
                section.model_copy(
                    update={"text": self._normalize_extracted_prose(section.text)}
                )
                for section in sections
            ]
            full_text = "\n\n".join(
                self._normalize_extracted_prose(page_text) for page_text in page_texts
            ).strip()
            meaningful = re.sub(r"\s+", "", full_text)
            if len(meaningful) < _MIN_MEANINGFUL_CHARS:
                raise PDFParseError(
                    f"No meaningful text extracted from '{path}'. "
                    "The PDF may be scanned/image-based, corrupted, "
                    "or otherwise unreadable without OCR."
                )

            return ParsedDocument(
                full_text=full_text,
                sections=sections,
                page_count=page_count,
            )
        finally:
            document.close()

    def _detect_sections(
        self,
        numbered_lines: list[tuple[int, str]],
        page_count: int,
    ) -> list[DocumentSection]:
        """Split numbered lines into sections using known academic headings."""
        hits: list[tuple[int, str, str]] = []
        # (line_index, canonical_title, optional same-line body)

        for index, (_page, line) in enumerate(numbered_lines):
            stripped = line.strip()
            if not stripped:
                continue

            match = _HEADING_LINE.match(stripped)
            if match:
                hits.append((index, self._canonical_title(match.group(1)), ""))
                continue

            inline = _INLINE_HEADING.match(stripped)
            if inline:
                hits.append(
                    (
                        index,
                        self._canonical_title(inline.group(1)),
                        inline.group(2).strip(),
                    )
                )

        if not hits:
            all_text = "\n".join(line for _, line in numbered_lines).strip()
            return [
                DocumentSection(
                    title="Full Document",
                    text=all_text,
                    page_start=1,
                    page_end=page_count,
                )
            ]

        sections: list[DocumentSection] = []

        # Preamble before the first heading (title block, authors, etc.).
        first_hit_index = hits[0][0]
        if first_hit_index > 0:
            preamble_lines = numbered_lines[:first_hit_index]
            preamble_text = "\n".join(line for _, line in preamble_lines).strip()
            if preamble_text:
                pages = [page for page, _ in preamble_lines]
                sections.append(
                    DocumentSection(
                        title="Preamble",
                        text=preamble_text,
                        page_start=min(pages),
                        page_end=max(pages),
                    )
                )

        for hit_pos, (start_index, title, inline_body) in enumerate(hits):
            end_index = (
                hits[hit_pos + 1][0]
                if hit_pos + 1 < len(hits)
                else len(numbered_lines)
            )
            body_lines = numbered_lines[start_index + 1 : end_index]
            parts: list[str] = []
            if inline_body:
                parts.append(inline_body)
            parts.extend(line for _, line in body_lines)
            text = "\n".join(parts).strip()

            # Page span includes the heading line and any body lines.
            span_lines = numbered_lines[start_index:end_index]
            pages = [page for page, _ in span_lines] or [
                numbered_lines[start_index][0]
            ]

            sections.append(
                DocumentSection(
                    title=title,
                    text=text,
                    page_start=min(pages),
                    page_end=max(pages),
                )
            )

        return sections

    @staticmethod
    def _canonical_title(matched: str) -> str:
        """Map a matched heading to its canonical casing from the known list."""
        lowered = matched.strip().lower()
        for title in _SECTION_TITLES:
            if title.lower() == lowered:
                return title
        return matched.strip().title()

    @staticmethod
    def _normalize_extracted_prose(text: str) -> str:
        """Apply minimal PDF prose cleanup without rewriting equations.

        - Rejoins hyphenated line breaks (``distri-\\nbution`` → ``distribution``)
        - Inserts a space when a lowercase prose line continues on the next line
          (avoids accidental ``channelmatrix``-style gluing when consumers join
          lines later)

        Mathematical notation and remaining layout quirks may still appear;
        aggressive NLP cleanup is intentionally avoided.
        """
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
        cleaned = re.sub(r"(?<=[a-z0-9,;:.)\"'\]])\n(?=[a-z])", " ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
