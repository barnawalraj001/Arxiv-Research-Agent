"""PDF download and local caching for arXiv papers."""

from __future__ import annotations

from pathlib import Path

import requests

from app.state import PaperMetadata

# Project root: app/services/pdf.py -> parents[2]
_DEFAULT_PAPERS_DIR = Path(__file__).resolve().parents[2] / "data" / "papers"
_DEFAULT_TIMEOUT_SECONDS = 60.0


class PDFServiceError(Exception):
    """Raised when a PDF cannot be downloaded, written, or validated."""


class PDFService:
    """Download arXiv PDFs and cache them under ``data/papers/``.

    Caching is keyed by the paper's normalized ``arxiv_id``. Existing non-empty
    local files are reused so repeated runs avoid redundant downloads.
    """

    def __init__(
        self,
        papers_dir: Path | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the PDF service.

        Args:
            papers_dir: Directory for cached PDFs. Defaults to
                ``<project_root>/data/papers``.
            timeout: HTTP request timeout in seconds.
        """
        self._papers_dir = papers_dir or _DEFAULT_PAPERS_DIR
        self._timeout = timeout

    def pdf_path(self, paper: PaperMetadata) -> Path:
        """Return the expected local path for a paper's PDF (may not exist)."""
        return self._papers_dir / self._filename_for(paper.arxiv_id)

    def has_cached_pdf(self, paper: PaperMetadata) -> bool:
        """Return True if a non-empty cached PDF already exists locally."""
        path = self.pdf_path(paper)
        return path.is_file() and path.stat().st_size > 0

    def download_pdf(self, paper: PaperMetadata) -> Path:
        """Download a paper's PDF, or reuse a valid local cache.

        Args:
            paper: Paper metadata whose ``pdf_url`` and ``arxiv_id`` are used.

        Returns:
            Path to the local PDF file under ``data/papers/``.

        Raises:
            PDFServiceError: On HTTP/network failures, empty downloads, or
                filesystem errors.
        """
        destination = self.pdf_path(paper)
        if self.has_cached_pdf(paper):
            return destination

        self._papers_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(
                paper.pdf_url,
                timeout=self._timeout,
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise PDFServiceError(
                f"Failed to download PDF for '{paper.arxiv_id}' "
                f"from {paper.pdf_url}: {exc}"
            ) from exc

        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)

            if partial.stat().st_size == 0:
                raise PDFServiceError(
                    f"Downloaded PDF for '{paper.arxiv_id}' is empty"
                )

            partial.replace(destination)
        except PDFServiceError:
            partial.unlink(missing_ok=True)
            raise
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise PDFServiceError(
                f"Failed to save PDF for '{paper.arxiv_id}' to {destination}: {exc}"
            ) from exc

        return destination

    @staticmethod
    def _filename_for(arxiv_id: str) -> str:
        """Build a safe PDF filename from a normalized arXiv ID.

        Legacy IDs may contain ``/`` (e.g. ``quant-ph/0201082v1``); those are
        flattened so all files live directly under ``data/papers/``.
        """
        safe_id = arxiv_id.strip().replace("/", "_")
        return f"{safe_id}.pdf"
