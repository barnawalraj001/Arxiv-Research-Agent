"""arXiv API access for paper metadata retrieval."""

from __future__ import annotations

from datetime import datetime

import arxiv

from app.state import PaperMetadata


class ArxivServiceError(Exception):
    """Raised when an arXiv API request fails or a paper cannot be found."""


class ArxivService:
    """Fetch and normalize paper metadata via the official arXiv API.

    Uses the ``arxiv`` Python package (Atom API wrapper). Does not scrape
    arxiv.org HTML pages.
    """

    def __init__(self, client: arxiv.Client | None = None) -> None:
        """Initialize the service with an optional reusable API client.

        Args:
            client: Existing ``arxiv.Client`` instance. A default client is
                created when omitted.
        """
        self._client = client or arxiv.Client()

    def get_paper(self, arxiv_id: str) -> PaperMetadata:
        """Retrieve metadata for a single paper by arXiv identifier.

        Args:
            arxiv_id: Paper ID (e.g. ``2401.12345`` or ``2401.12345v3``).
                Full abs URLs are also accepted and normalized.

        Returns:
            Metadata for the requested paper.

        Raises:
            ArxivServiceError: If the paper is not found or the API request
                fails.
        """
        normalized_id = self._normalize_arxiv_id(arxiv_id)
        search = arxiv.Search(id_list=[normalized_id], max_results=1)

        try:
            results = list(self._client.results(search))
        except arxiv.ArxivError as exc:
            raise ArxivServiceError(
                f"Failed to fetch arXiv paper '{normalized_id}': {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surface unexpected network failures
            raise ArxivServiceError(
                f"Unexpected error fetching arXiv paper '{normalized_id}': {exc}"
            ) from exc

        if not results:
            raise ArxivServiceError(f"No arXiv paper found for ID '{normalized_id}'")

        return self._to_paper_metadata(results[0])

    def search_papers(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[PaperMetadata]:
        """Search arXiv and return matching paper metadata.

        Args:
            query: arXiv search query string.
            max_results: Maximum number of papers to return (default 10).

        Returns:
            A list of matching papers, or an empty list when nothing matches.
            The list length is at most ``max_results``.

        Raises:
            ArxivServiceError: If the API request fails.
        """
        if max_results < 1:
            return []

        search = arxiv.Search(query=query, max_results=max_results)

        try:
            results = list(self._client.results(search))
        except arxiv.ArxivError as exc:
            raise ArxivServiceError(
                f"Failed to search arXiv for '{query}': {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surface unexpected network failures
            raise ArxivServiceError(
                f"Unexpected error searching arXiv for '{query}': {exc}"
            ) from exc

        return [self._to_paper_metadata(result) for result in results[:max_results]]

    def _to_paper_metadata(self, result: arxiv.Result) -> PaperMetadata:
        """Convert an ``arxiv.Result`` into our ``PaperMetadata`` model."""
        arxiv_id = self._normalize_arxiv_id(result.entry_id)
        abs_url = result.entry_id
        pdf_url = result.pdf_url or abs_url.replace("/abs/", "/pdf/")

        return PaperMetadata(
            title=result.title.strip(),
            authors=[author.name for author in result.authors],
            arxiv_id=arxiv_id,
            abstract=result.summary.strip(),
            published_date=self._format_date(result.published),
            updated_date=self._format_date(result.updated),
            categories=list(result.categories),
            pdf_url=pdf_url,
            abs_url=abs_url,
        )

    @staticmethod
    def _normalize_arxiv_id(entry_id_or_id: str) -> str:
        """Extract a normalized arXiv ID from a raw ID or abs URL.

        Preserves version suffixes (e.g. ``v3``) and legacy archive prefixes
        (e.g. ``quant-ph/0201082v1``). Never returns a full URL.
        """
        value = entry_id_or_id.strip()
        marker = "arxiv.org/abs/"
        if marker in value:
            value = value.split(marker, maxsplit=1)[1]
        return value.rstrip("/")

    @staticmethod
    def _format_date(value: datetime) -> str:
        """Serialize a datetime to an ISO-8601 string."""
        return value.isoformat()
