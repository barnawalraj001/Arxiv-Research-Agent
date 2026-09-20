"""Persistent local vector storage for document chunks via ChromaDB."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from app.state import DocumentChunk

_DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"


class VectorStoreError(Exception):
    """Raised when vector store operations fail or a collection is missing."""


class VectorStore:
    """Store and search ``DocumentChunk`` embeddings in a local ChromaDB.

    Uses Chroma's persistent client under ``data/chroma/``. Embeddings are
    supplied by the caller (typically ``EmbeddingService``); this class does
    not compute vectors itself.
    """

    def __init__(self, persist_directory: Path | None = None) -> None:
        """Open or create a persistent Chroma database on disk.

        Args:
            persist_directory: Directory for Chroma files. Defaults to
                ``<project_root>/data/chroma``.
        """
        self._persist_directory = Path(persist_directory or _DEFAULT_CHROMA_DIR)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_directory))

    @property
    def persist_directory(self) -> Path:
        """Filesystem path of the persistent Chroma database."""
        return self._persist_directory

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks and their embeddings into a named collection.

        Args:
            collection_name: Chroma collection to write to (created if needed).
            chunks: Document chunks to store.
            embeddings: Embedding vectors aligned 1:1 with ``chunks``.

        Raises:
            VectorStoreError: If lengths mismatch or Chroma rejects the write.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        try:
            collection = self._client.get_or_create_collection(name=collection_name)
            collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=[self._chunk_metadata(chunk) for chunk in chunks],
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface Chroma failures clearly
            raise VectorStoreError(
                f"Failed to add chunks to collection '{collection_name}': {exc}"
            ) from exc

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """Run semantic similarity search against a collection.

        Args:
            collection_name: Existing Chroma collection to query.
            query_embedding: Query vector from the same embedding model used
                when indexing chunks.
            top_k: Maximum number of results to return.

        Returns:
            Matching ``DocumentChunk`` objects in similarity order. Returns an
            empty list when ``query_embedding`` is empty or ``top_k`` < 1.

        Raises:
            VectorStoreError: If the collection does not exist or the query
                fails.
        """
        if not query_embedding or top_k < 1:
            return []

        try:
            collection = self._client.get_collection(name=collection_name)
        except NotFoundError as exc:
            raise VectorStoreError(
                f"Collection '{collection_name}' does not exist"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Failed to open collection '{collection_name}': {exc}"
            ) from exc

        try:
            # Do not request more neighbors than the collection contains.
            n_results = min(top_k, max(collection.count(), 1))
            if collection.count() == 0:
                return []

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Search failed in collection '{collection_name}': {exc}"
            ) from exc

        return self._results_to_chunks(results)

    @staticmethod
    def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        """Build Chroma-compatible primitive metadata for a chunk."""
        return {
            "section": chunk.section,
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "paper_id": chunk.paper_id,
        }

    @staticmethod
    def _results_to_chunks(results: dict) -> list[DocumentChunk]:
        """Reconstruct ``DocumentChunk`` objects from a Chroma query response."""
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        chunks: list[DocumentChunk] = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            meta = metadata or {}
            chunks.append(
                DocumentChunk(
                    chunk_id=str(chunk_id),
                    paper_id=str(meta.get("paper_id", "")),
                    text=text or "",
                    section=str(meta.get("section", "")),
                    page_start=int(meta.get("page_start", 0)),
                    page_end=int(meta.get("page_end", 0)),
                )
            )
        return chunks
