"""Local text embedding via sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate dense embeddings locally with a shared sentence-transformers model.

    The same model is used for document chunks and user queries so vectors live
    in one comparable embedding space. No external embedding API or API key is
    required.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        """Load an embedding model from the local Hugging Face cache/disk.

        Args:
            model_name: sentence-transformers model id. Defaults to
                ``all-MiniLM-L6-v2``.
        """
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        """Name of the loaded embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimensionality of vectors produced by this model."""
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or getattr(self._model, "get_sentence_embedding_dimension")
        return int(get_dim())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one batch.

        Args:
            texts: Strings to embed (document chunks or other passages).

        Returns:
            One embedding per input string as a list of floats. Returns an
            empty list when ``texts`` is empty.
        """
        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query with the same model as documents.

        Args:
            text: Query string.

        Returns:
            Embedding as a list of floats. An empty/whitespace-only string
            yields a zero vector of the model dimension.
        """
        if not text.strip():
            return [0.0] * self.dimension

        return self.embed_texts([text])[0]
