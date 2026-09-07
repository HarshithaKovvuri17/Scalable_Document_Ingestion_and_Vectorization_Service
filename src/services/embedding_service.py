"""
Embedding Service – wraps Sentence Transformers for vector generation.

The model is loaded **once** at construction time (i.e., when the Celery
worker process starts) and reused for all subsequent embedding calls.
This satisfies the requirement that the model must not be reloaded per task.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages a pre-trained Sentence Transformer model for generating
    fixed-size dense vector embeddings from text.

    Attributes:
        model_name: Identifier of the Sentence Transformer model in use.
        _model:     The loaded SentenceTransformer instance (private).

    Example::

        svc = EmbeddingService("all-MiniLM-L6-v2")
        vec = svc.get_embedding("Hello, world!")
        # vec is a list[float] of dimension 384
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the Sentence Transformer model.

        Args:
            model_name: HuggingFace model identifier or a local path.
        """
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        logger.info(
            "Embedding model loaded successfully. Embedding dimension: %d",
            self.embedding_dimension,
        )

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding vector for the given text.

        Args:
            text: Input string to embed. Must be non-empty.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        logger.debug("Generating embedding for text of length %d", len(text))
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # type: ignore[union-attr]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts in a single model call.

        Batch inference is significantly faster than calling
        ``get_embedding`` in a loop for large numbers of chunks.

        Args:
            texts: A list of non-empty strings.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            ValueError: If ``texts`` is empty or any element is blank.
        """
        if not texts:
            raise ValueError("texts list must not be empty.")

        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace-only.")

        logger.info("Generating batch embeddings for %d text(s).", len(texts))
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]  # type: ignore[union-attr]

# Model initialization caching

# Model caching enhancement

# Model caching enhancement
