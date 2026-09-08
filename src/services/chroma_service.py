"""
ChromaDB Service – manages the vector store collection.

Uses the ChromaDB HTTP client to communicate with the Chroma container.
All writes use ``upsert`` so that retried tasks remain idempotent: if a
chunk was already stored, it is simply overwritten with identical data
rather than causing a duplicate-key error.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class ChromaService:
    """
    Manages interactions with a ChromaDB collection for storing and
    retrieving document chunk embeddings.

    The collection is created (or retrieved if it already exists) during
    ``__init__``, making startup idempotent.

    Attributes:
        host:            ChromaDB server hostname.
        port:            ChromaDB server port.
        collection_name: Name of the ChromaDB collection to use.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "rag_document_embeddings",
    ) -> None:
        """
        Connect to ChromaDB and obtain (or create) the target collection.

        Args:
            host:            ChromaDB hostname.
            port:            ChromaDB port.
            collection_name: Name of the vector collection.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name

        logger.info(
            "Connecting to ChromaDB at %s:%d, collection=%s",
            host,
            port,
            collection_name,
        )

        self._client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # get_or_create_collection is idempotent
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
        )

        logger.info(
            "ChromaDB collection '%s' ready. Current document count: %d",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        chunk_id: str,
        embedding: List[float],
        document: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Upsert a single chunk embedding into the collection.

        Using ``upsert`` instead of ``add`` makes the operation idempotent:
        retried Celery tasks will overwrite rather than duplicate data.

        Args:
            chunk_id:  Unique, deterministic ID for this chunk
                       (e.g. ``"{document_id}_chunk_{index}"``).
            embedding: The dense vector for this chunk.
            document:  The raw text of the chunk (stored as ChromaDB document).
            metadata:  Arbitrary key-value metadata (document_id, chunk_index,
                       source_url, etc.).
        """
        # ChromaDB metadata values must be str | int | float | bool
        sanitized_metadata = _sanitize_metadata(metadata)

        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[sanitized_metadata],
        )
        logger.debug("Upserted chunk '%s' into collection '%s'.", chunk_id, self.collection_name)

    def add_embeddings_batch(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Batch-upsert multiple chunk embeddings in a single ChromaDB call.

        Significantly more efficient than calling ``add_embedding`` in a loop
        for documents with many chunks.

        Args:
            chunk_ids:  List of unique chunk identifiers.
            embeddings: Corresponding list of embedding vectors.
            documents:  Corresponding list of raw chunk texts.
            metadatas:  Corresponding list of metadata dicts.
        """
        if not chunk_ids:
            logger.warning("add_embeddings_batch called with empty lists – nothing to do.")
            return

        sanitized = [_sanitize_metadata(m) for m in metadatas]

        self._collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=sanitized,
        )
        logger.info(
            "Batch upserted %d chunk(s) into collection '%s'.",
            len(chunk_ids),
            self.collection_name,
        )

    # ------------------------------------------------------------------
    # Read / utility operations
    # ------------------------------------------------------------------

    def get_collection_count(self) -> int:
        """Return the total number of embeddings in the collection."""
        return self._collection.count()

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Semantic similarity search against the collection.

        Provided as a utility method for future retrieval endpoints or
        integration tests.

        Args:
            query_embedding: The query vector.
            n_results:       Maximum number of results to return.
            where:           Optional metadata filter dict.

        Returns:
            Raw ChromaDB query result dict.
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._collection.query(**kwargs)  # type: ignore[return-value]

    def health_check(self) -> bool:
        """
        Verify the ChromaDB connection is alive.

        Returns:
            ``True`` if the heartbeat succeeds, ``False`` otherwise.
        """
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("ChromaDB health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all metadata values are of types accepted by ChromaDB
    (str, int, float, bool).  None values are converted to empty strings.
    """
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized

# Exception handling optimizations
