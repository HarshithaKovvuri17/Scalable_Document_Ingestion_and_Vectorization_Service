"""
Text splitting utilities for document chunking.

Implements a character-level sliding window chunking strategy with
configurable chunk size and overlap. The overlap ensures contextual
continuity between successive chunks for better retrieval quality.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split a text document into overlapping chunks of approximately
    `chunk_size` characters with `chunk_overlap` characters of context
    carried over between consecutive chunks.

    Args:
        text:          The raw text to split. Whitespace is normalized.
        chunk_size:    Target size of each chunk in characters (default 500).
        chunk_overlap: Number of characters to overlap between chunks (default 50).

    Returns:
        A list of non-empty string chunks. Returns an empty list when
        ``text`` is empty or whitespace-only.

    Raises:
        ValueError: If ``chunk_size`` <= 0, ``chunk_overlap`` < 0, or
                    ``chunk_overlap`` >= ``chunk_size``.

    Example:
        >>> chunks = split_text_into_chunks("Hello world ...", chunk_size=10, chunk_overlap=2)
        >>> isinstance(chunks, list)
        True
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    # Normalize whitespace – collapse runs of spaces/newlines to a single space
    normalized = " ".join(text.split())

    if not normalized:
        logger.warning("split_text_into_chunks received empty or whitespace-only text.")
        return []

    total_len = len(normalized)

    # If the entire text fits in a single chunk, return it directly
    if total_len <= chunk_size:
        logger.debug("Text length %d fits in a single chunk.", total_len)
        return [normalized]

    chunks: List[str] = []
    stride = chunk_size - chunk_overlap  # how many characters to advance each iteration
    start = 0

    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
            logger.debug(
                "Created chunk %d: start=%d, end=%d, length=%d",
                len(chunks),
                start,
                end,
                len(chunk),
            )
        # Advance by stride; stop if we've already consumed everything
        if end == total_len:
            break
        start += stride

    logger.info(
        "split_text_into_chunks: text_length=%d → %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d)",
        total_len,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def split_text_recursive(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Enhanced recursive text splitter that tries to split on natural
    boundaries (paragraphs → sentences → words → characters) before
    falling back to hard character-level splitting.

    This is an *optional* enhancement over ``split_text_into_chunks`` and
    produces more semantically coherent chunks.

    Args:
        text:        The raw text to split.
        chunk_size:  Target maximum chunk size in characters.
        chunk_overlap: Number of characters to carry over between chunks.
        separators:  Ordered list of separator strings to try. Defaults to
                     ``['\\n\\n', '\\n', '. ', ' ', '']``.

    Returns:
        A list of non-empty string chunks.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    def _split(text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first separator that works."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        separator = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if separator:
            splits = text.split(separator)
        else:
            # Hard character split
            return split_text_into_chunks(text, chunk_size, chunk_overlap)

        result: List[str] = []
        current = ""

        for split in splits:
            candidate = current + (separator if current else "") + split
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    # Try to further split current if it's too long
                    if len(current) > chunk_size:
                        result.extend(_split(current, remaining_seps))
                    else:
                        result.append(current.strip())
                current = split

        if current.strip():
            if len(current) > chunk_size:
                result.extend(_split(current, remaining_seps))
            else:
                result.append(current.strip())

        # Apply overlap: prepend tail of previous chunk to next chunk
        if chunk_overlap > 0 and len(result) > 1:
            overlapped: List[str] = [result[0]]
            for i in range(1, len(result)):
                prev_tail = overlapped[-1][-chunk_overlap:]
                overlapped.append(prev_tail + " " + result[i])
            return overlapped

        return result

    chunks = _split(text, separators)
    logger.info(
        "split_text_recursive: %d chunk(s) produced (chunk_size=%d, chunk_overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks

# Optimized chunking logic

# Optimized chunk overlap
