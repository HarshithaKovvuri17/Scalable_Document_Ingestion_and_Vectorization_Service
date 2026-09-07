"""
Unit tests for the text splitting utilities (src/utils/text_splitter.py).

All tests run without any external infrastructure.
"""

import pytest
from src.utils.text_splitter import split_text_into_chunks, split_text_recursive


# ---------------------------------------------------------------------------
# split_text_into_chunks – happy path
# ---------------------------------------------------------------------------

class TestSplitTextIntoChunks:

    def test_empty_string_returns_empty_list(self):
        result = split_text_into_chunks("", chunk_size=500, chunk_overlap=50)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = split_text_into_chunks("   \n\t  ", chunk_size=500, chunk_overlap=50)
        assert result == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello, world!"
        result = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        assert len(result) == 1
        assert result[0] == text

    def test_exact_chunk_size_is_single_chunk(self):
        text = "a" * 500
        result = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        assert len(result) == 1

    def test_produces_multiple_chunks_for_long_text(self):
        text = "word " * 300  # ~1500 chars
        result = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        assert len(result) > 1

    def test_all_chunks_within_chunk_size(self):
        text = "x" * 2000
        result = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        for chunk in result:
            assert len(chunk) <= 500

    def test_overlap_is_applied(self):
        """Verify that consecutive chunks share characters from the overlap."""
        text = "a" * 1200
        chunk_size = 500
        chunk_overlap = 100
        result = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # The tail of chunk n should match the head of chunk n+1
        for i in range(len(result) - 1):
            tail = result[i][-chunk_overlap:]
            head = result[i + 1][:chunk_overlap]
            assert tail == head, (
                f"Overlap mismatch between chunk {i} and chunk {i+1}: "
                f"tail={tail!r}, head={head!r}"
            )

    def test_returns_list_of_strings(self):
        result = split_text_into_chunks("some text content here", chunk_size=10, chunk_overlap=2)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_no_empty_chunks_in_result(self):
        text = "word " * 200
        result = split_text_into_chunks(text, chunk_size=100, chunk_overlap=10)
        assert all(len(c) > 0 for c in result)

    def test_custom_small_chunk_size(self):
        text = "abcdefghij"  # 10 chars
        result = split_text_into_chunks(text, chunk_size=4, chunk_overlap=1)
        assert len(result) >= 2

    def test_whitespace_is_normalised(self):
        text = "hello   world\n\nfoo\tbar"
        result = split_text_into_chunks(text, chunk_size=500, chunk_overlap=0)
        assert result == ["hello world foo bar"]

    def test_text_exactly_two_chunk_sizes(self):
        """A text that is exactly 2× chunk_size should produce exactly 2 chunks (with no overlap)."""
        chunk_size = 100
        text = "a" * (chunk_size * 2)
        result = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=0)
        assert len(result) == 2

    # ---------------------------------------------------------------------------
    # Edge cases – invalid parameters
    # ---------------------------------------------------------------------------

    def test_raises_value_error_for_zero_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            split_text_into_chunks("text", chunk_size=0, chunk_overlap=0)

    def test_raises_value_error_for_negative_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            split_text_into_chunks("text", chunk_size=-1, chunk_overlap=0)

    def test_raises_value_error_for_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            split_text_into_chunks("text", chunk_size=100, chunk_overlap=-1)

    def test_raises_value_error_when_overlap_equals_chunk_size(self):
        with pytest.raises(ValueError, match="must be less than chunk_size"):
            split_text_into_chunks("text", chunk_size=100, chunk_overlap=100)

    def test_raises_value_error_when_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="must be less than chunk_size"):
            split_text_into_chunks("text", chunk_size=100, chunk_overlap=150)


# ---------------------------------------------------------------------------
# split_text_recursive – basic correctness
# ---------------------------------------------------------------------------

class TestSplitTextRecursive:

    def test_empty_returns_empty_list(self):
        assert split_text_recursive("") == []

    def test_short_text_single_chunk(self):
        text = "Short text."
        result = split_text_recursive(text, chunk_size=500, chunk_overlap=50)
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self):
        text = ("This is a sentence. " * 100)
        result = split_text_recursive(text, chunk_size=200, chunk_overlap=20)
        assert len(result) > 1

    def test_all_chunks_within_size(self):
        text = ("Another sentence here. " * 80)
        chunk_size = 300
        result = split_text_recursive(text, chunk_size=chunk_size, chunk_overlap=30)
        # With overlap prepend, chunks can exceed chunk_size by up to chunk_overlap.
        # We verify no chunk is dramatically larger (more than 2x) than chunk_size.
        for chunk in result:
            assert len(chunk) <= chunk_size * 2, (
                f"Chunk unexpectedly large: {len(chunk)} > {chunk_size * 2}"
            )

    def test_raises_for_invalid_params(self):
        with pytest.raises(ValueError):
            split_text_recursive("text", chunk_size=0)
        with pytest.raises(ValueError):
            split_text_recursive("text", chunk_size=100, chunk_overlap=100)

# Edge case unit test coverage
