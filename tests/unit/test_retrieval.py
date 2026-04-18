"""Unit tests for palav.retrieval: chunking, hashing, retrieve()."""

from __future__ import annotations

import os

from palav.retrieval import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    chunk_text,
    index_key,
    load_allowed_urls,
    retrieve,
)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_respects_overlap():
    text = "a" * (CHUNK_CHARS * 2 + 100)
    chunks = chunk_text(text)
    assert len(chunks) >= 3
    # Adjacent chunks share `overlap` chars at the boundary.
    for prev, curr in zip(chunks, chunks[1:]):
        assert prev[-CHUNK_OVERLAP:] == curr[:CHUNK_OVERLAP]


def test_chunk_text_short_returns_single():
    assert chunk_text("short") == ["short"]


def test_load_allowed_urls_filters_comments_and_blanks(tmp_path):
    p = tmp_path / "links.txt"
    p.write_text(
        "# a comment\n"
        "\n"
        "https://example.com/a\n"
        "   https://example.com/b  \n"
        "# https://example.com/skipped\n"
        "https://example.com/a\n",  # duplicate, should dedupe
        encoding="utf-8",
    )
    urls = load_allowed_urls(str(p))
    assert urls == ["https://example.com/a", "https://example.com/b"]


def test_load_allowed_urls_drops_non_url_lines_and_extracts_multiple(tmp_path):
    p = tmp_path / "links.txt"
    p.write_text(
        "SECTION HEADER:\n"
        "Title text with no URL\n"
        "Description: https://example.com/one\n"
        "Two links on one line: https://example.com/two https://example.com/three\n"
        "FAQ's\n"
        "POSTPARTUM SUPPORT workbook\n",
        encoding="utf-8",
    )
    urls = load_allowed_urls(str(p))
    assert urls == [
        "https://example.com/one",
        "https://example.com/two",
        "https://example.com/three",
    ]


def test_index_key_changes_with_file_contents(tmp_path):
    p = tmp_path / "links.txt"
    p.write_text("https://a", encoding="utf-8")
    k1 = index_key(str(p))
    p.write_text("https://b", encoding="utf-8")
    k2 = index_key(str(p))
    assert k1 != k2


def test_index_key_missing_file():
    assert index_key("/nonexistent/path/nope.txt") == "missing_links_file"


def test_retrieve_orders_by_similarity(tiny_bundle, stub_openai_factory):
    client = stub_openai_factory()
    # Query close to the "latch" chunk
    results = retrieve(client, tiny_bundle.index, tiny_bundle.chunks, "good latch areola", top_k=3)
    assert len(results) == 3
    # Highest similarity first; top hit should be the latch chunk.
    top_score, top_chunk = results[0]
    assert top_chunk.id == "a"
    # Scores are sorted descending.
    scores = [s for s, _ in results]
    assert scores == sorted(scores, reverse=True)
