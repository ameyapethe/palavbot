"""Shared pytest fixtures for palavbot tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from palav.retrieval import DocChunk, build_faiss_index


EMBED_DIM = 8


def _stable_bucket(tok: str, dim: int) -> int:
    return int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big") % dim


@dataclass
class _StubEmbeddings:
    """Deterministic fake embeddings: hashes tokens into a fixed-dim vector.

    Replaces the OpenAI embeddings API so unit tests stay offline. Uses sha1
    (not Python's hash()) so bucket assignments survive PYTHONHASHSEED
    randomization across processes. Vectors are normalized so cosine
    similarity (IndexFlatIP) behaves sensibly.
    """

    def create(self, model: str, input: List[str]):  # noqa: A002 - match OpenAI SDK
        vecs = []
        for text in input:
            v = np.zeros(EMBED_DIM, dtype=np.float32)
            for tok in text.lower().split():
                v[_stable_bucket(tok, EMBED_DIM)] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vecs.append(SimpleNamespace(embedding=v.tolist()))
        return SimpleNamespace(data=vecs)


class StubOpenAI:
    """Minimal stand-in for openai.OpenAI used in fast tests."""

    def __init__(self, chat_completion_text: str = "stubbed answer"):
        self.embeddings = _StubEmbeddings()
        self._chat_text = chat_completion_text
        self.last_messages: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))

    def _chat_create(self, model: str, messages: List[Dict[str, Any]], temperature: float = 0):
        self.last_messages = messages
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._chat_text))
            ]
        )


@pytest.fixture
def stub_openai_factory():
    """Factory so a test can configure the mocked completion text per case."""
    return StubOpenAI


@pytest.fixture
def tiny_chunks() -> List[DocChunk]:
    return [
        DocChunk(
            id="a",
            source_url="https://example.com/latch",
            title="Latching basics",
            text="A good latch means the baby's mouth covers more of the areola.",
        ),
        DocChunk(
            id="b",
            source_url="https://example.com/supply",
            title="Milk supply",
            text="Frequent nursing and pumping helps establish milk supply.",
        ),
        DocChunk(
            id="c",
            source_url="https://example.com/cars",
            title="Unrelated topic",
            text="Cars have wheels and engines; nothing about feeding.",
        ),
    ]


@pytest.fixture
def tiny_bundle(tiny_chunks):
    """In-memory index built from the stub embeddings for the tiny chunks."""
    stub = StubOpenAI()
    texts = [c.text for c in tiny_chunks]
    resp = stub.embeddings.create(model="stub", input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    index = build_faiss_index(vecs)
    return SimpleNamespace(index=index, vectors=vecs, chunks=tiny_chunks)
