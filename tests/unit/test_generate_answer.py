"""Unit tests for palav.retrieval.make_answer: guardrails + source parsing."""

from __future__ import annotations

from palav.retrieval import FALLBACK_TEXT, make_answer


def test_rejects_unrelated_question(tiny_bundle, stub_openai_factory):
    client = stub_openai_factory(chat_completion_text=FALLBACK_TEXT)
    retrieved = [(0.1, tiny_bundle.chunks[2])]
    result = make_answer(client, "gpt-4o-mini", "how do I fix my car?", retrieved)
    assert result.rejected is True
    assert result.answer == FALLBACK_TEXT
    assert result.sources == []


def test_source_urls_extracted_and_mapped_to_titles(tiny_bundle, stub_openai_factory):
    completion = (
        'A good latch means the baby covers the areola.\n'
        'USED_URLS: ["https://example.com/latch"]'
    )
    client = stub_openai_factory(chat_completion_text=completion)
    retrieved = [(0.9, tiny_bundle.chunks[0]), (0.4, tiny_bundle.chunks[1])]
    result = make_answer(client, "gpt-4o-mini", "how to latch?", retrieved)
    assert result.rejected is False
    assert result.external_knowledge is False
    assert result.sources == [
        {"url": "https://example.com/latch", "title": "Latching basics"}
    ]
    assert "USED_URLS" not in result.answer


def test_external_knowledge_flag_parsed(tiny_bundle, stub_openai_factory):
    client = stub_openai_factory(
        chat_completion_text="EXTERNAL_KNOWLEDGE: Typical feedings are 8-12 per day."
    )
    retrieved = [(0.3, tiny_bundle.chunks[0])]
    result = make_answer(client, "gpt-4o-mini", "how often to feed?", retrieved)
    assert result.external_knowledge is True
    assert not result.answer.startswith("EXTERNAL_KNOWLEDGE:")
    assert "Typical feedings" in result.answer


def test_no_sources_when_model_omits_used_urls(tiny_bundle, stub_openai_factory):
    client = stub_openai_factory(chat_completion_text="Plain answer with no tag.")
    retrieved = [(0.5, tiny_bundle.chunks[0])]
    result = make_answer(client, "gpt-4o-mini", "q?", retrieved)
    assert result.sources == []
    assert result.answer == "Plain answer with no tag."


def test_history_included_in_prompt(tiny_bundle, stub_openai_factory):
    client = stub_openai_factory(chat_completion_text="answer")
    history = [
        {"role": "user", "content": "my baby is 3 weeks old"},
        {"role": "assistant", "content": "got it"},
    ]
    make_answer(
        client,
        "gpt-4o-mini",
        "how often to feed?",
        [(0.5, tiny_bundle.chunks[0])],
        history=history,
    )
    roles = [m["role"] for m in client.last_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.last_messages[1]["content"] == "my baby is 3 weeks old"


def test_duplicate_urls_deduped(tiny_bundle, stub_openai_factory):
    completion = (
        'answer.\n'
        'USED_URLS: ["https://example.com/latch", "https://example.com/latch"]'
    )
    client = stub_openai_factory(chat_completion_text=completion)
    retrieved = [(0.9, tiny_bundle.chunks[0])]
    result = make_answer(client, "gpt-4o-mini", "q?", retrieved)
    assert len(result.sources) == 1
