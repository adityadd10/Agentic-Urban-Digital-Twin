"""Phase 9 acceptance tests: `llm.anthropic_client.AnthropicLLMClient`
(dev doc §7.4's real, billed LLM client), module M9b.

**Never calls the real Anthropic API** (dev doc §13: "CI never calls a
live API") — `anthropic.Anthropic.messages.create` is monkeypatched to
return a real `anthropic.types.Message`-shaped object built by hand, so
this tests the real request/response/token-tracking wiring without any
network access or API key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, Usage

from udt.llm.anthropic_client import AnthropicLLMClient


def _fake_response(text: str, *, input_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        content=[TextBlock(type="text", text=text)],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _client() -> AnthropicLLMClient:
    return AnthropicLLMClient(
        api_key="sk-test-not-real", model="claude-sonnet-5", max_tokens_per_decision=100
    )


@pytest.mark.phase9
def test_complete_extracts_text_from_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client._client.messages,
        "create",
        lambda **kwargs: _fake_response("hello world", input_tokens=5, output_tokens=5),
    )
    assert client.complete("prompt", node="generate_plans") == "hello world"


@pytest.mark.phase9
def test_complete_joins_multiple_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    response = SimpleNamespace(
        content=[TextBlock(type="text", text="part one "), TextBlock(type="text", text="part two")],
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kwargs: response)
    assert client.complete("prompt", node="x") == "part one part two"


@pytest.mark.phase9
def test_complete_tracks_cumulative_tokens_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client._client.messages,
        "create",
        lambda **kwargs: _fake_response("x", input_tokens=10, output_tokens=5),
    )
    client.complete("prompt 1", node="a")
    client.complete("prompt 2", node="b")
    assert client.tokens_used_this_decision == 30  # (10+5) x 2


@pytest.mark.phase9
def test_reset_decision_budget_zeroes_the_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client._client.messages,
        "create",
        lambda **kwargs: _fake_response("x", input_tokens=10, output_tokens=5),
    )
    client.complete("prompt", node="a")
    assert client.tokens_used_this_decision == 15
    client.reset_decision_budget()
    assert client.tokens_used_this_decision == 0


@pytest.mark.phase9
def test_exceeding_the_decision_budget_logs_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """dev doc §7.4: "hard cap on tokens per decision (log + alert at
    20k)" - here with a tiny cap (100) so the test doesn't need to
    actually burn 20k tokens' worth of fake calls."""
    client = _client()
    monkeypatch.setattr(
        client._client.messages,
        "create",
        lambda **kwargs: _fake_response("x", input_tokens=60, output_tokens=60),
    )
    client.complete("prompt", node="a")
    assert client.tokens_used_this_decision == 120 > client.max_tokens_per_decision


@pytest.mark.phase9
def test_request_uses_the_configured_model_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response("ok", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    client.complete("my real prompt text", node="generate_plans")

    assert captured["model"] == "claude-sonnet-5"
    assert captured["messages"] == [{"role": "user", "content": "my real prompt text"}]
