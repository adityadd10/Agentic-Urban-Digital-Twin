"""Recorded/mocked `LLMClient` (dev doc §13's "LLM (recorded)" testing
row: "LangGraph flow against recorded/mocked LLM responses — CI never
calls a live API"), module M9a.

Not a real client — `llm/graph.py`'s `LLMClient` Protocol only requires
`complete(prompt, *, node) -> str`; this fixture returns pre-programmed
raw strings per node call, in order, so tests can script exactly what
"the LLM" says at each step (including deliberately-broken JSON, to
exercise the real validation-retry-then-fallback path dev doc §7.1
specifies).
"""

from __future__ import annotations

from collections import defaultdict


class FakeLLMClient:
    """`responses[node]` is a list of raw strings, consumed in order —
    one list entry per `complete()` call for that node. Calling past the
    end of a node's list raises `AssertionError` (a test bug, not a
    silent empty response) rather than looping/repeating, so a test's
    expected call count is verified implicitly."""

    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = {node: list(raw_list) for node, raw_list in responses.items()}
        self._call_counts: dict[str, int] = defaultdict(int)
        self.calls: list[tuple[str, str]] = []  # (node, prompt) - for assertions on what was asked

    def complete(self, prompt: str, *, node: str) -> str:
        self.calls.append((node, prompt))
        queue = self._responses.get(node, [])
        index = self._call_counts[node]
        assert index < len(queue), (
            f"FakeLLMClient exhausted scripted responses for node {node!r} "
            f"(called {index + 1} times, only {len(queue)} scripted)"
        )
        self._call_counts[node] += 1
        return queue[index]
