"""Real, billed `LLMClient` (dev doc §7.4: "Prototyping: Claude Sonnet
(API)"), module M9b.

Implements `llm/graph.py`'s `LLMClient` Protocol (`complete(prompt, *,
node) -> str`) against the real Anthropic API. **Every other test/module
in this codebase uses a scripted client instead** (`FakeLLMClient` in
`tests/fixtures/`) — dev doc §13's own testing-strategy row is explicit:
"CI never calls a live API". This class is never imported by anything
that runs automatically; a caller has to explicitly construct one with
a real API key (`common/config.py`'s `Settings.llm_api_key`, `None` by
default) to make any network call at all.

**Budget guard (dev doc §7.4):** "hard cap on tokens per decision (log +
alert at 20k). Track llm_tokens_per_decision as an experiment metric."
One "decision" can involve several `complete()` calls (retries, the OOD
path's one refinement) — the cap is tracked cumulatively across calls
via `reset_decision_budget()`/`tokens_used_this_decision`, which the
caller (whatever invokes `run_planner` once per decision) is responsible
for resetting at the start of each new decision; this class has no way
to know on its own where one decision ends and the next begins.
"""

from __future__ import annotations

import structlog
from anthropic import Anthropic
from anthropic.types import TextBlock

log = structlog.get_logger()

DEFAULT_MAX_TOKENS_PER_CALL = 4096


class AnthropicLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens_per_decision: int,
        max_tokens_per_call: int = DEFAULT_MAX_TOKENS_PER_CALL,
    ) -> None:
        self._client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens_per_decision = max_tokens_per_decision
        self.max_tokens_per_call = max_tokens_per_call
        self.tokens_used_this_decision = 0

    def reset_decision_budget(self) -> None:
        """Call once per decision, before the first `complete()` for
        that decision — see class docstring's budget-guard note."""
        self.tokens_used_this_decision = 0

    def complete(self, prompt: str, *, node: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens_per_call,
            messages=[{"role": "user", "content": prompt}],
        )
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        self.tokens_used_this_decision += tokens_used
        log.info(
            "llm_call_complete",
            node=node,
            model=self.model,
            tokens_used=tokens_used,
            tokens_used_this_decision=self.tokens_used_this_decision,
        )
        if self.tokens_used_this_decision > self.max_tokens_per_decision:
            log.warning(
                "llm_tokens_per_decision_exceeded",
                tokens_used_this_decision=self.tokens_used_this_decision,
                cap=self.max_tokens_per_decision,
            )
        return "".join(block.text for block in response.content if isinstance(block, TextBlock))
