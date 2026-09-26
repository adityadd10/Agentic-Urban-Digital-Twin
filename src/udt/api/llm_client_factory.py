"""Picks the real `LLMClient` an API server process uses (dev doc §7.4),
module M10b.

If `UDT_LLM_API_KEY` is set (`common/config.py`'s `Settings`), every
episode gets a real, billed `AnthropicLLMClient` (M9b), wrapped in
`llm/cache.py`'s disk cache exactly like `scripts/run_planner_demo.py`
already does. Without a key, `NullLLMClient` is used instead — it
always returns invalid JSON, deliberately triggering `llm/graph.py`'s
own real validation-retry-then-fallback path (dev doc §7.1) rather than
refusing to start the server at all. This means the API is usable
out of the box with zero configuration (every decision falls back to
the rule-based/safe-hold defaults, visibly flagged `llm_fallback=true`
in every logged decision) and automatically starts making real LLM
calls the moment a key is added — no server restart needed beyond the
one that reads the new `.env` value.
"""

from __future__ import annotations

from udt.common.config import Settings
from udt.llm.anthropic_client import AnthropicLLMClient
from udt.llm.cache import CachedLLMClient, DiskCache
from udt.llm.graph import LLMClient


class NullLLMClient:
    """See module docstring — always returns invalid JSON, so every
    caller falls through to `llm/graph.py`'s own disclosed fallback
    plan/action rather than this class inventing a fake "good" answer."""

    def complete(self, prompt: str, *, node: str) -> str:
        return "no LLM configured (UDT_LLM_API_KEY unset) - falling back"


def build_llm_client(settings: Settings, *, cache_dir: str) -> LLMClient:
    if not settings.llm_api_key:
        return NullLLMClient()
    anthropic_client = AnthropicLLMClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_tokens_per_decision=settings.llm_max_tokens_per_decision,
    )
    return CachedLLMClient(anthropic_client, DiskCache(cache_dir), model=settings.llm_model)
