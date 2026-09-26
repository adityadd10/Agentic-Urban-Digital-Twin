"""Disk cache for LLM calls (dev doc §7.1), module M9b.

"All calls cached to disk keyed on (prompt_version, model,
state_digest) — ablation re-runs are free."

`CachedLLMClient` wraps any `LLMClient` (a real `AnthropicLLMClient`, or
even a scripted one, for testing the cache in isolation without needing
the real API at all) with this cache — a cache hit means the underlying
client's `complete()` is never called, so a repeated ablation run over
the same prompts/model/state never re-bills the API.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import structlog

from udt.llm.graph import LLMClient

log = structlog.get_logger()

_PROMPT_VERSION_RE = re.compile(r"<!--\s*prompt_version:\s*(\S+?)\s*-->")


def extract_prompt_version(prompt: str) -> str:
    """`llm/prompts/*.md`'s own convention (M9a): a leading `<!--
    prompt_version: vN -->` marker. Falls back to `"unknown"` rather
    than raising — a prompt built without that marker (e.g. a test
    fixture) still caches correctly, just under one shared "unknown"
    bucket instead of a version-specific one."""
    match = _PROMPT_VERSION_RE.search(prompt)
    return match.group(1) if match else "unknown"


def state_digest(prompt: str) -> str:
    """dev doc §7.1's "state_digest" — a stable hash of whatever
    actually determines the LLM's answer. Hashing the fully-rendered
    prompt itself (rather than a separately hand-picked subset of state
    fields) is the simplest reading that's guaranteed correct: if the
    rendered prompt text is identical, the model's input is identical,
    full stop — no risk of missing a field that secretly varies the
    prompt but wasn't included in a hand-picked digest."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per `(prompt_version, model)` pair holds
    `{state_digest: raw_response}` — read-modify-write, same
    simple-and-safe convention `agents/marl/registry.py`'s
    `append_entry` already uses for a similarly small, infrequently-
    written file (this cache is written once per unique prompt, not
    once per tick)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, prompt_version: str, model: str) -> Path:
        safe_model = model.replace("/", "_")
        return self._dir / f"{prompt_version}__{safe_model}.json"

    def get(self, *, prompt_version: str, model: str, prompt: str) -> str | None:
        cache_file = self._cache_file(prompt_version, model)
        if not cache_file.exists():
            return None
        data: dict[str, str] = json.loads(cache_file.read_text())
        return data.get(state_digest(prompt))

    def put(self, *, prompt_version: str, model: str, prompt: str, response: str) -> None:
        cache_file = self._cache_file(prompt_version, model)
        data: dict[str, str] = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        data[state_digest(prompt)] = response
        cache_file.write_text(json.dumps(data, indent=2))


class CachedLLMClient:
    """Wraps `inner` with `cache` — `complete()` checks the cache first,
    only calling `inner.complete(...)` (the expensive/billed path) on a
    miss, then writes the result back."""

    def __init__(self, inner: LLMClient, cache: DiskCache, *, model: str) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model

    def complete(self, prompt: str, *, node: str) -> str:
        prompt_version = extract_prompt_version(prompt)
        cached = self._cache.get(prompt_version=prompt_version, model=self._model, prompt=prompt)
        if cached is not None:
            log.info("llm_cache_hit", node=node, prompt_version=prompt_version, model=self._model)
            return cached

        log.info("llm_cache_miss", node=node, prompt_version=prompt_version, model=self._model)
        response = self._inner.complete(prompt, node=node)
        self._cache.put(
            prompt_version=prompt_version, model=self._model, prompt=prompt, response=response
        )
        return response
