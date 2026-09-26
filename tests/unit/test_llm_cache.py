"""Phase 9 acceptance tests: `llm.cache` (dev doc §7.1's disk cache,
"keyed on (prompt_version, model, state_digest) — ablation re-runs are
free"), module M9b."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from fake_llm_client import FakeLLMClient  # noqa: E402

from udt.llm.cache import (  # noqa: E402
    CachedLLMClient,
    DiskCache,
    extract_prompt_version,
    state_digest,
)

PROMPT_WITH_VERSION = "<!-- prompt_version: v1 -->\nHello, world."
PROMPT_WITHOUT_VERSION = "Hello, world."


@pytest.mark.phase9
def test_extract_prompt_version_finds_the_leading_marker() -> None:
    assert extract_prompt_version(PROMPT_WITH_VERSION) == "v1"


@pytest.mark.phase9
def test_extract_prompt_version_falls_back_to_unknown() -> None:
    assert extract_prompt_version(PROMPT_WITHOUT_VERSION) == "unknown"


@pytest.mark.phase9
def test_state_digest_is_stable_and_content_sensitive() -> None:
    assert state_digest("a") == state_digest("a")
    assert state_digest("a") != state_digest("b")


@pytest.mark.phase9
def test_disk_cache_round_trips(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get(prompt_version="v1", model="m1", prompt="hello") is None
    cache.put(prompt_version="v1", model="m1", prompt="hello", response="world")
    assert cache.get(prompt_version="v1", model="m1", prompt="hello") == "world"


@pytest.mark.phase9
def test_disk_cache_distinguishes_prompt_version_and_model(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put(prompt_version="v1", model="m1", prompt="hello", response="A")
    assert cache.get(prompt_version="v2", model="m1", prompt="hello") is None
    assert cache.get(prompt_version="v1", model="m2", prompt="hello") is None


@pytest.mark.phase9
def test_disk_cache_persists_across_instances(tmp_path: Path) -> None:
    DiskCache(tmp_path).put(prompt_version="v1", model="m1", prompt="hello", response="world")
    fresh = DiskCache(tmp_path)
    assert fresh.get(prompt_version="v1", model="m1", prompt="hello") == "world"


@pytest.mark.phase9
def test_cached_llm_client_only_calls_inner_once_for_repeated_prompts(tmp_path: Path) -> None:
    """dev doc §7.1: "ablation re-runs are free" - the whole point of
    the cache."""
    inner = FakeLLMClient({"generate_plans": ["response A"]})
    cache = DiskCache(tmp_path)
    client = CachedLLMClient(inner, cache, model="m1")

    first = client.complete(PROMPT_WITH_VERSION, node="generate_plans")
    second = client.complete(PROMPT_WITH_VERSION, node="generate_plans")

    assert first == second == "response A"
    assert len(inner.calls) == 1  # the second call was served from cache, not `inner`


@pytest.mark.phase9
def test_cached_llm_client_calls_inner_again_for_a_different_prompt(tmp_path: Path) -> None:
    inner = FakeLLMClient({"generate_plans": ["response A", "response B"]})
    cache = DiskCache(tmp_path)
    client = CachedLLMClient(inner, cache, model="m1")

    first = client.complete("<!-- prompt_version: v1 -->\nprompt one", node="generate_plans")
    second = client.complete("<!-- prompt_version: v1 -->\nprompt two", node="generate_plans")

    assert first == "response A"
    assert second == "response B"
    assert len(inner.calls) == 2
