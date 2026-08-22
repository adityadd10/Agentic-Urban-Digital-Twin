"""Phase 0 acceptance tests (dev doc §14): CI green on a hello-world test,
plus a real smoke test of the two pieces of plumbing M0 actually produces —
`common/config.py` and `common/seeding.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from udt.common.config import Settings, get_settings
from udt.common.seeding import make_rng, spawn_rngs


@pytest.mark.phase0
def test_hello_world() -> None:
    """CI green on a hello-world test (dev doc Phase 0 acceptance criterion)."""
    assert True


@pytest.mark.phase0
def test_settings_load_with_defaults() -> None:
    # Bypass .env on purpose: this checks the field *defaults* declared in
    # Settings, not whatever a developer's local .env happens to say (e.g.
    # UDT_POSTGRES_PORT may be remapped locally to dodge a port collision).
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(settings, Settings)
    assert settings.postgres_dsn == "postgresql://udt:udt@localhost:5432/udt"


@pytest.mark.phase0
def test_get_settings_returns_a_settings_instance() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)


@pytest.mark.phase0
def test_settings_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UDT_POSTGRES_DB", "udt_test")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.postgres_db == "udt_test"
    finally:
        get_settings.cache_clear()


@pytest.mark.phase0
def test_make_rng_is_reproducible() -> None:
    rng_a = make_rng(42)
    rng_b = make_rng(42)
    assert np.array_equal(rng_a.random(10), rng_b.random(10))


@pytest.mark.phase0
def test_spawn_rngs_are_independent_and_reproducible() -> None:
    a1, a2 = spawn_rngs(seed=7, n=2)
    b1, b2 = spawn_rngs(seed=7, n=2)

    # same seed -> same streams
    assert np.array_equal(a1.random(5), b1.random(5))
    assert np.array_equal(a2.random(5), b2.random(5))

    # different children -> different streams
    c1, c2 = spawn_rngs(seed=7, n=2)
    assert not np.array_equal(c1.random(5), c2.random(5))


@pytest.mark.phase0
def test_spawn_rngs_rejects_n_less_than_one() -> None:
    with pytest.raises(ValueError):
        spawn_rngs(seed=0, n=0)
