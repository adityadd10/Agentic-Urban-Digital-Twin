"""In-memory episode registry (dev doc §11.2), module M10b.

**Disclosed scope reduction:** dev doc §11.1's Postgres/PostGIS tables
(`episodes`, `decisions`, ...) aren't wired up — this manager keeps
every running `EpisodeSession` in a plain process-local dict. Fine for
a single-process demo server (dev doc §11.3's own framing: "this is a
demo surface, not a public product"); a restart loses all episode
state, and nothing here is shared across multiple server processes.
The decision log itself is still real and on disk regardless
(`logging/decision_log.py`'s JSONL file per episode, unaffected by this
in-memory layer disappearing).
"""

from __future__ import annotations

from udt.api.episode_session import EpisodeSession


class EpisodeManager:
    def __init__(self) -> None:
        self._sessions: dict[str, EpisodeSession] = {}

    def add(self, session: EpisodeSession) -> None:
        self._sessions[session.episode_id] = session

    def get(self, episode_id: str) -> EpisodeSession | None:
        return self._sessions.get(episode_id)

    def all_episode_ids(self) -> list[str]:
        return list(self._sessions)

    def all_sessions(self) -> list[EpisodeSession]:
        return list(self._sessions.values())
