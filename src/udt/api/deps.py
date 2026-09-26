"""Typed accessors for `Request.app.state` (module M10b).

FastAPI/Starlette's `app.state` is a plain dynamic bag (any attribute
access returns `Any`) — these thin, explicitly-typed wrappers are what
every route actually calls, so mypy strict can verify the rest of a
route's own logic instead of `Any` silently leaking through every
`req.app.state.whatever` access.
"""

from __future__ import annotations

from fastapi import Request

from udt.api.context import AppContext
from udt.api.episode_manager import EpisodeManager
from udt.api.ws import ConnectionManager
from udt.llm.graph import LLMClient


def get_context(req: Request) -> AppContext:
    context: AppContext = req.app.state.context
    return context


def get_episode_manager(req: Request) -> EpisodeManager:
    manager: EpisodeManager = req.app.state.episodes
    return manager


def get_ws_manager(req: Request) -> ConnectionManager:
    manager: ConnectionManager = req.app.state.ws_manager
    return manager


def get_llm_client(req: Request) -> LLMClient:
    client: LLMClient = req.app.state.llm_client
    return client
