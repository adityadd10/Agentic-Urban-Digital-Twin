"""WebSocket live-update channel (dev doc §11.2's `WS /ws/live`),
module M10b.

`ConnectionManager` is a plain in-memory pub/sub keyed by `episode_id`
— `episodes.py`'s `POST /episodes/{id}/step`/`POST /incidents` routes
call `broadcast(episode_id, message)` after they actually change that
episode's state, and every WebSocket currently subscribed to that
episode receives the message. **Disclosed:** single-process only (same
status as `api/episode_manager.py`'s in-memory episode store) — nothing
here works across multiple server workers/replicas.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, episode_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(episode_id, []).append(websocket)

    def disconnect(self, episode_id: str, websocket: WebSocket) -> None:
        connections = self._connections.get(episode_id, [])
        if websocket in connections:
            connections.remove(websocket)

    async def broadcast(self, episode_id: str, message: dict[str, Any]) -> None:
        for websocket in list(self._connections.get(episode_id, [])):
            try:
                await websocket.send_json(message)
            except Exception:  # noqa: BLE001 - a dead socket shouldn't break the broadcast loop
                self.disconnect(episode_id, websocket)


@router.websocket("/ws/live")
async def live_updates(websocket: WebSocket, episode_id: str) -> None:
    manager: ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(episode_id, websocket)
    try:
        while True:
            # This channel is server -> client push only (dev doc:
            # "tick-by-tick state + decision events") - still needs to
            # await *something* to detect a client disconnect, so it
            # just discards whatever the client sends.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(episode_id, websocket)
