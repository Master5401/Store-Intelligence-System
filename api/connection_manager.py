"""
src/api/connection_manager.py
───────────────────────────────
Manages all live WebSocket connections for the security dashboard.

Design decisions:
  • Each client gets its own asyncio.Queue (max 120 items).
  • If the queue is full (slow browser) old frames are dropped silently —
    server stability is prioritised over individual client fidelity.
  • Broadcast is done concurrently via asyncio.gather.
  • Disconnected sockets are purged atomically in _cleanup().
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Maximum pending messages per client before backpressure kicks in
_QUEUE_MAX = 120


@dataclass
class WSClient:
    websocket: WebSocket
    client_id: str
    store_id:  str
    queue:     asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=_QUEUE_MAX))
    connected_at: datetime = field(default_factory=datetime.utcnow)
    messages_sent: int = 0
    messages_dropped: int = 0


class ConnectionManager:
    """
    Thread-safe WebSocket connection manager.

    Usage
    -----
    manager = ConnectionManager()

    # In the WS endpoint:
    await manager.connect(websocket, client_id, store_id)
    try:
        async for frame_data in event_source:
            await manager.broadcast_to_store(store_id, frame_data)
    finally:
        await manager.disconnect(client_id)
    """

    def __init__(self) -> None:
        self._clients: dict[str, WSClient] = {}
        self._lock = asyncio.Lock()

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        store_id:  str,
    ) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients[client_id] = WSClient(
                websocket=websocket,
                client_id=client_id,
                store_id=store_id,
            )
        logger.info("WS connected: client=%s store=%s total=%d",
                    client_id, store_id, len(self._clients))

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            client = self._clients.pop(client_id, None)
        if client:
            logger.info(
                "WS disconnected: client=%s sent=%d dropped=%d",
                client_id, client.messages_sent, client.messages_dropped,
            )

    # ── Sending ───────────────────────────────────────────────────────────────

    async def send_personal(self, client_id: str, data: dict) -> bool:
        """Send a message to a specific client. Returns False if not connected."""
        async with self._lock:
            client = self._clients.get(client_id)
        if not client:
            return False
        return await self._enqueue(client, data)

    async def broadcast_to_store(self, store_id: str, data: dict) -> None:
        """Broadcast a message to every client watching a given store."""
        async with self._lock:
            targets = [c for c in self._clients.values() if c.store_id == store_id]

        if not targets:
            return

        # Push to each client's queue concurrently
        await asyncio.gather(
            *(self._enqueue(c, data) for c in targets),
            return_exceptions=True,
        )

    async def broadcast_all(self, data: dict) -> None:
        """Broadcast to every connected client."""
        async with self._lock:
            targets = list(self._clients.values())
        await asyncio.gather(
            *(self._enqueue(c, data) for c in targets),
            return_exceptions=True,
        )

    # ── Per-client sender task ────────────────────────────────────────────────

    async def run_sender(self, client_id: str) -> None:
        """
        Coroutine that drains a client's queue and sends messages over the WS.
        Should be run as a background task while the WS endpoint is active.
        """
        async with self._lock:
            client = self._clients.get(client_id)
        if not client:
            return

        try:
            while True:
                data = await client.queue.get()
                if data is None:   # sentinel → stop
                    break
                try:
                    text = json.dumps(data, default=str)
                    await client.websocket.send_text(text)
                    client.messages_sent += 1
                except Exception as exc:
                    logger.debug("WS send error client=%s: %s", client_id, exc)
                    break
        finally:
            await self.disconnect(client_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._clients)

    def get_stats(self) -> dict:
        stats = {
            "total_clients": len(self._clients),
            "clients": [],
        }
        for c in self._clients.values():
            stats["clients"].append({
                "client_id":       c.client_id,
                "store_id":        c.store_id,
                "connected_at":    c.connected_at.isoformat(),
                "messages_sent":   c.messages_sent,
                "messages_dropped": c.messages_dropped,
                "queue_size":      c.queue.qsize(),
            })
        return stats

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    async def _enqueue(client: WSClient, data: dict) -> bool:
        """
        Try to put data into the client's queue.
        If the queue is full, drop the message (backpressure) rather than
        blocking the broadcast loop.
        """
        try:
            client.queue.put_nowait(data)
            return True
        except asyncio.QueueFull:
            # Evict oldest item to make room for the newest
            try:
                client.queue.get_nowait()
                client.messages_dropped += 1
            except asyncio.QueueEmpty:
                pass
            try:
                client.queue.put_nowait(data)
            except asyncio.QueueFull:
                pass
            return False


# ── Module-level singleton (shared across all API routes) ────────────────────
ws_manager = ConnectionManager()
