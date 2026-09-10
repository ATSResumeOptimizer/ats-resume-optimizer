"""Short-lived progress state for anonymous optimization requests."""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import Lock

_lock = Lock()
_states: dict[str, dict[str, str]] = {}


def set_stage(session_id: str, stage: str) -> None:
    with _lock:
        _states[session_id] = {"stage": stage, "updated_at": datetime.utcnow().isoformat()}


def get_stage(session_id: str | None) -> dict[str, str]:
    if not session_id:
        return {"stage": "idle"}
    with _lock:
        state = _states.get(session_id)
        if not state:
            return {"stage": "idle"}
        try:
            updated = datetime.fromisoformat(state["updated_at"])
            if datetime.utcnow() - updated > timedelta(minutes=10):
                _states.pop(session_id, None)
                return {"stage": "idle"}
        except ValueError:
            pass
        return dict(state)


def clear_stage(session_id: str) -> None:
    with _lock:
        _states.pop(session_id, None)
