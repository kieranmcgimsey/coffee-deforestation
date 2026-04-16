"""Tool: scratchpad — simple KV store for researcher agent state.

What: Persists key-value pairs across tool calls within a researcher session.
Why: The researcher agent needs to accumulate findings across 8 tool calls
and pass them to the writer as a structured scratchpad.
Assumes: Used within a single reporting session per AOI.
Produces: Stored/retrieved string values.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT

_SCRATCHPAD_DIR = PROJECT_ROOT / "outputs" / "cache" / "scratchpads"


def _pad_path(aoi_id: str, session_id: str) -> Path:
    _SCRATCHPAD_DIR.mkdir(parents=True, exist_ok=True)
    return _SCRATCHPAD_DIR / f"{aoi_id}_{session_id}.json"


def scratchpad_write(key: str, value: str, aoi_id: str, session_id: str = "default") -> str:
    """Write a value to the scratchpad. Returns confirmation."""
    path = _pad_path(aoi_id, session_id)
    data: dict = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Scratchpad write: {aoi_id}/{session_id}/{key}")
    return f"OK: wrote {len(value)} chars to key {key!r}"


def scratchpad_read(key: str, aoi_id: str, session_id: str = "default") -> str:
    """Read a value from the scratchpad. Returns empty string if not found."""
    path = _pad_path(aoi_id, session_id)
    if not path.exists():
        return ""
    with open(path) as f:
        data = json.load(f)
    return data.get(key, "")


def scratchpad_read_all(aoi_id: str, session_id: str = "default") -> dict:
    """Read the entire scratchpad as a dict."""
    path = _pad_path(aoi_id, session_id)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def scratchpad_clear(aoi_id: str, session_id: str = "default") -> None:
    """Clear the scratchpad for a session."""
    path = _pad_path(aoi_id, session_id)
    if path.exists():
        path.unlink()
