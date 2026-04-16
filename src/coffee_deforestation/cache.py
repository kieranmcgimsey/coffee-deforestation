"""Content-addressed caching for expensive pipeline operations.

What: Provides a decorator that caches function outputs based on SHA-256 hashes of inputs.
Why: GEE exports, feature stacks, and model training are expensive. Re-runs after config
tweaks should only recompute what changed.
Assumes: All cached functions are pure (same inputs → same outputs). Arguments are
JSON-serializable or pydantic models.
Produces: Cached artifacts under outputs/cache/{stage}/ with .meta.json sidecars.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from loguru import logger
from pydantic import BaseModel

from coffee_deforestation.config import PROJECT_ROOT

T = TypeVar("T")

# Thread-local force flag for --force CLI override
_force_local = threading.local()


def set_force(force: bool) -> None:
    """Set the thread-local force flag to bypass cache."""
    _force_local.force = force


def get_force() -> bool:
    """Get the current force flag."""
    return getattr(_force_local, "force", False)


def _serialize_arg(arg: Any) -> Any:
    """Convert an argument to a JSON-serializable form for hashing."""
    if isinstance(arg, BaseModel):
        return arg.model_dump(mode="json")
    if isinstance(arg, Path):
        return str(arg)
    if isinstance(arg, (list, tuple)):
        return [_serialize_arg(a) for a in arg]
    if isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in sorted(arg.items())}
    return arg


def compute_hash(inputs: dict[str, Any], hash_length: int = 16) -> str:
    """Compute a SHA-256 content hash from a dict of inputs."""
    serialized = json.dumps(
        {k: _serialize_arg(v) for k, v in sorted(inputs.items())},
        sort_keys=True,
        default=str,
    )
    full_hash = hashlib.sha256(serialized.encode()).hexdigest()
    return full_hash[:hash_length]


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception as e:
        logger.debug(f"Git hash lookup failed: {e}")
        return "unknown"


def _write_meta(
    meta_path: Path,
    stage: str,
    input_hash: str,
    input_components: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a .meta.json sidecar for a cached artifact."""
    meta = {
        "stage": stage,
        "input_hash": input_hash,
        "input_components": {k: str(v) for k, v in input_components.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "output_path": str(output_path),
    }

    # Add file-specific metadata
    if output_path.is_file():
        meta["file_size_bytes"] = output_path.stat().st_size
    elif output_path.is_dir():
        meta["file_count"] = sum(1 for _ in output_path.rglob("*") if _.is_file())

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def cached(
    stage: str,
    cache_dir: str = "outputs/cache",
    hash_length: int = 16,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for content-addressed caching of pipeline stages.

    The decorated function must return a Path to its output.
    Arguments must be JSON-serializable or pydantic models.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Build input dict from function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            input_dict = {"__stage__": stage, "__func__": func.__name__}
            input_dict.update(bound.arguments)

            content_hash = compute_hash(input_dict, hash_length)
            cache_base = PROJECT_ROOT / cache_dir / stage
            cache_base.mkdir(parents=True, exist_ok=True)

            # Build a human-readable prefix from common args
            parts = [stage]
            for key in ["aoi", "year"]:
                if key in bound.arguments:
                    val = bound.arguments[key]
                    if isinstance(val, BaseModel) and hasattr(val, "id"):
                        parts.append(str(val.id))  # type: ignore[attr-defined]
                    elif isinstance(val, str):
                        parts.append(val)
                    else:
                        parts.append(str(val))
            prefix = "_".join(parts)
            cache_key = f"{prefix}_{content_hash}"

            # Check for existing cached output
            meta_path = cache_base / f"{cache_key}.meta.json"
            if not get_force() and meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                output_path = Path(meta["output_path"])
                if output_path.exists():
                    logger.debug(f"Cache hit: {stage}/{cache_key}")
                    return output_path  # type: ignore[return-value]

            # Cache miss — run the function
            logger.info(f"Cache miss: {stage}/{cache_key} — computing...")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"Computed {stage}/{cache_key} in {elapsed:.1f}s")

            # Write metadata
            if isinstance(result, Path):
                _write_meta(meta_path, stage, content_hash, input_dict, result)

            return result

        return wrapper

    return decorator


def clear_cache(stage: str | None = None, cache_dir: str = "outputs/cache") -> int:
    """Clear cached artifacts. Returns count of items removed."""
    cache_base = PROJECT_ROOT / cache_dir
    if not cache_base.exists():
        return 0

    count = 0
    if stage:
        stage_dir = cache_base / stage
        if stage_dir.exists():
            for item in stage_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                count += 1
    else:
        for stage_dir in cache_base.iterdir():
            if stage_dir.is_dir():
                for item in stage_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    count += 1
    return count
