"""Helpers for running async pipelines from sync entrypoints."""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any


def run_coro_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine from sync code, even if the caller already has an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
