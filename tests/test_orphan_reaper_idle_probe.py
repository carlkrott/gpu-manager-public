"""Proof: the orphan reaper must not force a full VRAM probe on a stale cache.

``_orphan_reaper_tick`` awaits ``_poll_vram_background(ORPHAN_REAPER_INTERVAL_S)``,
whose reuse window is the caller-supplied reaper cadence of
``ORPHAN_REAPER_INTERVAL_S`` = 30 s. The reaper runs every 30 s and the idle
pusher only refreshes every 5th tick, so the reaper always lands on a cache
older than the 5 s idle window; binding the window to the reaper cadence lets it
read the shared idle cache instead of forcing a fresh ``amdgpu_top`` scan on
every idle tick.
"""
from __future__ import annotations

import asyncio

from test_poll_vram_background_throttle import facade, _tick_reaper  # noqa: F401


def test_orphan_reaper_reads_idle_cache_without_probing(facade, monkeypatch):
    """A 6 s-old (stale) non-empty cache must be reused, not re-probed."""
    polls: list[str] = []
    result = {"per_device": [], "gpus": []}

    async def fake_poll_once():
        polls.append("poll")
        return result

    monkeypatch.setattr(facade, "_poll_vram_once", fake_poll_once)
    monkeypatch.setattr(facade, "_vram_poll_lock", None)
    monkeypatch.setattr(facade, "_vram_cache", result)
    # 6.0 s old: outside the 5 s idle reuse window (VRAM_IDLE_POLL_EVERY *
    # VRAM_POLL_INTERVAL) and outside the 1 s _VRAM_POLL_TTL, so any probe is
    # genuinely forced by the caller, not served from a cache.
    monkeypatch.setattr(facade, "_vram_cache_ts", facade.time.monotonic() - 6.0)

    asyncio.run(_tick_reaper(facade, monkeypatch))

    assert polls == []
