"""Offline reproduction for warning spam during a claimed-parent GPU wait."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_coordinator import (  # noqa: E402
    PipelineRunState,
    RedisCoordinatorStore,
)


class _Pipeline:
    def __init__(self):
        self.calls = []

    def xadd(self, *args):
        self.calls.append(("xadd", args))
        return "2000-1"

    def xack(self, *args):
        self.calls.append(("xack", args))

    def xdel(self, *args):
        self.calls.append(("xdel", args))

    def execute(self):
        return ["2000-1", 1, 1]


class _Redis:
    def pipeline(self, **_kwargs):
        return _Pipeline()

    def xlen(self, _key):
        return 0


def test_claimed_parent_wait_requeues_keep_backoff_without_per_tick_warnings(
    monkeypatch, caplog,
):
    """Two hundred normal wait ticks retain bounded delays, not 200 warnings."""
    store = object.__new__(RedisCoordinatorStore)
    store._r = _Redis()
    store._entry_ids = {"two-stage-image": "1000-1"}
    store._consecutive_requeues = {}
    store._last_requeue_warning = {}
    delays = []
    monkeypatch.setattr("pipeline_coordinator.time.sleep", delays.append)
    state = PipelineRunState(
        parent_job_id="image-job",
        run_id="two-stage-image",
        pipeline_id="krea-then-qwen-edit",
        revision_hash="fixture",
        status="running",
    )

    with caplog.at_level(logging.WARNING, logger="pipeline_coordinator"):
        for _ in range(200):
            assert store.requeue_parent(state)
            # Reclaimed parent has a new current stream entry on the next tick.
            store._entry_ids[state.run_id] = "1000-1"

    warnings = [r for r in caplog.records if "coordinator requeue loop" in r.getMessage()]
    # Before the warning threshold, the existing policy does not sleep.
    assert len(delays) == max(0, 200 - store.DEFAULT_REQUEUE_LIMIT)
    assert all(0 < delay <= store.REQUEUE_BACKOFF_MAX_SECONDS for delay in delays)
    assert len(warnings) <= 1


def test_requeue_warning_uses_per_run_monotonic_summary_interval(monkeypatch, caplog):
    """Warn immediately, summarize at 60s, and reset after ack."""
    store = object.__new__(RedisCoordinatorStore)
    store._consecutive_requeues = {}
    store._last_requeue_warning = {}
    now = [100.0]
    monkeypatch.setattr("pipeline_coordinator.time.monotonic", lambda: now[0])
    monkeypatch.setattr("pipeline_coordinator.time.sleep", lambda _delay: None)

    with caplog.at_level(logging.WARNING, logger="pipeline_coordinator"):
        store._requeue_backoff("run")
        store._requeue_backoff("run")
        assert len([r for r in caplog.records if "coordinator requeue loop" in r.getMessage()]) == 0
        # Cross the existing threshold, then verify no repeated warning before 60s.
        for _ in range(store.DEFAULT_REQUEUE_LIMIT):
            store._requeue_backoff("run")
        warnings = [r for r in caplog.records if "coordinator requeue loop" in r.getMessage()]
        assert len(warnings) == 1
        now[0] = 159.9
        store._requeue_backoff("run")
        assert len([r for r in caplog.records if "coordinator requeue loop" in r.getMessage()]) == 1
        now[0] = 160.0
        store._requeue_backoff("run")
        warnings = [r for r in caplog.records if "coordinator requeue loop" in r.getMessage()]
        assert len(warnings) == 2

    store._r = _Redis()
    # No pending entry is needed to exercise the acknowledgement bookkeeping reset.
    store._entry_ids = {}
    assert store.ack_parent("run")
    assert "run" not in store._last_requeue_warning
    assert "run" not in store._consecutive_requeues
    store._requeue_backoff("run")
    assert store._consecutive_requeues["run"] == 1
