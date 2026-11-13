"""Cross-boot process generations in FileRuntimeStateStore must not wedge LoadBundle.

``process_generation`` is the process start time in clock ticks since boot, so a
record persisted in a previous boot can hold a value no live process can report
again.  The loader must rebase such records; a genuinely older observation in
the same boot must still be rejected.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import runtime_state  # noqa: E402
from runtime_state import (  # noqa: E402
    FileRuntimeStateStore,
    RuntimeConflict,
    RuntimeDesiredState,
    RuntimeInstanceRecord,
)

PROFILE = {
    "schema_version": "runtime-profile.v1",
    "engine": "audiocpp",
    "adapter": "audiocpp",
    "revision": 1,
}
INSTANCE = "audiocpp-minimax-music3-candidate"
PREVIOUS_BOOT_GENERATION = 41236067  # start time from a prior boot, ticks
BOOT_TICKS = 89710  # ticks since the current boot (about 14.9 minutes)


def _observation(generation: int, state: str = "loading") -> dict:
    return {
        "profile_name": "minimax-music3-audiocpp",
        "profile_fingerprint": runtime_state.runtime_profile_fingerprint(PROFILE),
        "state": state,
        "process_generation": generation,
        "process_ready": False,
        "model_ready": False,
        "accepting": False,
    }


def _write_state(path: Path, generation: int) -> None:
    record = RuntimeInstanceRecord(
        instance_id=INSTANCE,
        profile_name="minimax-music3-audiocpp",
        profile_fingerprint=runtime_state.runtime_profile_fingerprint(PROFILE),
        desired_state=RuntimeDesiredState.STOPPED,
        state=runtime_state.RuntimeControlState.STOPPED,
        process_generation=generation,
        transition_fence=110,
        observed_at=1791660403.0,
    )
    document = {
        "schema_version": FileRuntimeStateStore.SCHEMA_VERSION,
        "records": {INSTANCE: json.loads(json.dumps(asdict(record), default=str))},
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _publish(store: FileRuntimeStateStore, generation: int):
    async def run():
        _, lease = await store.begin_transition(
            INSTANCE,
            "minimax-music3-audiocpp",
            PROFILE,
            owner="test-owner",
            desired_state=RuntimeDesiredState.READY,
            now=1000.0,
        )
        return await store.publish_observation(
            lease, PROFILE, _observation(generation), now=1000.0
        )

    return asyncio.run(run())


def test_prior_boot_generation_is_rebased_and_accepted(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "runtime-state.json"
    _write_state(state_path, PREVIOUS_BOOT_GENERATION)
    monkeypatch.setattr(runtime_state, "_boot_ticks", lambda: BOOT_TICKS, raising=False)

    store = FileRuntimeStateStore(state_path, clock=lambda: 1000.0)
    record = _publish(store, generation=1200)

    assert record.process_generation == 1200


def test_genuinely_older_generation_in_same_boot_is_still_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "runtime-state.json"
    _write_state(state_path, 4000)
    monkeypatch.setattr(runtime_state, "_boot_ticks", lambda: BOOT_TICKS, raising=False)

    store = FileRuntimeStateStore(state_path, clock=lambda: 1000.0)
    with pytest.raises(RuntimeConflict, match="older process generation"):
        _publish(store, generation=3000)


def test_unreadable_boot_ticks_does_not_rebase(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "runtime-state.json"
    _write_state(state_path, PREVIOUS_BOOT_GENERATION)
    monkeypatch.setattr(runtime_state, "_boot_ticks", lambda: None, raising=False)

    store = FileRuntimeStateStore(state_path, clock=lambda: 1000.0)
    with pytest.raises(RuntimeConflict, match="older process generation"):
        _publish(store, generation=1200)
