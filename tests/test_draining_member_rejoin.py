"""A member left in DRAINING with zero leases must be able to rejoin (2026-10-10 RX6950XT incident).

A drain wait timed out while leases were stranded; once the leases were gone nothing completed the
DRAINING member, reconcile skipped it, and every preload failed with "cannot rejoin from 'draining'".
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gemma_broker.contracts import MemberSnapshot, MemberState  # noqa: E402
from gemma_broker.drain import MemberDrainCoordinator  # noqa: E402
from gemma_broker.redis_store import InMemoryJobRepository  # noqa: E402


def _draining_member(version: int) -> MemberSnapshot:
    return MemberSnapshot(
        name="Gemma-RX6950XT",
        gpu_id="rx6950xt",
        state=MemberState.DRAINING,
        accepting=False,
        configured_slots=4,
        context_per_slot=65536,
        state_version=version,
        observed_at=0.0,
        generation_fence=7,
        compatible_free_slots=0,
    )


def test_stale_draining_member_with_zero_leases_completes_to_offline():
    repo = InMemoryJobRepository()
    repo.put_member(_draining_member(258560))
    coordinator = MemberDrainCoordinator(repo)

    # The caller read an older version; the refresh loop has advanced it since.
    result = asyncio.run(
        coordinator.begin_and_wait(
            "Gemma-RX6950XT", expected_state_version=258559, timeout=0.5, retry_stale=True
        )
    )

    assert result.drained is True
    assert result.lease_count == 0
    assert repo.get_member("Gemma-RX6950XT").state is MemberState.OFFLINE


def test_stale_draining_member_is_strict_without_retry_stale():
    repo = InMemoryJobRepository()
    repo.put_member(_draining_member(10))
    coordinator = MemberDrainCoordinator(repo)

    try:
        asyncio.run(
            coordinator.begin_and_wait("Gemma-RX6950XT", expected_state_version=9, timeout=0.1)
        )
    except Exception as exc:  # ReservationStale
        assert "STALE" in str(exc).upper()
    else:
        raise AssertionError("explicit stale version must stay strict")
    assert repo.get_member("Gemma-RX6950XT").state is MemberState.DRAINING


def test_faulted_member_rejoins_through_fenced_joining(monkeypatch):
    """Audit 2026-11-12 B5: FAULTED was refused ("cannot rejoin"), so it was terminal."""
    import dataclasses
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "gpm_faulted_rejoin", Path(__file__).resolve().parents[1] / "scripts" / "gpu-manager.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    repo = InMemoryJobRepository()
    repo.put_member(dataclasses.replace(_draining_member(5), state=MemberState.FAULTED))

    async def refresh():
        cur = repo.get_member("Gemma-RX6950XT")
        assert cur.state is MemberState.JOINING
        repo.put_member(dataclasses.replace(cur, state=MemberState.READY_ACCEPTING, accepting=True))

    monkeypatch.setattr(m, "_combined_gemma_broker", SimpleNamespace(_repository=repo))
    monkeypatch.setattr(m, "_combined_gemma_broker_owns_service", lambda *_a: True)
    monkeypatch.setattr(m, "_combined_gemma_broker_embedded", lambda: True)
    monkeypatch.setattr(m, "_refresh_combined_gemma_member_snapshots", refresh)
    assert asyncio.run(m._rejoin_combined_gemma_member_after_start("Gemma-RX6950XT", timeout=1)) is True
