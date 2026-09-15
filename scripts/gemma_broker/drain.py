"""Member-local, non-forced drain coordination for the combined Gemma broker."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from .contracts import ContractError, MemberSnapshot, MemberState, ReasonCode
from .redis_store import ReservationStale


DRAIN_LEASE_RECHECK_SECONDS = 0.25


class DrainRepository(Protocol):
    def begin_member_drain(self, member_name: str, *, expected_state_version: int) -> MemberSnapshot: ...
    def lease_count(self, member_name: str) -> int: ...
    def release_member_lease(
        self,
        member_name: str,
        attempt_id: str,
        *,
        claimed_member_state_version: int,
    ) -> MemberSnapshot: ...
    def complete_member_drain(
        self, member_name: str, *, expected_state_version: int
    ) -> MemberSnapshot: ...
    def get_member(self, member_name: str) -> MemberSnapshot | None: ...
    def begin_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        generation_fence: int,
    ) -> MemberSnapshot: ...
    def complete_member_rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        readiness: MemberSnapshot,
    ) -> MemberSnapshot: ...


@dataclass(frozen=True, slots=True)
class DrainResult:
    member: MemberSnapshot
    drained: bool
    lease_count: int
    blockers: tuple[str, ...]


class MemberDrainCoordinator:
    """Coordinate one member without consulting whole-group queue depth."""

    def __init__(self, repository: DrainRepository) -> None:
        self.repository = repository
        self._condition = asyncio.Condition()

    async def begin_and_wait(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        timeout: float,
        retry_stale: bool = False,
    ) -> DrainResult:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")

        current = self.repository.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state is MemberState.OFFLINE:
            leases = self.repository.lease_count(member_name)
            return DrainResult(
                current,
                leases == 0,
                leases,
                () if leases == 0 else (f"member_leases:{leases}",),
            )

        # The standalone observer legitimately advances state_version while
        # refreshing readiness.  When the caller did not provide an explicit
        # CAS version, rebase a bounded number of times only while the member
        # remains drainable.  Explicit stale versions remain strict.
        draining: MemberSnapshot | None = None
        begin_version = expected_state_version
        for attempt in range(3):
            try:
                draining = self.repository.begin_member_drain(
                    member_name, expected_state_version=begin_version
                )
                break
            except ReservationStale:
                if not retry_stale or attempt == 2:
                    raise
                current = self.repository.get_member(member_name)
                if current is None or current.state not in (
                    MemberState.READY_ACCEPTING,
                    MemberState.JOINING,
                    MemberState.UNHEALTHY,
                ):
                    raise
                begin_version = current.state_version
        if draining is None:
            raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        async with self._condition:
            while True:
                leases = self.repository.lease_count(member_name)
                if leases == 0:
                    current = self.repository.get_member(member_name) or draining
                    complete_version = current.state_version
                    offline: MemberSnapshot | None = None
                    for attempt in range(3):
                        try:
                            offline = self.repository.complete_member_drain(
                                member_name, expected_state_version=complete_version
                            )
                            break
                        except ReservationStale:
                            if not retry_stale or attempt == 2:
                                raise
                            refreshed = self.repository.get_member(member_name)
                            if (
                                refreshed is None
                                or refreshed.state is not MemberState.DRAINING
                                or self.repository.lease_count(member_name) != 0
                            ):
                                raise
                            complete_version = refreshed.state_version
                    if offline is None:
                        raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
                    return DrainResult(offline, True, 0, ())
                remaining = deadline - loop.time()
                if remaining <= 0:
                    current = self.repository.get_member(member_name) or draining
                    return DrainResult(
                        current,
                        False,
                        leases,
                        (f"member_leases:{leases}",),
                    )
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=min(remaining, DRAIN_LEASE_RECHECK_SECONDS),
                    )
                except asyncio.TimeoutError:
                    # A lease may disappear through Redis expiry, a fenced
                    # reconciliation, or another process. Those paths cannot
                    # notify this in-process condition, so re-check the live
                    # lease set at a bounded cadence instead of returning a
                    # false blocker at the overall deadline.
                    continue

    async def release_lease(
        self,
        member_name: str,
        attempt_id: str,
        *,
        claimed_member_state_version: int,
    ) -> MemberSnapshot:
        member = self.repository.release_member_lease(
            member_name,
            attempt_id,
            claimed_member_state_version=claimed_member_state_version,
            enforce_state_version_fence=True,
        )
        async with self._condition:
            self._condition.notify_all()
        return member


class MemberRejoinCoordinator:
    """Rejoin only after a new process generation proves full readiness."""

    def __init__(self, repository: DrainRepository) -> None:
        self.repository = repository

    def rejoin(
        self,
        member_name: str,
        *,
        expected_state_version: int,
        generation_fence: int,
        readiness: MemberSnapshot,
    ) -> MemberSnapshot:
        current = self.repository.get_member(member_name)
        if current is None:
            raise ContractError(f"UNKNOWN_MEMBER:{member_name}")
        if current.state is MemberState.READY_ACCEPTING and current.accepting:
            return current
        if (
            current.state is not MemberState.JOINING
            and generation_fence <= current.generation_fence
        ):
            raise ContractError("GENERATION_FENCE_STALE")
        if readiness.name != member_name:
            raise ContractError("READINESS_MEMBER_MISMATCH")
        if readiness.gpu_id and readiness.idle_service_effective is not True:
            raise ContractError(ReasonCode.MEMBER_IDLE_SERVICE_NOT_EFFECTIVE.value)
        if readiness.state is not MemberState.READY_ACCEPTING or not readiness.accepting:
            raise ContractError("MEMBER_NOT_READY")
        if readiness.blockers or readiness.reason_codes:
            raise ContractError("MEMBER_NOT_READY")

        if current.state is MemberState.JOINING:
            if expected_state_version != current.state_version:
                raise ReservationStale(ReasonCode.RESERVATION_STATE_STALE.value)
            joining = current
        else:
            joining = self.repository.begin_member_rejoin(
                member_name,
                expected_state_version=expected_state_version,
                generation_fence=generation_fence,
            )
        return self.repository.complete_member_rejoin(
            member_name,
            expected_state_version=joining.state_version,
            readiness=readiness,
        )
