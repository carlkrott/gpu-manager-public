"""One-attempt dispatcher with an explicit backend acceptance boundary."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Callable, Protocol

from .contracts import JobRecord, JobState, ReasonCode
from .compatibility import normalize_chat_request
from .redis_store import (
    InMemoryJobRepository,
    LeadershipLost,
    RepositoryUnavailable,
    Reservation,
)


class Transport(Protocol):
    async def send(
        self,
        member: Any,
        body: dict[str, Any],
        on_accepted: Callable[[], None],
    ) -> dict[str, Any]: ...


class Dispatcher:
    def __init__(
        self,
        *,
        repo: InMemoryJobRepository,
        transport: Transport,
        request_timeout: float,
        clock: Callable[[], float],
    ) -> None:
        self.repo = repo
        self.transport = transport
        self.request_timeout = request_timeout
        self.clock = clock

    async def dispatch(self, reservation: Reservation) -> JobRecord:
        current = self.repo.get(reservation.job.job_id) or reservation.job
        if current.cancel_requested and not current.accepted_boundary_crossed:
            cancelled = replace(
                current,
                state=JobState.CANCELLED,
                completed_at=self.clock(),
                reason_code=ReasonCode.CANCELLED_BEFORE_ACCEPTANCE,
                state_version=current.state_version + 1,
            )
            return self.repo.replace_job(cancelled)

        accepted = False

        def on_accepted() -> None:
            nonlocal accepted, current
            if accepted:
                return
            accepted = True
            current = replace(
                current,
                state=JobState.ACCEPTED,
                accepted_boundary_crossed=True,
                forward_started_at=self.clock(),
                state_version=current.state_version + 1,
            )
            self.repo.replace_job(current)

        try:
            response = await asyncio.wait_for(
                self.transport.send(
                    reservation.member,
                    normalize_chat_request(current.request_body),
                    on_accepted,
                ),
                timeout=self.request_timeout,
            )
        except (RepositoryUnavailable, LeadershipLost):
            raise
        except Exception as exc:
            if accepted:
                unknown = replace(
                    current,
                    state=JobState.OUTCOME_UNKNOWN,
                    completed_at=self.clock(),
                    reason_code=ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN,
                    error=f"{type(exc).__name__}:{exc}",
                    state_version=current.state_version + 1,
                )
                replaced = self.repo.replace_job(unknown)
                self._release_reservation_lease(reservation)
                return replaced
            return self.repo.requeue_claim(
                current,
                reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
                terminal_at=self.clock(),
            )

        if not accepted:
            # A complete successful response is itself evidence that the backend
            # accepted the request even if an adapter omitted the early callback.
            on_accepted()

        # Post-acceptance CANCELLED path: between the on_accepted() callback
        # above (or the early one inside try) and the terminal outcome, the
        # client may have requested cancellation. Convert it to CANCELLED
        # rather than COMPLETED so the lease is released exactly once.
        if current.cancel_requested:
            cancelled = replace(
                current,
                state=JobState.CANCELLED,
                completed_at=self.clock(),
                reason_code=ReasonCode.CANCELLED_AFTER_ACCEPTANCE,
                state_version=current.state_version + 1,
            )
            replaced = self.repo.replace_job(cancelled)
            self._release_reservation_lease(reservation)
            return replaced

        completed = replace(
            current,
            state=JobState.COMPLETED,
            completed_at=self.clock(),
            result=response,
            error=None,
            reason_code=None,
            state_version=current.state_version + 1,
        )
        replaced = self.repo.replace_job(completed)
        self._release_reservation_lease(reservation)
        return replaced

    def _release_reservation_lease(self, reservation: Reservation) -> None:
        """Release the lease tied to this reservation's attempt.

        Called from terminal post-acceptance branches (COMPLETED, OUTCOME_UNKNOWN,
        post-acceptance CANCELLED) AFTER replace_job returns. Not called on the
        pre-acceptance requeue path: requeue_claim.lua already removes the lease.
        """
        attempt_id = reservation.job.current_attempt_id
        claimed_version = reservation.job.member_state_version
        if not attempt_id or claimed_version is None:
            return
        self.repo.release_member_lease(
            reservation.member.name,
            attempt_id,
            claimed_member_state_version=claimed_version,
        )
