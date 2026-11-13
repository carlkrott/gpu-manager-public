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
    ReservationStale,
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
        terminal_write_attempts: int = 3,
        terminal_write_delay: float = 0.05,
    ) -> None:
        self.repo = repo
        self.transport = transport
        self.request_timeout = request_timeout
        self.clock = clock
        self.terminal_write_attempts = max(1, int(terminal_write_attempts))
        self.terminal_write_delay = max(0.0, float(terminal_write_delay))
        # Post-acceptance reservations whose terminal write/release hit a
        # repository outage. Keyed by attempt id, so reconciliation can never
        # release a successor attempt's lease on the same member.
        self.unreleased: dict[str, Reservation] = {}
        # Attempts whose backend acceptance callback fired but whose durable
        # ACCEPTED write hit an outage: they must converge to OUTCOME_UNKNOWN,
        # never be requeued, because the backend may be running the work.
        self._accepted_unrecorded: set[str] = set()

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
            try:
                replaced = self.repo.replace_job(cancelled)
            except RepositoryUnavailable:
                self.unreleased[self._attempt_id(reservation)] = reservation
                raise
            self._release_reservation_lease(reservation)
            return replaced

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
        except RepositoryUnavailable:
            # The claim's lease is still held and no terminal record exists:
            # remember it so reconcile_unreleased converges it once the
            # repository returns instead of waiting for a process restart.
            attempt = self._attempt_id(reservation)
            self.unreleased[attempt] = reservation
            if accepted:
                self._accepted_unrecorded.add(attempt)
            raise
        except LeadershipLost:
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
                return await self._terminalize(reservation, unknown)
            error_str = f"{type(exc).__name__}:{exc}"
            try:
                return self.repo.requeue_claim(
                    current,
                    reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
                    terminal_at=self.clock(),
                    error=error_str,
                )
            except RepositoryUnavailable:
                self.unreleased[self._attempt_id(reservation)] = reservation
                raise

        if not accepted:
            # A complete successful response is itself evidence that the backend
            # accepted the request even if an adapter omitted the early callback.
            on_accepted()

        # A cancel recorded while the request was in flight lives only in the
        # durable record; merge it so the stale local copy cannot clobber it.
        latest = self.repo.get(current.job_id)
        if latest is not None and latest.cancel_requested and not current.cancel_requested:
            current = replace(current, cancel_requested=True)

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
            return await self._terminalize(reservation, cancelled)

        completed = replace(
            current,
            state=JobState.COMPLETED,
            completed_at=self.clock(),
            result=response,
            error=None,
            reason_code=None,
            state_version=current.state_version + 1,
        )
        return await self._terminalize(reservation, completed)

    async def _terminalize(
        self, reservation: Reservation, terminal: JobRecord
    ) -> JobRecord:
        """Write a post-acceptance terminal record, then free its lease.

        A repository outage is retried a bounded number of times. If it
        persists, the attempt is remembered in ``unreleased`` so
        :meth:`reconcile_unreleased` frees it once the repository returns; the
        outage is still re-raised. ``LeadershipLost`` is never retried or
        released here: a successor fence owns reconciliation of this attempt.
        """
        replaced: JobRecord | None = None
        for attempt in range(self.terminal_write_attempts):
            try:
                replaced = self.repo.replace_job(terminal)
                self._release_reservation_lease(reservation)
                return replaced
            except RepositoryUnavailable:
                if replaced is not None:
                    # Terminal record is durable; only the release is pending.
                    self.unreleased[self._attempt_id(reservation)] = reservation
                    raise
                if attempt + 1 >= self.terminal_write_attempts:
                    self.unreleased[self._attempt_id(reservation)] = reservation
                    raise
                await asyncio.sleep(self.terminal_write_delay)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _attempt_id(reservation: Reservation) -> str:
        return str(reservation.job.current_attempt_id or reservation.job.job_id)

    async def reconcile_unreleased(self) -> list[str]:
        """Finish terminalization/release for attempts hit by an outage.

        Accepted work is never replayed: a job still ACCEPTED/IN_FLIGHT after
        the outage becomes OUTCOME_UNKNOWN. Returns the reconciled attempt ids.
        """
        done: list[str] = []
        for attempt_id, reservation in list(self.unreleased.items()):
            try:
                current = self.repo.get(reservation.job.job_id)
                same_attempt = (
                    current is not None
                    and current.current_attempt_id == reservation.job.current_attempt_id
                )
                if current is not None and current.state is JobState.CLAIMED and same_attempt:
                    if attempt_id in self._accepted_unrecorded:
                        # Backend acceptance was observed; the durable write was
                        # lost. Never replay: record the unknown outcome.
                        self.repo.replace_job(
                            replace(
                                current,
                                state=JobState.OUTCOME_UNKNOWN,
                                accepted_boundary_crossed=True,
                                completed_at=self.clock(),
                                reason_code=ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN,
                                error="acceptance_write_lost_to_repository_outage",
                                state_version=current.state_version + 1,
                            )
                        )
                    elif current.cancel_requested:
                        self.repo.replace_job(
                            replace(
                                current,
                                state=JobState.CANCELLED,
                                completed_at=self.clock(),
                                reason_code=ReasonCode.CANCELLED_BEFORE_ACCEPTANCE,
                                state_version=current.state_version + 1,
                            )
                        )
                    else:
                        # Pre-acceptance claim abandoned by this dispatcher:
                        # requeue_claim frees the lease atomically.
                        self.repo.requeue_claim(
                            current,
                            reason=ReasonCode.PRE_ACCEPTANCE_REQUEUE,
                            terminal_at=self.clock(),
                        )
                        self._accepted_unrecorded.discard(attempt_id)
                        self.unreleased.pop(attempt_id, None)
                        done.append(attempt_id)
                        continue
                elif current is not None and current.state in {
                    JobState.ACCEPTED,
                    JobState.IN_FLIGHT,
                }:
                    if current.current_attempt_id == reservation.job.current_attempt_id:
                        self.repo.replace_job(
                            replace(
                                current,
                                state=JobState.OUTCOME_UNKNOWN,
                                completed_at=self.clock(),
                                reason_code=ReasonCode.POST_ACCEPTANCE_OUTCOME_UNKNOWN,
                                error="terminal_write_lost_to_repository_outage",
                                state_version=current.state_version + 1,
                            )
                        )
                self._release_reservation_lease(reservation)
            except ReservationStale:
                pass  # lease already freed (reconciler/successor): nothing to do
            except (RepositoryUnavailable, LeadershipLost):
                continue  # retry next tick; a successor fence reconciles on loss
            self._accepted_unrecorded.discard(attempt_id)
            self.unreleased.pop(attempt_id, None)
            done.append(attempt_id)
        return done

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
