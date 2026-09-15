"""Freshness-aware bounded admission for combined Gemma jobs."""
from __future__ import annotations

from dataclasses import dataclass

from .compatibility import RequestRequirements, evaluate
from .contracts import JobRecord, MemberSnapshot, MemberState


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admitted: bool
    reason_codes: tuple[str, ...]
    eta_seconds: float | None
    member_profiles: dict[str, str]


class AdmissionController:
    def __init__(
        self,
        *,
        queue_cap: int,
        token_cap: int,
        freshness_seconds: float,
        service_rate: float,
        eta_cap_seconds: float = 3600.0,
    ) -> None:
        if queue_cap <= 0 or token_cap <= 0:
            raise ValueError("admission caps must be positive")
        if freshness_seconds <= 0 or service_rate <= 0 or eta_cap_seconds <= 0:
            raise ValueError("freshness, service rate, and ETA cap must be positive")
        self.queue_cap = queue_cap
        self.token_cap = token_cap
        self.freshness_seconds = freshness_seconds
        self.service_rate = service_rate
        self.eta_cap_seconds = eta_cap_seconds

    def evaluate(
        self,
        job: JobRecord,
        members: list[MemberSnapshot],
        *,
        now: float,
        queue_depth: int,
        queued_tokens: int,
    ) -> AdmissionResult:
        profiles: dict[str, str] = {}
        requirements = RequestRequirements(
            input_tokens_estimate=int(job.input_tokens_estimate or 0),
            max_output_tokens=int(job.max_output_tokens or 0),
            required_capabilities=tuple(job.required_capabilities),
            estimate_source="durable_job",
        )
        eligible = False
        fresh_count = 0
        stale_count = 0
        for member in members:
            fresh = now - member.observed_at <= self.freshness_seconds
            if not fresh:
                profiles[member.name] = "stale"
                stale_count += 1
                continue
            fresh_count += 1
            compatible = evaluate(requirements, member).compatible
            has_capacity = int(member.compatible_free_slots or 0) > 0
            ready = member.state is MemberState.READY_ACCEPTING and member.accepting
            if compatible and has_capacity and ready:
                profiles[member.name] = "eligible"
                eligible = True
            elif compatible and ready:
                profiles[member.name] = "no_capacity"
            else:
                profiles[member.name] = "incompatible"

        reasons: tuple[str, ...] = ()
        if queue_depth >= self.queue_cap:
            reasons = ("queue_depth_cap",)
        elif queued_tokens + int(job.input_tokens_estimate or 0) + int(job.max_output_tokens or 0) > self.token_cap:
            reasons = ("queue_token_cap",)
        elif not eligible:
            if stale_count and not fresh_count:
                reasons = ("capacity_stale",)
            else:
                reasons = ("no_eligible_member",)

        if reasons:
            return AdmissionResult(False, reasons, None, profiles)
        eta = min((max(0, queue_depth) + 1) / self.service_rate, self.eta_cap_seconds)
        return AdmissionResult(True, (), float(eta), profiles)
