"""Measured ETA ranges and timeout guidance for the Gemma broker."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from collections import deque
from typing import Iterable


def token_bucket(input_tokens: int) -> str:
    value = max(0, int(input_tokens))
    if value <= 4096:
        return "small"
    if value <= 32768:
        return "medium"
    return "large"


def quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    index = max(0, ceil(q * len(ordered)) - 1)
    value = ordered[index]
    return int(value) if value.is_integer() else value


@dataclass(frozen=True, slots=True)
class WorkSample:
    observed_at: float
    member_name: str
    input_tokens: int
    max_tokens_present: bool
    queue_seconds: float
    first_result_seconds: float
    completion_seconds: float
    succeeded: bool

    @property
    def token_bucket(self) -> str:
        return token_bucket(self.input_tokens)


@dataclass(frozen=True, slots=True)
class ETAEstimate:
    available: bool
    member_name: str
    token_bucket: str
    window: str | None = None
    confidence: str = "unavailable"
    sample_count: int = 0
    p50_queue_seconds: float | None = None
    p90_queue_seconds: float | None = None
    p50_first_result_seconds: float | None = None
    p90_first_result_seconds: float | None = None
    p50_completion_seconds: float | None = None
    p90_completion_seconds: float | None = None
    recent_error_rate: float | None = None
    suggested_queue_timeout: float | None = None
    suggested_first_result_timeout: float | None = None
    suggested_completion_timeout: float | None = None


class BoundedSampleStore:
    def __init__(
        self,
        *,
        max_samples: int,
        qualified_baselines: dict[tuple[str, str], list[WorkSample]] | None = None,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._samples: deque[WorkSample] = deque(maxlen=max_samples)
        self._baselines = qualified_baselines or {}

    def add(self, sample: WorkSample) -> None:
        for value in (
            sample.queue_seconds,
            sample.first_result_seconds,
            sample.completion_seconds,
        ):
            if value < 0:
                raise ValueError("durations must be non-negative")
        self._samples.append(sample)

    def _matching(
        self, *, member_name: str, bucket: str, now: float, seconds: float
    ) -> list[WorkSample]:
        return [
            sample
            for sample in self._samples
            if sample.member_name == member_name
            and sample.token_bucket == bucket
            and 0 <= now - sample.observed_at <= seconds
        ]

    def estimate(
        self,
        *,
        now: float,
        member_name: str,
        input_tokens: int,
        max_tokens_present: bool,
    ) -> ETAEstimate:
        bucket = token_bucket(input_tokens)
        recent = self._matching(
            member_name=member_name, bucket=bucket, now=now, seconds=600.0
        )
        if len(recent) >= 8:
            selected = recent
            window = "10m"
            confidence = "high" if max_tokens_present else "medium"
        else:
            hourly = self._matching(
                member_name=member_name, bucket=bucket, now=now, seconds=3600.0
            )
            if len(hourly) >= 3:
                selected = hourly
                window = "1h"
                confidence = "low"
            else:
                baseline = list(self._baselines.get((member_name, bucket), ()))
                if baseline:
                    selected = baseline
                    window = "qualified_baseline"
                    confidence = "low"
                else:
                    return ETAEstimate(False, member_name, bucket)

        p50_queue = quantile((item.queue_seconds for item in selected), 0.5)
        p90_queue = quantile((item.queue_seconds for item in selected), 0.9)
        p50_first = quantile((item.first_result_seconds for item in selected), 0.5)
        p90_first = quantile((item.first_result_seconds for item in selected), 0.9)
        p50_completion = quantile((item.completion_seconds for item in selected), 0.5)
        p90_completion = quantile((item.completion_seconds for item in selected), 0.9)
        error_rate = sum(not item.succeeded for item in selected) / len(selected)
        assert p90_queue is not None and p90_first is not None and p90_completion is not None
        return ETAEstimate(
            available=True,
            member_name=member_name,
            token_bucket=bucket,
            window=window,
            confidence=confidence,
            sample_count=len(selected),
            p50_queue_seconds=p50_queue,
            p90_queue_seconds=p90_queue,
            p50_first_result_seconds=p50_first,
            p90_first_result_seconds=p90_first,
            p50_completion_seconds=p50_completion,
            p90_completion_seconds=p90_completion,
            recent_error_rate=error_rate,
            suggested_queue_timeout=float(p90_queue * 2),
            suggested_first_result_timeout=float(p90_first * 2),
            suggested_completion_timeout=float(p90_completion * 2),
        )
