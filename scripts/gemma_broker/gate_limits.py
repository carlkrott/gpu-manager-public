"""Gate limits for broker dispatch loops."""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PRE_ACCEPTANCE_REQUEUE_LIMIT = 5


@dataclass(frozen=True, slots=True)
class BrokerGateLimits:
    """Immutable limits that govern broker dispatch gates.

    Attributes
    ----------
    pre_acceptance_requeue_limit:
        Maximum number of PRE_ACCEPTANCE_REQUEUE cycles a job may undergo
        before it is terminalized with reason PRE_ACCEPTANCE_REQUEUE_EXHAUSTED.
        Jobs that have crossed the acceptance boundary are unaffected.
    """

    pre_acceptance_requeue_limit: int = DEFAULT_PRE_ACCEPTANCE_REQUEUE_LIMIT


# Module-level singleton for callers that do not require custom limits.
DEFAULT_GATE_LIMITS = BrokerGateLimits()
