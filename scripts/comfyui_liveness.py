"""Bounded lost-completion detection for accepted ComfyUI prompts.

A ``/prompt`` ACK is a hand-off, not a completion. When the backend later
forgets the prompt (history gone, queue empty) or stops answering, the worker
used to keep polling until the full generation deadline (up to two hours) while
holding its worker lease and the GPU bundle. This module gives the poll loop a
short, configurable, typed way out: the job ends ``failed`` with
``error_type="lost_completion"`` and is never replayed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

LOST_COMPLETION_GRACE_DEFAULT_S = 120
LOST_COMPLETION_GRACE_MIN_S = 30
LOST_COMPLETION_GRACE_MAX_S = 1800
LOST_COMPLETION_ERROR_TYPE = "lost_completion"


class PromptLiveness(StrEnum):
    PRESENT = "present"        # /history lists the prompt
    ACTIVE = "active"          # running or pending in /queue
    ABSENT = "absent"          # backend answered; prompt in neither
    UNREACHABLE = "unreachable"


def lost_completion_grace_s(metadata: Mapping[str, Any] | None) -> int:
    """Clamped grace before an unseen prompt is declared lost."""
    try:
        requested = int((metadata or {}).get("lost_completion_grace_s"))
    except (TypeError, ValueError):
        requested = LOST_COMPLETION_GRACE_DEFAULT_S
    return max(LOST_COMPLETION_GRACE_MIN_S, min(LOST_COMPLETION_GRACE_MAX_S, requested))


@dataclass
class LostCompletionTracker:
    """Tracks how long a prompt has been continuously unseen."""

    grace_s: float
    unseen_since: float | None = None
    last_state: PromptLiveness | None = None

    def observe(self, state: PromptLiveness, now: float) -> None:
        self.last_state = state
        if state in (PromptLiveness.PRESENT, PromptLiveness.ACTIVE):
            self.unseen_since = None
        elif self.unseen_since is None:
            self.unseen_since = now

    def lost_reason(self, now: float) -> str | None:
        if self.unseen_since is None or now - self.unseen_since < self.grace_s:
            return None
        kind = (
            "backend_unreachable"
            if self.last_state is PromptLiveness.UNREACHABLE
            else "prompt_absent_from_history_and_queue"
        )
        return f"{kind} for {int(now - self.unseen_since)}s"


async def probe_prompt(session: Any, base: str, prompt_id: str) -> PromptLiveness:
    """Ask ComfyUI whether ``prompt_id`` is still known (history, then queue)."""
    try:
        async with session.get(f"{base}/history/{prompt_id}", timeout=_timeout()) as resp:
            history = await resp.json(content_type=None)
        if isinstance(history, dict) and history.get(prompt_id):
            return PromptLiveness.PRESENT
        async with session.get(f"{base}/queue", timeout=_timeout()) as resp:
            queue = await resp.json(content_type=None)
    except Exception:
        return PromptLiveness.UNREACHABLE
    if not isinstance(queue, dict):
        return PromptLiveness.UNREACHABLE
    for key in ("queue_running", "queue_pending"):
        for item in queue.get(key) or []:
            if isinstance(item, (list, tuple)) and prompt_id in item[:2]:
                return PromptLiveness.ACTIVE
    return PromptLiveness.ABSENT


def _timeout() -> Any:
    from aiohttp import ClientTimeout

    return ClientTimeout(total=10)
