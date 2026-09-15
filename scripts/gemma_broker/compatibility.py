"""Conservative request/member compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from .contracts import MemberSnapshot, ReasonCode

TokenCounter = Callable[[dict[str, Any]], int | None]


@dataclass(frozen=True, slots=True)
class RequestRequirements:
    input_tokens_estimate: int
    max_output_tokens: int
    required_capabilities: tuple[str, ...]
    estimate_source: str


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    compatible: bool
    required_context: int
    reason_codes: tuple[ReasonCode, ...]


def normalize_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy completion-shaped input for the chat backend.

    Older callers used ``prompt`` while the internal llama endpoint requires
    ``messages``. Keep all other caller fields intact and only synthesize the
    missing chat messages when a usable prompt is present.
    """
    normalized = dict(body)
    messages = normalized.get("messages")
    if isinstance(messages, list) and messages:
        return normalized
    prompt = normalized.get("prompt")
    if isinstance(prompt, str) and prompt:
        normalized["messages"] = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and prompt and all(isinstance(item, str) for item in prompt):
        normalized["messages"] = [{"role": "user", "content": "\n".join(prompt)}]
    return normalized


def _utf8_upper_bound(body: dict[str, Any]) -> int:
    messages = body.get("messages", [])
    encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(encoded)


def build_requirements(
    body: dict[str, Any],
    *,
    required_capabilities: tuple[str, ...] = ("chat",),
    input_tokens_hint: int | None = None,
    token_counter: TokenCounter | None = None,
) -> RequestRequirements:
    estimate: int | None = None
    source = "unknown"
    if input_tokens_hint is not None:
        estimate = max(0, int(input_tokens_hint))
        source = "caller_hint"
    elif token_counter is not None:
        counted = token_counter(body)
        if counted is not None:
            estimate = max(0, int(counted))
            source = "tokenizer"
    if estimate is None:
        estimate = _utf8_upper_bound(body)
        source = "utf8_upper_bound"
    max_output = body.get("max_completion_tokens", body.get("max_tokens", 0))
    return RequestRequirements(
        input_tokens_estimate=estimate,
        max_output_tokens=max(0, int(max_output or 0)),
        required_capabilities=tuple(required_capabilities),
        estimate_source=source,
    )


def evaluate(
    requirements: RequestRequirements,
    member: MemberSnapshot,
    *,
    safety_margin_tokens: int = 1024,
) -> CompatibilityResult:
    reasons: list[ReasonCode] = []
    required_context = (
        requirements.input_tokens_estimate
        + requirements.max_output_tokens
        + max(0, int(safety_margin_tokens))
    )
    if required_context > member.context_per_slot:
        reasons.append(ReasonCode.REQUEST_CONTEXT_INCOMPATIBLE)
    if not set(requirements.required_capabilities).issubset(member.capabilities):
        reasons.append(ReasonCode.REQUEST_CAPABILITY_INCOMPATIBLE)
    return CompatibilityResult(
        compatible=not reasons,
        required_context=required_context,
        reason_codes=tuple(reasons),
    )
