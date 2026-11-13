"""Conservative request/member compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable

from .contracts import MemberSnapshot, ReasonCode

TokenCounter = Callable[[dict[str, Any]], int | None]


@dataclass(frozen=True, slots=True)
class RequestRequirements:
    input_tokens_estimate: int
    max_output_tokens: int
    required_capabilities: tuple[str, ...]
    estimate_source: str
    # Restricts eligibility to one member; the job still shares the queue,
    # priority order and slot accounting with unpinned work.
    pinned_member: str | None = None


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


# Context allowance for one image part. Image bytes are not text tokens: a
# base64 data URL would otherwise count as hundreds of thousands of "tokens"
# and make every image job incompatible with the GPU members.
IMAGE_TOKEN_ALLOWANCE = 1536
# shortcut: conservative lower bound on tokens-per-byte for mixed text (real
# ASCII is ~4 bytes/token); upgrade to the per-member tokenizer (estimate_source
# "tokenizer") once every routing member advertises one.
BYTES_PER_TOKEN_ESTIMATE = 3
VISION_CAPABILITY = "vision"
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _is_image_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    if part.get("type") in _IMAGE_PART_TYPES:
        return True
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return isinstance(url, str) and url.startswith("data:image/")


def _content_parts(body: dict[str, Any]):
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            yield from content


def request_has_image(body: dict[str, Any]) -> bool:
    """True when any chat message carries an image part (URL or data URL)."""
    return any(_is_image_part(part) for part in _content_parts(body))


def _text_token_estimate(body: dict[str, Any]) -> int:
    messages = body.get("messages", [])
    images = 0
    if isinstance(messages, list):
        stripped = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                kept = [part for part in content if not _is_image_part(part)]
                images += len(content) - len(kept)
                message = {**message, "content": kept}
            stripped.append(message)
        messages = stripped
    encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return math.ceil(len(encoded) / BYTES_PER_TOKEN_ESTIMATE) + images * IMAGE_TOKEN_ALLOWANCE


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
        estimate = _text_token_estimate(body)
        source = "text_bytes_token_estimate"
    max_output = body.get("max_completion_tokens", body.get("max_tokens", 0))
    capabilities = tuple(required_capabilities)
    # Detected here, not at the callers, so submit-time admission and the
    # dispatch-time requirements factory always agree (plan 5.1).
    if VISION_CAPABILITY not in capabilities and request_has_image(body):
        capabilities += (VISION_CAPABILITY,)
    return RequestRequirements(
        input_tokens_estimate=estimate,
        max_output_tokens=max(0, int(max_output or 0)),
        required_capabilities=capabilities,
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
    if requirements.pinned_member and member.name != requirements.pinned_member:
        reasons.append(ReasonCode.REQUEST_PINNED_MEMBER_MISMATCH)
    return CompatibilityResult(
        compatible=not reasons,
        required_context=required_context,
        reason_codes=tuple(reasons),
    )
