"""
pipeline_provider_runtime.py — Service-registry-driven pipeline provider runtime.

This module closes the seam between the validated pipeline metadata
(compiled by service_pipeline.py) and the actual execution of individual
stages against named providers resolved from the GPU Manager's services.json
configuration.

It provides:

1. ProviderRuntimeError  — base fail-closed error.
2. UnknownProviderError  — provider ID not in registry or disabled.
3. AmbiguousProviderError — provider ID resolves to multiple endpoints.
4. EndpointResolutionError — endpoint_ref not found in services config.
5. DisabledProviderError  — provider is explicitly disabled.
6. KindCapabilityError    — stage kind not in provider's declared capabilities.

7. PipelineProviderRuntime — the main runtime class.

   Usage::

       runtime = PipelineProviderRuntime(
           services_config=services_json,   # full services.json dict
           provider_registry=frozenset({"open-design", "n8n", "comfyui", ...}),
       )
       executor = runtime.build_executor(
           provider_id="open-design",
           contract=worker_pipeline_dict,
           transport=my_transport,
       )
       ctx, results = await executor.dispatch(stages, context)

   For gpu_manager_generation::

       adapter = runtime.build_gpu_adapter(
           provider_id="comfyui",
           transport=my_transport,
       )
       receipt = await adapter.submit(assignment, state_store=store)

8. A simple factory callable::

       make_provider_runtime(services_config_path, provider_registry) -> PipelineProviderRuntime

No live network I/O occurs in tests when a fake transport is injected.
No arbitrary URLs, shell commands, or credentials are produced or followed.
Secrets are redacted from all error strings.
"""

from __future__ import annotations

import asyncio
import json as _json
import base64 as _base64
import dataclasses
import io as _io
import ipaddress as _ipaddress
import mimetypes as _mimetypes
from pathlib import Path as _Path
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Callable, Awaitable

from stage_dispatch import (
    DispatchError,
    DispatchProviderError,
    DispatchKindError,
    DispatchContextError,
    StageContext,
    StageResult,
    GenerationStageExecutor,
    StageExecutorFactory,
    MissingExecutorFactoryError,
)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# ── Constants ─────────────────────────────────────────────────────────────────

# Re-use the CLOSED_KINDS from service_pipeline (imported at bottom to avoid
# circular import; re-exported here for convenience).
_CLOSED_KINDS: frozenset[str] = frozenset({
    "prompt_enhance",
    "generate",
    "qc",
    "correct",
    "delegate",
})
_ENDPOINT_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_ENDPOINT_SHARED_NETWORK = _ipaddress.ip_network("100.64.0.0/10")

# Redaction patterns (mirrored from pipeline_http_provider for consistency).
_AUTH_RE = re.compile(r"(Bearer\s+)([^\s\"']+)", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?<![a-zA-Z0-9_])(api[_-]?key|secret|password|auth|token)(?![a-zA-Z0-9_])",
    re.IGNORECASE,
)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ProviderRuntimeError(DispatchProviderError):
    """Base fail-closed error for the provider runtime."""
    pass


class UnknownProviderError(ProviderRuntimeError):
    """Raised when the provider ID is not in the provider registry."""
    pass


class AmbiguousProviderError(ProviderRuntimeError):
    """Raised when the provider ID resolves ambiguously (multiple endpoints)."""
    pass


class DisabledProviderError(ProviderRuntimeError):
    """Raised when the provider is explicitly disabled in services.json."""
    pass


class EndpointResolutionError(ProviderRuntimeError):
    """Raised when the endpoint_ref cannot be resolved from services.json."""
    pass


class KindCapabilityError(ProviderRuntimeError):
    """Raised when a stage kind is not in the provider's declared capabilities."""


class HttpFlowOperationError(ProviderRuntimeError):
    """Raised when an http_flow operation (poll/fetch) fails or times out."""


class DuplicatePipelineError(ProviderRuntimeError):
    """Raised when the same pipeline ID appears under both 'pipelines' and 'service_pipelines'."""


# ── Redaction helpers ──────────────────────────────────────────────────────────

def _redact_secrets(text: str) -> str:
    """Redact bearer tokens and secret-like substrings from an error string."""
    text = _AUTH_RE.sub(r"\1<redacted>", text)
    def _secret_replacer(m: re.Match) -> str:
        before = text[m.start() - 12 : m.start()]
        if "<redacted>" in before:
            return m.group(0)
        return "<redacted>"
    text = _SECRET_RE.sub(_secret_replacer, text)
    return text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_service_endpoint(
    endpoint_ref: str,
    base_url: str,
    service: dict[str, Any],
) -> None:
    """Reject endpoint overrides that escape the declared execution boundary."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} has an invalid HTTP endpoint"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} endpoint may not contain credentials, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} endpoint has an invalid port"
        ) from exc
    if port is not None and not 1 <= port <= 65535:
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} endpoint has an invalid port"
        )

    host = parsed.hostname.lower().strip("[]")
    if host in _ENDPOINT_LOCAL_HOSTS:
        return
    try:
        address = _ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address in _ENDPOINT_SHARED_NETWORK
    ):
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} endpoint host is not permitted"
        )
    if service.get("endpoint_public") is not True:
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} public endpoint requires endpoint_public=true"
        )


def _resolve_service_endpoint(
    services_config: dict,
    endpoint_ref: str,
) -> tuple[str, int | None]:
    """
    Resolve an endpoint_ref to a (url, port) pair.

    Returns (base_url, port) where base_url is built from the service's
    ``endpoint`` field (or ``host:port`` construction), or raises
    ``EndpointResolutionError`` fail-closed.

    Port is returned separately so callers can combine with ``request_path``
    from the provider config.
    """
    services = services_config.get("services", {})
    if not isinstance(services, dict):
        raise EndpointResolutionError(
            f"services config has no 'services' dict (endpoint_ref={endpoint_ref!r})"
        )

    svc = services.get(endpoint_ref)
    if svc is None:
        raise EndpointResolutionError(
            f"endpoint_ref {endpoint_ref!r} not found in services config; "
            f"available services: {sorted(services.keys())}"
        )

    if not isinstance(svc, dict):
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} is not a dict"
        )

    if not svc.get("enabled", True):
        raise DisabledProviderError(
            f"service {endpoint_ref!r} is disabled in services config"
        )

    # Build the base URL from endpoint or host+port
    endpoint: object = svc.get("endpoint")
    host: object = svc.get("host")
    port: object = svc.get("port")

    if endpoint is not None:
        if not isinstance(endpoint, str) or not endpoint:
            raise EndpointResolutionError(
                f"service {endpoint_ref!r} endpoint must be a non-empty string"
            )
        base_url = endpoint.rstrip("/")
    elif host and port:
        if not isinstance(host, str) or isinstance(port, bool) or not isinstance(port, int):
            raise EndpointResolutionError(
                f"service {endpoint_ref!r} host/port has invalid types"
            )
        if not 1 <= port <= 65535:
            raise EndpointResolutionError(
                f"service {endpoint_ref!r} port is outside 1..65535"
            )
        scheme = "https" if svc.get("tls") else "http"
        formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        base_url = f"{scheme}://{formatted_host}:{port}"
    elif port:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise EndpointResolutionError(
                f"service {endpoint_ref!r} port is outside 1..65535"
            )
        base_url = f"http://127.0.0.1:{port}"
    else:
        raise EndpointResolutionError(
            f"service {endpoint_ref!r} has no 'endpoint', 'host', or 'port' — "
            f"cannot construct URL"
        )

    _validate_service_endpoint(endpoint_ref, base_url, svc)
    return base_url, port if isinstance(port, int) and not isinstance(port, bool) else None


def _build_provider_url(
    services_config: dict,
    provider_config: dict,
) -> str:
    """
    Build the full URL for a pipeline provider from its config entry.

    Combines the service's ``endpoint`` (resolved via ``endpoint_ref``) with
    the provider's ``request_path``.

    Raises ``EndpointResolutionError`` fail-closed on any resolution failure.
    Raises ``DisabledProviderError`` if the referenced service is disabled.
    """
    endpoint_ref: str | None = provider_config.get("endpoint_ref")
    if not endpoint_ref:
        raise EndpointResolutionError(
            f"provider config is missing 'endpoint_ref': {provider_config}"
        )

    base_url, _ = _resolve_service_endpoint(services_config, endpoint_ref)

    request_path: str = provider_config.get("request_path", "/")
    if not request_path.startswith("/"):
        request_path = "/" + request_path

    return base_url.rstrip("/") + request_path


def _check_capabilities(
    provider_config: dict,
    stage_kind: str,
    provider_id: str,
) -> None:
    """
    Validate that ``stage_kind`` is in the provider's declared capabilities.

    Raises ``KindCapabilityError`` fail-closed.
    """
    capabilities: list[str] = provider_config.get("capabilities", [])
    if not isinstance(capabilities, list):
        capabilities = []
    if stage_kind not in capabilities:
        raise KindCapabilityError(
            f"stage kind {stage_kind!r} not in provider {provider_id!r} "
            f"capabilities: {capabilities}"
        )


# ── input_tokens_hint helper ───────────────────────────────────────────────────

def _text_only_input_tokens_hint(messages: list[dict]) -> int:
    """
    Compute a conservative text-only UTF-8 token estimate for the messages list.

    This is used as ``input_tokens_hint`` in OpenAI-compatible multimodal requests.
    It intentionally excludes all image_url / data-URL bytes from the count so
    that the broker admission gate (which uses this field to budget a request)
    cannot be thrown off by a large base64 payload in a multimodal message.

    The algorithm counts each UTF-8 text byte as one token.  This deliberately
    conservative upper bound avoids tokenizer assumptions while allowing image
    blocks to be excluded from the broker's text-context budget.

    Parameters
    ----------
    messages:
        OpenAI-style ``messages`` list, e.g.::

            [{"role": "system", "content": "..."},
             {"role": "user", "content": [{"type": "image_url", ...},
                                          {"type": "text", "text": "..."}]}]

    Returns
    -------
    int
        A non-negative integer upper-bound estimate of the text token count.
        The value is the UTF-8 byte length of text-only content.
    """
    def _content_tokens(content: Any) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content.encode("utf-8"))
        if isinstance(content, list):
            # list (multimodal): skip image_url blocks, count text blocks
            total = 0
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        # image_url carries a data: URL or http URL — do NOT count
                        continue
                    # text part or any other: recurse
                    text = part.get("text", "") if isinstance(part, dict) else ""
                    total += _content_tokens(text)
                elif isinstance(part, str):
                    total += _content_tokens(part)
            return total
        # any other type (int, float, bool, …): zero contribution
        return 0

    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += _content_tokens(msg.get("content"))
    return max(0, total)


# ── Transport wrappers ────────────────────────────────────────────────────────

class _OpenAICompatibleTransport:
    """
    Thin sync wrapper that makes an injectable async transport callable
    produce OpenAI-compatible /v1/chat/completions shaped requests.

    This exists because the ``HttpPipelineProvider`` builds its own POST body
    internally. We intercept the transport call and re-shape the payload
    to the OpenAI completions format before passing it along.

    Usage::

        transport = MyAsyncTransport()
        wrapped = OpenAICompatibleTransport(transport, model="gemma-4")
        # Now use ``wrapped`` as the session for HttpPipelineProvider.
    """

    def __init__(
        self,
        inner: Callable[
            [dict, str, str, bytes | None, float],
            Awaitable[tuple[int, bytes, dict]],
        ],
        model: str = "gemma-4",
        max_inline_image_bytes: int | None = None,
    ) -> None:
        self._inner = inner
        self._model = model
        self._max_inline_image_bytes = (
            int(max_inline_image_bytes)
            if max_inline_image_bytes is not None
            else None
        )

    def _encode_local_image(self, artifact_path: _Path, mime: str) -> tuple[str, bytes]:
        """Return a bounded image payload without modifying the source artifact."""
        raw = artifact_path.read_bytes()
        limit = self._max_inline_image_bytes
        if limit is None or len(raw) <= limit:
            return mime, raw
        if limit <= 0:
            raise ProviderRuntimeError("max_inline_image_bytes must be positive")

        try:
            from PIL import Image

            with Image.open(artifact_path) as source:
                source.load()
                if source.mode in {"RGBA", "LA"}:
                    rgba = source.convert("RGBA")
                    image = Image.new("RGB", rgba.size, "white")
                    image.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    image = source.convert("RGB")
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                for quality in (92, 85, 78, 70, 60, 50):
                    output = _io.BytesIO()
                    image.save(
                        output,
                        format="JPEG",
                        quality=quality,
                        optimize=True,
                    )
                    preview = output.getvalue()
                    if len(preview) <= limit:
                        return "image/jpeg", preview
        except ProviderRuntimeError:
            raise
        except Exception as exc:
            raise ProviderRuntimeError(
                f"failed to create bounded QC preview: {type(exc).__name__}"
            ) from exc
        raise ProviderRuntimeError(
            f"QC preview exceeds max_inline_image_bytes: {limit}"
        )

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict | None = None,
        headers: dict | None = None,
        timeout: Any = None,
    ):
        """
        Convert the pipeline-payload to an OpenAI chat completions body,
        call ``self._inner``, then convert the response back to the
        pipeline provider shape ``{stage_id, status, data, error}``.
        """
        if json is not None:
            # Transform to OpenAI completions payload.
            # The pipeline payload shape (from HttpPipelineProvider._build_payload):
            #   {pipeline: {schema, original_prompt}, stage: {id, kind, ...},
            #    context: {original_prompt, enhanced_prompts, artifacts, qc_results}}
            pipeline_ctx = json.get("context", {})
            stage_info = json.get("stage", {})
            original_prompt = (
                pipeline_ctx.get("original_prompt", "")
                or json.get("pipeline", {}).get("original_prompt", "")
            )
            # Build messages in OpenAI format. Artifact references originate
            # from completed, trusted pipeline stages; QC must inspect the
            # pixels rather than silently judging only the prompt text.
            artifact_root = pipeline_ctx.get("artifacts", {}) if stage_info.get("kind") == "qc" else {}
            artifact_refs: list[str] = []

            def _collect_artifacts(value: Any) -> None:
                if len(artifact_refs) >= 4:
                    return
                if isinstance(value, dict):
                    for key in (
                        "path_or_url", "artifact_path", "artifact_url",
                        "file_path", "path", "url", "image",
                    ):
                        candidate = value.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            artifact_refs.append(candidate.strip())
                            break
                    for nested in value.values():
                        if isinstance(nested, (dict, list, tuple)):
                            _collect_artifacts(nested)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        _collect_artifacts(item)

            _collect_artifacts(artifact_root)
            image_parts: list[dict] = []
            seen_refs: set[str] = set()
            for artifact_ref in artifact_refs:
                if artifact_ref in seen_refs:
                    continue
                seen_refs.add(artifact_ref)
                parsed = urlsplit(artifact_ref)
                if parsed.scheme in {"http", "https", "data"}:
                    image_url = artifact_ref
                elif parsed.scheme:
                    raise ProviderRuntimeError(
                        f"unsupported QC artifact scheme: {parsed.scheme!r}"
                    )
                else:
                    artifact_path = _Path(artifact_ref)
                    if not artifact_path.is_absolute() or not artifact_path.is_file():
                        raise ProviderRuntimeError(
                            "QC artifact is not an existing absolute file"
                        )
                    size = artifact_path.stat().st_size
                    if size <= 0 or size > 32 * 1024 * 1024:
                        raise ProviderRuntimeError(
                            f"QC artifact size outside allowed range: {size}"
                        )
                    mime = _mimetypes.guess_type(artifact_path.name)[0]
                    if not mime or not mime.startswith("image/"):
                        raise ProviderRuntimeError("QC artifact is not an image")
                    inline_mime, inline_bytes = self._encode_local_image(
                        artifact_path,
                        mime,
                    )
                    encoded = _base64.b64encode(inline_bytes).decode("ascii")
                    image_url = f"data:{inline_mime};base64,{encoded}"
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

            if artifact_root and not image_parts:
                raise ProviderRuntimeError("QC stage has artifacts but no usable image input")

            params = stage_info.get("params", {})
            param_str = ""
            if params:
                param_str = ", ".join(
                    f"{k}={v}"
                    for k, v in params.items()
                    if k not in (
                        "endpoint", "allow_http", "timeout", "headers",
                        "auth_header",
                    )
                )
            qc_instruction = (
                "Inspect every attached image against the target prompt and QC "
                "requirements. The image is attached to this message; do not ask "
                "the caller to provide it.\n"
                f"Target prompt: {original_prompt}\n"
            )
            if param_str:
                qc_instruction += f"QC requirements: {param_str}\n"
            qc_instruction += (
                "Return JSON only, with no Markdown fences or surrounding prose, "
                "using exactly this shape: "
                '{"qc_pass": true, "report": "specific visual findings"}. '
                "Set qc_pass to false if any requirement fails."
            )
            user_content: str | list[dict]
            if image_parts:
                user_content = [
                    *image_parts,
                    {"type": "text", "text": qc_instruction},
                ]
            else:
                user_content = qc_instruction
            messages: list[dict] = [
                {
                    "role": "system",
                    "content": (
                        "You are a visual quality-control assistant. Inspect the "
                        "attached image and return a structured JSON verdict."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

            # This is also the reviewed text-preparation transport. Non-QC
            # services must not receive a visual-verdict prompt just because
            # they share Gemma's OpenAI-compatible protocol.
            if stage_info.get("kind") == "prompt_enhance":
                enhanced = pipeline_ctx.get("enhanced_prompts") or {}
                if enhanced:
                    original_prompt = list(enhanced.values())[-1]
                messages = [
                    {"role": "system", "content": (
                        "Turn the supplied brief into a clear, faithful production prompt. "
                        "Preserve constraints and do not invent facts. Return JSON only: "
                        '{"prompt": "the refined prompt"}.'
                    )},
                    {"role": "user", "content": (
                        str(original_prompt) + "\nPreparation requirements: " + param_str
                    )},
                ]

            oai_body = {
                "model": self._model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
            # input_tokens_hint: broker-admission token budget counts text only,
            # excluding base64 image bytes so a large multimodal payload does not
            # cause NO_ELIGIBLE_MEMBER.
            oai_body["input_tokens_hint"] = _text_only_input_tokens_hint(messages)
            body_bytes = _json.dumps(oai_body).encode("utf-8") if isinstance(oai_body, dict) else b""
        else:
            body_bytes = b""

        status, raw_body, resp_headers = await self._inner(
            dict(headers) if headers else {},
            method,
            url,
            body_bytes,
            timeout.total if hasattr(timeout, "total") else (timeout or 30.0),
        )

        # Parse response — expect OpenAI error shape or completion
        try:
            resp_data = _json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            resp_data = {}

        stage_id = stage_info.get("id", "unknown")

        if status == 200:
            # Accept both the native OpenAI chat-completion wire shape and the
            # already-normalized pipeline result shape used by hermetic
            # transports.  The latter is important for adapter composition:
            # it must not be converted into the default qc_pass=True merely
            # because it has no ``choices`` array.
            content = ""
            embedded_status = "ok"
            embedded_data: dict = {}
            if isinstance(resp_data, dict) and (
                "stage_id" in resp_data or "data" in resp_data
            ):
                raw_data = resp_data.get("data", {})
                embedded_data = dict(raw_data) if isinstance(raw_data, dict) else {}
                embedded_status = str(resp_data.get("status", "ok"))
                content = str(embedded_data.get("content", ""))
            else:
                embedded_data = {}
                if isinstance(resp_data, dict):
                    choices = resp_data.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("message", {})
                        content = delta.get("content", "")
                        # Parse the QC result from the content string.
                        # The QC provider embeds its result as a JSON string in
                        # content.  If that string contains a "status" field, use
                        # it to propagate the failure signal; otherwise default
                        # to "ok" for pass-through providers.
                        if content:
                            try:
                                json_content = content.strip()
                                if json_content.startswith(("```json\n", "```\n")) and json_content.endswith("```"):
                                    json_content = json_content.split("\n", 1)[1][:-3].strip()
                                parsed = _json.loads(json_content)
                                if isinstance(parsed, dict):
                                    embedded_status = str(parsed.get("status", "ok"))
                                    embedded_data = dict(parsed)
                                    # Propagate a qc_pass=false verdict as a failed gate,
                                    # mirroring the logic used for native OAI responses.
                                    # This ensures the correction loop is triggered when
                                    # the external QC provider embeds its verdict as
                                    # {"qc_pass": false} in the content string.
                                    if embedded_data.get("qc_pass") is False:
                                        embedded_status = "failed"
                            except Exception:
                                pass  # non-JSON content fails the QC gate below
            invalid_qc_verdict = (
                stage_info.get("kind") == "qc"
                and not isinstance(embedded_data.get("qc_pass"), bool)
            )
            invalid_prompt = (stage_info.get("kind") == "prompt_enhance" and (
                not isinstance(embedded_data.get("prompt"), str)
                or not embedded_data.get("prompt", "").strip()
            ))
            if invalid_prompt:
                embedded_status = "failed"
            if invalid_qc_verdict:
                embedded_data.pop("qc_pass", None)
                embedded_status = "failed"
            # A false QC verdict is a failed quality gate.  Keep the structured
            # verdict in the data payload and expose the failed status so
            # dependency gates can prevent downstream stages from proceeding.
            if embedded_data.get("qc_pass") is False:
                embedded_status = "failed"
            return _OpenAIResponse(
                status=status,
                body=_json.dumps({
                    "stage_id": stage_id,
                    "status": embedded_status,
                    "data": {**embedded_data, "content": content, "raw": resp_data},
                    "error": (
                        "QC provider response missing boolean qc_pass"
                        if invalid_qc_verdict else
                        "Preparation provider response missing prompt" if invalid_prompt else ""
                    ),
                }).encode("utf-8"),
                headers=resp_headers,
            )
        else:
            error_value = resp_data.get("error") if isinstance(resp_data, dict) else None
            if isinstance(error_value, dict):
                error_msg = str(error_value.get("message") or f"HTTP {status}")
            elif error_value is not None:
                error_msg = str(error_value)
            else:
                error_msg = f"HTTP {status}"
            return _OpenAIResponse(
                status=status,
                body=_json.dumps({
                    "stage_id": stage_id,
                    "status": "failed",
                    "data": {},
                    "error": _redact_secrets(error_msg),
                }).encode("utf-8"),
                headers=resp_headers,
            )


class _OpenAIResponse:
    """Minimal stand-in for aiohttp.Response to satisfy HttpPipelineProvider.dispatch."""
    def __init__(self, status: int, body: bytes, headers: dict) -> None:
        self.status = status
        self._body = body
        self._headers = headers

    @property
    def headers(self) -> dict:
        return self._headers

    @property
    def reason(self) -> str:
        return ""

    async def json(self) -> Any:
        return _json.loads(self._body.decode("utf-8"))

    async def text(self) -> str:
        return self._body.decode("utf-8")

    async def __aenter__(self) -> "_OpenAIResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def close(self) -> None:
        pass


# ── Typed dicts for config ────────────────────────────────────────────────────

class PipelineProviderConfig(TypedDict, total=False):
    """Shape of a single entry in pipeline_providers."""
    adapter: str
    capabilities: list[str]
    enabled: bool
    endpoint_ref: str
    request_path: str


# ── Declarative HTTP flow contract ────────────────────────────────────────────

class HttpFlowPollConfig(TypedDict, total=False):
    """
    Declarative poll-step configuration for ``http_flow`` adapter.

    All fields are optional and have safe defaults — the config is
    fail-closed on required fields at dispatch time.
    """
    # Path appended to service base URL for the poll request.
    # Supports {run_id} interpolation from the submit response.
    path: str
    # HTTP method for the poll request.
    method: str
    # JSONPath-like dotted path into the poll response whose value is compared
    # against ``status_values`` to determine completion.
    # Example: "data.status" extracts resp["data"]["status"].
    status_path: str
    # Sequence of string values (in order of progression) that ``status_path``
    # visits during the async operation.  The last value is the terminal "done"
    # state.  Any value not in this list that is encountered is treated as a
    # polling failure (timeout or unexpected state).
    status_values: list[str]
    # Optional fixed interval between poll requests, in seconds.
    poll_interval: float
    # Optional timeout in seconds for the entire poll/fetch phase.
    timeout: float


class HttpFlowFetchConfig(TypedDict, total=False):
    """
    Declarative fetch-step configuration for ``http_flow`` adapter.
    """
    # Path appended to service base URL for the fetch request.
    # Supports {run_id} interpolation.
    path: str
    # HTTP method for the fetch request.
    method: str
    # Dotted path into the fetch response whose value is extracted as the
    # stage result data.  If the extracted value is not a dict, it is wrapped
    # as ``{"value": <extracted>}``.
    extract_path: str
    # Optional selector: when the fetch returns a collection, select one object
    # by matching a field value against a previously captured identifier.
    # When absent, the entire fetch response is used as-is.
    select: "HttpFlowSelectConfig"


class HttpFlowSelectConfig(TypedDict, total=False):
    """
    Declarative selector for choosing one object from a fetched collection.

    Used inside ``HttpFlowFetchConfig.select`` when the fetch endpoint returns
    a list and the desired result is a specific item identified by metadata
    captured during earlier flow steps (e.g. ``submit_response`` IDs).
    """
    # Dotted path to the collection within the fetch response body.
    # Example: ``"messages"`` extracts ``body["messages"]``.
    collection_path: str
    # Dotted path to the matching field within each collection item.
    # Example: ``"id"`` extracts ``item["id"]``.
    match_path: str
    # Reference to the value to match, in ``${}`` template syntax.
    # Supports ``submit_response.<field>`` and ``path_values.<key>``.
    # Example: ``"submit_response.assistantMessageId"``.
    match_value_ref: str


class HttpFlowConfig(TypedDict, total=False):
    """
    Declarative multi-step HTTP flow for ``http_flow`` adapter.

    Describes a bounded request → poll → fetch flow entirely through config.
    All step definitions are optional; when absent the step is skipped.

    Example (services.json)::

        "my-provider": {
            "adapter": "http_flow",
            "request": {
                "method": "POST",
                "path": "/api/runs",
                "status_path": "data.status",
                "status_values": ["pending", "running", "completed"],
                "poll_interval": 1.0,
                "timeout": 60.0
            },
            "poll": {
                "path": "/api/runs/{run_id}/status",
                "method": "GET",
                "status_path": "data.status",
                "status_values": ["pending", "running", "completed"],
                "poll_interval": 1.0,
                "timeout": 60.0
            },
            "fetch": {
                "path": "/api/runs/{run_id}/result",
                "method": "GET",
                "extract_path": "data.result"
            }
        }
    """
    # Request-step configuration.  When absent, behaves identically to
    # ``http_json`` (synchronous POST + immediate 2xx response is the result).
    request: dict  # dict form to avoid circular-ref with TypedDict
    # Explicit poll-step configuration.  Required when the async flow has
    # a distinct poll phase (separate from request).  When absent and the
    # request does not return a terminal status, fail-closed
    # ``ProviderRuntimeError`` is raised at dispatch time.
    poll: HttpFlowPollConfig
    # Explicit fetch-step configuration.  Required when the async flow has
    # a distinct fetch phase.  When absent and the poll step returned a
    # non-terminal status, fail-closed ``ProviderRuntimeError`` is raised.
    fetch: HttpFlowFetchConfig


# ── Main runtime ──────────────────────────────────────────────────────────────

_PROVIDER_ADAPTERS = frozenset({
    "http_json",
    "http_openai_compatible",
    "http_flow",
    "gpu_manager_generation",
    "local_pipeline",
})


def _resolve_http_provider_timing(cfg: dict) -> tuple[float, float, float, float]:
    """Validate bounded HTTP request and retry timing from provider config."""
    try:
        request_timeout = float(cfg.get("request_timeout_seconds", 30.0))
        retry_delay = float(cfg.get("retry_delay_seconds", 0.0))
        retry_backoff = float(cfg.get("retry_backoff_multiplier", 1.0))
        retry_max_delay = float(cfg.get("retry_max_delay_seconds", retry_delay))
    except (TypeError, ValueError) as exc:
        raise ProviderRuntimeError(f"invalid HTTP provider timing value: {exc}") from exc

    if not 1.0 <= request_timeout <= 900.0:
        raise ProviderRuntimeError("request_timeout_seconds must be within [1, 900]")
    if not 0.0 <= retry_delay <= 300.0:
        raise ProviderRuntimeError("retry_delay_seconds must be within [0, 300]")
    if not 1.0 <= retry_backoff <= 10.0:
        raise ProviderRuntimeError("retry_backoff_multiplier must be within [1, 10]")
    if not retry_delay <= retry_max_delay <= 300.0:
        raise ProviderRuntimeError(
            "retry_max_delay_seconds must be within [retry_delay_seconds, 300]"
        )
    return request_timeout, retry_delay, retry_backoff, retry_max_delay


class PipelineProviderRuntime:
    """
    Service-registry-driven provider runtime for arbitrary pipeline providers.

    This runtime resolves named providers from the GPU Manager's ``services.json``
    configuration and produces concrete executors for the ``http_json`` and
    ``http_openai_compatible`` adapters, and typed GPU generation adapters
    for the ``gpu_manager_generation`` adapter.

    Parameters
    ----------
    services_config : dict
        The parsed services.json dict. Required top-level keys:
        ``pipeline_providers`` (mapping of provider_id → config) and
        ``services`` (mapping of service_name → service config with endpoint/port).
    provider_registry : frozenset[str]
        The authoritative set of known/approved provider IDs.
        Unknown provider IDs in stages raise ``UnknownProviderError``.
    """

    def __init__(
        self,
        services_config: dict,
        provider_registry: frozenset[str],
    ) -> None:
        self._services: dict[str, Any] = dict(services_config.get("services", {}))

        # pipeline_providers: always from the canonical key
        self._providers: dict[str, PipelineProviderConfig] = dict(
            services_config.get("pipeline_providers", {})
        )

        # Pipelines: support both 'service_pipelines' (canonical) and legacy
        # 'pipelines' keys, with deterministic precedence.
        #
        # Precedence: 'service_pipelines' wins when both are absent/present
        # (the real services.json format uses 'service_pipelines').
        # If a pipeline ID appears under BOTH keys, raise fail-closed
        # DuplicatePipelineError to prevent silent shadowing.
        legacy = services_config.get("pipelines", {})
        canonical = services_config.get("service_pipelines", {})

        if legacy and canonical:
            overlap = set(legacy.keys()) & set(canonical.keys())
            if overlap:
                raise DuplicatePipelineError(
                    f"pipeline IDs appear under both 'pipelines' and 'service_pipelines': "
                    f"{sorted(overlap)}; remove the duplicate key(s) from 'pipelines'"
                )

        # service_pipelines takes precedence; fall back to pipelines only when
        # canonical is absent (legacy callers that never populated service_pipelines).
        self._pipelines: dict[str, dict] = (
            dict(canonical) if canonical else dict(legacy)
        )
        self._registry = provider_registry

    # ── Provider resolution ─────────────────────────────────────────────────

    def _resolve_provider_config(self, provider_id: str) -> PipelineProviderConfig:
        """Resolve and validate a provider_id against registry + config."""
        if provider_id not in self._registry:
            raise UnknownProviderError(
                f"provider {provider_id!r} not in provider registry; "
                f"known providers: {sorted(self._registry)}"
            )
        cfg = self._providers.get(provider_id)
        if cfg is None:
            raise UnknownProviderError(
                f"provider {provider_id!r} not found in pipeline_providers config"
            )
        if not cfg.get("enabled", True):
            raise DisabledProviderError(
                f"provider {provider_id!r} is disabled"
            )
        return cfg

    def _resolve_url(self, provider_id: str, cfg: PipelineProviderConfig) -> str:
        """Build the full URL for a provider.

        For http_json / http_openai_compatible adapters: returns the resolved
        full URL (endpoint + request_path).

        For gpu_manager_generation adapters: returns "" (the adapter routes
        internally via the GPU Manager generation API), but the endpoint_ref
        is still validated fail-closed — an unknown or disabled endpoint_ref
        raises EndpointResolutionError / DisabledProviderError rather than
        silently returning "" with no validation.
        """
        adapter = cfg.get("adapter", "")

        # Validate endpoint_ref canonical existence even for gpu_manager_generation.
        # This ensures a stale/mismatched endpoint_ref fails at resolution time,
        # not silently at dispatch time.
        endpoint_ref = cfg.get("endpoint_ref", "")
        if endpoint_ref:
            _resolve_service_endpoint({"services": self._services}, endpoint_ref)

        if adapter == "gpu_manager_generation":
            # gpu_manager_generation has no request_path — it's handled separately
            # via the GPU Manager generation API (submit/poll/cancel).
            return ""
        if adapter == "local_pipeline":
            # local_pipeline resolves a named contract from the config registry;
            # it does not make an HTTP request or follow an arbitrary URL.
            return ""
        return _build_provider_url(
            {"services": self._services},
            cfg,
        )

    # ── Stage kind validation ──────────────────────────────────────────────

    def validate_stage_kind(self, provider_id: str, stage_kind: str) -> None:
        """
        Validate that ``stage_kind`` is declared in the provider's capabilities.

        Raises ``KindCapabilityError`` fail-closed if the kind is not declared.
        Raises ``UnknownProviderError`` / ``DisabledProviderError`` as applicable.
        """
        cfg = self._resolve_provider_config(provider_id)
        _check_capabilities(cfg, stage_kind, provider_id)

    def validate_pipeline_stages(
        self,
        provider_id: str,
        stages: list[dict],
    ) -> None:
        """
        Validate every stage in ``stages`` against the provider's capabilities.

        Raises ``KindCapabilityError`` fail-closed on the first stage kind
        not in the provider's declared capabilities.
        Raises ``UnknownProviderError`` / ``DisabledProviderError`` as applicable.
        """
        cfg = self._resolve_provider_config(provider_id)
        for stage in stages:
            kind = stage.get("kind", "")
            if kind not in _CLOSED_KINDS:
                raise DispatchKindError(f"unknown stage kind {kind!r}")
            _check_capabilities(cfg, kind, provider_id)

    # ── Executor factory ────────────────────────────────────────────────────

    def build_executor(
        self,
        provider_id: str,
        contract: dict,
        transport: Any,  # async transport callable used by HttpPipelineProvider
    ) -> "KreaLikeExecutor":
        """
        Build a pipeline executor for ``provider_id`` using the given transport.

        The executor is one of:
          - ``HttpPipelineExecutor``  — for ``http_json`` and
            ``http_openai_compatible`` adapters.
          - ``GpuGenerationExecutor`` — for ``gpu_manager_generation`` adapter.

        Parameters
        ----------
        provider_id : str
            Named provider (e.g. ``"open-design"``, ``"combined-gemma"``).
        contract : dict
            The compiled worker pipeline dict (from
            ``service_pipeline.compile_generation_pipeline``).
        transport : callable
            An async callable matching the ``AdapterTransportCallable`` signature
            (headers, method, url, body, timeout) -> Awaitable[AdapterHttpResponse].
            Used for http_json / http_openai_compatible providers.

        Returns
        -------
        KreaLikeExecutor
            An async executor with a ``dispatch(stages, context)`` method
            returning ``tuple[StageContext, list[StageResult]]``.

        Raises
        ------
        UnknownProviderError
            Provider ID not in registry.
        DisabledProviderError
            Provider is explicitly disabled.
        EndpointResolutionError
            endpoint_ref not found in services.
        KindCapabilityError
            A stage kind is not in the provider's capabilities.
        """
        cfg = self._resolve_provider_config(provider_id)
        adapter = cfg.get("adapter", "")

        if adapter not in _PROVIDER_ADAPTERS:
            raise ProviderRuntimeError(
                f"provider {provider_id!r} has unsupported adapter {adapter!r}; "
                f"supported: {sorted(_PROVIDER_ADAPTERS)}"
            )

        if adapter == "gpu_manager_generation":
            raise ProviderRuntimeError(
                f"provider {provider_id!r} uses gpu_manager_generation adapter; "
                f"use build_gpu_adapter() instead"
            )

        # Validate all stages against provider capabilities before building
        stages = contract.get("stages", [])
        self.validate_pipeline_stages(provider_id, stages)

        # Build the URL and wrap the transport
        url = self._resolve_url(provider_id, cfg)
        (
            request_timeout,
            retry_delay,
            retry_backoff,
            retry_max_delay,
        ) = _resolve_http_provider_timing(cfg)

        if adapter == "http_openai_compatible":
            wrapped_transport = _OpenAICompatibleTransport(
                transport,
                model=str(cfg.get("model") or "gemma-4"),
                max_inline_image_bytes=cfg.get("max_inline_image_bytes"),
            )
            provider = _HttpPipelineProviderWrapper(
                contract=contract,
                session=wrapped_transport,
                endpoint_url=url,
                request_timeout_seconds=request_timeout,
            )
        elif adapter == "http_flow":
            # http_flow: declarative request/poll/fetch flow defined entirely in cfg
            provider = _HttpFlowExecutor(
                provider_id=provider_id,
                provider_config=cfg,
                services_config={"services": self._services},
                session=transport,
                endpoint_url=url,
            )
        else:
            # http_json
            provider = _HttpPipelineProviderWrapper(
                contract=contract,
                session=transport,
                endpoint_url=url,
                request_timeout_seconds=request_timeout,
            )

        return _HttpExecutorWrapper(
            provider,
            retry_delay_seconds=retry_delay,
            retry_backoff_multiplier=retry_backoff,
            retry_max_delay_seconds=retry_max_delay,
        )

    # ── GPU generation adapter factory ─────────────────────────────────────

    def build_gpu_adapter(
        self,
        provider_id: str,
        transport: Any,
    ) -> "GPUMgrLikeAdapter":
        """
        Build a GPU generation adapter for ``provider_id``.

        Returns an object with:
          - ``submit(assignment, *, state_store=None, params=None) -> GPUMgrSubmitReceipt``
          - ``poll(job_id, deadline_seconds=None) -> GPUMgrJobSnapshot``
          - ``cancel(job_id) -> bool``

        Parameters
        ----------
        provider_id : str
            Named provider (e.g. ``"comfyui"``, ``"comfyui-qwen"``,
            ``"acestep-candidate"``).
        transport : AdapterTransportCallable
            An async callable matching the ``AdapterTransportCallable`` signature.

        Raises
        ------
        UnknownProviderError
            Provider ID not in registry or not a gpu_manager_generation provider.
        DisabledProviderError
            Provider is disabled.
        KindCapabilityError
            A stage kind is not in the provider's capabilities.
        """
        cfg = self._resolve_provider_config(provider_id)
        adapter_type = cfg.get("adapter", "")
        if adapter_type != "gpu_manager_generation":
            raise ProviderRuntimeError(
                f"provider {provider_id!r} has adapter {adapter_type!r}, not gpu_manager_generation; "
                f"use build_executor() instead"
            )

        # gpu_manager_generation providers use a fixed base URL (always 127.0.0.1:8091)
        # and the endpoint_ref names the service in the gpu_manager's own service registry.
        endpoint_ref = cfg.get("endpoint_ref", provider_id)

        return _GpuMgrAdapterWrapper(
            provider_id=provider_id,
            endpoint_ref=endpoint_ref,
            transport=transport,
            config={
                "endpoint_ref": endpoint_ref,
                "capabilities": cfg.get("capabilities", []),
            },
        )

    # ── Multi-provider executor ───────────────────────────────────────────────

    def build_multi_provider_executor(
        self,
        contract: dict,
        transport_map: dict[str, Any],
        *,
        stage_executor_factory: StageExecutorFactory | None = None,
        max_delegate_depth: int = 3,
        max_retries: int = 3,
        max_corrections: int | None = None,
    ) -> "CompositeProviderExecutor":
        """
        Build a stage-wise multi-provider executor from a single contract.

        Unlike ``build_executor`` which is bound to one provider, this method
        accepts a ``transport_map`` covering all providers referenced in the
        pipeline.  Each stage is dispatched to the executor for its declared
        ``provider`` field, enabling pipelines that span multiple providers.

        When ``stage_executor_factory`` is provided, gpu_manager_generation
        providers are also supported.  The factory is called with
        ``(provider_id, transport)`` and must return a
        ``GenerationStageExecutor`` that implements the typed dispatch
        interface.  If a gpu_manager_generation provider is encountered
        without a factory, ``MissingExecutorFactoryError`` is raised
        fail-closed.

        Parameters
        ----------
        contract : dict
            The compiled worker pipeline dict (from
            ``service_pipeline.compile_generation_pipeline``).
        transport_map : dict[str, callable]
            Mapping of provider_id → async transport callable.  Required keys
            are exactly the set of provider IDs referenced in ``contract``.
            All referenced providers must be present; missing keys raise
            ``UnknownProviderError`` at dispatch time.
        stage_executor_factory : StageExecutorFactory | None
            Optional factory callback for gpu_manager_generation providers.
            Signature: ``(provider_id: str, transport) -> GenerationStageExecutor``.
            If None, gpu_manager_generation providers cannot be used and
            attempting to use one raises ``MissingExecutorFactoryError``.
        max_corrections : int | None
            Override for the pipeline's ``max_loop_count`` / ``max_corrections``
            bound.  When set (the normal case for nested pipeline dispatch),
            this cap is passed to child ``CompositeProviderExecutor`` instances
            so that correction loops in delegated pipelines cannot exceed the
            parent's bound regardless of the child's own contract value.
            When None (the top-level default), the value is read from
            ``contract.get("max_corrections", contract.get("max_loop_count", 1))``.

        Returns
        -------
        CompositeProviderExecutor
            An async executor with a ``dispatch(stages, context)`` method.

        Raises
        -------
        KindCapabilityError
            A stage kind is not in the provider's declared capabilities.
        UnknownProviderError
            A provider ID in ``transport_map`` is not in the registry or is
            disabled.
        DisabledProviderError
            A provider in ``transport_map`` is explicitly disabled.
        MissingExecutorFactoryError
            A gpu_manager_generation provider is referenced but
            ``stage_executor_factory`` was not provided.
        """
        # Pre-build one executor per transport-map entry, validating each
        # provider.  The contract stages are NOT validated here because
        # dispatch() validates per-stage kind vs provider capabilities at
        # runtime (enabling lazy/gated contracts that may add stages later).
        executor_map: dict[str, Any] = {}
        gpu_executor_map: dict[str, GenerationStageExecutor] = {}
        for provider_id, transport in transport_map.items():
            cfg = self._resolve_provider_config(provider_id)
            adapter = cfg.get("adapter", "")
            if adapter == "gpu_manager_generation":
                if stage_executor_factory is None:
                    raise MissingExecutorFactoryError(
                        f"provider {provider_id!r} uses gpu_manager_generation adapter "
                        f"but stage_executor_factory was not provided; "
                        f"pass stage_executor_factory to allow gpu_manager_generation stages"
                    )
                gpu_executor_map[provider_id] = stage_executor_factory(
                    provider_id, transport
                )
            elif adapter == "local_pipeline":
                # local_pipeline adapters resolve a named child pipeline from the
                # config registry and dispatch it via _dispatch_local_pipeline.
                # They do NOT make HTTP requests and do NOT use the HTTP executor
                # path.  Do NOT add them to executor_map — they are handled
                # exclusively by _dispatch_local_pipeline in CompositeProviderExecutor.
                # An empty URL is a symptom of the bug: the local_pipeline adapter
                # would otherwise fall through to the HTTP path and make a bogus POST
                # to an empty URL (or to http://127.0.0.1:NONE/run if a default port
                # were inferred).  Fail-closed by explicitly skipping the URL path
                # and not registering this provider in executor_map.
                continue
            else:
                # http_json / http_openai_compatible / http_flow need a resolved URL.
                # Worker-owned HTTP providers (open-design, combined-gemma) may lack
                # endpoint_ref in their config (they use the ComfyUI/OD worker path,
                # not the HTTP transport path).  Skip building an HTTP executor for
                # such providers so build_multi_provider_executor does not crash on
                # RECONCILE pipelines that contain them in transport_map.
                try:
                    url = self._resolve_url(provider_id, cfg)
                except EndpointResolutionError:
                    # Provider lacks endpoint_ref → it is worker-owned via HTTP adapter.
                    # Do not add it to executor_map so dispatch raises on use.
                    continue
                # Extract retry timing from provider config (same logic as build_executor)
                (
                    _timing_req_timeout,
                    _timing_retry_delay,
                    _timing_retry_backoff,
                    _timing_retry_max_delay,
                ) = _resolve_http_provider_timing(cfg)
                if adapter == "http_openai_compatible":
                    wrapped = _OpenAICompatibleTransport(
                        transport,
                        model=str(cfg.get("model") or "gemma-4"),
                        max_inline_image_bytes=cfg.get("max_inline_image_bytes"),
                    )
                    provider = _HttpPipelineProviderWrapper(
                        contract=contract,
                        session=wrapped,
                        endpoint_url=url,
                        request_timeout_seconds=_timing_req_timeout,
                    )
                elif adapter == "http_flow":
                    provider = _HttpFlowExecutor(
                        provider_id=provider_id,
                        provider_config=cfg,
                        services_config={"services": self._services},
                        session=transport,
                        endpoint_url=url,
                    )
                else:
                    provider = _HttpPipelineProviderWrapper(
                        contract=contract,
                        session=transport,
                        endpoint_url=url,
                        request_timeout_seconds=_timing_req_timeout,
                    )
                executor_map[provider_id] = _HttpExecutorWrapper(
                    provider,
                    retry_delay_seconds=_timing_retry_delay,
                    retry_backoff_multiplier=_timing_retry_backoff,
                    retry_max_delay_seconds=_timing_retry_max_delay,
                )

        # Also pre-build gpu executors for gpu_manager_generation providers
        # referenced in the contract stages but NOT in transport_map.
        # These providers route internally via the GPU Manager generation API
        # and do not use the transport_map transport.
        contract_stages = contract.get("stages", [])
        for stage in contract_stages:
            provider_id = str(stage.get("provider", ""))
            if not provider_id:
                continue
            if provider_id in gpu_executor_map:
                continue  # already built above
            try:
                cfg = self._resolve_provider_config(provider_id)
            except (UnknownProviderError, DisabledProviderError):
                continue  # will raise at dispatch time if actually used
            adapter = cfg.get("adapter", "")
            if adapter == "gpu_manager_generation":
                if stage_executor_factory is None:
                    raise MissingExecutorFactoryError(
                        f"provider {provider_id!r} uses gpu_manager_generation adapter "
                        f"but stage_executor_factory was not provided; "
                        f"pass stage_executor_factory to allow gpu_manager_generation stages"
                    )
                # gpu_manager_generation providers don't use transport_map transport;
                # pass None so the factory can use its own transport strategy.
                gpu_executor_map[provider_id] = stage_executor_factory(
                    provider_id, None
                )

        return CompositeProviderExecutor(
            executor_map=executor_map,
            gpu_executor_map=gpu_executor_map,
            runtime=self,
            max_corrections=max_corrections
            if max_corrections is not None
            else contract.get("max_corrections", contract.get("max_loop_count", 1)),
            max_retries=max_retries,
            transport_map=transport_map,
            stage_executor_factory=stage_executor_factory,
            max_delegate_depth=max_delegate_depth,
            loops=contract.get("loops"),
            execution_mode=contract.get("execution_mode", "generic"),
        )

    # ── Diagnostics ────────────────────────────────────────────────────────

    def provider_summary(self) -> dict[str, Any]:
        """Return a read-only summary of resolved providers for diagnostics."""
        out = {}
        for pid, cfg in self._providers.items():
            try:
                url = self._resolve_url(pid, cfg)
            except (EndpointResolutionError, DisabledProviderError):
                url = "<unresolvable>"
            out[pid] = {
                "adapter": cfg.get("adapter", ""),
                "enabled": cfg.get("enabled", False),
                "capabilities": cfg.get("capabilities", []),
                "endpoint_ref": cfg.get("endpoint_ref", ""),
                "resolved_url": url,
                "in_registry": pid in self._registry,
            }
        return out


# ── Internal executor/adapter wrappers ────────────────────────────────────────
# These bridge the HttpPipelineProvider / GPUMgrGenerationAdapter interfaces
# to the stage_dispatch contract.

def _result_has_artifact(result: "StageResult") -> bool:
    """
    Return True if the StageResult contains an artifact in its data.

    An artifact is considered present if the data dict is non-empty AND
    contains at least one of the common artifact keys (artifact_path,
    artifact_url, file_path).  This prevents a stage that returns e.g.
    {{"qc_pass": True}} from being mistaken for a generate-stage artifact.
    """
    if not isinstance(result.data, dict) or not result.data:
        return False
    return any(
        k in result.data for k in ("artifact_path", "artifact_url", "file_path")
    )


def _qc_result_is_failing(result: "StageResult") -> bool:
    """
    Return True if the stage result carries QC-fail evidence.

    A QC result is "failing" when:
      - kind == "qc"
      - data contains qc_pass=False
      - status is anything (ok, failed, or skipped)

    This captures the QC verdict even when status has been promoted to
    "failed" by _HttpPipelineProviderWrapper (qc_pass=false → status=failed)
    or when a QC stage is seeded from a worker-owned result.
    """
    if result.kind != "qc":
        return False
    if not isinstance(result.data, dict):
        return False
    return result.data.get("qc_pass") is False


def _join_flow_url(base_url: str, path: str) -> str:
    """Join a declarative flow endpoint and path without duplicating prefixes.

    A provider may declare either a host-only endpoint plus a path, or an
    endpoint that already includes the service prefix.  Both forms are valid;
    a path equal to (or below) the endpoint path must not be appended twice.
    Absolute paths remain absolute so a flow can deliberately target a
    different route on the same configured service.
    """
    base = str(base_url or "").strip()
    raw_path = str(path or "/").strip()
    if not raw_path:
        raw_path = "/"
    if raw_path.startswith(("http://", "https://")):
        return raw_path

    base_parts = urlsplit(base)
    path_parts = urlsplit(raw_path)
    requested_path = path_parts.path or "/"
    base_path = base_parts.path.rstrip("/")

    if raw_path.startswith("/"):
        # URL-root-relative flow paths replace any default provider request path.
        # This lets one declarative flow POST to e.g. /api/runs and later fetch
        # from a sibling route such as /api/projects/... without producing
        # /api/runs/api/projects/....
        joined_path = requested_path
    elif base_path and (
        requested_path == base_path
        or requested_path.startswith(base_path + "/")
    ):
        joined_path = requested_path
    elif base_path:
        joined_path = base_path + "/" + requested_path.lstrip("/")
    else:
        joined_path = "/" + requested_path.lstrip("/")

    return urlunsplit((
        base_parts.scheme,
        base_parts.netloc,
        joined_path,
        path_parts.query,
        path_parts.fragment,
    ))


class _CallableTransportAdapter:
    """
    Wrap a bare ``__call__``-only transport (e.g. ``_AiohttpTransport``)
    so it presents the ``request()`` interface that ``_HttpPipelineProviderWrapper``
    expects.

    Transport signature (unchanged)::

        (headers, method, url, body_bytes, timeout) -> Awaitable[tuple[int, bytes, dict]]

    This adapter bridges to the session.request(method, url, *, json=..., headers=..., timeout=...)
    call by unpacking the json body and re-packaging the raw tuple response
    into a minimal aiohttp-Response-like object.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict | None = None,
        headers: dict | None = None,
        timeout: Any = None,
    ):
        """Translate a session.request() call into the inner transport's __call__()."""
        body_bytes = _json.dumps(json).encode("utf-8") if json is not None else b""
        raw_timeout = timeout.total if hasattr(timeout, "total") else (timeout or 30.0)

        status, raw_body, resp_headers = await self._inner(
            dict(headers) if headers else {},
            method,
            url,
            body_bytes,
            raw_timeout,
        )

        return _MinimalResponse(status=status, body=raw_body, headers=resp_headers)


class _MinimalResponse:
    """Minimal stand-in for aiohttp.Response used by _CallableTransportAdapter."""

    def __init__(self, status: int, body: bytes, headers: dict) -> None:
        self.status = status
        self._body = body
        self._headers = headers

    @property
    def headers(self) -> dict:
        return self._headers

    @property
    def reason(self) -> str:
        return ""

    async def json(self) -> Any:
        return _json.loads(self._body.decode("utf-8")) if self._body else {}

    async def text(self) -> str:
        return self._body.decode("utf-8")

    async def __aenter__(self) -> "_MinimalResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def close(self) -> None:
        pass


class _HttpPipelineProviderWrapper:
    """
    Internal wrapper that binds an injectable transport to the
    HttpPipelineProvider interface so the runtime stays agnostic to
    the exact HTTP stack.
    """

    def __init__(
        self,
        contract: dict,
        session: Any,
        endpoint_url: str,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self._contract = contract
        # Bridge the transport interface mismatch:
        # - _AiohttpTransport implements __call__ only.
        # - _HttpPipelineProviderWrapper calls session.request().
        # Detect a bare __call__ transport and wrap it.
        if callable(session) and not hasattr(session, "request"):
            session = _CallableTransportAdapter(session)
        self._session = session
        self._endpoint_url = endpoint_url
        self._request_timeout_seconds = float(request_timeout_seconds)

    async def dispatch(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        """
        Dispatch via the injectable transport + HTTP call using the pre-resolved URL.

        The URL is pre-resolved from the service registry at build_executor() time,
        so we do NOT read ``endpoint`` from stage params (which is what
        ``HttpPipelineProvider`` does — that would fail since stage params
        only contain user-facing stage parameters, not the resolved service URL).
        """
        stage_map = {s["id"]: s for s in self._contract.get("stages", [])}
        stage = stage_map.get(stage_id)
        if stage is None:
            raise EndpointResolutionError(
                f"stage {stage_id!r} not found in pipeline contract"
            )

        if not self._endpoint_url:
            raise EndpointResolutionError(
                f"provider has no resolved URL (gpu_manager_generation adapter? "
                f"use build_gpu_adapter() instead)"
            )

        # Build the HTTP payload from the contract context
        # (mirrors HttpPipelineProvider._build_payload but without http-internal params)
        http_keys = frozenset(("endpoint", "allow_http", "timeout", "headers", "auth_header"))
        clean_params = {k: v for k, v in params.items() if k not in http_keys}

        payload = {
            "pipeline": {
                "schema": self._contract.get("schema", ""),
                "original_prompt": context.original_prompt,
            },
            "stage": {
                "id": stage_id,
                "kind": kind,
                "provider": provider,
                "params": clean_params,
                "depends_on": list(stage.get("depends_on", [])),
                "retries": stage.get("retries"),
            },
            "context": {
                "original_prompt": context.original_prompt,
                "enhanced_prompts": dict(context.enhanced_prompts),
                "artifacts": dict(context.artifacts),
                "qc_results": dict(context.qc_results),
            },
        }

        # Use the transport's request() interface (aiohttp-compatible)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        timeout = self._request_timeout_seconds

        response = await self._session.request(
            "POST",
            self._endpoint_url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        # Parse the response and convert to StageResult
        status = response.status
        try:
            response_body = await response.json()
        except Exception:
            response_body = None

        if not (200 <= status < 300):
            error_msg = ""
            if isinstance(response_body, dict):
                error_msg = str(response_body.get("error", f"HTTP {status}"))
            elif isinstance(response_body, str):
                error_msg = response_body[:500]
            else:
                error_msg = f"HTTP {status} {response.reason}"
            raise DispatchProviderError(
                f"provider HTTP error for stage {stage_id!r} at {self._endpoint_url}: "
                f"{_redact_secrets(error_msg)}"
            )

        if response_body is None or not isinstance(response_body, dict):
            raise DispatchProviderError(
                f"provider returned 2xx for stage {stage_id!r} but response is not JSON dict"
            )

        result_status = response_body.get("status", "ok")
        result_data = response_body.get("data", {})
        if not isinstance(result_data, dict):
            raise DispatchProviderError(
                f"provider returned non-object data for stage {stage_id!r}"
            )
        result_error = _redact_secrets(str(response_body.get("error", "")))

        # A generic HTTP 2xx response is not proof that the provider finished
        # the operation.  Treat only explicit terminal aliases as terminal;
        # queued/running/in-flight responses must fail closed unless the
        # provider is declared as an ``http_flow`` with a bounded poll/fetch
        # contract.  Otherwise the coordinator could ACK a stage that is still
        # executing remotely and has no durable resume handle.
        result_status = str(result_status or "ok").strip().lower()
        if result_status in {"completed", "success", "succeeded"}:
            result_status = "ok"
        elif result_status not in ("ok", "failed", "skipped"):
            raise DispatchProviderError(
                f"provider returned nonterminal status {result_status!r} "
                f"for stage {stage_id!r}; declare a bounded poll/fetch flow"
            )
        # A false QC verdict is an execution result with a failed quality gate.
        # Preserve the structured verdict in result.data, but expose status="failed"
        # so dependency gates can stop downstream stages until the configured
        # correction path explicitly handles the failure.
        if kind == "qc" and result_status == "ok" and result_data.get("qc_pass") is False:
            if not result_error:
                result_error = "quality gate returned qc_pass=false"
            result_status = "failed"

        return StageResult(
            stage_id=stage_id,  # use the parameter, not the response body stage_id
            kind=kind,
            provider=provider,
            status=result_status,
            data=dict(result_data),
            error=result_error,
            retries=0,
        )


def _update_context_from_result(
    stage_id: str,
    kind: str,
    result: "StageResult",
    context: "StageContext",
) -> "StageContext":
    """
    Update StageContext from a successful StageResult.

    Mirrors the update logic from CompositeProviderExecutor._update_context.
    This is extracted into a standalone helper so _HttpExecutorWrapper.dispatch
    can apply context updates without duplicating the logic.
    """
    if kind == "prompt_enhance":
        prompt = result.data.get("prompt", "")
        return context.with_prompt(stage_id, prompt)
    if kind in ("generate", "delegate"):
        return context.with_artifact(stage_id, dict(result.data))
    if kind == "qc":
        return context.with_qc(stage_id, dict(result.data))
    if kind == "correct":
        # Increment loop counter and record the correction
        return (context
                .with_loop_increment()
                .with_artifact(stage_id, dict(result.data)))
    return context


# ── http_flow: declarative multi-step HTTP executor ───────────────────────────

def _interpolate_path(
    path_template: str,
    run_id: str = "",
    variables: dict[str, Any] | None = None,
) -> str:
    """Replace declared path placeholders with URL-encoded values.

    ``run_id`` remains a compatibility argument for the original flow contract.
    Additional placeholders are supplied through ``variables`` and are resolved
    without provider-specific knowledge.
    """
    import re
    from urllib.parse import quote

    values = {"run_id": run_id}
    values.update(dict(variables or {}))

    def _replace(match):
        key = match.group(1)
        value = values.get(key)
        return quote(str(value), safe="") if value is not None else ""

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", _replace, str(path_template))


def _flow_lookup(root: Any, path: str) -> Any:
    """Resolve a dotted config path from mappings, objects, or lists."""
    if not path:
        return root
    current = root
    for segment in str(path).split("."):
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
            except (TypeError, ValueError):
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            current = getattr(current, segment, None)
        if current is None:
            return None
    return current


def _render_flow_value(
    value: Any,
    *,
    payload: dict,
    params: dict,
    context: StageContext,
    submit_response: Any = None,
    path_values: dict[str, Any] | None = None,
) -> Any:
    """Render a declarative flow value from stage/context/response data.

    Exact ``${path}`` values preserve the referenced type; embedded placeholders
    are converted to strings. Missing placeholders fail closed rather than being
    silently sent as the literal token.
    """
    import re

    roots: dict[str, Any] = {
        "stage": {"params": dict(params)},
        "context": payload.get("context", {}),
        "pipeline": payload.get("pipeline", {}),
        "submit_response": submit_response,
    }
    if path_values:
        roots["path_values"] = dict(path_values)
    exact = re.fullmatch(r"\$\{([^{}]+)\}", value) if isinstance(value, str) else None
    if exact:
        path = exact.group(1).strip()
        rendered = _flow_lookup(roots, path)
        if rendered is None:
            raise HttpFlowOperationError(
                f"http_flow template reference {path!r} is unavailable"
            )
        return rendered
    if isinstance(value, str):
        def _replace(match):
            path = match.group(1).strip()
            rendered = _flow_lookup(roots, path)
            if rendered is None:
                raise HttpFlowOperationError(
                    f"http_flow template reference {path!r} is unavailable"
                )
            return str(rendered)
        return re.sub(r"\$\{([^{}]+)\}", _replace, value)
    if isinstance(value, dict):
        return {
            key: _render_flow_value(
                item,
                payload=payload,
                params=params,
                context=context,
                submit_response=submit_response,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _render_flow_value(
                item,
                payload=payload,
                params=params,
                context=context,
                submit_response=submit_response,
            )
            for item in value
        ]
    return value


def _extract_field(data: Any, path: str) -> Any:
    """
    Extract a dotted-path field from a nested dict/list structure.

    ``path`` is a dot-separated sequence of keys/indexes.
    Example: ``"data.status"`` → ``data["data"]["status"]``
    Example: ``"items.0.name"`` → ``data["items"][0]["name"]``

    Returns ``None`` if any path segment is missing.
    """
    if not path:
        return data
    current: Any = data
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list):
            try:
                idx = int(segment)
                current = current[idx] if 0 <= idx < len(current) else None
            except ValueError:
                return None
        else:
            return None
    return current


class _HttpFlowExecutor:
    """
    Declarative multi-step HTTP executor driven entirely by provider config.

    Supports two modes:

    1. **Synchronous passthrough** — when neither ``poll`` nor ``fetch`` is
       declared in the provider config, the request is sent and its 2xx response
       is returned as the stage result exactly as ``http_json`` does today.

    2. **Bounded async flow** — request → poll (bounded) → fetch.
       All step parameters (method, path, body fields, poll interval, timeout,
       status extraction, response extraction) are declared in the provider config.
       The runtime raises ``HttpFlowOperationError`` fail-closed on any
       unexpected state, poll timeout, or fetch failure.

    Parameters
    ----------
    provider_id : str
        Named provider ID for error messages.
    provider_config : dict
        The full pipeline provider config dict (contains ``http_flow`` key).
    services_config : dict
        Subset of services.json needed for endpoint resolution.
    session : callable
        An async transport callable with a ``request()`` method.
    endpoint_url : str
        Pre-resolved base URL for the provider.
    """

    def __init__(
        self,
        provider_id: str,
        provider_config: dict,
        services_config: dict,
        session: Any,
        endpoint_url: str,
    ) -> None:
        self._provider_id = provider_id
        self._config: dict = dict(provider_config.get("http_flow") or {})
        self._services = dict(services_config.get("services", {}))
        # Bridge callable-only transport (same pattern as _HttpPipelineProviderWrapper)
        if callable(session) and not hasattr(session, "request"):
            session = _CallableTransportAdapter(session)
        self._session = session
        self._base_url = endpoint_url

    # ── Public dispatch interface ─────────────────────────────────────────────

    async def dispatch(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> "StageResult":
        """
        Dispatch a stage using the declarative http_flow contract.

        The method / path / body field mappings / poll / timeout / response
        extraction are all taken from the provider's ``http_flow`` config.
        No hardcoded provider names or IDs.
        """
        # Build the provider-neutral context payload.  The request body may be
        # replaced by a declarative template below; this payload remains the
        # source for `${context.*}` and `${pipeline.*}` references.
        http_keys = frozenset(("endpoint", "allow_http", "timeout", "headers", "auth_header"))
        clean_params = {k: v for k, v in params.items() if k not in http_keys}

        payload = {
            "pipeline": {
                "schema": "",
                "original_prompt": context.original_prompt,
            },
            "stage": {
                "id": stage_id,
                "kind": kind,
                "provider": provider,
                "params": clean_params,
            },
            "context": {
                "original_prompt": context.original_prompt,
                "enhanced_prompts": dict(context.enhanced_prompts),
                "artifacts": dict(context.artifacts),
                "qc_results": dict(context.qc_results),
            },
        }

        request_cfg: dict = self._config.get("request") or {}
        poll_cfg: dict = self._config.get("poll") or {}
        fetch_cfg: dict = self._config.get("fetch") or {}

        request_body = request_cfg.get("body", payload)
        request_body = _render_flow_value(
            request_body,
            payload=payload,
            params=clean_params,
            context=context,
        )
        if not isinstance(request_body, dict):
            raise HttpFlowOperationError(
                f"http_flow request body for stage {stage_id!r} must resolve to an object"
            )

        # ── Step 1: Request ───────────────────────────────────────────────────
        submit_resp = await self._do_request(request_body, request_cfg)

        # Identifiers are declarative.  The legacy ``run_id`` response field is
        # retained as the default so existing flow contracts remain valid.
        run_id_path = str(
            self._config.get("run_id_path", request_cfg.get("run_id_path", "run_id"))
        )
        run_id = _extract_field(submit_resp, run_id_path) if isinstance(submit_resp, dict) else None
        path_values: dict[str, Any] = {}
        if run_id is not None:
            path_values["run_id"] = run_id
        for key, path in dict(self._config.get("path_values") or {}).items():
            value = _render_flow_value(
                "${" + str(path) + "}",
                payload=payload,
                params=clean_params,
                context=context,
                submit_response=submit_resp,
            )
            path_values[str(key)] = value
        if "conversation_id_path" in self._config:
            path_values["conversation_id"] = _extract_field(
                submit_resp, str(self._config["conversation_id_path"])
            )

        # ── Step 2: Poll (if poll config is declared) ───────────────────────
        if poll_cfg:
            poll_status = await self._do_poll(
                run_id=str(run_id) if run_id is not None else None,
                submit_response=submit_resp,
                poll_cfg=poll_cfg,
                stage_id=stage_id,
                path_values=path_values,
            )
        elif fetch_cfg:
            poll_status = submit_resp
        else:
            poll_status = None

        # ── Step 3: Fetch (if fetch config is declared) ────────────────────
        if fetch_cfg:
            if not run_id:
                run_id = str(_extract_field(poll_status, run_id_path)
                             or _extract_field(poll_status, "data.run_id")
                             or _extract_field(poll_status, "run_id")
                             or "")
                path_values["run_id"] = run_id
            if not run_id:
                raise HttpFlowOperationError(
                    f"http_flow fetch for stage {stage_id!r}: "
                    f"run_id could not be determined from submit or poll response; "
                    f"declare run_id_path in the flow contract"
                )
            fetch_result = await self._do_fetch(
                run_id=str(run_id),
                fetch_cfg=fetch_cfg,
                stage_id=stage_id,
                path_values=path_values,
                submit_response=submit_resp,
            )
        else:
            fetch_result = None

        # ── Build final result ───────────────────────────────────────────────
        result_data: dict
        result_status: str
        result_error: str

        if fetch_result is not None:
            # Extract data from fetch result
            extract_path = fetch_cfg.get("extract_path", "data")
            extracted = _extract_field(fetch_result, extract_path)
            if extracted is None:
                raise HttpFlowOperationError(
                    f"http_flow fetch for stage {stage_id!r}: "
                    f"extract_path {extract_path!r} resolved to None "
                    f"in response: {fetch_result!r}"
                )
            if isinstance(extracted, dict):
                result_data = dict(extracted)
            else:
                result_key = fetch_cfg.get("result_key")
                result_data = {
                    str(result_key) if result_key else "value": extracted
                }
            result_status = "ok"
            result_error = ""
        elif poll_status is not None:
            # No fetch: use poll response as result data
            if isinstance(poll_status, dict):
                result_data = dict(poll_status.get("data", poll_status))
            else:
                result_data = {"value": poll_status}
            # Check for failure signal in poll status
            status_val = _extract_field(poll_status, "data.status") if isinstance(poll_status, dict) else None
            result_status = "failed" if status_val in ("failed", "error") else "ok"
            result_error = _redact_secrets(str(poll_status.get("error", "")))
        elif isinstance(submit_resp, dict):
            # Synchronous passthrough (no poll, no fetch)
            result_data = dict(submit_resp.get("data", submit_resp))
            result_status = str(submit_resp.get("status", "ok"))
            result_error = _redact_secrets(str(submit_resp.get("error", "")))
        else:
            result_data = {}
            result_status = "ok"
            result_error = ""

        # Normalize status
        # Do not turn a queued/running response into a completed stage.  A
        # declared http_flow with poll/fetch has already reduced the response
        # to a terminal observation; anything else is an unsafe resume gap.
        result_status = str(result_status or "ok").strip().lower()
        if result_status in {"completed", "success", "succeeded"}:
            result_status = "ok"
        elif result_status not in ("ok", "failed", "skipped"):
            raise HttpFlowOperationError(
                f"http_flow provider returned nonterminal status "
                f"{result_status!r} for stage {stage_id!r}; declare poll/fetch"
            )

        return StageResult(
            stage_id=stage_id,
            kind=kind,
            provider=provider,
            status=result_status,
            data=result_data,
            error=result_error,
            retries=0,
        )

    # ── Internal step methods ────────────────────────────────────────────────

    async def _do_request(
        self,
        payload: dict,
        request_cfg: dict,
    ) -> Any:
        """Perform the request step (synchronous POST)."""
        method = request_cfg.get("method", "POST").upper()
        path = request_cfg.get("path", "/")
        url = _join_flow_url(self._base_url, path)

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        timeout = float(request_cfg.get("timeout", 30.0))

        response = await self._session.request(
            method,
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        status = response.status
        try:
            body = await response.json()
        except Exception:
            body = None

        if not (200 <= status < 300):
            error_msg = ""
            if isinstance(body, dict):
                error_msg = str(body.get("error", f"HTTP {status}"))
            raise DispatchProviderError(
                f"http_flow request failed for stage: HTTP {status}: "
                f"{_redact_secrets(error_msg)}"
            )

        if not isinstance(body, dict):
            raise HttpFlowOperationError(
                f"http_flow request: expected JSON dict response, got {type(body).__name__}"
            )
        return body

    async def _do_poll(
        self,
        run_id: str | None,
        submit_response: Any,
        poll_cfg: dict,
        stage_id: str,
        path_values: dict[str, Any] | None = None,
    ) -> Any:
        """
        Poll until status_path reaches the terminal value or timeout expires.

        Fail-closed on unexpected status values, timeout, or non-2xx responses.
        """
        status_path = poll_cfg.get("status_path", "")
        status_values = list(poll_cfg.get("status_values", []))
        poll_interval = float(poll_cfg.get("poll_interval", 1.0))
        timeout = float(poll_cfg.get("timeout", 60.0))
        path_template = poll_cfg.get("path", "/api/runs/{run_id}/status")
        method = poll_cfg.get("method", "GET").upper()

        if not status_values:
            raise HttpFlowOperationError(
                f"http_flow poll for stage {stage_id!r}: "
                f"status_values is required and must be non-empty"
            )

        terminal_status = status_values[-1] if status_values else None

        # Check if submit response already reached terminal status
        current_status = None
        if status_path and isinstance(submit_response, dict):
            current_status = _extract_field(submit_response, status_path)

        if current_status == terminal_status:
            return submit_response

        # Polling loop
        import asyncio
        import time

        deadline = time.monotonic() + timeout
        last_response = submit_response

        while True:
            elapsed = time.monotonic()
            if elapsed >= deadline:
                raise HttpFlowOperationError(
                    f"http_flow poll for stage {stage_id!r} timed out after {timeout}s; "
                    f"last status: {current_status!r}; "
                    f"expected terminal status: {terminal_status!r}"
                )

            remaining = deadline - elapsed
            poll_url = _join_flow_url(
                self._base_url,
                _interpolate_path(
                    path_template,
                    run_id or "",
                    variables=path_values,
                ),
            )

            headers = {"Accept": "application/json"}
            timeout_val = min(poll_interval, remaining)

            try:
                response = await asyncio.wait_for(
                    self._session.request(
                        method,
                        poll_url,
                        headers=headers,
                        timeout=timeout_val,
                    ),
                    timeout=timeout_val + 5.0,  # generous wall-clock grace
                )
            except asyncio.TimeoutError:
                raise HttpFlowOperationError(
                    f"http_flow poll for stage {stage_id!r}: "
                    f"poll request timed out after {timeout_val}s"
                )

            status = response.status
            try:
                body = await response.json()
            except Exception:
                body = None

            if not (200 <= status < 300):
                error_msg = ""
                if isinstance(body, dict):
                    error_msg = str(body.get("error", f"HTTP {status}"))
                raise HttpFlowOperationError(
                    f"http_flow poll for stage {stage_id!r}: "
                    f"poll request returned HTTP {status}: "
                    f"{_redact_secrets(error_msg)}"
                )

            last_response = body

            # A 2xx response with an explicit error object is still a failed
            # operation.  Do not keep polling until timeout; preserve only the
            # redacted provider error for deterministic diagnostics.
            if isinstance(body, dict) and body.get("error"):
                raise HttpFlowOperationError(
                    f"http_flow poll for stage {stage_id!r}: "
                    f"provider error: {_redact_secrets(str(body.get('error')))}"
                )

            if status_path and isinstance(body, dict):
                current_status = _extract_field(body, status_path)
            else:
                current_status = None

            # Fail-closed: unexpected status not in status_values progression
            if current_status is not None and status_values:
                if current_status not in status_values:
                    raise HttpFlowOperationError(
                        f"http_flow poll for stage {stage_id!r}: "
                        f"unexpected status {current_status!r}; "
                        f"expected one of {status_values!r}; "
                        f"run_id={run_id!r}"
                    )

            if current_status == terminal_status:
                return body

            # Sleep for poll interval, respecting deadline
            sleep_time = min(poll_interval, deadline - time.monotonic())
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _do_fetch(
        self,
        run_id: str,
        fetch_cfg: dict,
        stage_id: str,
        path_values: dict[str, Any] | None = None,
        submit_response: dict | None = None,
    ) -> Any:
        """Perform the fetch step and return the raw response."""
        path_template = fetch_cfg.get("path", "/api/runs/{run_id}/result")
        method = fetch_cfg.get("method", "GET").upper()
        url = _join_flow_url(
            self._base_url,
            _interpolate_path(
                path_template,
                run_id or "",
                variables=path_values,
            ),
        )

        headers = {"Accept": "application/json"}
        timeout_val = float(fetch_cfg.get("timeout", 30.0))

        response = await self._session.request(
            method,
            url,
            headers=headers,
            timeout=timeout_val,
        )

        status = response.status
        try:
            body = await response.json()
        except Exception:
            body = None

        if not (200 <= status < 300):
            error_msg = ""
            if isinstance(body, dict):
                error_msg = str(body.get("error", f"HTTP {status}"))
            raise HttpFlowOperationError(
                f"http_flow fetch for stage {stage_id!r}: "
                f"fetch request returned HTTP {status}: "
                f"{_redact_secrets(error_msg)}"
            )

        if not isinstance(body, dict):
            raise HttpFlowOperationError(
                f"http_flow fetch for stage {stage_id!r}: "
                f"expected JSON dict response, got {type(body).__name__}"
            )

        # ── Optional selector: pick one object from a collection ─────────────
        select_cfg: dict | None = fetch_cfg.get("select")
        if select_cfg:
            collection_path = select_cfg.get("collection_path")
            match_path = select_cfg.get("match_path")
            match_value_ref = select_cfg.get("match_value_ref")

            if not (collection_path and match_path and match_value_ref):
                raise HttpFlowOperationError(
                    f"http_flow fetch for stage {stage_id!r}: "
                    f"select requires collection_path, match_path, and match_value_ref"
                )

            collection = _extract_field(body, collection_path)
            if not isinstance(collection, list):
                raise HttpFlowOperationError(
                    f"http_flow fetch for stage {stage_id!r}: "
                    f"select.collection_path {collection_path!r} did not resolve to a list; "
                    f"got {type(collection).__name__}"
                )

            declared_values = dict(path_values or {})
            if match_value_ref not in declared_values:
                raise HttpFlowOperationError(
                    f"http_flow fetch for stage {stage_id!r}: "
                    f"select.match_value_ref {match_value_ref!r} is unavailable"
                )
            match_value = declared_values[match_value_ref]

            matches = [
                item for item in collection
                if isinstance(item, dict) and _extract_field(item, match_path) == match_value
            ]

            if len(matches) != 1:
                raise HttpFlowOperationError(
                    f"http_flow fetch for stage {stage_id!r}: "
                    f"select must match exactly one object, got {len(matches)}; "
                    f"collection={collection_path!r}, match_path={match_path!r}, "
                    f"match_value={match_value!r}"
                )

            body = dict(matches[0])

        return body


class _HttpExecutorWrapper:
    """Wraps _HttpPipelineProviderWrapper into a KreaLikeExecutor shape."""
    def __init__(
        self,
        provider: "_HttpPipelineProviderWrapper | _HttpFlowExecutor",
        *,
        retry_delay_seconds: float = 0.0,
        retry_backoff_multiplier: float = 1.0,
        retry_max_delay_seconds: float = 0.0,
    ) -> None:
        self._provider = provider
        self._retry_delay_seconds = float(retry_delay_seconds)
        self._retry_backoff_multiplier = float(retry_backoff_multiplier)
        self._retry_max_delay_seconds = float(retry_max_delay_seconds)

    async def dispatch(
        self,
        stages: list[dict],
        context: StageContext,
        *,
        stage_map: dict[str, "StageResult"] | None = None,
    ):
        """
        Execute stages using the injected HTTP provider.

        Parameters
        ----------
        stages : list[dict]
            Stage definitions to execute.
        context : StageContext
            Shared execution context.
        stage_map : dict[str, StageResult] | None
            Optional map of completed stage_id → StageResult.  When provided,
            the executor uses it for depends_on resolution instead of
            maintaining its own internal map.  This enables the
            CompositeProviderExecutor to share dependency state across
            per-provider executor instances.
        """
        from stage_dispatch import (
            DispatchProviderError,
            StageResult,
        )

        results: list[StageResult] = []
        _stage_map: dict[str, StageResult] = stage_map if stage_map is not None else {}

        for stage in stages:
            sid = stage["id"]
            kind = stage["kind"]
            provider = stage["provider"]
            params = dict(stage.get("params", {}))
            stage_retries = stage.get("retries", 3)

            # depends_on fan-in gate — use shared stage_map when available
            for dep in stage.get("depends_on", []):
                dep_result = _stage_map.get(dep)
                if dep_result is None:
                    from stage_dispatch import DispatchContextError
                    raise DispatchContextError(
                        f"stage {sid!r} depends on {dep!r} which has no result"
                    )
                if dep_result.status in ("failed", "skipped"):
                    # QC verdict failures are expected inputs to a correction
                    # stage only when that stage explicitly opts in.  A
                    # transport/adapter failure (a failed result without
                    # qc_pass=False evidence) always blocks dependency flow.
                    qc_failure_consumable = (
                        _qc_result_is_failing(dep_result)
                        and kind == "correct"
                        and params.get("run_on_qc_fail") is True
                    )
                    if not qc_failure_consumable:
                        result = StageResult(
                            stage_id=sid, kind=kind, provider=provider,
                            status="skipped", data={},
                            error=f"depends_on stage {dep!r} is {dep_result.status}",
                            retries=0,
                        )
                        _stage_map[sid] = result
                        results.append(result)
                        break
            else:
                # Bounded retry loop
                attempt = 0
                consumed = 0
                last_error = ""
                last_data: dict = {}
                result: "StageResult | None" = None
                while True:
                    try:
                        result = await self._provider.dispatch(
                            sid, kind, provider, params, context
                        )
                        result = StageResult(
                            stage_id=result.stage_id,
                            kind=result.kind,
                            provider=result.provider,
                            status=result.status,
                            data=result.data,
                            error=result.error,
                            retries=consumed,
                        )
                        _stage_map[sid] = result
                        results.append(result)
                        # Apply context updates so subsequent stages (or the caller's
                        # CompositeProviderExecutor) receive the updated context.
                        # Mirror the update logic from CompositeProviderExecutor._update_context.
                        if result.is_ok():
                            context = _update_context_from_result(sid, kind, result, context)
                        elif kind == "qc":
                            # Record failed QC result as evidence even when QC fails;
                            # the correction loop needs the qc_results to reason about
                            # whether to proceed.
                            context = context.with_qc(sid, dict(result.data))
                        # ── QC-pass correction skip ─────────────────────────────
                        # Mirror CompositeProviderExecutor.dispatch logic (lines 2345-2370).
                        # When a correct stage with run_on_qc_fail=True succeeds because
                        # QC passed the fan-in gate (status=ok), it must still be skipped
                        # since run_on_qc_fail=True only means "don't block at fan-in";
                        # it does NOT mean "run correction when QC passes".
                        if (
                            kind == "correct"
                            and result.is_ok()
                            and params.get("run_on_qc_fail") is True
                        ):
                            for dep_id in stage.get("depends_on", []):
                                dep_res = _stage_map.get(dep_id)
                                if dep_res is not None and dep_res.kind == "qc":
                                    if dep_res.data.get("qc_pass") is True:
                                        # QC passed; correction is not needed.
                                        skip = StageResult(
                                            stage_id=sid,
                                            kind=kind,
                                            provider=provider,
                                            status="skipped",
                                            data={},
                                            error="QC passed; correction not needed",
                                            retries=0,
                                        )
                                        _stage_map[sid] = skip
                                        results.append(skip)
                                        break
                        break
                    except DispatchProviderError as exc:
                        last_error = _redact_secrets(str(exc))
                        attempt += 1
                        consumed += 1
                        if attempt > stage_retries:
                            result = StageResult(
                                stage_id=sid, kind=kind, provider=provider,
                                status="failed", data=last_data,
                                error=f"retries exhausted ({stage_retries}): {last_error}",
                                retries=consumed,
                            )
                            _stage_map[sid] = result
                            results.append(result)
                            break
                        if self._retry_delay_seconds > 0:
                            delay = min(
                                self._retry_delay_seconds
                                * (self._retry_backoff_multiplier ** (attempt - 1)),
                                self._retry_max_delay_seconds,
                            )
                            await asyncio.sleep(delay)

        return context, results


# Type alias for executors returned by build_executor
KreaLikeExecutor = _HttpExecutorWrapper


class CompositeProviderExecutor:
    """
    Stage-wise multi-provider executor.

    Resolves each stage's ``provider`` field to the appropriate pre-built
    executor and routes the dispatch call accordingly.  One pipeline may
    contain stages targeting completely different providers (e.g. ``open-design``
    for ``prompt_enhance`` and ``comfyui`` for ``generate`` via the
    gpu_manager_generation adapter).

    gpu_manager_generation stages are dispatched through the
    ``GenerationStageExecutor`` returned by the factory, ensuring a typed
    dispatch interface without exposing the raw adapter.

    Usage::

        runtime = PipelineProviderRuntime(services_config, registry)
        multi = runtime.build_multi_provider_executor(
            contract=pipeline_contract,
            transport_map={
                "open-design": http_transport,
                "combined-gemma": openai_transport,
            },
            stage_executor_factory=my_gpu_factory,
        )
        ctx, results = await multi.dispatch(stages, context)

    Raises
    ------
    UnknownProviderError
        A stage references a provider ID not in the transport map.
    KindCapabilityError
        A stage kind is not in the provider's declared capabilities.
    MissingExecutorFactoryError
        A gpu_manager_generation stage is encountered but no factory was
        provided at construction time.
    """

    def __init__(
        self,
        executor_map: dict[str, Any],  # provider_id -> KreaLikeExecutor
        runtime: "PipelineProviderRuntime",
        gpu_executor_map: dict[str, GenerationStageExecutor] | None = None,
        max_corrections: int = 1,
        max_retries: int = 3,
        transport_map: dict[str, Any] | None = None,
        stage_executor_factory: StageExecutorFactory | None = None,
        max_delegate_depth: int = 3,
        loops: list[dict] | None = None,
        execution_mode: str = "generic",
    ) -> None:
        self._executors = dict(executor_map)
        self._gpu_executors = dict(gpu_executor_map) if gpu_executor_map else {}
        self._runtime = runtime
        self._transport_map = dict(transport_map or {})
        self._stage_executor_factory = stage_executor_factory
        self._execution_mode = execution_mode
        if isinstance(max_corrections, bool) or not isinstance(max_corrections, int):
            raise ProviderRuntimeError(
                f"max_corrections must be an integer, got {type(max_corrections).__name__}"
            )
        if max_corrections < 0 or max_corrections > 5:
            raise ProviderRuntimeError(
                f"max_corrections {max_corrections} is outside the bounded range 0..5"
            )
        self._max_corrections = max_corrections
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ProviderRuntimeError(
                f"max_retries must be an integer, got {type(max_retries).__name__}"
            )
        if max_retries < 0:
            raise ProviderRuntimeError(
                f"max_retries must be non-negative, got {max_retries}"
            )
        self._max_retries = max_retries
        if isinstance(max_delegate_depth, bool) or not isinstance(max_delegate_depth, int):
            raise ProviderRuntimeError(
                "max_delegate_depth must be an integer"
            )
        if max_delegate_depth < 0 or max_delegate_depth > 5:
            raise ProviderRuntimeError(
                f"max_delegate_depth {max_delegate_depth} is outside the bounded range 0..5"
            )
        self._max_delegate_depth = max_delegate_depth
        # Explicit loops contract from compiled pipeline.
        # Each entry: {"id", "trigger_stage_id", "correction_stage_id",
        #              "on_correction": [stage_ids], "max_iterations": int}
        self._loops: list[dict] = list(loops) if loops else []
        # Build a lookup: trigger_stage_id -> loop entry
        self._loop_by_trigger: dict[str, dict] = {
            lp["trigger_stage_id"]: lp for lp in self._loops
        }

    @property
    def max_retries(self) -> int:
        """Return the per-stage retry bound (forwarded from the pipeline contract)."""
        return getattr(self, "_max_retries", 3)

    @property
    def max_loop_count(self) -> int:
        """Return the correction-loop bound (forwarded from the pipeline contract)."""
        return self._max_corrections

    @property
    def max_corrections(self) -> int:
        """Alias for max_loop_count / max_corrections for backward compatibility."""
        return self._max_corrections

    @property
    def execution_mode(self) -> str:
        """Return the pipeline execution mode (generic or legacy_reconcile)."""
        return getattr(self, "_execution_mode", "generic")

    @property
    def gpu_executors(self) -> dict[str, GenerationStageExecutor]:
        """Return the map of gpu_manager_generation provider_id → GenerationStageExecutor.

        Useful for tests that need to inspect call logs or inject specific
        responses after construction.
        """
        return self._gpu_executors

    def _update_context(
        self,
        stage_id: str,
        kind: str,
        result: "StageResult",
        context: "StageContext",
    ) -> "StageContext":
        """Update StageContext with stage outputs, mirroring PipelineExecutor._update_context."""
        if kind == "prompt_enhance":
            prompt = result.data.get("prompt", "")
            return context.with_prompt(stage_id, prompt)
        if kind in ("generate", "delegate"):
            return context.with_artifact(stage_id, dict(result.data))
        if kind == "qc":
            return context.with_qc(stage_id, dict(result.data))
        if kind == "correct":
            # Increment loop counter and record the correction
            return (context
                    .with_loop_increment()
                    .with_artifact(stage_id, dict(result.data)))
        return context

    async def _dispatch_single_stage(
        self,
        stage: dict,
        context: "StageContext",
        stage_map: dict[str, "StageResult"],
    ) -> tuple["StageContext", "StageResult"]:
        """
        Dispatch one stage through the appropriate executor.

        Returns (updated_context, result). Does NOT update stage_map or results;
        the caller is responsible for that after the QC loop check.

        This is a split of the per-iteration dispatch logic so that
        loop re-entry can re-use the same stage-execution path without
        duplicating the fan-in, artifact-gate, and retry machinery.
        """
        sid = stage["id"]
        kind = stage["kind"]
        provider = stage["provider"]
        params = dict(stage.get("params", {}))

        # ── Bounded correction loop gating ─────────────────────────────────
        # A "correct" stage increments the loop counter and is the mechanism
        # by which max_corrections is enforced.  Skip it when the bound is
        # already reached (loop_count starts at 0; first correct increments to 1;
        # with max_corrections=1, loop_count=1 means the bound is reached).
        if kind == "correct" and context.correction_loop_count >= self._max_corrections:
            skipped = StageResult(
                stage_id=sid,
                kind=kind,
                provider=provider,
                status="skipped",
                data={},
                error=(
                    f"correction loop bound: max_corrections "
                    f"{self._max_corrections} reached; max_loop_count "
                    f"{self._max_corrections} reached"
                ),
                retries=0,
            )
            return context, skipped

        # ── Shared fan-in dependency gate ────────────────────────────────────
        # HTTP executors enforce this internally, but GPU/local providers must
        # receive the same fail-closed dependency semantics.  A QC verdict of
        # qc_pass=false is intentionally consumable only by a correction stage
        # explicitly marked run_on_qc_fail.
        for dep_id in stage.get("depends_on", []):
            dep_result = stage_map.get(dep_id)
            if dep_result is None:
                from stage_dispatch import DispatchContextError
                raise DispatchContextError(
                    f"stage {sid!r} depends on {dep_id!r} which has no result"
                )
            if dep_result.status in ("failed", "skipped"):
                qc_failure_consumable = (
                    kind == "correct"
                    and (
                        params.get("run_on_qc_fail") is True
                        or any(
                            loop.get("correction_stage_id") == sid
                            for loop in self._loops
                        )
                    )
                    and _qc_result_is_failing(dep_result)
                )
                if not qc_failure_consumable:
                    skipped = StageResult(
                        stage_id=sid,
                        kind=kind,
                        provider=provider,
                        status="skipped",
                        data={},
                        error=f"depends_on stage {dep_id!r} is {dep_result.status}",
                        retries=0,
                    )
                    return context, skipped

        # ── QC-pass correction skip (no explicit loop) ───────────────────────
        # When a pipeline has no explicit "loops" contract, correction stages
        # with run_on_qc_fail=True should be skipped when QC passed, because
        # there is no retry mechanism to re-run the QC.  The run_on_qc_fail
        # flag only means "don't block at the fan-in gate"; it does NOT mean
        # "force correction when QC passes".
        if kind == "correct" and params.get("run_on_qc_fail"):
            # Check if any depends_on QC stage passed.
            qc_passed = False
            for dep_id in stage.get("depends_on", []):
                dep_result = stage_map.get(dep_id)
                if dep_result is not None and dep_result.kind == "qc":
                    if dep_result.data.get("qc_pass") is True:
                        qc_passed = True
                        break
            if qc_passed:
                skipped = StageResult(
                    stage_id=sid,
                    kind=kind,
                    provider=provider,
                    status="skipped",
                    data={},
                    error=f"QC passed; correction not needed",
                    retries=0,
                )
                return context, skipped

        # ── Generate artifact gate ───────────────────────────────────────────
        # A "generate" stage requires its depends_on generate stages to have
        # produced artifacts.  This prevents a downstream generate from receiving
        # an empty artifact dict (which would be forwarded to the GPU worker
        # as a malformed request).
        if kind == "generate":
            for dep_id in stage.get("depends_on", []):
                dep_result = stage_map.get(dep_id)
                if dep_result is not None and dep_result.kind == "generate":
                    if not _result_has_artifact(dep_result):
                        failed = StageResult(
                            stage_id=sid,
                            kind=kind,
                            provider=provider,
                            status="failed",
                            data={},
                            error=(
                                f"depends_on stage '{dep_id}' produced no artifact; "
                                f"generate stage cannot proceed"
                            ),
                            retries=0,
                        )
                        return context, failed

        cfg = self._runtime._providers.get(provider, {})
        adapter_type = cfg.get("adapter", "")

        if adapter_type == "local_pipeline":
            self._runtime.validate_stage_kind(provider, kind)
            result = await self._dispatch_local_pipeline(sid, provider, params, context)
            upd_ctx = context
            if result.is_ok():
                upd_ctx = self._merge_nested_context(sid, result, upd_ctx)
            else:
                upd_ctx = upd_ctx.with_error(
                    f"stage '{sid}' returned status={result.status}: {result.error}"
                )
            return upd_ctx, result

        if adapter_type == "gpu_manager_generation":
            gpu_executor = self._gpu_executors.get(provider)
            if gpu_executor is None:
                raise MissingExecutorFactoryError(
                    f"stage {sid!r} uses provider {provider!r} "
                    f"(gpu_manager_generation) but no executor was provided"
                )
            try:
                result = await gpu_executor.dispatch(sid, kind, provider, params, context)
            except DispatchProviderError as exc:
                result = StageResult(
                    stage_id=sid, kind=kind, provider=provider,
                    status="failed", data={},
                    error=str(exc), retries=0,
                )
            upd_ctx = context
            if result.is_ok():
                upd_ctx = self._update_context(sid, kind, result, upd_ctx)
            else:
                upd_ctx = upd_ctx.with_error(
                    f"stage '{sid}' returned status={result.status}: {result.error}"
                )
            if kind == "qc":
                upd_ctx = upd_ctx.with_qc(sid, dict(result.data))
            return upd_ctx, result

        # HTTP provider path
        executor = self._executors.get(provider)
        if executor is None:
            raise UnknownProviderError(
                f"stage {sid!r} references provider {provider!r} "
                f"which is not in the transport map; "
                f"available providers: {sorted(self._executors.keys())}"
            )
        try:
            self._runtime.validate_stage_kind(provider, kind)
        except KindCapabilityError:
            raise
        context, stage_results = await executor.dispatch(
            [stage], context, stage_map=stage_map
        )
        result = stage_results[0] if stage_results else StageResult(
            stage_id=sid, kind=kind, provider=provider,
            status="failed", data={}, error="no result from executor"
        )
        upd_ctx = context
        if kind == "qc":
            upd_ctx = upd_ctx.with_qc(sid, dict(result.data))
            if not result.is_ok():
                upd_ctx = upd_ctx.with_error(
                    f"stage '{sid}' returned status={result.status}: {result.error}"
                )
        elif not result.is_ok():
            upd_ctx = upd_ctx.with_error(
                f"stage '{sid}' returned status={result.status}: {result.error}"
            )
        return upd_ctx, result

    def _record_stage_result(
        self,
        sid: str,
        kind: str,
        result: "StageResult",
        stage_map: dict[str, "StageResult"],
        results: list["StageResult"],
    ) -> None:
        """Record a newly dispatched result and make it the latest dependency value.

        Seeded worker evidence is inserted into ``stage_map`` before dispatch and
        is never passed to this helper.  Re-entered stages deliberately have the
        same declared ID, so they must replace the dependency value while still
        appending a distinct per-dispatch trace entry to ``results``.
        """
        stage_map[sid] = result
        results.append(result)

    def _merge_nested_context(
        self,
        stage_id: str,
        result: "StageResult",
        context: "StageContext",
    ) -> "StageContext":
        """Record a child context and namespace its outputs for later stages."""
        context = context.with_artifact(stage_id, dict(result.data))
        nested = result.data.get("child_context")
        if not isinstance(nested, dict):
            return context
        artifacts = dict(context.artifacts)
        for child_id, data in dict(nested.get("artifacts", {})).items():
            artifacts[f"{stage_id}.{child_id}"] = dict(data)
        enhanced = dict(context.enhanced_prompts)
        for child_id, prompt in dict(nested.get("enhanced_prompts", {})).items():
            enhanced[f"{stage_id}.{child_id}"] = str(prompt)
        qc_results = dict(context.qc_results)
        for child_id, qc in dict(nested.get("qc_results", {})).items():
            qc_results[f"{stage_id}.{child_id}"] = dict(qc)
        context = dataclasses.replace(
            context,
            artifacts=artifacts,
            enhanced_prompts=enhanced,
            qc_results=qc_results,
        )
        return context.with_nested_context(stage_id, nested)

    async def _dispatch_local_pipeline(
        self,
        stage_id: str,
        provider: str,
        params: dict,
        context: "StageContext",
    ) -> "StageResult":
        """Resolve and execute one config-defined child pipeline."""
        pipeline_id = params.get("pipeline_id", params.get("delegate_pipeline_id"))
        if not isinstance(pipeline_id, str) or not pipeline_id.strip():
            return StageResult(
                stage_id=stage_id, kind="delegate", provider=provider,
                status="failed", data={},
                error="delegate stage requires a non-empty pipeline_id",
            )
        if context.delegate_depth >= self._max_delegate_depth:
            return StageResult(
                stage_id=stage_id, kind="delegate", provider=provider,
                status="failed", data={},
                error=(
                    f"delegate depth {context.delegate_depth} reached "
                    f"max_delegate_depth {self._max_delegate_depth}"
                ),
            )
        child_contract = self._runtime._pipelines.get(pipeline_id)
        if not isinstance(child_contract, dict):
            return StageResult(
                stage_id=stage_id, kind="delegate", provider=provider,
                status="failed", data={},
                error=f"unknown delegated pipeline {pipeline_id!r}",
            )
        child_stages = child_contract.get("stages")
        if not isinstance(child_stages, list) or not child_stages:
            return StageResult(
                stage_id=stage_id, kind="delegate", provider=provider,
                status="failed", data={},
                error=f"delegated pipeline {pipeline_id!r} has no stages",
            )
        try:
            child_executor = self._runtime.build_multi_provider_executor(
                child_contract,
                self._transport_map,
                stage_executor_factory=self._stage_executor_factory,
                max_delegate_depth=self._max_delegate_depth,
                max_corrections=self._max_corrections,
            )
            child_context, child_results = await child_executor.dispatch(
                child_stages,
                context.with_delegate_depth(context.delegate_depth + 1),
            )
        except (DispatchError, KeyError, TypeError, ValueError) as exc:
            return StageResult(
                stage_id=stage_id, kind="delegate", provider=provider,
                status="failed", data={},
                error=_redact_secrets(f"delegated pipeline failed: {exc}"),
            )
        child_payload = dataclasses.asdict(child_context)
        child_payload["results"] = [dataclasses.asdict(item) for item in child_results]
        failed = next(
            (item for item in child_results if item.status in ("failed", "skipped")),
            None,
        )
        return StageResult(
            stage_id=stage_id,
            kind="delegate",
            provider=provider,
            status="failed" if failed else "ok",
            data={"pipeline_id": pipeline_id, "child_context": child_payload},
            error=(f"delegated pipeline stage {failed.stage_id!r} was {failed.status}" if failed else ""),
        )

    async def dispatch(
        self,
        stages: list[dict],
        context: StageContext,
        *,
        seeded_results: list["StageResult"] | None = None,
        stage_defs: dict[str, dict] | None = None,
    ):
        """
        Dispatch each stage to the executor matching its ``provider`` field.

        Stages are executed sequentially; fan-in dependencies (``depends_on``)
        are resolved within each stage's own executor, not across executors.
        Each executor's retry policy applies independently.

        Context is propagated stage-to-stage: after each stage completes
        the shared StageContext is updated via _update_context (prompt,
        artifact, qc_results, correction_loop_count).  The updated context
        is passed to the next stage's executor dispatch call.

        gpu_manager_generation stages are dispatched through the
        registered GenerationStageExecutor for that provider, using the
        typed dispatch interface.  If no factory was provided at construction
        time, MissingExecutorFactoryError is raised fail-closed.

        seeded_results : list[StageResult] | None
            Optional pre-executed stage results from the GPU Manager worker
            (used in mixed-ownership handoff).  When provided, each result
            is pre-populated into the fan-in dependency map so that external
            stages can depend on worker-owned stages without re-executing them.
            Results are consumed in order; a result for each worker-owned
            stage ID must be present in the list.  QC stages with qc_pass=False
            are recorded normally (the qc_pass flag is in result.data) — the
            correction logic in the pipeline executor handles whether to proceed
            or abort based on the correction_loop_count and the qc verdict.

        Loops
        -----
        When a compiled ``loops`` contract is present, the dispatch loop
        detects ``qc_pass=false`` at a loop trigger stage and enters the
        declared correction cycle:

            1. Execute ``correction_stage_id``.
            2. Increment the loop counter (via ``with_loop_increment()``).
            3. Re-enter ``on_correction`` stage IDs in declaration order.
            4. Re-execute the trigger QC stage.
            5. Repeat up to ``max_iterations`` times.

        On ``qc_pass=true`` the loop is exited immediately (no correction,
        no re-entry).  On bound exhaustion an explicit terminal result with
        evidence is appended and the loop exits; the pipeline continues with
        whatever stages follow.

        No stage ID collisions occur during re-entry: each iteration of a
        re-entered stage produces a fresh result appended to ``results``
        (the per-iteration result representation).  The ``stage_map`` keeps
        only the latest result for dependency resolution; prior evidence
        remains in ``context.qc_results`` and ``context.artifacts``.
        """
        results: list[StageResult] = []
        stage_map: dict[str, StageResult] = {}
        _defs: dict[str, dict] = stage_defs if stage_defs is not None else {s["id"]: s for s in stages}

        # Pre-populate fan-in map with worker-owned stage results (handoff seam).
        # This lets external stages depend on worker stages without re-executing.
        if seeded_results:
            for result in seeded_results:
                if result.stage_id in stage_map:
                    # Duplicate entry — worker submitted same stage twice;
                    # keep the first entry (original evidence takes priority).
                    continue
                stage_map[result.stage_id] = result
                if result.is_ok():
                    context = self._update_context(
                        result.stage_id, result.kind, result, context
                    )
                if result.kind == "qc":
                    context = context.with_qc(result.stage_id, dict(result.data))

        for stage in stages:
            sid = stage["id"]
            kind = stage["kind"]

            # A mixed-ownership handoff seeds results for stages already
            # executed by the live WorkerPool. Preserve their evidence and
            # dependency edges, but never dispatch them again. A seeded QC
            # trigger is the exception: its verdict still has to be consumed
            # by the explicit correction-loop controller below.
            seeded = sid in stage_map

            # A correction explicitly marked run_on_qc_fail is conditional on
            # its declared QC dependency.  When that dependency passed, omit
            # the correction entirely: it was not executed and must not appear
            # as a synthetic skipped result in the dispatch trace.  This applies
            # to both normal and seeded QC results.
            if (
                kind == "correct"
                and stage.get("params", {}).get("run_on_qc_fail")
                and any(
                    lp.get("correction_stage_id") == sid
                    for lp in self._loops
                )
                and any(
                    isinstance(stage_map.get(dep_id), StageResult)
                    and stage_map[dep_id].kind == "qc"
                    and stage_map[dep_id].data.get("qc_pass") is True
                    for dep_id in stage.get("depends_on", [])
                )
            ):
                continue

            # ── Bug 2 fix (non-loop correct seeded → recorded as skipped): ────────
            # A correct stage NOT in any loop's correction_stage_id that was
            # seeded from the worker handoff must be recorded as 'skipped',
            # not passed through with its seeded 'ok' status.  A correct stage
            # that isn't a loop correction has no business appearing in results
            # with its original status — the handoff replaced it.
            if (
                kind == "correct"
                and seeded
                and not any(
                    lp.get("correction_stage_id") == sid
                    for lp in self._loops
                )
            ):
                skipped = StageResult(
                    stage_id=sid,
                    kind=kind,
                    provider=stage["provider"],
                    status="skipped",
                    data={},
                    error="non-loop correct stage seeded from worker handoff; "
                          "correction was applied by worker and is not re-executed",
                    retries=0,
                )
                self._record_stage_result(sid, kind, skipped, stage_map, results)
                continue

            if seeded:
                result = stage_map[sid]
                # A seeded QC trigger still belongs in the dispatch trace so
                # the explicit loop controller can consume and expose the
                # live verdict without redispatching the provider.
                if (
                    kind == "qc"
                    and sid in self._loop_by_trigger
                    and result.data.get("qc_pass") is True
                ):
                    self._record_stage_result(sid, kind, result, stage_map, results)
            else:
                # Execute the stage via the shared helper (shared with loop re-entry)
                context, result = await self._dispatch_single_stage(
                    stage, context, stage_map
                )
                self._record_stage_result(sid, kind, result, stage_map, results)

            # ── Loop re-entry logic ────────────────────────────────────────────
            # Only a QC stage that declares a loop trigger participates.
            if kind == "qc" and result.kind == "qc":
                loop_entry = self._loop_by_trigger.get(sid)
                if loop_entry is not None:
                    qc_pass = result.data.get("qc_pass")
                    if qc_pass is False:
                        # QC failed → enter the correction loop
                        correction_sid = loop_entry["correction_stage_id"]
                        max_iters = loop_entry["max_iterations"]
                        on_correction_ids = loop_entry["on_correction"]  # usually [generate, qc]

                        # Each iteration of this loop body corresponds to one
                        # loop iteration declared in the pipeline contract.
                        iteration = 0
                        while True:
                            iteration += 1
                            # Enforce declared max_iterations bound (fail-closed)
                            if iteration > max_iters:
                                # Bound exhausted: append terminal skipped result
                                # with evidence rather than silently claiming success.
                                terminal = StageResult(
                                    stage_id=sid,
                                    kind="qc",
                                    provider=stage["provider"],
                                    status="skipped",
                                    data={
                                        "qc_pass": False,
                                        "loop_evidence": {
                                            "loop_id": loop_entry["id"],
                                            "iteration": iteration - 1,
                                            "reason": (
                                                f"loop {loop_entry['id']}: "
                                                f"max_iterations {max_iters} exhausted; "
                                                f"qc_pass=false persisted in qc_results"
                                            ),
                                        },
                                    },
                                    error=(
                                        f"loop {loop_entry['id']}: max_iterations "
                                        f"{max_iters} reached (max_loop_count bound); qc_pass=false; "
                                        f"correction loop bound exhausted"
                                    ),
                                    retries=0,
                                )
                                self._record_stage_result(
                                    sid, kind, terminal, stage_map, results
                                )
                                # Overwrite the latest QC result with the terminal one
                                # so downstream stages see the bound-exhaustion evidence
                                # in qc_results (but the PASS of the final re-run QC
                                # is what actually gates downstream — which won't happen
                                # since we just hit the bound).
                                context = context.with_qc(sid, dict(terminal.data))
                                break

                            # ── Step 1: Execute correction stage ─────────────────
                            correction_stage = _defs.get(correction_sid)
                            if correction_stage is None:
                                # Defensive: declared correction stage not in pipeline
                                context = context.with_error(
                                    f"loop {loop_entry['id']}: correction stage "
                                    f"{correction_sid!r} not found in pipeline"
                                )
                                break

                            # The correction stage is already in stage_defs (it's part
                            # of the pipeline).  Check if it was already executed this
                            # iteration via seeded_results or a prior loop pass.
                            # Use stage_map (latest result) for dependency resolution.
                            context, correction_result = await self._dispatch_single_stage(
                                correction_stage, context, stage_map
                            )
                            self._record_stage_result(
                                correction_sid,
                                correction_stage["kind"],
                                correction_result,
                                stage_map,
                                results,
                            )

                            # If correction itself failed critically, stop the loop
                            if correction_result.status == "failed":
                                context = context.with_error(
                                    f"loop {loop_entry['id']}: correction stage "
                                    f"{correction_sid!r} failed: {correction_result.error}"
                                )
                                break

                            # The correction-stage dispatcher already increments
                            # correction_loop_count through _update_context.  Do
                            # not increment it a second time here.

                            # ── Step 2: Re-enter on_correction stages in order ────
                            reentered_qc_result = None
                            for re_entry_sid in on_correction_ids:
                                # ``stage_defs`` is an optional public argument;
                                # ``_defs`` is the normalized map used by the
                                # whole dispatch call.  Re-entry must use the
                                # normalized map so callers that omit
                                # ``stage_defs`` still get a fully executable
                                # declarative loop.
                                re_entry_stage = _defs.get(re_entry_sid)
                                if re_entry_stage is None:
                                    context = context.with_error(
                                        f"loop {loop_entry['id']}: on_correction stage "
                                        f"{re_entry_sid!r} not found in pipeline"
                                    )
                                    continue

                                context, re_result = await self._dispatch_single_stage(
                                    re_entry_stage, context, stage_map
                                )
                                self._record_stage_result(
                                    re_entry_sid,
                                    re_entry_stage["kind"],
                                    re_result,
                                    stage_map,
                                    results,
                                )
                                if re_entry_sid == sid and re_result.kind == "qc":
                                    # The declarative re-entry list may include
                                    # the trigger QC stage. Reuse this result in
                                    # Step 3 instead of dispatching QC twice.
                                    reentered_qc_result = re_result
                                # If a critical stage in on_correction fails, stop
                                if re_result.status == "failed":
                                    context = context.with_error(
                                        f"loop {loop_entry['id']}: on_correction stage "
                                        f"{re_entry_sid!r} failed: {re_result.error}"
                                    )
                                    break

                            # ── Step 3: Re-run trigger QC when not re-entered ──
                            # Some declarations use [generate] in on_correction
                            # and rely on the loop controller to run QC here;
                            # declarations using [generate, qc] are already done.
                            if reentered_qc_result is None:
                                context, re_qc_result = await self._dispatch_single_stage(
                                    stage, context, stage_map
                                )
                                self._record_stage_result(
                                    sid, kind, re_qc_result, stage_map, results
                                )
                                context = context.with_qc(sid, dict(re_qc_result.data))
                            else:
                                re_qc_result = reentered_qc_result

                            # Check the re-run verdict
                            if re_qc_result.kind == "qc" and re_qc_result.data.get("qc_pass") is True:
                                # Loop exited successfully
                                break
                            if iteration >= max_iters:
                                # The final allowed QC attempt itself is the
                                # terminal failure. Replace that trace entry
                                # with explicit bound evidence instead of
                                # appending a duplicate synthetic QC result on
                                # the next while iteration.
                                terminal = StageResult(
                                    stage_id=sid,
                                    kind="qc",
                                    provider=stage["provider"],
                                    status="skipped",
                                    data={
                                        "qc_pass": False,
                                        "loop_evidence": {
                                            "loop_id": loop_entry["id"],
                                            "iteration": iteration,
                                            "reason": (
                                                f"loop {loop_entry['id']}: "
                                                f"max_iterations {max_iters} exhausted; "
                                                f"qc_pass=false persisted in qc_results"
                                            ),
                                        },
                                    },
                                    error=(
                                        f"loop {loop_entry['id']}: max_iterations "
                                        f"{max_iters} reached (max_loop_count bound); qc_pass=false; "
                                        f"correction loop bound exhausted"
                                    ),
                                    retries=0,
                                )
                                if results and results[-1].stage_id == sid:
                                    results[-1] = terminal
                                else:
                                    results.append(terminal)
                                stage_map[sid] = terminal
                                context = context.with_qc(sid, dict(terminal.data))
                                break
                            # else: qc_pass still false → continue to next iteration

        return context, results


# ── GPU adapter wrapper ─────────────────────────────────────────────────────────


class _GpuMgrAdapterWrapper:
    """
    Internal wrapper that presents a minimal submit/poll/cancel interface
    from the gpu_manager_generation adapter, bridged to the pipeline stage
    contract.

    This does NOT silently use a fake — it wraps the real
    ``GPUMgrGenerationAdapter`` from capability_agent.
    """

    def __init__(
        self,
        provider_id: str,
        endpoint_ref: str,
        transport: Any,
        config: dict,
    ) -> None:
        self._provider_id = provider_id
        self._endpoint_ref = endpoint_ref
        self._transport = transport
        self._config = config

    async def submit(
        self,
        assignment: Any,
        *,
        state_store: Any = None,
        params: Any = None,
    ):
        """
        Submit a generation request via the gpu_manager_generation adapter.

        ``assignment`` is an ``Assignment`` envelope from the capability agent.
        Returns a ``GPUMgrGenerationSubmitReceipt``.
        """
        from capability_agent.gpu_manager_generation_adapter import (
            GPUMgrGenerationAdapter,
            GPUMgrGenerationAdapterConfig,
        )
        from capability_agent.client import Assignment
        from capability_agent.adapters import (
            AdapterHttpResponse,
        )

        # Build a minimal transport wrapper that bridges the
        # AdapterTransportCallable signature to what GPUMgrGenerationAdapter expects.
        adapter_cfg = GPUMgrGenerationAdapterConfig()
        adapter = GPUMgrGenerationAdapter(
            config=adapter_cfg,
            transport=self._transport,
        )
        return await adapter.submit(
            assignment=assignment,
            state_store=state_store,
            params=params,
        )

    async def poll(self, job_id: str, deadline_seconds: float | None = None):
        """Poll job status."""
        from capability_agent.gpu_manager_generation_adapter import (
            GPUMgrGenerationAdapter,
            GPUMgrGenerationAdapterConfig,
        )
        adapter = GPUMgrGenerationAdapter(
            config=GPUMgrGenerationAdapterConfig(),
            transport=self._transport,
        )
        return await adapter.poll(job_id=job_id, deadline_seconds=deadline_seconds)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        from capability_agent.gpu_manager_generation_adapter import (
            GPUMgrGenerationAdapter,
            GPUMgrGenerationAdapterConfig,
        )
        adapter = GPUMgrGenerationAdapter(
            config=GPUMgrGenerationAdapterConfig(),
            transport=self._transport,
        )
        return await adapter.cancel(job_id=job_id)


# Type alias for GPU adapters
GPUMgrLikeAdapter = _GpuMgrAdapterWrapper


# ── Factory callable ──────────────────────────────────────────────────────────

def make_provider_runtime(
    services_config: dict | str,
    provider_registry: frozenset[str],
) -> PipelineProviderRuntime:
    """
    Factory callable usable by daemon integration.

    Parameters
    ----------
    services_config : dict | str
        Either the parsed services.json dict, or a path string pointing to
        the JSON file (which will be loaded).
    provider_registry : frozenset[str]
        The set of known provider IDs.

    Returns
    -------
    PipelineProviderRuntime
    """
    if isinstance(services_config, str):
        with open(services_config) as fh:
            services_config = _json.load(fh)
    return PipelineProviderRuntime(
        services_config=services_config,
        provider_registry=provider_registry,
    )


# ── Public re-exports ─────────────────────────────────────────────────────────

__all__ = [
    "ProviderRuntimeError",
    "UnknownProviderError",
    "AmbiguousProviderError",
    "DisabledProviderError",
    "EndpointResolutionError",
    "KindCapabilityError",
    "HttpFlowOperationError",
    "DuplicatePipelineError",
    "PipelineProviderRuntime",
    "make_provider_runtime",
    "KreaLikeExecutor",
    "GPUMgrLikeAdapter",
    "CompositeProviderExecutor",
    "MissingExecutorFactoryError",
    "GenerationStageExecutor",
    "StageExecutorFactory",
]
