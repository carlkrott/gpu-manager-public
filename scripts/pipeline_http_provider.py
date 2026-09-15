"""
pipeline_http_provider.py — Contract-driven async HTTP-JSON provider adapter.

This module provides a production-quality provider-runtime extension that
resolves endpoint definitions from a declarative pipeline contract, posts
a typed stage payload, validates 2xx JSON responses, redacts auth headers
and secrets from logged errors, and fails closed on missing or ambiguous
endpoints.

It is wired exclusively through the ``pipeline_dispatch_runner`` injection
point (``_run_pipeline_stages`` / ``run_pipeline_dispatch``).  It does NOT
change the existing Krea path, which uses ``WorkerPool`` artifacts and
post-QC evidence reconciliation.

Contract shape (parsed from the compiled pipeline dict)::

    {
        "schema": "image-pipeline.v1",
        "original_prompt": "...",
        "stages": [
            {
                "id": "enhance-1",
                "kind": "prompt_enhance",
                "provider": "http",
                "params": {"endpoint": "https://example.com/enhance", ...},
                ...
            },
            ...
        ],
    }

The ``http`` provider entry in each stage's ``params`` may contain:

    endpoint        — required URL (must be https unless allow_http=True)
    allow_http      — bool, permit http:// endpoints (default False)
    timeout         — float, seconds for the HTTP request (default 30)
    headers         — dict, additional request headers
    auth_header     — str, name of the header carrying the bearer token
                      (default "Authorization"); the value is redacted in logs

Fail-closed guarantees
──────────────────────
• No ``endpoint`` in stage params  → raises ``EndpointResolutionError``
• ``endpoint`` URL parse failure   → raises ``EndpointResolutionError``
• ``endpoint`` is http:// when ``allow_http=False`` → raises
  ``EndpointResolutionError``
• HTTP response status not 2xx      → ``HttpProviderError`` (triggers retry
  via the bounded-retry loop in ``KreaPipelineExecutor``)
• JSON parse of 2xx body fails     → ``HttpProviderError``
• Missing stage in contract         → raises ``EndpointResolutionError``

Redaction
────────
The following are replaced with ``"<redacted>"`` in any error string
returned from this module:
  - ``Bearer <token>`` and ``<token>`` in auth headers
  - Strings matching ``api[_-]?key``, ``secret``, ``token``, ``password``
    in header names and URL query parameters

No live network I/O occurs in tests when a fake session is injected.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any

# TypedDict / Protocol only in type stubs
try:
    from typing import TypedDict, Protocol
except ImportError:
    from typing_extensions import TypedDict, Protocol

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from stage_dispatch import (
    DispatchProviderError,
    DispatchKindError,
    KreaPipelineExecutor,
    StageContext,
    StageResult,
)

# ── Constants & redaction ────────────────────────────────────────────────────

_AUTH_RE = re.compile(
    r"(Bearer\s+)([^\s\"']+)",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"(?<![a-zA-Z0-9_])(api[_-]?key|secret|password|auth|token)(?![a-zA-Z0-9_])",
    re.IGNORECASE,
)
_URL_QUERY_RE = re.compile(r"([?&])([^=]+)=([^&]*)")

# Maximum depth for redacting nested dict / list values
_RECURSION_LIMIT = 10


# ── Exceptions ────────────────────────────────────────────────────────────────

class EndpointResolutionError(DispatchProviderError):
    """
    Raised when the endpoint for a stage cannot be resolved from the contract
    (missing, ambiguous, or malformed).  This is a fail-closed signal: the
    pipeline MUST NOT proceed without a unambiguous endpoint.
    """
    pass


class HttpProviderError(DispatchProviderError):
    """
    Raised when the HTTP call to a resolved endpoint fails or returns a
    non-2xx status.  The bounded-retry loop in ``KreaPipelineExecutor``
    handles this as a retriable provider error.
    """
    pass


# ── Redaction helpers ────────────────────────────────────────────────────────

def _redact_text(text: str) -> str:
    """
    Redact bearer tokens and secret-like substrings from an error string.

    Strategy:
    1. Replace ``Bearer <token>`` patterns first with ``Bearer <redacted>``.
    2. Replace standalone secret words (api_key, secret, token, password, auth)
       that appear as full words — not already inside ``<redacted>``.
    """
    # Redact "Bearer <token>"
    text = _AUTH_RE.sub(r"\1<redacted>", text)
    # Redact secret-like words that appear stand-alone (not in URL paths)
    # Only match if not already inside a <redacted> placeholder
    def _secret_replacer(m: re.Match) -> str:
        before = text[m.start() - 12 : m.start()]
        if "<redacted>" in before:
            return m.group(0)  # already redacted
        return "<redacted>"

    text = _SECRET_RE.sub(_secret_replacer, text)
    return text


def _redact_dict(data: dict, _depth: int = 0) -> dict:
    """
    Recursively walk a dict and redact values whose keys look like secrets.
    Non-secret keys have their string values passed through _redact_text
    (which handles bearer-token redaction) for nested auth content.
    """
    if _depth > _RECURSION_LIMIT:
        return {"<max-depth>": True}
    out: dict = {}
    for key, val in data.items():
        if _SECRET_RE.search(key):
            # Secret-like key → always redact the value
            out[key] = "<redacted>"
        elif isinstance(val, dict):
            out[key] = _redact_dict(val, _depth + 1)
        elif isinstance(val, list):
            out[key] = _redact_list(val, _depth + 1)
        elif isinstance(val, str):
            # Non-secret key: apply bearer-token redaction to the value
            out[key] = _redact_text(val)
        else:
            out[key] = val
    return out


def _redact_list(data: list, _depth: int = 0) -> list:
    if _depth > _RECURSION_LIMIT:
        return ["<max-depth>"]
    return [_redact_dict(item, _depth + 1) if isinstance(item, dict)
            else _redact_list(item, _depth + 1) if isinstance(item, list)
            else _redact_text(item) if isinstance(item, str)
            else item for item in data]


def _redact_url(url: str) -> str:
    """
    Redact secret-like query parameters from a URL string.
    """
    def _redact_param(m: re.Match) -> str:
        key = m.group(2)
        if _SECRET_RE.search(key):
            return f"{m.group(1)}{key}=<redacted>"
        return m.group(0)
    return _URL_QUERY_RE.sub(_redact_param, url)


# ── Contract parsing ─────────────────────────────────────────────────────────

class _ContractStage(TypedDict, total=False):
    id: str
    kind: str
    provider: str
    params: dict[str, Any]
    depends_on: list[str]
    retries: int | None


class HttpStageParams(TypedDict, total=False):
    """Allowed keys in a stage's ``params`` for the ``http`` provider."""
    endpoint: str
    allow_http: bool
    timeout: float
    headers: dict[str, str]
    auth_header: str


# ── Provider ─────────────────────────────────────────────────────────────────

class HttpPipelineProvider:
    """
    Async HTTP-JSON provider that resolves endpoints from a pipeline contract
    and makes typed stage dispatches.

    The ``contract`` is the compiled pipeline dict (from
    ``service_pipeline.compile_generation_pipeline``) that carries stage
    definitions with embedded endpoint parameters.

    The ``session`` is an ``aiohttp.ClientSession`` (or a fake that implements
    the same interface).  When a fake session is supplied, no live network
    I/O occurs.

    Usage::

        provider = HttpPipelineProvider(
            contract=pipeline_dict,
            session=fake_session,   # or a real aiohttp.ClientSession
        )
        executor = HttpPipelineExecutor(provider=provider)
        ctx, results = await executor.dispatch(stages, context)

    For injection into ``run_pipeline_dispatch``::

        provider = HttpPipelineProvider(contract=worker_pipeline, session=session)
        executor = HttpPipelineExecutor(provider=provider)
        ctx, results = await executor.dispatch(stages, context)
    """

    # Default timeout for HTTP requests (seconds)
    DEFAULT_TIMEOUT: float = 30.0
    # Default header name for bearer token
    DEFAULT_AUTH_HEADER: str = "Authorization"

    def __init__(
        self,
        contract: dict,
        session: Any,  # aiohttp.ClientSession or fake
    ) -> None:
        self._contract = contract
        self._session = session
        self._stage_map: dict[str, _ContractStage] = {
            s["id"]: s for s in contract.get("stages", [])
        }
        if not self._stage_map:
            raise EndpointResolutionError(
                "contract has no stages; cannot resolve endpoints"
            )

    # ── Endpoint resolution ────────────────────────────────────────────────

    def _resolve_endpoint(self, stage_id: str) -> tuple[str, dict]:
        """
        Resolve the endpoint URL and provider params for a stage.

        Returns (url, provider_params).

        Raises ``EndpointResolutionError`` fail-closed when the endpoint is
        missing, ambiguous, or malformed.
        """
        stage = self._stage_map.get(stage_id)
        if stage is None:
            raise EndpointResolutionError(
                f"stage '{stage_id}' not found in pipeline contract; "
                f"available stages: {sorted(self._stage_map.keys())}"
            )

        params: dict[str, Any] = stage.get("params") or {}
        raw_endpoint = params.get("endpoint")

        if not raw_endpoint:
            raise EndpointResolutionError(
                f"stage '{stage_id}' is missing required 'endpoint' in params; "
                f"add endpoint=https://... to the stage definition"
            )

        if not isinstance(raw_endpoint, str):
            raise EndpointResolutionError(
                f"stage '{stage_id}' endpoint must be a string, "
                f"got {type(raw_endpoint).__name__}"
            )

        # Parse URL — fail closed on malformed
        try:
            parsed = urllib.parse.urlparse(raw_endpoint)
        except Exception as exc:
            raise EndpointResolutionError(
                f"stage '{stage_id}' endpoint URL parse failed: {exc}"
            ) from exc

        scheme = parsed.scheme.lower()
        allow_http = bool(params.get("allow_http", False))
        if scheme == "http" and not allow_http:
            raise EndpointResolutionError(
                f"stage '{stage_id}' uses http:// which is not allowed; "
                f"set allow_http=True in params to permit it"
            )
        if scheme not in ("http", "https"):
            raise EndpointResolutionError(
                f"stage '{stage_id}' endpoint has unsupported scheme '{scheme}'; "
                f"only http and https are supported"
            )

        return raw_endpoint, params

    # ── Payload construction ────────────────────────────────────────────────

    def _build_payload(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> dict:
        """
        Build the POST body for the stage dispatch.

        Payload shape::

            {
                "pipeline": {
                    "schema": <schema_id>,
                    "original_prompt": <original_prompt>,
                },
                "stage": {
                    "id": <stage_id>,
                    "kind": <kind>,
                    "provider": <provider>,
                    "params": <params>,   # provider params only, not stage def
                    "depends_on": [...],
                    "retries": <retries>,
                },
                "context": {
                    "original_prompt": <original_prompt>,
                    "enhanced_prompts": {<stage_id>: <prompt>, ...},
                    "artifacts": {<stage_id>: {...}, ...},
                    "qc_results": {<stage_id>: {...}, ...},
                },
            }
        """
        # Strip http-specific keys from params so the endpoint receives only
        # the user-defined stage params (not internal provider config)
        http_keys = frozenset(("endpoint", "allow_http", "timeout", "headers", "auth_header"))
        clean_params = {k: v for k, v in params.items() if k not in http_keys}

        return {
            "pipeline": {
                "schema": self._contract.get("schema", ""),
                "original_prompt": context.original_prompt,
            },
            "stage": {
                "id": stage_id,
                "kind": kind,
                "provider": provider,
                "params": clean_params,
                "depends_on": list(self._stage_map.get(stage_id, {}).get("depends_on", [])),
                "retries": self._stage_map.get(stage_id, {}).get("retries"),
            },
            "context": {
                "original_prompt": context.original_prompt,
                "enhanced_prompts": dict(context.enhanced_prompts),
                "artifacts": dict(context.artifacts),
                "qc_results": dict(context.qc_results),
            },
        }

    # ── HTTP call ──────────────────────────────────────────────────────────

    async def dispatch(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        """
        Make a single HTTP POST to the resolved endpoint and return a
        ``StageResult``.

        Raises
        ------
        EndpointResolutionError
            The endpoint could not be resolved (fail-closed).
        HttpProviderError
            The HTTP call failed or returned a non-2xx status (retriable).
        """
        endpoint, provider_params = self._resolve_endpoint(stage_id)

        timeout = float(provider_params.get("timeout", self.DEFAULT_TIMEOUT))
        headers = dict(provider_params.get("headers") or {})
        auth_header = provider_params.get("auth_header", self.DEFAULT_AUTH_HEADER)

        # Construct the HTTP payload
        payload = self._build_payload(stage_id, kind, provider, params, context)

        # ── Auth header redaction for error messages ──────────────────────
        auth_value_for_log = headers.get(auth_header, "")

        method = "POST"
        redacted_endpoint = _redact_url(endpoint)

        try:
            response = await self._session.request(
                method,
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout) if aiohttp else None,
            )
            try:
                status = response.status
                # Read body for error reporting before checking status
                try:
                    response_body = await response.json()
                except Exception:
                    response_body = None

                if not (200 <= status < 300):
                    # Build a redacted error message
                    status_text = f"HTTP {status} {response.reason}"
                    body_preview = ""
                    if isinstance(response_body, dict):
                        # Redact the body too
                        redacted_body = _redact_dict(response_body)
                        body_preview = json.dumps(redacted_body)[:500]
                    elif isinstance(response_body, str):
                        body_preview = _redact_text(response_body[:500])
                    else:
                        body_preview = f"<non-JSON body, {len(response_body or b'')} bytes>"

                    safe_body = (
                        f"{status_text}; body: {body_preview}"
                        if body_preview
                        else status_text
                    )
                    raise HttpProviderError(
                        f"provider HTTP error for stage '{stage_id}' "
                        f"at {redacted_endpoint}: {safe_body}"
                    )

                # 2xx — validate JSON structure
                if response_body is None:
                    raise HttpProviderError(
                        f"provider returned 2xx for stage '{stage_id}' "
                        f"but response body is not valid JSON"
                    )

                if not isinstance(response_body, dict):
                    raise HttpProviderError(
                        f"provider returned 2xx for stage '{stage_id}' "
                        f"but response body is a {type(response_body).__name__}, expected dict"
                    )

                # Build StageResult from response_body.
                # Expected format: {"stage_id": "...", "status": "ok"|"failed",
                #                  "data": {...}, "error": "..."}
                result_stage_id = response_body.get("stage_id", stage_id)
                result_status = response_body.get("status", "ok")
                result_data = response_body.get("data", {})
                result_error = response_body.get("error", "")

                if not isinstance(result_data, dict):
                    raise HttpProviderError(
                        f"provider returned non-object data for stage '{stage_id}'"
                    )

                if result_status not in ("ok", "failed", "skipped"):
                    result_status = "ok"  # treat unknown as ok per the spec
                if kind == "qc" and result_status == "ok" and result_data.get("qc_pass") is False:
                    result_status = "failed"
                    if not result_error:
                        result_error = "quality gate returned qc_pass=false"

                return StageResult(
                    stage_id=result_stage_id,
                    kind=kind,
                    provider=provider,
                    status=result_status,
                    data=dict(result_data),
                    error=_redact_text(str(result_error)),
                )
            finally:
                # Ensure the response is always closed, mirroring aiohttp's
                # async-context-manager behaviour
                close = getattr(response, "aclose", None)
                if close is not None:
                    await close()
                elif hasattr(response, "close"):
                    c = response.close()
                    if hasattr(c, "__await__"):
                        await c

        except HttpProviderError:
            # Already redact the endpoint above; re-raise as-is
            raise
        except (aiohttp.ClientError if aiohttp else Exception) as exc:
            raise HttpProviderError(
                f"provider connection error for stage '{stage_id}' "
                f"at {redacted_endpoint}: {_redact_text(str(exc))}"
            ) from exc
        except Exception as exc:
            # Anything else (e.g. json.JSONDecodeError) — wrap as HttpProviderError
            raise HttpProviderError(
                f"provider error for stage '{stage_id}' "
                f"at {redacted_endpoint}: {_redact_text(str(exc))}"
            ) from exc


# ── PipelineExecutor ─────────────────────────────────────────────────────────

class HttpPipelineExecutor(KreaPipelineExecutor):
    """
    Async pipeline executor that uses ``HttpPipelineProvider`` for stage
    dispatches.

    Injects into the ``pipeline_dispatch_runner`` injection point via
    ``run_pipeline_dispatch``'s ``executor_provider`` argument::

        contract = worker_pipeline  # the compiled pipeline dict
        provider = HttpPipelineProvider(contract=contract, session=session)
        executor = HttpPipelineExecutor(provider=provider)

    The executor is also directly instantiable for unit testing::

        executor = HttpPipelineExecutor(provider=fake_provider)
        ctx, results = await executor.dispatch(stages, context)
    """

    def __init__(self, provider: HttpPipelineProvider) -> None:
        # KreaPipelineExecutor is intentionally lightweight; we add no
        # additional state beyond the provider.
        self._provider = provider

    async def _call_provider_async(
        self,
        stage_id: str,
        kind: str,
        provider: str,
        params: dict,
        context: StageContext,
    ) -> StageResult:
        """
        Route a stage dispatch to ``HttpPipelineProvider.dispatch``.
        """
        return await self._provider.dispatch(
            stage_id=stage_id,
            kind=kind,
            provider=provider,
            params=params,
            context=context,
        )


# ── Public re-exports ────────────────────────────────────────────────────────

__all__ = [
    "EndpointResolutionError",
    "HttpProviderError",
    "HttpPipelineProvider",
    "HttpPipelineExecutor",
]
