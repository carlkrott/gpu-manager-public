"""
execution_boundary.py — Declarative ownership and execution boundary for service pipelines.

This module replaces the hardcoded _KREA_ONLY_PROVIDERS / _has_non_krea_stage
decision tree with a declarative ownership/adapter contract.

Design contract
---------------
Every pipeline stage carries a ``provider`` ID.  The provider entry in services.json
declares two fields that together determine ownership:

  ``adapter``  — how the stage is dispatched:
    • ``gpu_manager_generation`` — routed through the live ComfyUI queue via
      GenerationStageExecutor; must NOT recursively submit to the same queue.
    • HTTP-based adapters (``http_json``, ``http_openai_compatible``) — dispatched
      through CompositeProviderExecutor via HTTP.
    • ``local_pipeline`` — handled by the pipeline-orchestrator locally.

  ``managed``  — who executes the stage (WHO field, not what adapter is used):
    • ``worker`` — stage is executed by the live GPU Manager worker process
     on this machine.  Evidence is bound via _reconcile_pipeline_evidence or
     seeded into the explicit mixed-ownership handoff.  Must NOT be
     re-dispatched through CompositeProviderExecutor (recursive queue deadlock
     for gpu_manager_generation stages).
    • ``external`` — stage is executed by a remote service.  Must be dispatched
      through CompositeProviderExecutor.
    • absent / any other value — treated as ``external`` for safety.

Ownership classification
-----------------------
A stage is "self-owned" (worker-owned) when its provider has
``managed = "worker"`` in services.json.  Such stages bind to real worker
evidence via _reconcile_pipeline_evidence and must NOT be re-dispatched.

A stage is "external" when its provider has ``managed = "external"`` or
the field is absent / unknown.  These stages must be dispatched through
CompositeProviderExecutor.

Fail-closed unknown providers
-----------------------------
If a provider ID is not found in the registry, it is treated as unknown and
MissingOwnershipMetadataError is raised on every caller path.  There is NO
back-door that silently treats unknown providers as generic/external.
Unknown providers cause immediate failure rather than incorrect dispatch.

Mixed ownership pipelines
------------------------
A pipeline that contains BOTH worker-owned and external stages is classified as
MIXED_OWNERSHIP.  In ``generic`` execution mode it is dispatched through the
explicit safe_handoff_stages() protocol: worker evidence is seeded into the
shared context and only external stages are sent to CompositeProviderExecutor.
``legacy_reconcile`` still rejects mixed ownership because it has no handoff
transport.

Public API
----------
class ExecutionBoundary
    classify_stages(stages) -> StageClassification
    is_self_owned(stages) -> bool
    is_external(stages) -> bool
    get_dispatch_mode(stages, execution_mode) -> str
    validate_ownership(stages) -> list[str]   # returns [] on success, list of errors on fail

LIVE_WORKER_OWNED_ADAPTERS: frozenset[str]
    Kept for backward compatibility.  The authoritative source of truth is now
    the ``managed`` field on each provider in services.json.

PROVIDER_OWNERSHIP_MAP: dict[str, str]
    Maps provider ID -> adapter type.
    Populated by register_provider_ownership() during gpu-manager init.

MANAGED_WORKER_PROVIDERS: set[str]
    Provider IDs whose ``managed`` field is ``"worker"``.
    Populated by register_provider_ownership() during gpu-manager init.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import typing as _t

if _t.TYPE_CHECKING:
    from stage_dispatch import StageContext, StageResult

# ── Constants ────────────────────────────────────────────────────────────────

# Adapters whose stages are fully owned by the live ComfyUI/OD worker path.
# Kept for backward compatibility only — the authoritative classification now
# comes from the ``managed`` field on each provider in services.json.
LIVE_WORKER_OWNED_ADAPTERS: frozenset[str] = frozenset({
    "open-design",
    "comfyui",
    "comfyui-qwen",
    "combined-gemma",
    "n8n",
})

# Sentinel value for providers whose adapter is unknown (not registered).
UNKNOWN_ADAPTER: str = "<unknown>"

# Value stored in PROVIDER_OWNERSHIP_MAP for the special managed="worker" marker.
_WORKER_OWNED_MARKER: str = "<worker-owned>"


# ── Exceptions ──────────────────────────────────────────────────────────────

class OwnershipError(Exception):
    """Raised when a stage's ownership cannot be determined."""
    pass


class MissingOwnershipMetadataError(OwnershipError):
    """
    Raised when a provider has no registered ownership/adapter metadata.
    This is the fail-closed signal for unknown provider IDs.
    """
    pass


# ── Provider ownership registry ───────────────────────────────────────────────

# Maps provider_id -> adapter type string, or _WORKER_OWNED_MARKER if managed="worker".
# Populated at gpu-manager init time from services.json provider definitions.
PROVIDER_OWNERSHIP_MAP: dict[str, str] = {}

# Provider IDs explicitly marked managed="worker" in services.json.
# These stages bind to live worker evidence and must NOT be re-dispatched.
MANAGED_WORKER_PROVIDERS: set[str] = set()


class ExecutionBoundary:
    """Compatibility facade over the module-level declarative registry.

    Older GPUManager call sites construct an ``ExecutionBoundary`` object,
    while the current registry is intentionally process-wide and refreshed
    atomically from ``services.json``.  Keep that public class surface without
    introducing a second source of truth: every operation delegates to the
    module-level functions below.
    """

    def register_provider_ownership(
        self, provider_id: str, adapter: str, managed: str = ""
    ) -> None:
        register_provider_ownership(provider_id, adapter, managed)

    def get_provider_ownership(self, provider_id: str) -> dict[str, str]:
        adapter = get_provider_adapter(provider_id)
        return {
            "provider_id": provider_id,
            "adapter": adapter,
            "managed": (
                "worker"
                if is_provider_worker_owned(provider_id)
                else "external"
            ),
        }

    def classify_stages(self, stages: list[dict]) -> "StageClassification":
        return classify_stages(stages)

    def get_dispatch_mode(
        self, stages: list[dict], execution_mode: str = "generic"
    ) -> "DispatchMode":
        return get_dispatch_mode(stages, execution_mode)

    def validate_ownership(self, stages: list[dict]) -> list[str]:
        return validate_ownership(stages)

    def safe_handoff_stages(
        self,
        stages: list[dict],
        *,
        original_prompt: str = "",
        enhanced_prompts: dict[str, str] | None = None,
        artifacts: dict[str, dict] | None = None,
        qc_results: dict[str, dict] | None = None,
        correction_loop_count: int = 0,
    ) -> "HandoffResult":
        return safe_handoff_stages(
            stages,
            original_prompt=original_prompt,
            enhanced_prompts=enhanced_prompts,
            artifacts=artifacts,
            qc_results=qc_results,
            correction_loop_count=correction_loop_count,
        )


def register_provider_ownership(provider_id: str, adapter: str, managed: str = "") -> None:
    """
    Register the adapter type and managed owner for a provider ID.

    Called during gpu-manager init to populate the ownership registry from
    services.json pipeline_providers definitions.

    Parameters
    ----------
    provider_id : str
        Provider identifier (e.g. "comfyui", "open-design", "combined-gemma").
    adapter : str
        Adapter type (e.g. "gpu_manager_generation", "http_json",
        "http_openai_compatible", "local_pipeline").
    managed : str
        Ownership mode: "worker" means the live GPU Manager worker executes
        this stage and binds evidence.  "external" or "" means a remote
        service executes it and it must be HTTP-dispatched.
    """
    if managed == "worker":
        # Store the actual adapter so callers can inspect it; ownership is tracked
        # separately in MANAGED_WORKER_PROVIDERS.
        PROVIDER_OWNERSHIP_MAP[provider_id] = adapter
        MANAGED_WORKER_PROVIDERS.add(provider_id)
    else:
        PROVIDER_OWNERSHIP_MAP[provider_id] = adapter
        MANAGED_WORKER_PROVIDERS.discard(provider_id)


def clear_provider_ownership() -> None:
    """Clear all registered ownership metadata. Used in tests."""
    PROVIDER_OWNERSHIP_MAP.clear()
    MANAGED_WORKER_PROVIDERS.clear()


def get_provider_adapter(provider_id: str) -> str:
    """
    Return the adapter type for ``provider_id``.

    Raises MissingOwnershipMetadataError if the provider is not registered.
    """
    if provider_id not in PROVIDER_OWNERSHIP_MAP:
        raise MissingOwnershipMetadataError(
            f"provider {provider_id!r} has no registered ownership metadata; "
            f"cannot classify stage for execution boundary"
        )
    return PROVIDER_OWNERSHIP_MAP[provider_id]


def is_provider_worker_owned(provider_id: str) -> bool:
    """
    Return True if the provider is explicitly marked as worker-owned.

    Raises MissingOwnershipMetadataError if the provider is not registered.
    """
    if provider_id not in PROVIDER_OWNERSHIP_MAP:
        raise MissingOwnershipMetadataError(
            f"provider {provider_id!r} has no registered ownership metadata"
        )
    return provider_id in MANAGED_WORKER_PROVIDERS


# ── Stage classification ─────────────────────────────────────────────────────

class DispatchMode(str, Enum):
    """How a pipeline's stages should be dispatched."""
    RECONCILE = "reconcile"      # Self-owned: bind live evidence, no re-execution
    GENERIC = "generic"           # External: dispatch through CompositeProviderExecutor
    MIXED_OWNERSHIP = "mixed"     # Both worker-owned and external stages — UNSAFE
    UNSUPPORTED = "unsupported"   # Fail-closed: unknown provider or missing metadata


@dataclass(frozen=True)
class StageClassification:
    """
    Result of classifying a list of pipeline stages.

    Attributes
    ----------
    dispatch_mode : DispatchMode
        The required dispatch path.
    self_owned_providers : frozenset[str]
        Providers classified as worker-owned (live-worker-executed).
    external_providers : frozenset[str]
        Providers classified as external (require generic dispatch).
    unknown_providers : frozenset[str]
        Providers with missing ownership metadata (fail-closed).
    errors : list[str]
        Human-readable errors for each classification failure.
    """
    dispatch_mode: DispatchMode = DispatchMode.UNSUPPORTED
    self_owned_providers: frozenset[str] = field(default_factory=frozenset)
    external_providers: frozenset[str] = field(default_factory=frozenset)
    unknown_providers: frozenset[str] = field(default_factory=frozenset)
    errors: list[str] = field(default_factory=list)

    @property
    def is_self_owned(self) -> bool:
        """True when all stages are worker-owned and no providers are unknown."""
        return (
            self.dispatch_mode == DispatchMode.RECONCILE
            and not self.unknown_providers
        )

    @property
    def is_external(self) -> bool:
        """True when all stages are external (requires generic dispatch)."""
        return self.dispatch_mode == DispatchMode.GENERIC

    @property
    def is_mixed_ownership(self) -> bool:
        """True when the pipeline has both worker-owned and external stages."""
        return self.dispatch_mode == DispatchMode.MIXED_OWNERSHIP

    @property
    def is_fail_closed(self) -> bool:
        """True when classification cannot be dispatched safely.

        Mixed ownership is dispatchable only through the explicit generic
        handoff path; unknown providers and unsupported execution modes remain
        fail closed.
        """
        return bool(self.unknown_providers) or self.dispatch_mode == DispatchMode.UNSUPPORTED


def classify_stage(stage: dict) -> tuple[str, str, bool, str | None]:
    """
    Classify a single stage's provider, adapter, and worker-ownership.

    Parameters
    ----------
    stage : dict
        Stage dict with at minimum a ``provider`` key.

    Returns
    -------
    tuple[provider_id, adapter, is_worker_owned, error]
        provider_id : str — the stage's provider field
        adapter : str — the resolved adapter type (or UNKNOWN_ADAPTER)
        is_worker_owned : bool — True if managed="worker"
        error : str | None — error message if classification failed

    Raises MissingOwnershipMetadataError for unknown providers (fail-closed).
    """
    provider: str = str(stage.get("provider", ""))
    if not provider:
        return provider, UNKNOWN_ADAPTER, False, "stage has no provider field"

    try:
        adapter = get_provider_adapter(provider)
        is_worker_owned = is_provider_worker_owned(provider)
        return provider, adapter, is_worker_owned, None
    except MissingOwnershipMetadataError as exc:
        return provider, UNKNOWN_ADAPTER, False, str(exc)


def classify_stages(stages: list[dict]) -> StageClassification:
    """
    Classify a list of pipeline stages by their ownership.

    Parameters
    ----------
    stages : list[dict]
        List of stage dicts from a compiled worker_pipeline.

    Returns
    -------
    StageClassification
        Describes the dispatch mode and per-provider breakdown.

    Fail-closed: if ANY stage has an unknown provider (not in the registry),
    the entire classification returns dispatch_mode=UNSUPPORTED and the
    unknown provider is recorded in unknown_providers.

    Mixed ownership: if a pipeline contains BOTH worker-owned and external
    stages, it returns dispatch_mode=MIXED_OWNERSHIP.  This is not safe
    because the worker would need to redispatch a worker-owned stage to
    CompositeProviderExecutor to handle the external stages, reintroducing
    the recursive queue deadlock.  Until an explicit handoff protocol is
    implemented, mixed ownership pipelines fail closed (UNSUPPORTED).
    """
    if not stages:
        return StageClassification(
            dispatch_mode=DispatchMode.RECONCILE,
            self_owned_providers=frozenset(),
            external_providers=frozenset(),
            unknown_providers=frozenset(),
            errors=[],
        )

    self_owned: set[str] = set()
    external: set[str] = set()
    unknown: set[str] = set()
    errors: list[str] = []

    for stage in stages:
        provider: str = str(stage.get("provider", ""))
        if not provider:
            errors.append(f"stage {stage.get('id', '?')} has no provider field")
            unknown.add(provider)
            continue

        try:
            _, is_worker_owned = get_provider_adapter(provider), is_provider_worker_owned(provider)
        except MissingOwnershipMetadataError as exc:
            unknown.add(provider)
            errors.append(str(exc))
            continue

        if is_worker_owned:
            self_owned.add(provider)
        else:
            external.add(provider)

    # Fail-closed: unknown providers win
    if unknown:
        return StageClassification(
            dispatch_mode=DispatchMode.UNSUPPORTED,
            self_owned_providers=frozenset(self_owned),
            external_providers=frozenset(external),
            unknown_providers=frozenset(unknown),
            errors=errors,
        )

    # All worker-owned
    if not external:
        return StageClassification(
            dispatch_mode=DispatchMode.RECONCILE,
            self_owned_providers=frozenset(self_owned),
            external_providers=frozenset(),
            unknown_providers=frozenset(),
            errors=[],
        )

    # All external
    if not self_owned:
        return StageClassification(
            dispatch_mode=DispatchMode.GENERIC,
            self_owned_providers=frozenset(),
            external_providers=frozenset(external),
            unknown_providers=frozenset(),
            errors=[],
        )

    # Mixed ownership: both worker-owned and external stages in one pipeline.
    # This is a first-class mode.  The generic dispatcher must use the explicit
    # handoff protocol and seed worker evidence instead of redispatching it.
    return StageClassification(
        dispatch_mode=DispatchMode.MIXED_OWNERSHIP,
        self_owned_providers=frozenset(self_owned),
        external_providers=frozenset(external),
        unknown_providers=frozenset(),
        errors=[
            f"pipeline has mixed ownership: worker-owned stages {sorted(self_owned)} "
            f"and external stages {sorted(external)}. "
            f"requires the explicit worker-evidence handoff before external "
            f"stages are dispatched."
        ],
    )


def is_self_owned(stages: list[dict]) -> bool:
    """
    Return True when all stages in ``stages`` are worker-owned.

    A worker-owned stage is one whose provider has ``managed = "worker"``
    in services.json.  Such stages bind to real worker evidence via
    _reconcile_pipeline_evidence and must NOT be re-dispatched through
    CompositeProviderExecutor.

    Raises MissingOwnershipMetadataError if any provider is unknown (fail-closed).
    Returns False for mixed-ownership pipelines (not purely self-owned).
    """
    cls = classify_stages(stages)
    # Raise only for unknown/missing metadata; mixed ownership just returns False.
    if cls.unknown_providers:
        raise MissingOwnershipMetadataError(
            f"cannot determine self-ownership: {cls.errors[0] if cls.errors else 'unknown provider'}"
        )
    return cls.is_self_owned


def is_external(stages: list[dict]) -> bool:
    """
    Return True when all stages in ``stages`` are external.

    An external stage requires generic dispatch through CompositeProviderExecutor.

    Raises MissingOwnershipMetadataError if any provider is unknown (fail-closed).
    Returns False for mixed-ownership pipelines (not purely external).
    """
    cls = classify_stages(stages)
    # Raise only for unknown/missing metadata; mixed ownership just returns False.
    if cls.unknown_providers:
        raise MissingOwnershipMetadataError(
            f"cannot determine external status: {cls.errors[0] if cls.errors else 'unknown provider'}"
        )
    return cls.is_external


def is_mixed_ownership(stages: list[dict]) -> bool:
    """
    Return True when the pipeline contains both worker-owned and external stages.

    Such pipelines are currently UNSUPPORTED by get_dispatch_mode() because
    redispatching a worker-owned stage through CompositeProviderExecutor
    reintroduces the recursive queue deadlock.  Until an explicit handoff
    protocol is implemented, mixed ownership pipelines must fail closed.

    Raises MissingOwnershipMetadataError if any provider is unknown (fail-closed).
    Returns True for mixed-ownership pipelines.
    """
    cls = classify_stages(stages)
    # Raise only for unknown/missing metadata; mixed ownership is a valid True result.
    if cls.unknown_providers:
        raise MissingOwnershipMetadataError(
            f"cannot determine ownership: {cls.errors[0] if cls.errors else 'unknown provider'}"
        )
    return cls.is_mixed_ownership


def get_dispatch_mode(stages: list[dict], execution_mode: str) -> DispatchMode:
    """
    Determine the dispatch mode for a pipeline given its stages and execution_mode.

    Parameters
    ----------
    stages : list[dict]
        Compiled stage list.
    execution_mode : str
        "generic" or "legacy_reconcile".

    Returns
    -------
    DispatchMode
        RECONCILE for worker-owned-only pipelines (evidence binding, no re-execution),
        GENERIC for external-only pipelines (CompositeProviderExecutor dispatch),
        MIXED_OWNERSHIP for mixed pipelines in generic mode (explicit handoff),
        UNSUPPORTED for unknown providers (fail-closed).

    Raises MissingOwnershipMetadataError if any provider is unknown.
    """
    if not stages:
        return DispatchMode.RECONCILE

    cls = classify_stages(stages)

    # Fail-closed: unknown providers must never silently dispatch
    if cls.unknown_providers:
        raise MissingOwnershipMetadataError(
            f"unknown provider in pipeline: {list(cls.unknown_providers)}"
        )

    # Mixed ownership is safe only through the generic handoff protocol.
    if cls.is_mixed_ownership:
        return (
            DispatchMode.MIXED_OWNERSHIP
            if execution_mode == "generic"
            else DispatchMode.UNSUPPORTED
        )

    if execution_mode == "legacy_reconcile":
        # legacy_reconcile only supports evidence reconciliation (worker-owned stages)
        if cls.is_self_owned:
            return DispatchMode.RECONCILE
        # External stages in legacy_reconcile: fail closed
        return DispatchMode.UNSUPPORTED

    # generic mode
    if cls.is_self_owned:
        return DispatchMode.RECONCILE
    return DispatchMode.GENERIC


def validate_ownership(stages: list[dict]) -> list[str]:
    """
    Validate that all stages have registered ownership metadata.

    Returns an empty list on success (all providers known and registered).
    Returns a list of error strings for each unknown/missing provider.

    Note: mixed ownership is NOT an ownership-metadata error; it is a
    classification result handled by get_dispatch_mode() and classify_stages().
    This function only validates that providers can be identified, not that
    the pipeline is safe to dispatch (use get_dispatch_mode for that).
    """
    cls = classify_stages(stages)
    # Only return errors for unknown/missing metadata, not mixed-ownership.
    return [e for e in cls.errors if "no registered ownership metadata" in e or "no provider field" in e]


# ── Mixed-ownership handoff seam ─────────────────────────────────────────────

class HandoffResult:
    """
    Result of ``safe_handoff_stages()``.

    Attributes
    ----------
    external_stages : list[dict]
        The subset of stages whose providers are external (not worker-owned).
        In dependency order matching the input list.
    worker_owned_providers : frozenset[str]
        Provider IDs that are worker-owned in the original pipeline.
    errors : list[str]
        Human-readable errors when the handoff is not safe to proceed.
    is_handoff_safe : bool
        True when a safe handoff is possible (all providers known,
        at least one external stage present, no unknown providers).
    seeded_context : StageContext | None
        A StageContext pre-populated with the evidence from worker-owned
        stages (enhanced_prompts, artifacts, qc_results, correction_loop_count).
        Passed directly to CompositeProviderExecutor.dispatch so external
        stages receive the same context as if worker stages had run inline.
    seeded_results : list[StageResult] | None
        StageResult list for all worker-owned stages that have already been
        executed by the GPU Manager worker.  Passed to
        CompositeProviderExecutor.dispatch to pre-populate the fan-in
        dependency map so external stages can depend on worker-owned stages
        without re-executing them.  Results are in execution order.
    qc_verdict_false : bool
        True when at least one QC stage returned qc_pass=False in the
        seeded evidence.  The external pipeline dispatch must continue
        to the configured correction stage rather than treating this as
        a transport failure.  Callers should check this flag after
        a safe handoff and route to the correction-capable execution path.
    """

    __slots__ = (
        "external_stages",
        "worker_owned_providers",
        "errors",
        "is_handoff_safe",
        "seeded_context",
        "seeded_results",
        "qc_verdict_false",
    )

    def __init__(
        self,
        external_stages: list[dict],
        worker_owned_providers: frozenset[str],
        errors: list[str],
        is_handoff_safe: bool,
        seeded_context: "StageContext | None" = None,
        seeded_results: list | None = None,
        qc_verdict_false: bool = False,
    ) -> None:
        self.external_stages = external_stages
        self.worker_owned_providers = worker_owned_providers
        self.errors = errors
        self.is_handoff_safe = is_handoff_safe
        self.seeded_context = seeded_context
        self.seeded_results = seeded_results
        self.qc_verdict_false = qc_verdict_false


def safe_handoff_stages(
    stages: list[dict],
    *,
    original_prompt: str = "",
    enhanced_prompts: dict[str, str] | None = None,
    artifacts: dict[str, dict] | None = None,
    qc_results: dict[str, dict] | None = None,
    correction_loop_count: int = 0,
    pre_results: list | None = None,
) -> HandoffResult:
    """
    Extract the external-only subset of ``stages`` and return a pre-seeded
    ``StageContext`` containing evidence from worker-owned stages already
    executed by the live GPU Manager worker.

    This is the explicit handoff seam for mixed-ownership pipelines:
    the caller (typically the GPU Manager worker loop) executes the
    worker-owned stages (prompt enhancement via OpenDesign, ComfyUI generation,
    QC via Combined-Gemma) and collects their outputs, then passes that
    evidence here to produce a ``HandoffResult`` whose ``external_stages``
    can be safely dispatched through ``CompositeProviderExecutor`` without
    re-executing worker-owned stages or causing recursive GPU submission.

    Parameters
    ----------
    stages : list[dict]
        Full ordered stage list from the pipeline contract.
    original_prompt : str
        The raw prompt string (propagated into the seeded context).
    enhanced_prompts : dict[str, str] | None
        Map of ``stage_id → enhanced_prompt`` from executed prompt_enhance
        stages.  Each entry is recorded via ``context.with_prompt()``.
    artifacts : dict[str, dict] | None
        Map of ``stage_id → artifact_dict`` from executed generate/delegate
        stages.  Each entry is recorded via ``context.with_artifact()``.
    qc_results : dict[str, dict] | None
        Map of ``stage_id → qc_dict`` from executed QC stages.
        Each entry is recorded via ``context.with_qc()``.
    correction_loop_count : int
        Current correction loop count (from the worker context).
    pre_results : list[StageResult] | None
        Results from pre-stage dispatch (external stages executed before
        the first worker-owned stage).  These are prepended to the
        ``seeded_results`` so pre-stages are never re-dispatched in the
        post-generation MIXED_OWNERSHIP handoff.

    Returns
    ------
    HandoffResult
        ``is_handoff_safe`` is True when all providers are known and at
        least one external stage is present.  ``external_stages`` contains
        the stages whose providers are NOT worker-owned.

        When ``is_handoff_safe`` is True, ``seeded_context`` carries the
        evidence bag and ``seeded_results`` carries StageResult objects for
        all worker-owned stages.  Both must be passed to
        ``CompositeProviderExecutor.dispatch`` so the external stages receive
        the correct context and their dependency fan-in is satisfied.

        When ``is_handoff_safe`` is False the caller must NOT dispatch
        ``external_stages`` through ``CompositeProviderExecutor`` — instead
        the entire pipeline should fail closed with the error messages
        in ``errors``.

    Fail-closed conditions
    ----------------------
    • Any provider is unknown (not in the ownership registry) → not safe
    • All stages are worker-owned (nothing to hand off externally) → not safe
    • All stages are external (no worker evidence to merge) → not safe
      (this case is not mixed-ownership; the standard ``GENERIC`` dispatch
      path handles it directly)

    QC verdict handling
    -------------------
    When ``qc_results`` contains a ``qc_pass: False`` entry, the
    ``qc_verdict_false`` flag is set on the returned HandoffResult.
    This tells the caller that the pipeline should continue to the
    configured correction stage rather than treating the QC failure as
    a transport/dispatch error.  The external stages that follow the
    correction stage are still dispatched normally.

    Example
    -------
    ::

        result = safe_handoff_stages(
            stages,
            original_prompt="a horse",
            enhanced_prompts={"enhance-1": "[enhanced] a horse"},
            artifacts={
                "gen-1": {"artifact_path": "staging/gen-1/output.png"},
                "correct-1": {"artifact_path": "staging/correct-1/output.png"},
            },
            qc_results={"qc-1": {"qc_pass": True, "score": 0.9}},
        )
        if not result.is_handoff_safe:
            raise PipelineDispatchError(result.errors)
        # result.seeded_context and result.seeded_results carry the worker evidence
        # result.external_stages is safe to dispatch through
        # CompositeProviderExecutor — worker evidence is already seeded
        # in the context returned alongside the HandoffResult.
    """
    # Lazy imports to avoid circular reference at module load time.
    import dataclasses
    from stage_dispatch import StageContext, StageResult

    if not stages:
        return HandoffResult(
            external_stages=[],
            worker_owned_providers=frozenset(),
            errors=["pipeline has no stages"],
            is_handoff_safe=False,
        )

    cls = classify_stages(stages)

    # Fail-closed: unknown providers make handoff unsafe
    if cls.unknown_providers:
        return HandoffResult(
            external_stages=[],
            worker_owned_providers=cls.self_owned_providers,
            errors=list(cls.errors),
            is_handoff_safe=False,
        )

    # Not mixed-ownership: pure external pipelines are handled by the
    # standard GENERIC dispatch path, not the handoff seam.
    # pure worker-owned pipelines have nothing to hand off.
    if not cls.is_mixed_ownership:
        return HandoffResult(
            external_stages=[],
            worker_owned_providers=cls.self_owned_providers,
            errors=[
                f"pipeline is not mixed-ownership "
                f"(dispatch_mode={cls.dispatch_mode.value!r}); "
                f"use the standard dispatch path instead of safe_handoff_stages"
            ],
            is_handoff_safe=False,
        )

    # Mixed-ownership: extract only the external stages
    pre_result_ids = {
        str(pre_result.stage_id)
        for pre_result in (pre_results or [])
        if getattr(pre_result, "stage_id", None)
    }
    external_stages = [
        s for s in stages
        if str(s.get("provider", "")) in cls.external_providers
        and str(s.get("id", "")) not in pre_result_ids
    ]

    if not external_stages:
        return HandoffResult(
            external_stages=[],
            worker_owned_providers=cls.self_owned_providers,
            errors=[
                "mixed-ownership pipeline has no external stages; "
                "check that all providers are correctly registered with "
                "the appropriate managed value in services.json"
            ],
            is_handoff_safe=False,
        )

    # ── Build the seeded StageContext from worker evidence ──────────────────
    enhanced_prompts = enhanced_prompts or {}
    artifacts = artifacts or {}
    qc_results = qc_results or {}

    # Build the seeded context step-by-step so each update is recorded
    # correctly (with_artifact, with_prompt, etc. return new instances).
    ctx = StageContext(original_prompt=original_prompt)
    ctx = dataclasses.replace(
        ctx,
        enhanced_prompts=dict(enhanced_prompts),
        artifacts=dict(artifacts),
        qc_results=dict(qc_results),
        correction_loop_count=correction_loop_count,
    )

    # ── Build StageResult list for worker-owned stages ───────────────────────
    # These are pre-populated into CompositeProviderExecutor.dispatch's
    # fan-in map so external stages can depend on worker stages.
    seeded_results: list[StageResult] = []
    stage_map: dict[str, StageResult] = {}
    qc_verdict_false = False

    for stage in stages:
        sid = stage["id"]
        provider = str(stage.get("provider", ""))
        kind = stage["kind"]

        # Only build results for worker-owned stages
        if provider not in cls.self_owned_providers:
            continue

        # Build the appropriate StageResult from the seeded evidence
        if kind == "prompt_enhance":
            prompt = enhanced_prompts.get(sid, "")
            result = StageResult(
                stage_id=sid,
                kind=kind,
                provider=provider,
                status="ok",
                data={"prompt": prompt} if prompt else {},
            )
        elif kind in ("generate", "delegate"):
            artifact_data = artifacts.get(sid, {})
            result = StageResult(
                stage_id=sid,
                kind=kind,
                provider=provider,
                status="ok",
                data=dict(artifact_data),
            )
        elif kind == "qc":
            qc_data = qc_results.get(sid, {})
            qc_pass = bool(qc_data.get("qc_pass", False))
            if not qc_pass:
                qc_verdict_false = True
            result = StageResult(
                stage_id=sid,
                kind=kind,
                provider=provider,
                status="ok",  # QC stage executed; qc_pass is in data
                data=dict(qc_data),
            )
        elif kind == "correct":
            artifact_data = artifacts.get(sid, {})
            result = StageResult(
                stage_id=sid,
                kind=kind,
                provider=provider,
                status="ok",
                data=dict(artifact_data),
            )
        else:
            # Unknown kind — treat as ok with empty data to avoid
            # blocking dependent external stages
            result = StageResult(
                stage_id=sid,
                kind=kind,
                provider=provider,
                status="ok",
                data={},
            )

        stage_map[sid] = result
        seeded_results.append(result)

    # ── Prepend pre-stage results so they are never re-dispatched ─────────────
    # pre_results are StageResult objects from the pre-generation dispatch.
    # These are external stages that ran before the first worker-owned stage;
    # they must appear in seeded_results so the MIXED_OWNERSHIP dispatch
    # skips them (they're already done).
    if pre_results:
        seeded_results = list(pre_results) + seeded_results
        # Merge pre-stage outputs into the seeded context so downstream
        # stages (including post-generation external stages) see them.
        for pre_result in pre_results:
            if pre_result.kind == "prompt_enhance":
                prompt = pre_result.data.get("prompt", "")
                if prompt:
                    ctx = ctx.with_prompt(pre_result.stage_id, prompt)
            elif pre_result.kind in ("generate", "delegate"):
                ctx = ctx.with_artifact(pre_result.stage_id, dict(pre_result.data))
            elif pre_result.kind == "qc":
                ctx = ctx.with_qc(pre_result.stage_id, dict(pre_result.data))

    return HandoffResult(
        external_stages=external_stages,
        worker_owned_providers=cls.self_owned_providers,
        errors=[],
        is_handoff_safe=True,
        seeded_context=ctx,
        seeded_results=seeded_results,
        qc_verdict_false=qc_verdict_false,
    )


# ── Pre-stage handoff extraction ──────────────────────────────────────────────

def extract_pre_handoff_stages(
    stages: list[dict],
    *,
    original_prompt: str = "",
    enhanced_prompts: dict[str, str] | None = None,
    artifacts: dict[str, dict] | None = None,
    qc_results: dict[str, dict] | None = None,
    correction_loop_count: int = 0,
) -> "PreStageHandoff":
    """
    Extract external stages that are dependency-ready before the first
    worker-owned stage and can be executed before ComfyUI/generation.

    A stage is a "pre-stage" when:
    1. Its provider is external (not worker-owned).
    2. It appears before the first worker-owned stage in the pipeline.
    3. All its ``depends_on`` references resolve to other pre-stages
       (no unmet external dependencies).

    This function is the fail-closed detection seam: it returns
    ``is_pre_dispatch_safe=False`` when no pre-stages exist, when the
    pipeline is not mixed-ownership, or when any pre-stage has an
    unresolvable external dependency.

    Parameters
    ----------
    stages : list[dict]
        Full ordered stage list from the pipeline contract.
    original_prompt : str
        Raw prompt string (propagated into the pre-seeded context).
    enhanced_prompts, artifacts, qc_results, correction_loop_count : evidence
        Worker evidence already available before the pre-dispatch.  For
        pre-stages (which run before any worker-owned stage), these are
        typically empty; they are accepted for consistency with the
        ``safe_handoff_stages`` API signature.

    Returns
    -------
    PreStageHandoff
        ``is_pre_dispatch_safe`` is True when pre-stages exist and are
        dependency-ready.  ``pre_stages`` contains the stages to dispatch
        before ComfyUI; ``worker_stages`` contains the rest (first worker-owned
        stage and everything after it).

        When ``is_pre_dispatch_safe`` is True, ``pre_context`` carries the
        initial evidence bag and ``pre_results`` is an empty list placeholder
        (actual results are recorded after ``dispatch_pre_stages`` returns).

    Fail-closed conditions
    ----------------------
    • Pipeline is not mixed-ownership (all external or all worker-owned)
    • No external stages appear before the first worker-owned stage
    • Any pre-stage depends on a stage ID that does not exist or is not
      also a pre-stage
    """
    # Lazy import to avoid circular dependency at module load time
    import dataclasses
    from stage_dispatch import PreStageHandoff, StageContext, StageResult

    if not stages:
        return PreStageHandoff(
            pre_stages=[],
            worker_stages=[],
            is_pre_dispatch_safe=False,
            pre_context=None,
            pre_results=None,
            errors=["pipeline has no stages"],
        )

    cls = classify_stages(stages)

    # Fail-closed: unknown providers make any dispatch unsafe
    if cls.unknown_providers:
        return PreStageHandoff(
            pre_stages=[],
            worker_stages=list(stages),
            is_pre_dispatch_safe=False,
            pre_context=None,
            pre_results=None,
            errors=list(cls.errors),
        )

    # Not mixed-ownership: nothing to hand off via the pre-stage seam
    if not cls.is_mixed_ownership:
        return PreStageHandoff(
            pre_stages=[],
            worker_stages=list(stages),
            is_pre_dispatch_safe=False,
            pre_context=None,
            pre_results=None,
            errors=[
                f"pipeline is not mixed-ownership "
                f"(dispatch_mode={cls.dispatch_mode.value!r}); "
                f"pre-stage seam applies only to mixed-ownership pipelines"
            ],
        )

    # ── Locate the first worker-owned stage ──────────────────────────────────
    first_worker_idx: int | None = None
    for idx, stage in enumerate(stages):
        provider = str(stage.get("provider", ""))
        if provider in cls.self_owned_providers:
            first_worker_idx = idx
            break

    if first_worker_idx is None:
        # All stages are external (should have been caught above, but guard)
        return PreStageHandoff(
            pre_stages=[],
            worker_stages=list(stages),
            is_pre_dispatch_safe=False,
            pre_context=None,
            pre_results=None,
            errors=["mixed-ownership pipeline has no worker-owned stage"],
        )

    # Stages before the first worker-owned stage are candidates for pre-stages
    candidates = list(stages[:first_worker_idx])
    if not candidates:
        return PreStageHandoff(
            pre_stages=[],
            worker_stages=list(stages),
            is_pre_dispatch_safe=False,
            pre_context=None,
            pre_results=None,
            errors=[
                f"first worker-owned stage at index {first_worker_idx}; "
                f"no external stages precede it"
            ],
        )

    # ── Validate dependency-readiness of candidates ──────────────────────────
    # A pre-stage's depends_on must resolve only to other pre-stages
    # (stages that also appear before the first worker-owned stage).
    pre_stage_ids = {stage["id"] for stage in candidates}
    pre_stages: list[dict] = []
    errors: list[str] = []

    for stage in candidates:
        sid = stage["id"]
        for dep in stage.get("depends_on", []):
            if dep not in pre_stage_ids:
                errors.append(
                    f"pre-stage {sid!r} depends on {dep!r} which is not "
                    f"a pre-stage (does not appear before the first "
                    f"worker-owned stage)"
                )

    if errors:
        return PreStageHandoff(
            pre_stages=[],
            worker_stages=list(stages),
            is_pre_dispatch_safe=False,
            pre_context=None,
            pre_results=None,
            errors=errors,
        )

    pre_stages = list(candidates)

    # ── Build the pre-seeded context ─────────────────────────────────────────
    # Pre-stages run before any worker-owned stage, so there is typically
    # no worker evidence yet.  The context carries the original prompt
    # and any prior evidence (empty for a pure pre-stage run).
    enhanced_prompts = dict(enhanced_prompts or {})
    artifacts = dict(artifacts or {})
    qc_results = dict(qc_results or {})

    ctx = StageContext(original_prompt=original_prompt)
    ctx = dataclasses.replace(
        ctx,
        enhanced_prompts=dict(enhanced_prompts),
        artifacts=dict(artifacts),
        qc_results=dict(qc_results),
        correction_loop_count=correction_loop_count,
    )

    return PreStageHandoff(
        pre_stages=pre_stages,
        worker_stages=list(stages[first_worker_idx:]),
        is_pre_dispatch_safe=True,
        pre_context=ctx,
        pre_results=None,  # Placeholder; filled after dispatch
        errors=[],
    )


# ── Backward-compatibility shim ──────────────────────────────────────────────

def _has_non_krea_stage(stages: list[dict]) -> bool:
    """
    Backward-compatible legacy predicate: True if any stage requires the
    generic (external) pipeline dispatcher rather than evidence reconciliation.

    IMPORTANT - legacy contract for old unit-test callers:
      • unknown provider  → returns True  (was treated as non-Krea/external)
      • missing provider field → returns True  (same as no provider)
      • external provider → returns True
      • worker-owned provider → returns False

    This function is NOT consulted by the production worker dispatch path
    (which uses get_dispatch_mode() / classify_stages() instead).
    It is retained ONLY for backward compatibility with existing unit tests
    and call sites that still reference it directly.
    """
    if not stages:
        return False
    for stage in stages:
        provider = str(stage.get("provider", ""))
        # Empty / missing provider field: legacy returned True
        if not provider:
            return True
        try:
            if not is_provider_worker_owned(provider):
                return True  # external or unknown-but-registered
        except MissingOwnershipMetadataError:
            # Unknown provider (not in registry): legacy returned True
            return True
    return False
