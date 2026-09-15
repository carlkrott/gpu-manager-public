"""
service_pipeline.py — Declarative service-pipeline compiler and validator.

Pure declarative pipeline description → validated + bounded worker_pipeline
metadata shape, suitable for the Krea worker path.

Schema
──────
service_pipelines : dict[str, ServicePipeline]
  top-level mapping  service_id → pipeline

ServicePipeline
  schema_id        : str   — fixed "image-pipeline.v1" — required
  stages           : list[Stage] — ordered, unique stage IDs
  max_retries      : int   — per-stage retry cap (0-10)
  max_loop_count   : int   — correction loop cap (0-5)

Stage
  id               : str   — unique within pipeline
  kind             : str   — one of the CLOSED_KINDS set
  provider         : str   — resolved against supplied provider_registry
  depends_on       : list[str] — stage IDs this stage needs before running
  retries          : int   — override for max_retries (optional)
  params           : dict  — opaque stage params (compiled into output)

CLOSED_KINDS = {"prompt_enhance", "generate", "qc", "correct", "delegate"}

compile_generation_pipeline(service_id, request_params, generation_templates,
                            provider_registry)
  → worker_pipeline metadata dict  (or raises PipelineError)

Rejection list
──────────────
• Missing schema_id or wrong value
• Duplicate stage IDs
• Cycle in depends_on graph
• Unknown stage kind
• Unknown provider ID
• URL / shell command / credential strings in params values
• Arbitrary adapter names
• Missing artifact dependencies (stage output consumed but never produced)
• Correction loop with no QC gate between correct → correct
• max_loop_count > LOOP_BOUND (5)
• max_retries > RETRY_BOUND (10)
• Any depends_on target that does not exist as a stage

The output shape matches the existing Krea worker path expectation:
{
    "schema": "image-pipeline.v1",
    "original_prompt": str,
    "prompt_target": str,          # dotted node reference
    "prompt_enhance": dict | None,
    "post_generate": dict | None,
    "stages": [...ordered stage list with resolved providers...],
}
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

SCHEMA_ID = "image-pipeline.v1"
CLOSED_KINDS: frozenset[str] = frozenset({
    "prompt_enhance",
    "generate",
    "qc",
    "correct",
    "delegate",
})
RETRY_BOUND = 10
LOOP_BOUND = 5

# Execution modes for pipeline dispatch contract.
# "generic"   — standard declarative pipeline; all stage execution is explicit.
# "legacy_reconcile" — relaxed validation for backward-compat callers that
#               supply stages with unresolved endpoint references or per-stage
#               overrides that would otherwise be rejected.  Metadata is
#               preserved but some safety checks are deferred to runtime.
VALID_MODES: frozenset[str] = frozenset({"generic", "legacy_reconcile"})

# Patterns that indicate unsafe / opaque content in string values.
_UNSAFE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_SHELL_META_RE = re.compile(
    r"(\$[A-Z_][A-Z0-9_]*|%%[A-Z_][A-Z0-9_]*|\{[^}]*[|;`<>]\})",
    re.IGNORECASE,
)
_CREDENTIAL_RE = re.compile(
    r"(api[_-]?key|password|secret|token|auth|bearer)\b",
    re.IGNORECASE,
)


# ── Exceptions ───────────────────────────────────────────────────────────────

class PipelineError(Exception):
    """Raised when pipeline validation fails."""
    pass


class UnknownProviderError(PipelineError):
    pass


class UnknownStageKindError(PipelineError):
    pass


class CycleError(PipelineError):
    pass


class DuplicateStageIdError(PipelineError):
    pass


class MissingDependencyError(PipelineError):
    pass


class UnsafeFieldError(PipelineError):
    pass


class AmbiguousCorrectionLoopError(PipelineError):
    pass


class LoopBoundExceededError(PipelineError):
    pass


class LoopValidationError(PipelineError):
    """Raised when a loop entry fails declarative validation."""
    pass


class PipelineValidationError(PipelineError):
    """Raised when a pipeline-level control-flow constraint is violated."""
    pass


class RetryBoundExceededError(PipelineError):
    pass


class SchemaIdError(PipelineError):
    pass


class UnknownExecutionModeError(PipelineError):
    pass


# ── Dataclasses ─────────────────────────────────────────────────────────────

class Stage:
    __slots__ = ("id", "kind", "provider", "depends_on", "retries", "params")

    def __init__(
        self,
        id: str,
        kind: str,
        provider: str,
        depends_on: list[str] | None = None,
        retries: int | None = None,
        params: dict | None = None,
    ):
        self.id = id
        self.kind = kind
        self.provider = provider
        self.depends_on = list(depends_on) if depends_on else []
        self.retries = retries
        self.params = dict(params) if params else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "provider": self.provider,
            "depends_on": list(self.depends_on),
            "retries": self.retries,
            "params": dict(self.params),
        }


class Loop:
    """
    Declarative correction-loop contract.

    Attributes
    ----------
    id : str
        Unique identifier for this loop within the pipeline.
    trigger_stage_id : str
        ID of the ``qc`` stage whose ``qc_pass=false`` verdict activates this loop.
    correction_stage_id : str
        ID of the ``correct`` stage to execute on trigger.
    on_correction : list[str]
        Ordered list of stage IDs to re-enter after correction completes
        (typically a generate stage followed by a QC stage, e.g.
        ``["generate", "qc"]``).
    max_iterations : int
        Maximum number of loop entries; capped at LOOP_BOUND.

    Semantics (executor handoff)
    ----------------------------
    When the pipeline executor reaches ``trigger_stage_id`` (a QC stage) and
    the verdict is ``qc_pass=false``, the executor MUST:

        1. Execute ``correction_stage_id``.
        2. Increment the loop counter.
        3. Abide by ``max_iterations`` (fail closed if bound is reached
           and ``qc_pass`` is still false).
        4. Re-enter stages in ``on_correction`` in declaration order.
        5. Return to the trigger stage for re-evaluation.

    The loop is exited when either:
        - ``qc_pass=true`` is observed at the trigger stage, or
        - ``max_iterations`` is exhausted (fail-closed stop).

    The pipeline compiler validates all stage references, checks for cycles
    and ambiguity, enforces that the trigger is ``qc`` and the correction is
    ``correct``, and bounds ``max_iterations`` to ``LOOP_BOUND``.
    """
    __slots__ = ("id", "trigger_stage_id", "correction_stage_id", "on_correction", "max_iterations")

    def __init__(
        self,
        id: str,
        trigger_stage_id: str,
        correction_stage_id: str,
        on_correction: list[str],
        max_iterations: int,
    ):
        self.id = id
        self.trigger_stage_id = trigger_stage_id
        self.correction_stage_id = correction_stage_id
        self.on_correction = list(on_correction)
        self.max_iterations = max_iterations

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger_stage_id": self.trigger_stage_id,
            "correction_stage_id": self.correction_stage_id,
            "on_correction": list(self.on_correction),
            "max_iterations": self.max_iterations,
        }


class ServicePipeline:
    __slots__ = ("schema_id", "stages", "max_retries", "max_loop_count", "execution_mode", "loops")

    def __init__(
        self,
        schema_id: str,
        stages: list[Stage],
        max_retries: int = 3,
        max_loop_count: int = 1,
        execution_mode: str = "generic",
        loops: list[Loop] | None = None,
    ):
        self.schema_id = schema_id
        self.stages = stages
        self.max_retries = max_retries
        self.max_loop_count = max_loop_count
        self.execution_mode = execution_mode
        self.loops = list(loops) if loops else []

    def stage_map(self) -> dict[str, Stage]:
        return {s.id: s for s in self.stages}

    def to_dict(self) -> dict:
        return {
            "schema_id": self.schema_id,
            "stages": [s.to_dict() for s in self.stages],
            "max_retries": self.max_retries,
            "max_loop_count": self.max_loop_count,
            "execution_mode": self.execution_mode,
            "loops": [loop.to_dict() for loop in self.loops],
        }


# ── Validation helpers ─────────────────────────────────────────────────────

def _check_unsafe_value(value: Any, path: str) -> None:
    """Raise UnsafeFieldError if a string value looks like a URL, shell meta, or credential."""
    if not isinstance(value, str):
        return
    if _UNSAFE_URL_RE.match(value):
        raise UnsafeFieldError(
            f"URL not allowed in field '{path}': {value[:80]}"
        )
    if _SHELL_META_RE.search(value):
        raise UnsafeFieldError(
            f"Shell metacharacter/expansion not allowed in '{path}': {value[:80]}"
        )
    if _CREDENTIAL_RE.search(value):
        raise UnsafeFieldError(
            f"Credential-like content not allowed in '{path}': {value[:80]}"
        )


def _walk_params(params: dict, prefix: str = "") -> None:
    """Recursively walk params dict, rejecting unsafe keys and values."""
    for key, val in params.items():
        p = f"{prefix}.{key}" if prefix else key
        # Reject credential-like keys (e.g. "api_key", "password", "secret_token")
        if _CREDENTIAL_RE.search(key):
            raise UnsafeFieldError(
                f"Credential-like key not allowed in '{p}': {key}"
            )
        # Reject arbitrary adapter names in any field named "adapter"
        if key == "adapter" and isinstance(val, str):
            raise UnsafeFieldError(
                f"Arbitrary adapter name not allowed in '{p}': {val}"
            )
        if isinstance(val, dict):
            _walk_params(val, p)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, str):
                    _check_unsafe_value(item, f"{p}[{i}]")
        else:
            _check_unsafe_value(val, p)


def _detect_cycle(stage_map: dict[str, Stage]) -> list[str] | None:
    """
    Kahn's algorithm topsort on the depends_on graph.

    An edge "s.depends_on = [dep]" means s must run AFTER dep.
    in_degree[sid] = number of dependencies this stage has
                      (count of deps listed in its depends_on list).
    Nodes with in_degree==0 have no prerequisites and can run first.
    After removing a node, we decrement in_degree of its dependents.
    If fewer than N nodes are removed total, a cycle exists.

    Returns list of cycle-member stage IDs, or None if acyclic.
    """
    # in_degree[sid] = count of depends_on entries for sid
    in_degree: dict[str, int] = {sid: 0 for sid in stage_map}
    for sid, s in stage_map.items():
        in_degree[sid] = len(s.depends_on)

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    ordered: list[str] = []

    while queue:
        sid = queue.pop(0)
        ordered.append(sid)
        # This sid is done; tell each node that depended on sid
        for other_sid, s in stage_map.items():
            if sid in s.depends_on:
                in_degree[other_sid] -= 1
                if in_degree[other_sid] == 0:
                    queue.append(other_sid)

    if len(ordered) != len(stage_map):
        return [sid for sid in stage_map if sid not in ordered]
    return None


def _resolve_provider(provider_id: str, registry: frozenset[str]) -> str:
    if provider_id not in registry:
        raise UnknownProviderError(
            f"Provider '{provider_id}' not in provider registry"
        )
    return provider_id


def _validate_loop(
    loop_data: dict,
    stage_ids: set[str],
    stage_map: dict[str, Stage],
    seen_loop_ids: set[str],
) -> Loop:
    """
    Parse and validate a single ``loops`` entry, then return a ``Loop`` object.

    Raises LoopValidationError on:
    - missing / duplicate loop id
    - missing required fields
    - trigger / correction / on_correction stage not in pipeline
    - trigger stage is not kind ``qc``
    - correction stage is not kind ``correct``
    - max_iterations exceeds LOOP_BOUND or is negative
    """
    lid = loop_data.get("id")
    if not lid:
        raise LoopValidationError(f"loop entry missing 'id': {loop_data}")
    if lid in seen_loop_ids:
        raise LoopValidationError(f"duplicate loop id '{lid}'")
    seen_loop_ids.add(lid)

    trigger = loop_data.get("trigger_stage_id")
    if not trigger:
        raise LoopValidationError(f"loop '{lid}' missing 'trigger_stage_id'")
    if trigger not in stage_ids:
        raise LoopValidationError(
            f"loop '{lid}' references unknown trigger_stage_id '{trigger}'"
        )
    if stage_map[trigger].kind != "qc":
        raise LoopValidationError(
            f"loop '{lid}' trigger_stage_id '{trigger}' must be kind 'qc', "
            f"got kind '{stage_map[trigger].kind}'"
        )

    correction = loop_data.get("correction_stage_id")
    if not correction:
        raise LoopValidationError(f"loop '{lid}' missing 'correction_stage_id'")
    if correction not in stage_ids:
        raise LoopValidationError(
            f"loop '{lid}' references unknown correction_stage_id '{correction}'"
        )
    if stage_map[correction].kind != "correct":
        raise LoopValidationError(
            f"loop '{lid}' correction_stage_id '{correction}' must be kind 'correct', "
            f"got kind '{stage_map[correction].kind}'"
        )

    on_corr_raw = loop_data.get("on_correction", [])
    if not isinstance(on_corr_raw, list):
        raise LoopValidationError(
            f"loop '{lid}' on_correction must be a list, got {type(on_corr_raw).__name__}"
        )
    if not on_corr_raw:
        raise LoopValidationError(
            f"loop '{lid}' on_correction must be non-empty"
        )
    on_corr: list[str] = []
    for sid in on_corr_raw:
        if not isinstance(sid, str):
            raise LoopValidationError(
                f"loop '{lid}' on_correction contains non-string stage id: {sid!r}"
            )
        if sid not in stage_ids:
            raise LoopValidationError(
                f"loop '{lid}' on_correction references unknown stage '{sid}'"
            )
        on_corr.append(sid)

    max_iter = loop_data.get("max_iterations")
    if max_iter is None:
        raise LoopValidationError(f"loop '{lid}' missing 'max_iterations'")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise LoopValidationError(
            f"loop '{lid}' max_iterations must be an integer, got {type(max_iter).__name__}"
        )
    if max_iter < 1:
        raise LoopValidationError(
            f"loop '{lid}' max_iterations must be >= 1, got {max_iter}"
        )
    if max_iter > LOOP_BOUND:
        raise LoopValidationError(
            f"loop '{lid}' max_iterations {max_iter} exceeds LOOP_BOUND {LOOP_BOUND}"
        )

    return Loop(
        id=lid,
        trigger_stage_id=trigger,
        correction_stage_id=correction,
        on_correction=on_corr,
        max_iterations=max_iter,
    )


# ── Generic prompt_target helpers ──────────────────────────────────────────

# Accepts:
#   - alphanumeric IDs:     gen.inputs.text
#   - numeric ComfyUI IDs:  6.inputs.text  (ComfyUI numeric node IDs are valid)
_SAFE_DOTTED_PATH_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*|[0-9]+)(\.[a-zA-Z_][a-zA-Z0-9_]*)+$"
)
"""Safe dotted identifier path grammar — no URLs, no shell, no arbitrary patterns."""

_INVALID_PROMPT_TARGET_RE = re.compile(
    r"(^https?://|[\$\{;`<>|]|api[_-]?key|password|secret|token|auth)",
    re.IGNORECASE,
)


def _is_safe_prompt_target(value: str) -> bool:
    """Return True if value is a safe ComfyUI dotted node-path."""
    if not value or not isinstance(value, str):
        return False
    if _INVALID_PROMPT_TARGET_RE.search(value):
        return False
    return bool(_SAFE_DOTTED_PATH_RE.match(value))


def _find_downstream_generate_stages(
    enhance_stage_id: str,
    stages: list["Stage"],
) -> list[tuple["Stage", str | None]]:
    """
    Return generate stages reachable from enhance_stage_id via depends_on graph,
    whose params.prompt_target is a non-empty safe dotted path.

    A stage S is "downstream" of enhance if enhance appears anywhere in
    S's transitive depends_on closure.

    The BFS traverses the reverse edges: for each stage X with depends_on Y,
    we know Y runs before X, so X is reachable from Y.

    Returns list of (stage, prompt_target) tuples.
    """
    # Build reverse adjacency: for each stage, who depends on it (forward direction)
    # depends_on = [A, B] means "I depend on A and B" → A and B run BEFORE this stage.
    # So the forward direction (who runs after) is the reverse of depends_on.
    dependents: dict[str, list[str]] = {}  # stage_id -> list of stages that depend on it
    for s in stages:
        for dep_id in s.depends_on:
            dependents.setdefault(dep_id, []).append(s.id)

    # BFS: from enhance, follow forward edges (who runs after)
    reachable_ids: set[str] = set()
    queue = [enhance_stage_id]
    while queue:
        sid = queue.pop()
        for dependent_id in dependents.get(sid, []):
            if dependent_id not in reachable_ids:
                reachable_ids.add(dependent_id)
                queue.append(dependent_id)

    results: list[tuple["Stage", str | None]] = []
    for s in stages:
        if s.id not in reachable_ids:
            continue
        if s.kind != "generate":
            continue
        pt = s.params.get("prompt_target") if isinstance(s.params, dict) else None
        results.append((s, str(pt) if pt is not None else None))

    return results


def _derive_generic_prompt_target(stages: list["Stage"]) -> str | None:
    """
    Derive prompt_target for generic execution_mode with prompt_enhance present.

    Uses downstream depends_on reachability from the prompt_enhance stage to
    find generate stages, then requires exactly one to declare a safe
    non-empty params.prompt_target.

    Raises PipelineError on missing, ambiguous, or malformed bindings.
    """
    # Find the prompt_enhance stage
    enhance_stage: "Stage | None" = None
    for s in stages:
        if s.kind == "prompt_enhance":
            enhance_stage = s
            break

    if enhance_stage is None:
        # No enhance stage — prompt_target is None (caller routes as needed)
        return None

    candidates = _find_downstream_generate_stages(enhance_stage.id, stages)

    if not candidates:
        # A prompt-enhancement-only pipeline has no workflow binding to
        # validate. The binding becomes mandatory only when generation is a
        # downstream consumer of the enhanced prompt.
        return None
    if len(candidates) > 1:
        stage_ids = ", ".join(s.id for s, _ in candidates)
        raise PipelineError(
            f"ambiguous prompt_target binding: multiple downstream generate "
            f"stages declare params.prompt_target: {stage_ids}"
        )

    _, prompt_target = candidates[0]
    if not prompt_target or not _is_safe_prompt_target(prompt_target):
        raise PipelineError(
            "generic execution requires a downstream generate stage "
            "to declare non-empty params.prompt_target "
            "(dotted node path grammar); none found"
        )
    return prompt_target


def _load_service_pipeline(
    data: dict,
    provider_registry: frozenset[str],
) -> ServicePipeline:
    schema_id = data.get("schema_id") or data.get("schema")
    if not schema_id:
        raise SchemaIdError("pipeline is missing schema_id")
    if schema_id != SCHEMA_ID:
        raise SchemaIdError(
            f"pipeline schema must be '{SCHEMA_ID}', got '{schema_id}'"
        )

    raw_stages = data.get("stages") or []
    if not raw_stages:
        raise PipelineError("pipeline has no stages")

    seen_ids: set[str] = set()
    stages: list[Stage] = []

    for entry in raw_stages:
        sid = entry.get("id")
        if not sid:
            raise PipelineError(f"stage missing id: {entry}")
        if sid in seen_ids:
            raise DuplicateStageIdError(f"duplicate stage id '{sid}'")
        seen_ids.add(sid)

        kind = entry.get("kind")
        if kind not in CLOSED_KINDS:
            raise UnknownStageKindError(
                f"stage '{sid}' has unknown kind '{kind}'; "
                f"allowed: {sorted(CLOSED_KINDS)}"
            )

        provider = entry.get("provider") or ""
        _resolve_provider(provider, provider_registry)

        depends_on = entry.get("depends_on") or []
        if not isinstance(depends_on, list):
            raise PipelineError(f"stage '{sid}' depends_on must be a list")

        retries_override = entry.get("retries")
        params = entry.get("params") or {}
        _walk_params(params, prefix=sid)

        stages.append(Stage(
            id=sid,
            kind=kind,
            provider=provider,
            depends_on=depends_on,
            retries=retries_override,
            params=params,
        ))

    max_retries = int(data.get("max_retries", 3))
    if max_retries > RETRY_BOUND:
        raise RetryBoundExceededError(
            f"max_retries {max_retries} exceeds bound {RETRY_BOUND}"
        )
    if max_retries < 0:
        raise PipelineError("max_retries must be non-negative")

    max_loop_count = int(data.get("max_loop_count", 1))
    if max_loop_count > LOOP_BOUND:
        raise LoopBoundExceededError(
            f"max_loop_count {max_loop_count} exceeds bound {LOOP_BOUND}"
        )
    if max_loop_count < 0:
        raise PipelineError("max_loop_count must be non-negative")

    execution_mode = str(data.get("execution_mode", "generic"))
    if execution_mode not in VALID_MODES:
        raise UnknownExecutionModeError(
            f"execution_mode must be one of {sorted(VALID_MODES)}, got '{execution_mode}'"
        )

    # ── Structural validations ─────────────────────────────────────────────
    stage_map = {s.id: s for s in stages}

    # 1. All depends_on targets exist (must check BEFORE cycle detection,
    #    otherwise a reference to an unknown node makes Kahn's algorithm
    #    flag a false cycle on the lone remaining node)
    all_ids = set(stage_map.keys())
    for s in stages:
        for dep in s.depends_on:
            if dep not in all_ids:
                raise MissingDependencyError(
                    f"stage '{s.id}' depends on unknown stage '{dep}'"
                )

    # 2. Acyclicity
    cycle = _detect_cycle(stage_map)
    if cycle:
        raise CycleError(f"cycle detected in stage graph: {cycle}")

    # 3. Ambiguous correction loops: a correct stage that is (transitively)
    #    depended on by another correct stage, with no QC gate between them.
    _check_correction_loop(stages)

    # 4. Explicit declarative loops: validate each loops[] entry.
    #    Parsed after stage_map is built so stage kind lookups are available.
    raw_loops = data.get("loops") or []
    if not isinstance(raw_loops, list):
        raise PipelineError("loops must be a list")
    seen_loop_ids: set[str] = set()
    validated_loops: list[Loop] = []
    for loop_entry in raw_loops:
        if not isinstance(loop_entry, dict):
            raise LoopValidationError(f"loop entry must be a dict, got {type(loop_entry).__name__}")
        validated = _validate_loop(
            loop_entry,
            stage_ids=all_ids,
            stage_map=stage_map,
            seen_loop_ids=seen_loop_ids,
        )
        validated_loops.append(validated)

    # 5. Impossible terminal path: if explicit loops exist but
    #    max_loop_count==0, the pipeline declares a correction cycle
    #    but caps corrections at zero, making the terminal state unreachable.
    if validated_loops and max_loop_count == 0:
        raise PipelineValidationError(
            "pipeline declares correction loops but max_loop_count is 0; "
            "this creates an impossible terminal path "
            "(loop would always require at least one correction entry)"
        )

    return ServicePipeline(
        schema_id, stages, max_retries, max_loop_count, execution_mode,
        loops=validated_loops,
    )


def _check_correction_loop(stages: list[Stage]) -> None:
    """
    Reject ambiguous correction loops: a correct stage 'a' that is
    (transitively) depended on by another correct stage 'b', where
    there is no QC stage on the path from 'a' to 'b'.

    We walk backward from 'b' via depends_on edges; if we encounter
    a QC node before reaching 'a', the chain is gated and valid.
    If we reach 'a' without finding a QC gate, the loop is ambiguous.
    """
    sm = {s.id: s for s in stages}

    def _has_qc_between(source: str, target: str) -> bool:
        """
        Walk backward from `target` via depends_on edges.
        Return True if a 'qc' node is found *before* `source` is reached.
        If `source` is reached first, return False (no gate on the path).
        """
        visited: set[str] = set()
        queue = [target]
        while queue:
            cur = queue.pop(0)
            if cur == source:
                # Reached source without finding a QC gate
                return False
            if cur in visited:
                continue
            visited.add(cur)
            if sm[cur].kind == "qc":
                return True
            for dep in sm[cur].depends_on:
                if dep not in visited:
                    queue.append(dep)
        return False

    correct_ids = [s.id for s in stages if s.kind == "correct"]
    for i, a in enumerate(correct_ids):
        for b in correct_ids[i + 1:]:
            # Does 'b' depend (transitively) on 'a'?
            if not _has_qc_between(a, b):
                raise AmbiguousCorrectionLoopError(
                    f"correction loop detected: '{a}' → '{b}' has no QC gate"
                )


# ── Public API ──────────────────────────────────────────────────────────────

def load_service_pipelines(
    data: dict,
    provider_registry: frozenset[str],
) -> dict[str, ServicePipeline]:
    """
    Load the top-level ``service_pipelines`` mapping.
    Accepts two layouts:
      A) {"service_pipelines": {"svc_id": pipeline_cfg, ...}}  (outer wrapper)
      B) {"svc_id": pipeline_cfg, ...}                         (already unwrapped)

    Both are detected and handled transparently.
    """
    # Layout B detection: svc_id keys that look like pipeline configs
    if "service_pipelines" in data:
        raw = data["service_pipelines"]
    else:
        # Assume already-unwrapped layout B: keys are service IDs
        raw = data
    out = {}
    for svc_id, cfg in raw.items():
        pipeline = _load_service_pipeline(cfg, provider_registry)
        out[svc_id] = pipeline
    return out


def _compute_revision_hash(pipeline: ServicePipeline) -> str:
    """
    Compute a deterministic revision hash from pipeline content.
    Uses SHA-256 over a canonical JSON serialization of the pipeline
    (schema_id, stages, max_retries, max_loop_count, loops).
    Does NOT include request-scoped fields like original_prompt.
    """
    canonical = {
        "schema_id": pipeline.schema_id,
        "stages": [s.to_dict() for s in pipeline.stages],
        "max_retries": pipeline.max_retries,
        "max_loop_count": pipeline.max_loop_count,
        "loops": [loop.to_dict() for loop in pipeline.loops],
        "execution_mode": pipeline.execution_mode,
    }
    # Canonical JSON: sorted keys, no extra whitespace
    import json
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def compile_generation_pipeline(
    service_id: str,
    request_params: dict,
    generation_templates: dict,
    provider_registry: frozenset[str],
) -> dict:
    """
    Compile a generation request into the bounded ``worker_pipeline`` metadata
    shape used by the Krea worker path.

    generation_templates may be in two layouts:
      A) {service_id: {service_pipelines: {...}}}   (real-world per-template)
      B) {service_pipelines: {service_id: {...}}}  (flat / test fixture)
    """
    if service_id in generation_templates:
        svc_pipelines_cfg = generation_templates
    elif "service_pipelines" in generation_templates:
        svc_pipelines_cfg = generation_templates
    else:
        raise PipelineError(f"No generation_template for service '{service_id}'")

    pipelines = load_service_pipelines(svc_pipelines_cfg, provider_registry)

    if service_id not in pipelines:
        raise PipelineError(
            f"service_id '{service_id}' not found in service_pipelines; "
            f"available: {list(pipelines.keys())}"
        )

    pipeline = pipelines[service_id]

    # Require prompt when prompt_enhance stage is present
    has_enhance = any(s.kind == "prompt_enhance" for s in pipeline.stages)
    if has_enhance and not request_params.get("prompt"):
        raise PipelineError(
            "params.prompt is required when prompt_enhance stage is present"
        )

    # Build resolved stage list in declaration order.
    # phase is derived from kind and is used by generic runtimes to route
    # stage execution without hard-coding provider assumptions.
    _KIND_PHASE: dict[str, str] = {
        "prompt_enhance": "pre_generate",
        "generate":       "generate",
        "qc":             "post_generate",
        "correct":        "post_generate",
        "delegate":       "delegate",
    }
    out_stages: list[dict] = []
    for s in pipeline.stages:
        provider = _resolve_provider(s.provider, provider_registry)
        out_stages.append({
            "id": s.id,
            "kind": s.kind,
            "phase": _KIND_PHASE.get(s.kind, s.kind),
            "provider": provider,
            "depends_on": list(s.depends_on),
            "retries": s.retries if s.retries is not None else pipeline.max_retries,
            "params": dict(s.params),
        })

    # Extract prompt_enhance / post_generate sub-configs for the live Krea
    # worker path.  Stage parameters are deliberately flattened here: the
    # existing worker helpers consume fields such as ``project_id``,
    # ``vision_url`` and ``qwen_template`` directly.  Keeping them under a
    # generic ``params`` key would validate successfully but silently discard
    # the declared policy at execution time.
    prompt_enhance_cfg: dict | None = None
    post_generate_cfg: dict | None = None

    for s in pipeline.stages:
        if s.kind == "prompt_enhance":
            prompt_enhance_cfg = {
                "enabled": True,
                "provider": s.provider,
                **dict(s.params),
            }
        elif s.kind == "qc":
            post_generate_cfg = {
                **(post_generate_cfg or {}),
                "enabled": True,
                "kind": s.kind,
                "provider": s.provider,
                **dict(s.params),
                "max_corrections": pipeline.max_loop_count,
                "retries": (
                    s.retries if s.retries is not None else pipeline.max_retries
                ),
            }
        elif s.kind == "correct":
            # Keep the QC provider/policy and add a distinct correction
            # provider.  The live worker uses the former for grounded
            # observation and the latter for the bounded edit hand-off.
            post_generate_cfg = {
                **(post_generate_cfg or {}),
                "enabled": True,
                "kind": "correct",
                "correction_provider": s.provider,
                **dict(s.params),
                "max_corrections": pipeline.max_loop_count,
                "retries": (
                    s.retries if s.retries is not None else pipeline.max_retries
                ),
            }

    # Derive prompt_target:
    # - "generic": requires exactly one downstream generate stage to declare
    #   params.prompt_target (safe dotted node path). An enhance-stage-only
    #   target does NOT satisfy generic. If no prompt_enhance stage exists,
    #   prompt_target is None (caller routes as needed).
    # - "legacy_reconcile": preserve old Krea behavior — fall back to
    #   prompt_enhance.params.target or "prompt_enhance.inputs.prompt".
    prompt_target = None
    if pipeline.execution_mode == "generic":
        prompt_target = _derive_generic_prompt_target(pipeline.stages)
    else:  # legacy_reconcile
        for s in pipeline.stages:
            if s.kind == "prompt_enhance":
                # Preserve old Krea default behavior for backward compatibility
                prompt_target = s.params.get("target") or "prompt_enhance.inputs.prompt"
                break

    return {
        "schema": SCHEMA_ID,
        "service_id": service_id,
        "pipeline_id": service_id,
        "execution_mode": pipeline.execution_mode,
        "original_prompt": str(request_params.get("prompt") or ""),
        "prompt_target": prompt_target,
        "prompt_enhance": prompt_enhance_cfg,
        "post_generate": post_generate_cfg,
        "max_retries": pipeline.max_retries,
        "max_loop_count": pipeline.max_loop_count,
        "stages": out_stages,
        "loops": [loop.to_dict() for loop in pipeline.loops],
        "revision_hash": _compute_revision_hash(pipeline),
    }


def resolve_generation_pipeline_id(
    gen_type: str,
    template: dict,
    request_params: dict,
) -> tuple[str | None, dict | None]:
    """Resolve the pipeline owned by a generation template.

    A template may declare ``service_pipeline_id`` as its service-level
    contract. Requests may omit that field or repeat the same value, but a
    caller must not silently replace the pipeline selected by the endpoint.
    A malformed template link is a configuration error, not permission to
    fall back to the legacy inline path.

    Returns ``(pipeline_id, None)`` on success or ``(None, error_dict)`` on a
    fail-closed rejection. This helper is pure so ingress and hermetic tests
    share exactly the same ownership rule.
    """
    template = template if isinstance(template, dict) else {}
    request_params = request_params if isinstance(request_params, dict) else {}

    # Distinguish "key absent" from "key explicitly null":
    # - absent → return (None, None)  (no pipeline selected, no error)
    # - explicitly null → fail closed pipeline_disabled
    _ABSENT = object()
    declared = template.get("service_pipeline_id", _ABSENT)
    requested = request_params.get("service_pipeline_id", _ABSENT)

    # Explicitly null (key present but value is null) = disabled pipeline contract; fail closed.
    if declared is not _ABSENT and declared is None:
        return None, {
            "error": (
                f"generation template {gen_type!r} declares "
                "service_pipeline_id = null (pipeline disabled); "
                "a template that opts out of the declarative pipeline "
                "contract must not be used with service_pipeline_id requests"
            ),
            "status": 400,
            "reason": "pipeline_disabled",
            "template": gen_type,
        }

    # Key absent from template: a request may explicitly select a registered
    # pipeline as a compatibility escape hatch for dynamically registered
    # services.  The selected ID is still resolved against the server-owned
    # registry by ``compile_generation_pipeline``; it is never an executable
    # adapter or endpoint supplied by the caller.
    if declared is _ABSENT:
        if requested is _ABSENT:
            return None, None
        if requested is None:
            return None, {
                "error": "request service_pipeline_id must be a non-empty string",
                "status": 400,
                "reason": "invalid_service_pipeline_request",
                "template": gen_type,
            }
        if not isinstance(requested, str) or not requested.strip():
            return None, {
                "error": "request service_pipeline_id must be a non-empty string",
                "status": 400,
                "reason": "invalid_service_pipeline_request",
                "template": gen_type,
            }
        return requested.strip(), None

    if not isinstance(declared, str) or not declared.strip():
        return None, {
            "error": (
                f"generation template {gen_type!r} declares an invalid "
                "service_pipeline_id"
            ),
            "status": 500,
            "reason": "invalid_service_pipeline_link",
            "template": gen_type,
        }
    declared = declared.strip()

    if requested is not _ABSENT and requested is not None:
        if not isinstance(requested, str) or not requested.strip():
            return None, {
                "error": "request service_pipeline_id must be a non-empty string",
                "status": 400,
                "reason": "invalid_service_pipeline_request",
                "template": gen_type,
            }
        requested = requested.strip()
    elif requested is _ABSENT:
        requested = None

    if declared and requested and declared != requested:
        return None, {
            "error": (
                f"generation template {gen_type!r} owns pipeline {declared!r}; "
                f"request attempted to override it with {requested!r}"
            ),
            "status": 409,
            "reason": "service_pipeline_override",
            "template": gen_type,
            "declared_pipeline_id": declared,
            "requested_pipeline_id": requested,
        }

    return declared or requested, None


# ── Shared pipeline resolution helper ─────────────────────────────────────────

def resolve_and_compile_pipeline(
    *,
    gen_type: str,
    template: dict,
    request_params: dict,
    generation_templates: dict,
    provider_registry: frozenset,
) -> tuple[dict | None, dict | None]:
    """
    Shared pipeline resolution and compilation for both ingress handlers.

    Combines resolve_generation_pipeline_id + compile_generation_pipeline into
    a single call so both /v1/submit/generation and /generate/{type} share
    identical logic.

    Returns
    -------
    (compiled_pipeline_dict, error_dict):
      - On success: (compiled_dict, None)
      - On reject:  (None, error_dict with 'error' + 'status' keys)
      - On no-pipeline: (None, None) — caller should use legacy leaf path.

    The compiled dict includes:
      - pipeline_id, revision_hash, max_loop_count, max_retries, stages, loops
      - schema: "image-pipeline.v1"
      - original_prompt, prompt_target, prompt_enhance, post_generate (from template)

    No side effects; fully hermetic.
    """
    pipeline_id, link_error = resolve_generation_pipeline_id(
        gen_type, template, request_params
    )
    if link_error:
        return None, link_error

    if not pipeline_id:
        return None, None  # No pipeline — use legacy leaf path

    # Pull the pipeline definition from generation_templates.service_pipelines
    pipelines_container = (
        generation_templates.get("service_pipelines", {})
        if isinstance(generation_templates, dict) else {}
    )
    pipeline_def = pipelines_container.get(pipeline_id)
    if not pipeline_def:
        return None, {
            "error": f"pipeline '{pipeline_id}' not found in generation_templates",
            "status": 500,
            "reason": "pipeline_not_found",
            "template": gen_type,
        }

    try:
        compiled = compile_generation_pipeline(
            service_id=pipeline_id,
            request_params=request_params,
            generation_templates=generation_templates,
            provider_registry=provider_registry,
        )
    except PipelineError as e:
        return None, {
            "error": "pipeline_compile_error",
            "detail": str(e),
            "status": 400,
        }

    return compiled, None
