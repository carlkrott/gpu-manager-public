"""
runtime_dependency_resolver.py — Fail-closed external dependency resolution.

This module provides fail-closed resolution for external runtime dependencies
that the gpu-manager daemon requires at startup or during request handling.

Principles:
- Every external path MUST be overridable via an environment variable.
- If the environment variable is absent AND the path doesn't exist, raise
  DependencyNotFound at runtime (not import time), preventing silent fallback
  to a wrong path.
- Use os.environ.get() + os.path.exists() for resolution.

Environment variables:
    MEDIA_PIPELINE_ROOT      Root of the media-pipeline project checkout.
                              Used to resolve: workflows/, scripts/, output/
    ACE_STEP_ROOT            Root of the ACE-Step project checkout.
                              Used to resolve: ACE-Step-1.5/, ACE-Step-1.5-dualgpu-candidate/
    MEDIA_PIPELINE_VENV      Python interpreter inside media-pipeline's venv.
                              Default: {MEDIA_PIPELINE_ROOT}/.venv/bin/python3

Paths resolved here are RUNTIME coupling — they are subprocess calls and
workflow template files that legitimately live in the external project trees.
This is NOT import-time coupling; import-time coupling is tested by
test_source_closure.py and test_hermetic_relocation.py.

Usage:
    from runtime_dependency_resolver import (
        resolve_nginx_manager_sh,
        resolve_pipeline_script,
        resolve_venv_python,
        resolve_workflow_template,
        MEDIA_PIPELINE_ROOT,
        ACE_STEP_ROOT,
    )

    # Raises RuntimeError if not configured and the default doesn't exist:
    nginx_sh = resolve_nginx_manager_sh()

    # Returns the path string even if it doesn't exist (defer check to call site):
    workflow = resolve_workflow_template(
        "qwen-image-edit-2511.json",
        default="workflows/qwen-image-edit-2511.json",
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = [
    "resolve_nginx_manager_sh",
    "resolve_pipeline_script",
    "resolve_pipeline_output_path",
    "resolve_venv_python",
    "resolve_workflow_template",
    "resolve_workflow_path",
    "resolve_acestep_python",
    "WorkflowResolutionError",
    "MEDIA_PIPELINE_ROOT",
    "ACE_STEP_ROOT",
    "DependencyNotFound",
]


class DependencyNotFound(RuntimeError):
    """
    Raised when an external runtime dependency cannot be resolved.

    Either the required environment variable is unset AND the default path
    does not exist, OR the path exists but is not a file.
    """
    pass


# ── Env-var names ────────────────────────────────────────────────────────────

_ENV_MEDIA_PIPELINE_ROOT = "GPU_MANAGER_MEDIA_PIPELINE_ROOT"
_ENV_ACE_STEP_ROOT = "GPU_MANAGER_ACE_STEP_ROOT"
_ENV_MEDIA_PIPELINE_VENV = "GPU_MANAGER_MEDIA_PIPELINE_VENV"


# ── Env-root getters (fail-closed on misconfiguration, not on absence) ───────

def _get_env_var(name: str, must_exist: bool = False) -> Optional[str]:
    """Get an environment variable value, or None if unset."""
    val = os.environ.get(name)
    if val is not None:
        return val
    if must_exist:
        raise DependencyNotFound(
            f"Required environment variable {name!r} is not set. "
            f"This dependency must be configured via the environment."
        )
    return None


def _get_path_env(name: str, must_exist: bool = False) -> Optional[Path]:
    """Get an environment variable as a Path, validating it exists if must_exist."""
    val = _get_env_var(name, must_exist=False)
    if val is None:
        return None
    p = Path(val)
    if must_exist and not p.exists():
        raise DependencyNotFound(
            f"{name}={val!r} is set but the path does not exist. "
            f"Verify {name} points to a valid directory."
        )
    return p


# ── Root resolution ──────────────────────────────────────────────────────────

def _resolve_media_pipeline_root() -> Optional[Path]:
    """
    Resolve MEDIA_PIPELINE_ROOT.

    Returns None if not configured. External project roots must be explicit;
    this resolver never reaches into a host checkout by convention.
    """
    env = _get_env_var(_ENV_MEDIA_PIPELINE_ROOT)
    if env is not None:
        p = Path(env)
        if not p.exists():
            raise DependencyNotFound(
                f"{_ENV_MEDIA_PIPELINE_ROOT}={env!r} is set but the path does not exist."
            )
        return p
    return None


def _resolve_acestep_root() -> Optional[Path]:
    """
    Resolve ACE_STEP_ROOT.

    Returns None if not configured.
    """
    env = _get_env_var(_ENV_ACE_STEP_ROOT)
    if env is not None:
        p = Path(env)
        if not p.exists():
            raise DependencyNotFound(
                f"{_ENV_ACE_STEP_ROOT}={env!r} is set but the path does not exist."
            )
        return p
    return None


# Cached root resolution (lazily evaluated on first call)
_media_pipeline_root: Optional[Path] = None
_acestep_root: Optional[Path] = None


def _get_media_pipeline_root() -> Path:
    """Return the media-pipeline root as a Path.

    Fail-closed: raises DependencyNotFound when GPU_MANAGER_MEDIA_PIPELINE_ROOT
    is unset. The external project is optional and must be explicitly selected.
    """
    global _media_pipeline_root
    if _media_pipeline_root is None:
        root = _resolve_media_pipeline_root()
        if root is None:
            raise DependencyNotFound(
                "GPU_MANAGER_MEDIA_PIPELINE_ROOT is not set. "
                "External media-pipeline integrations require an explicit "
                "checkout root; no host default is used."
            )
        _media_pipeline_root = root
    return _media_pipeline_root


def _get_acestep_root() -> Path:
    """Return the ACE-Step root as a Path.

    Fail-closed: raises DependencyNotFound when GPU_MANAGER_ACE_STEP_ROOT
    is unset. The external ACE-Step checkout must be explicitly selected.
    """
    global _acestep_root
    if _acestep_root is None:
        root = _resolve_acestep_root()
        if root is None:
            raise DependencyNotFound(
                "GPU_MANAGER_ACE_STEP_ROOT is not set. "
                "External ACE-Step integrations require an explicit checkout "
                "root; no host default is used."
            )
        _acestep_root = root
    return _acestep_root


# ── Public accessors ──────────────────────────────────────────────────────────

def MEDIA_PIPELINE_ROOT() -> str:
    """Return the configured media-pipeline root as a string."""
    return str(_get_media_pipeline_root())


def ACE_STEP_ROOT() -> str:
    """Return the configured ACE-Step root as a string."""
    return str(_get_acestep_root())


# ── Individual resolvers ──────────────────────────────────────────────────────

def resolve_nginx_manager_sh() -> str:
    """
    Resolve the path to nginx-manager.sh.

    Resolution order:
      1. GPU_MANAGER_MEDIA_PIPELINE_ROOT/scripts/nginx-manager.sh
      2. no path (unset external root fails closed)

    Raises DependencyNotFound if the resolved path does not exist.
    """
    root = _get_media_pipeline_root()
    path = root / "scripts" / "nginx-manager.sh"
    if path.exists() and path.is_file():
        return str(path)
    raise DependencyNotFound(
        f"nginx-manager.sh not found at {path}. "
        f"Set {_ENV_MEDIA_PIPELINE_ROOT} to an external media-pipeline root "
        f"containing scripts/nginx-manager.sh."
    )


def resolve_pipeline_script() -> str:
    """
    Resolve the path to pipeline-orchestrator.py.

    Resolution order:
      1. GPU_MANAGER_MEDIA_PIPELINE_ROOT/scripts/pipeline-orchestrator.py
      2. no path (unset external root fails closed)

    Raises DependencyNotFound if the resolved path does not exist.
    """
    root = _get_media_pipeline_root()
    path = root / "scripts" / "pipeline-orchestrator.py"
    if path.exists() and path.is_file():
        return str(path)
    raise DependencyNotFound(
        f"pipeline-orchestrator.py not found at {path}. "
        f"Set {_ENV_MEDIA_PIPELINE_ROOT} to an external media-pipeline root "
        f"containing scripts/pipeline-orchestrator.py."
    )


def resolve_pipeline_output_path(run_id: str) -> str:
    """Resolve one pipeline result path under the explicitly configured root.

    The pipeline orchestrator is an optional external integration.  Its result
    directory must therefore be derived from the same explicit
    ``GPU_MANAGER_MEDIA_PIPELINE_ROOT`` used to launch it; no host checkout is
    consulted implicitly.  ``run_id`` is deliberately restricted to a single
    safe path component so a status request cannot escape the output directory.
    """
    value = str(run_id or "").strip()
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
        raise DependencyNotFound(
            "Pipeline run_id must be a non-empty alphanumeric, dash, or underscore value."
        )
    root = _get_media_pipeline_root()
    return str(root / "output" / value / f"{value}_result.json")


def resolve_venv_python() -> str:
    """
    Resolve the Python interpreter inside the media-pipeline venv.

    Resolution order:
      1. GPU_MANAGER_MEDIA_PIPELINE_VENV if explicitly set
      2. GPU_MANAGER_MEDIA_PIPELINE_ROOT/.venv/bin/python3
      3. no path (unset external root fails closed)

    Raises DependencyNotFound if the resolved path does not exist.
    """
    # Check explicit venv override first
    explicit_venv = _get_env_var(_ENV_MEDIA_PIPELINE_VENV)
    if explicit_venv is not None:
        p = Path(explicit_venv)
        if p.exists() and p.is_file():
            return str(p)
        raise DependencyNotFound(
            f"{_ENV_MEDIA_PIPELINE_VENV}={explicit_venv!r} is set but the file does not exist."
        )

    # Fall back to root/.venv/bin/python3
    root = _get_media_pipeline_root()
    path = root / ".venv" / "bin" / "python3"
    if path.exists() and path.is_file():
        return str(path)
    raise DependencyNotFound(
        f"Python interpreter not found at {path}. "
        f"Set {_ENV_MEDIA_PIPELINE_VENV} to the full path of the venv python, "
        f"or ensure the explicit external root contains .venv/bin/python3."
    )


def resolve_workflow_template(
    template_name: str,
    default_subpath: Optional[str] = None,
) -> str:
    """
    Resolve a workflow template path.

    Resolution order:
      1. GPU_MANAGER_MEDIA_PIPELINE_ROOT/workflows/<template_name>
      2. no path (unset external root fails closed)

    Parameters:
        template_name: The filename of the workflow template (e.g. "krea2_multigpu.json").
        default_subpath: If provided, use this subpath under workflows/ as the default
                         instead of template_name directly. Useful when the default
                         subpath differs from the filename.

    Returns:
        The absolute path to the workflow template.

    Raises DependencyNotFound if the resolved path does not exist.
    """
    root = _get_media_pipeline_root()
    subpath = default_subpath or template_name
    path = root / "workflows" / subpath
    if path.exists() and path.is_file():
        return str(path)
    raise DependencyNotFound(
        f"Workflow template {template_name!r} not found at {path}. "
        f"Set {_ENV_MEDIA_PIPELINE_ROOT} to an external media-pipeline root "
        f"containing the workflow file."
    )


# ── Consumer-facing workflow resolver ─────────────────────────────────────────

_ENV_ALLOW_ABSOLUTE_WORKFLOW_PATHS = "GPU_MANAGER_ALLOW_ABSOLUTE_WORKFLOW_PATHS"

# Candidate-owned workflows subdirectory. Workflows stored here are owned by
# the candidate and are resolved relative to the candidate's staging root.
_CANDIDATE_WORKFLOW_SUBDIR = "workflows"


def _get_candidate_root() -> Path:
    """Return the candidate staging root.

    Derived from the directory containing this module (runtime_dependency_resolver.py).
    This is the canonical candidate-owned root — NOT the live media-pipeline checkout.
    """
    return Path(__file__).resolve().parent.parent


def _resolve_candidate_workflow(workflow_name: str) -> Path:
    """Resolve a candidate workflow name without permitting path escape."""
    value = str(workflow_name or "").strip()
    if value.startswith("workflows/"):
        value = value[len("workflows/"):]
    if not value or Path(value).is_absolute():
        raise WorkflowResolutionError(
            f"Candidate workflow name {workflow_name!r} is empty or absolute. "
            "Use a relative file name under workflows/."
        )
    workflows_root = (_get_candidate_root() / _CANDIDATE_WORKFLOW_SUBDIR).resolve()
    resolved = (workflows_root / value).resolve()
    try:
        resolved.relative_to(workflows_root)
    except ValueError as exc:
        raise WorkflowResolutionError(
            f"Candidate workflow {workflow_name!r} escapes the candidate workflows/ directory."
        ) from exc
    return resolved


def resolve_workflow_path(
    workflow_template_value: Optional[str],
    candidate_workflow_name: Optional[str] = None,
) -> str:
    """
    Resolve a workflow_template value from services.json (or equivalent config)
    to an absolute filesystem path, fail-closed.

    Resolution rules (in order of evaluation):

      1. Candidate-owned workflow (candidate_workflow_name is set):
         Resolved as  <candidate_root>/workflows/<candidate_workflow_name>.
         Never falls back to media-pipeline; the candidate root is always
         computed from this module's location.

      2. Relative workflow name (no path separator, e.g. "krea2_multigpu.json"):
         Resolved as <candidate_root>/workflows/<name>.
         This allows candidate-relative workflow references without an env override.

      3. Absolute external path (starts with "/" and not a candidate path):
         - Allowed ONLY if GPU_MANAGER_ALLOW_ABSOLUTE_WORKFLOW_PATHS=1 is set.
         - Otherwise raises WorkflowResolutionError (fail-closed).
         - If the allowed env is set, the path is returned unchanged.

    Legacy subprocess dependencies (nginx-manager.sh, pipeline-orchestrator.py,
    venv python) are handled by their own dedicated resolvers and are NOT
    affected by this function.

    Parameters:
        workflow_template_value: The raw workflow_template string from config.
                                 May be None, a relative name, or an absolute path.
        candidate_workflow_name: The canonical candidate-owned workflow name
                                 (e.g. "krea2_multigpu.json"). If provided,
                                 takes precedence for candidate-owned resolution.

    Returns:
        An absolute path to the workflow file.

    Raises:
        WorkflowResolutionError: The value could not be safely resolved to a
                                 candidate-owned workflow path without an
                                 explicit env override.

    Example (candidate-owned, no override needed):
        resolve_workflow_path(None, "krea2_multigpu.json")
        # -> <candidate_root>/workflows/krea2_multigpu.json

    Example (relative name from config):
        resolve_workflow_path("my-workflow.json")
        # -> <candidate_root>/workflows/my-workflow.json

    Example (absolute path, env override required):
        resolve_workflow_path("/absolute/path/to/media-pipeline/workflows/krea2.json")
        # -> raises WorkflowResolutionError
        # With GPU_MANAGER_ALLOW_ABSOLUTE_WORKFLOW_PATHS=1:
        # -> returns the path unchanged
    """
    # ── Case 1: candidate-owned workflow name ───────────────────────────────
    if candidate_workflow_name:
        resolved = _resolve_candidate_workflow(candidate_workflow_name)
        if resolved.is_file():
            return str(resolved)
        raise WorkflowResolutionError(
            f"Candidate workflow {candidate_workflow_name!r} not found at {resolved}. "
            f"Place the workflow under workflows/ in the candidate staging root, "
            f"or use a candidate_workflow_name that exists there."
        )

    # ── Case 2: no value provided ──────────────────────────────────────────
    if not workflow_template_value:
        raise WorkflowResolutionError(
            "workflow_template value is empty and no candidate_workflow_name was provided. "
            "Configure a workflow_template in the service definition or use a "
            "candidate_workflow_name."
        )

    # ── Case 3: absolute path ──────────────────────────────────────────────
    if workflow_template_value.startswith("/"):
        allow_env = os.environ.get(_ENV_ALLOW_ABSOLUTE_WORKFLOW_PATHS, "")
        if allow_env == "1":
            # Explicit opt-in: trust the configured absolute path
            return workflow_template_value
        raise WorkflowResolutionError(
            f"Absolute workflow path {workflow_template_value!r} is not permitted "
            f"without GPU_MANAGER_ALLOW_ABSOLUTE_WORKFLOW_PATHS=1. "
            f"Copy the workflow into the candidate workflows/ directory and "
            f"refer to it by name, or set GPU_MANAGER_ALLOW_ABSOLUTE_WORKFLOW_PATHS=1 "
            f"as an explicit override."
        )

    # ── Case 4: relative name ─────────────────────────────────────────────
    # Treat as a candidate-relative workflow name
    resolved = _resolve_candidate_workflow(workflow_template_value)
    if resolved.is_file():
        return str(resolved)
    raise WorkflowResolutionError(
        f"Workflow {workflow_template_value!r} not found at {resolved}. "
        f"Place the workflow under workflows/ in the candidate staging root."
    )


class WorkflowResolutionError(RuntimeError):
    """
    Raised when a workflow_template value from config cannot be safely resolved
    to a candidate-owned workflow path.

    Fail-closed: absolute paths require an explicit env override; missing
    candidate-relative paths raise an explicit error rather than falling back
    to the live media-pipeline checkout.
    """
    pass


def resolve_acestep_python() -> str:
    """
    Resolve the Python interpreter inside the ACE-Step venv.

    Resolution order:
      1. GPU_MANAGER_ACE_STEP_ROOT/.venv/bin/python if explicitly set
      2. GPU_MANAGER_ACE_STEP_ROOT/ACE-Step-1.5-dualgpu-candidate/.venv/bin/python
      3. no path (unset external root fails closed)

    Raises DependencyNotFound if the resolved path does not exist.
    """
    root = _get_acestep_root()
    # Accept either the candidate checkout itself or its parent directory.
    # The documented override points at the candidate checkout, while older
    # deployments used a parent containing ACE-Step-1.5-dualgpu-candidate/.
    candidates = (
        root / ".venv" / "bin" / "python",
        root / "ACE-Step-1.5-dualgpu-candidate" / ".venv" / "bin" / "python",
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    attempted = ", ".join(str(path) for path in candidates)
    raise DependencyNotFound(
        f"ACE-Step Python interpreter not found; attempted: {attempted}. "
        f"Set {_ENV_ACE_STEP_ROOT} to the ACE-Step root, "
        f"or ensure the candidate .venv/bin/python exists."
    )
