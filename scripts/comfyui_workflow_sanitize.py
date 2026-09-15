"""
Phase 5.2.79 (R2): canonical ComfyUI workflow sanitizer.

Single source of truth for stripping non-node keys from a workflow dict
before it is POSTed to ComfyUI's /prompt endpoint.

Before this module existed, the same sanitization logic lived inline in
exactly one place (gpu-manager._build_comfyui_workflow_entry, lines 9909-9980),
and was bypassed by 4 other entry points -- leading to "Failed to validate
prompt for output N" or HTTP 400 "missing_node_type" when a workflow carried
a top-level _meta dict. This module extracts the rules into one callable so
they're DRY, testable, and reusable at every choke point.

The rules (kept identical to the original 5.2.74 logic):
  1. Drop any top-level key whose name is in _META_KEYS.
  2. Drop any top-level key whose value is not a dict OR is missing
     `class_type` -- these are the only shapes ComfyUI accepts at the
     top level of a workflow.

Usage:
    from comfyui_workflow_sanitize import sanitize_comfyui_workflow
    safe = sanitize_comfyui_workflow(workflow_dict, source="<caller-tag>")

The `source` tag is used for log lines so the operator can see WHERE a
stripped key came from (operator visibility per the R2 risk note:
"sanitizing at Job level could mask future bugs").
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_META_KEYS = ("_meta", "_comment", "_notes", "_doc", "_description")


def sanitize_comfyui_workflow(workflow: Any, *, source: str = "unknown") -> dict:
    """Return a NEW dict safe to POST to ComfyUI /prompt.

    - If `workflow` is not a dict, returns {} (the caller should reject).
    - Strips top-level keys in _META_KEYS.
    - Strips top-level keys whose value is not a dict with `class_type`.
    - Logs every stripped key with its source tag (operator visibility).
    - NEVER mutates the input dict.
    """
    if not isinstance(workflow, dict):
        logger.warning(
            "sanitize_comfyui_workflow[%s]: input is not a dict (%r)",
            source, type(workflow).__name__,
        )
        return {}
    safe = dict(workflow)  # shallow copy; values are the same objects but we never mutate them
    stripped_meta: list[str] = []
    for mk in _META_KEYS:
        if mk in safe:
            stripped_meta.append(mk)
            safe.pop(mk)
    invalid = [
        k for k, v in list(safe.items())
        if not isinstance(v, dict) or "class_type" not in v
    ]
    for ik in invalid:
        safe.pop(ik, None)
    if stripped_meta or invalid:
        logger.warning(
            "sanitize_comfyui_workflow[%s]: stripped %d _meta keys (%s) "
            "and %d invalid top-level keys (%s)",
            source, len(stripped_meta), stripped_meta,
            len(invalid), invalid,
        )
    return safe
