"""Read-only reconciliation preview for split GPU Manager registries.

The controller and broker must eventually agree on one routing revision, but
copying either ``services.json`` over the other is unsafe: broker-only
deployment settings and local host overlays may intentionally differ.  This
module compares two explicit registry documents and emits only fingerprints,
JSON paths, types, digests, and legacy finding metadata.  It never returns
configuration values and never writes a file.

The report is suitable for a GUI preview, an operator review, or a CI drift
gate.  A caller still has to choose the authority for every shared conflict
before activation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

try:  # Support ``PYTHONPATH=scripts`` and package-style imports.
    from gpu_manager_contracts import (
        registry_fingerprint,
        registry_legacy_findings,
        validate_registry,
    )
except ImportError:  # pragma: no cover - package import path
    from .gpu_manager_contracts import (  # type: ignore[no-redef]
        registry_fingerprint,
        registry_legacy_findings,
        validate_registry,
    )


# These sections affect scheduling and runtime admission.  They are shared
# policy even when a broker has additional deployment-local settings.
SHARED_SECTIONS = frozenset(
    {
        "services",
        "generation_templates",
        "routing_groups",
        "bundles",
        "runtime_profiles",
        "model_sets",
        "workflows",
        "scheduling",
        "gpu_devices",
        "pipeline_providers",
        "service_pipelines",
    }
)

# These sections are read by the Gemma broker and may legitimately be local to
# its deployment.  They are still reported so an operator can document the
# partition instead of mistaking it for shared policy.
BROKER_LOCAL_SECTIONS = frozenset(
    {"combined_gemma_broker", "combined_gemma_deployment"}
)


def _section_for_path(path: str) -> str:
    first = path.split(".", 1)[0] if path else "$"
    if first in SHARED_SECTIONS:
        return "shared"
    if first in BROKER_LOCAL_SECTIONS:
        return "broker_local"
    return "local"


def _shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _digest(value: Any) -> str:
    """Return a comparison digest without exposing the compared value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda item: repr(item),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_code(error: str) -> str:
    # Validation messages contain useful paths, but the report only needs a
    # stable category for aggregation.  Avoid copying user-provided values.
    tail = str(error).split(":", 1)[-1].strip()
    return tail.split(None, 1)[0] if tail else "invalid"


def _finding_summary(finding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code": str(finding.get("code", "unknown")),
        "severity": str(finding.get("severity", "info")),
        "path": str(finding.get("path", "$")),
        "detail": str(finding.get("detail", "")),
    }


def _compare(left: Any, right: Any, path: str, changes: list[dict[str, Any]]) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                changes.append(
                    {
                        "path": child,
                        "kind": "right_only",
                        "owner": _section_for_path(child),
                        "right_type": _shape(right[key]),
                        "right_digest": _digest(right[key]),
                    }
                )
            elif key not in right:
                changes.append(
                    {
                        "path": child,
                        "kind": "left_only",
                        "owner": _section_for_path(child),
                        "left_type": _shape(left[key]),
                        "left_digest": _digest(left[key]),
                    }
                )
            else:
                _compare(left[key], right[key], child, changes)
        return
    if left != right:
        changes.append(
            {
                "path": path or "$",
                "kind": "value_conflict",
                "owner": _section_for_path(path),
                "left_type": _shape(left),
                "right_type": _shape(right),
                "left_digest": _digest(left),
                "right_digest": _digest(right),
            }
        )


def _side_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    errors = [str(error) for error in validate_registry(config)]
    findings = [
        _finding_summary(item) for item in registry_legacy_findings(config)
    ]
    return {
        "fingerprint": registry_fingerprint(config),
        "validation_error_count": len(errors),
        "validation_error_codes": dict(sorted(Counter(map(_validation_code, errors)).items())),
        "legacy_finding_count": len(findings),
        "legacy_finding_codes": dict(
            sorted(Counter(item["code"] for item in findings).items())
        ),
        "legacy_findings": findings,
    }


def reconcile_registry_views(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    max_changes: int = 512,
) -> dict[str, Any]:
    """Build a sanitized, deterministic reconciliation report.

    ``left`` and ``right`` are explicit snapshots, not implicit live paths.
    ``max_changes`` bounds the report for GUI/health consumers while retaining
    the total count and a deterministic truncation marker.
    """

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("registry snapshots must be objects")
    if isinstance(max_changes, bool) or not isinstance(max_changes, int) or max_changes < 0:
        raise ValueError("max_changes must be a non-negative integer")

    changes: list[dict[str, Any]] = []
    _compare(left, right, "", changes)
    changes.sort(key=lambda item: (str(item["path"]), str(item["kind"])))
    total_changes = len(changes)
    shown_changes = changes[:max_changes]
    owner_counts = Counter(str(item["owner"]) for item in changes)
    shared_changes = [item for item in changes if item["owner"] == "shared"]
    local_changes = [item for item in changes if item["owner"] != "shared"]
    left_summary = _side_summary(left)
    right_summary = _side_summary(right)

    recommendations: list[dict[str, Any]] = []
    if left_summary["validation_error_count"]:
        recommendations.append(
            {
                "code": "left_registry_invalid",
                "severity": "error",
                "action": "repair_left_before_reconciliation",
            }
        )
    if right_summary["validation_error_count"]:
        recommendations.append(
            {
                "code": "right_registry_invalid",
                "severity": "error",
                "action": "repair_right_before_reconciliation",
            }
        )
    if shared_changes:
        recommendations.append(
            {
                "code": "shared_policy_drift",
                "severity": "error",
                "count": len(shared_changes),
                "action": "choose_an_authority_for_each_path_then_publish_one_revision",
            }
        )
    if local_changes:
        recommendations.append(
            {
                "code": "partitioned_local_drift",
                "severity": "warning",
                "count": len(local_changes),
                "action": "document_intentional_local_overlay_or_review",
            }
        )
    if left_summary["legacy_finding_count"] or right_summary["legacy_finding_count"]:
        recommendations.append(
            {
                "code": "legacy_review_required",
                "severity": "warning",
                "action": "review_inactive_or_orphaned_records_before_retirement",
            }
        )

    if left_summary["validation_error_count"] or right_summary["validation_error_count"]:
        status = "blocked"
    elif shared_changes:
        status = "review_required"
    elif local_changes or left_summary["legacy_finding_count"] or right_summary["legacy_finding_count"]:
        status = "partition_or_legacy_review"
    else:
        status = "aligned"

    return {
        "schema_version": 1,
        "status": status,
        "activation_ready": status == "aligned",
        "left": left_summary,
        "right": right_summary,
        "drift": {
            "change_count": total_changes,
            "shown_change_count": len(shown_changes),
            "truncated": total_changes > len(shown_changes),
            "owner_counts": dict(sorted(owner_counts.items())),
            "changes": shown_changes,
        },
        "recommendations": recommendations,
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load one explicit JSON snapshot; no default or environment path exists."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"registry snapshot is not a file: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read registry snapshot: {candidate}") from exc
    if not isinstance(value, dict):
        raise ValueError("registry snapshot must be a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two explicit GPU Manager registry snapshots without writing either one."
    )
    parser.add_argument("left", type=Path, help="first registry snapshot")
    parser.add_argument("right", type=Path, help="second registry snapshot")
    parser.add_argument(
        "--max-changes",
        type=int,
        default=512,
        help="maximum sanitized change records to emit (default: 512)",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit 1 unless both snapshots are valid and fully aligned",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = reconcile_registry_views(
            load_registry(args.left),
            load_registry(args.right),
            max_changes=args.max_changes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if args.fail_on_drift and not report["activation_ready"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    sys.exit(main())


__all__ = [
    "BROKER_LOCAL_SECTIONS",
    "SHARED_SECTIONS",
    "load_registry",
    "main",
    "reconcile_registry_views",
]
