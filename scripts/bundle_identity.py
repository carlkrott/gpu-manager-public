"""
bundle_identity.py — Fail-closed bundle resource identity validation.

This module provides pure validation functions that enforce the strict
bundle identity contract at the loader/scheduler boundary. It is safe to
import in both the live daemon and hermetic tests.

Contract invariants enforced
──────────────────────────
1. Every non-idle, non-reservation bundle with services[] (a top-level
   entry point) must declare capacity_slots (a positive integer).

2. Every non-idle bundle that is a primary (linked_to is non-empty,
   linked_from is empty) must declare claim_resources listing itself and
   all of its linked secondary bundles.

3. Every non-idle bundle that is a secondary (linked_from is non-empty)
   must declare claim_resources listing only itself.

4. Every claim_resources reference must name a bundle that actually exists
   in bundles{} (no dangling claims).

5. resource_group and resource_role must be present on every non-idle
   bundle. resource_role must be "primary" iff the bundle is a primary
   (linked_to non-empty, linked_from empty); "secondary" iff the bundle
   is a secondary (linked_from non-empty).

6. Idle bundles are exempt from all resource identity requirements
   (they are pure GPU identity state, not scheduling targets).

7. resource_contract must be "exclusive" for all non-idle bundles that
   participate in dual-GPU scheduling.

These checks are fail-closed: any violation raises BundleIdentityError
rather than silently returning a default or None.
"""

from __future__ import annotations

from typing import Any


class BundleIdentityError(ValueError):
    """Raised when a bundle violates the resource identity contract."""


def _is_primary(bundle_cfg: dict[str, Any]) -> bool:
    linked_to = bundle_cfg.get("linked_to", [])
    linked_from = bundle_cfg.get("linked_from", [])
    return bool(linked_to) and not linked_from


def _is_secondary(bundle_cfg: dict[str, Any]) -> bool:
    return bool(bundle_cfg.get("linked_from", []))


def _is_idle(bundle_cfg: dict[str, Any]) -> bool:
    return bool(bundle_cfg.get("idle", False))


def _is_reservation_only(bundle_cfg: dict[str, Any]) -> bool:
    """Reservation-only bundles have no services and are secondaries."""
    services = bundle_cfg.get("services", [])
    return not services and _is_secondary(bundle_cfg)


# ── Per-bundle validators ──────────────────────────────────────────────────────


def validate_primary_claims(bundle_name: str, bundle_cfg: dict[str, Any]) -> None:
    """
    Validate that a primary bundle declares the full dual-GPU claim set.

    A primary must have:
      - claim_resources listing itself AND all linked secondaries
      - resource_role == "primary"

    Raises BundleIdentityError on violation.
    """
    if not _is_primary(bundle_cfg):
        return  # Not a primary — let _validate_secondary handle it

    claim_resources = bundle_cfg.get("claim_resources")
    if not isinstance(claim_resources, list) or not claim_resources:
        raise BundleIdentityError(
            f"primary bundle '{bundle_name}' is missing or has empty "
            f"claim_resources — must list itself and all linked secondaries"
        )

    # Must include self
    if bundle_name not in claim_resources:
        raise BundleIdentityError(
            f"primary bundle '{bundle_name}' claim_resources must include "
            f"'{bundle_name}' itself; got {claim_resources}"
        )

    # Must include all secondaries (everything in linked_to)
    linked_to = bundle_cfg.get("linked_to", [])
    for secondary in linked_to:
        if secondary not in claim_resources:
            raise BundleIdentityError(
                f"primary bundle '{bundle_name}' claim_resources must include "
                f"linked secondary '{secondary}'; got {claim_resources}"
            )

    # Must not list any bundle that doesn't exist in linked_to
    for claimed in claim_resources:
        if claimed != bundle_name and claimed not in linked_to:
            raise BundleIdentityError(
                f"primary bundle '{bundle_name}' claim_resources references "
                f"'{claimed}' which is not in its linked_to list {linked_to}"
            )


def validate_secondary_claims(bundle_name: str, bundle_cfg: dict[str, Any]) -> None:
    """
    Validate that a secondary bundle declares its own identity claim only.

    A secondary must have:
      - claim_resources listing only itself
      - resource_role == "secondary"

    Raises BundleIdentityError on violation.
    """
    if not _is_secondary(bundle_cfg):
        return  # Not a secondary

    claim_resources = bundle_cfg.get("claim_resources")
    if not isinstance(claim_resources, list) or not claim_resources:
        raise BundleIdentityError(
            f"secondary bundle '{bundle_name}' is missing or has empty "
            f"claim_resources — must list only itself"
        )

    if claim_resources != [bundle_name]:
        raise BundleIdentityError(
            f"secondary bundle '{bundle_name}' claim_resources must list "
            f"only itself; got {claim_resources}"
        )


def validate_capacity_slots(bundle_name: str, bundle_cfg: dict[str, Any]) -> None:
    """
    Validate that a top-level non-idle bundle declares capacity_slots.

    A "top-level non-idle bundle" is one that:
      - is not idle
      - has services[] (is a scheduling entry point)

    Reservation-only secondaries (no services) are exempt because they
    are activated by their primary's linked_to claim, not scheduled directly.

    Raises BundleIdentityError on violation.
    """
    if _is_idle(bundle_cfg):
        return  # Idle bundles are pure identity state

    services = bundle_cfg.get("services", [])
    if not services:
        return  # Reservation-only bundle — activated by primary, not scheduled directly

    capacity_slots = bundle_cfg.get("capacity_slots")
    if not isinstance(capacity_slots, int) or capacity_slots < 1:
        raise BundleIdentityError(
            f"top-level non-idle bundle '{bundle_name}' (services={services}) "
            f"must declare capacity_slots as a positive integer; "
            f"got {capacity_slots!r}"
        )


def validate_resource_role(bundle_name: str, bundle_cfg: dict[str, Any]) -> None:
    """
    Validate that resource_role is consistent with the bundle's position
    in the linked chain (primary vs secondary).

    Raises BundleIdentityError on violation.
    """
    if _is_idle(bundle_cfg):
        return  # Idle bundles don't participate in dual-GPU resource contracts

    resource_role = bundle_cfg.get("resource_role")
    is_primary = _is_primary(bundle_cfg)
    is_secondary = _is_secondary(bundle_cfg)

    if not is_primary and not is_secondary:
        # Not part of a dual-GPU linked chain — no resource_role required
        return

    if is_primary and resource_role != "primary":
        raise BundleIdentityError(
            f"bundle '{bundle_name}' is a primary (linked_to is non-empty, "
            f"linked_from is empty) but resource_role is {resource_role!r} "
            f"instead of 'primary'"
        )

    if is_secondary and resource_role != "secondary":
        raise BundleIdentityError(
            f"bundle '{bundle_name}' is a secondary (linked_from is non-empty) "
            f"but resource_role is {resource_role!r} instead of 'secondary'"
        )


def validate_resource_group(bundle_name: str, bundle_cfg: dict[str, Any]) -> None:
    """
    Validate that a non-idle bundle declares resource_group.

    Raises BundleIdentityError on violation.
    """
    if _is_idle(bundle_cfg):
        return

    resource_group = bundle_cfg.get("resource_group")
    if not isinstance(resource_group, str) or not resource_group:
        raise BundleIdentityError(
            f"non-idle bundle '{bundle_name}' must declare resource_group; "
            f"got {resource_group!r}"
        )


def validate_resource_contract(bundle_name: str, bundle_cfg: dict[str, Any]) -> None:
    """
    Validate that a non-idle bundle that participates in dual-GPU scheduling
    declares resource_contract == "exclusive".

    Raises BundleIdentityError on violation.
    """
    if _is_idle(bundle_cfg):
        return

    # Bundles that are primaries or secondaries in a linked chain
    # must use exclusive contract
    is_primary = _is_primary(bundle_cfg)
    is_secondary = _is_secondary(bundle_cfg)

    if not is_primary and not is_secondary:
        return  # Standalone bundle, no dual-GPU requirement

    resource_contract = bundle_cfg.get("resource_contract")
    if resource_contract != "exclusive":
        raise BundleIdentityError(
            f"bundle '{bundle_name}' participates in dual-GPU scheduling "
            f"(is primary or secondary) but resource_contract is "
            f"{resource_contract!r} instead of 'exclusive'"
        )


# ── Routing-group consistency ─────────────────────────────────────────────────


def validate_routing_group_presence(
    bundle_name: str, bundle_cfg: dict[str, Any]
) -> None:
    """
    Validate that a non-idle, non-reservation bundle declares routing_group.

    A top-level non-idle bundle (has services[]) is a scheduling entry point
    and must declare routing_group so the scheduler can route jobs to it.

    Raises BundleIdentityError on violation.
    """
    if _is_idle(bundle_cfg):
        return

    services = bundle_cfg.get("services", [])
    if not services:
        return  # Reservation-only secondary — activated by primary, not scheduled directly

    routing_group = bundle_cfg.get("routing_group")
    if not isinstance(routing_group, str) or not routing_group:
        raise BundleIdentityError(
            f"non-idle top-level bundle '{bundle_name}' (services={services}) "
            f"must declare routing_group; got {routing_group!r}"
        )


def validate_routing_group_chain_consistency(
    bundle_name: str,
    bundle_cfg: dict[str, Any],
    all_bundles: dict[str, dict[str, Any]],
) -> None:
    """
    Validate that a bundle's routing_group is consistent with its linked-chain
    peers.

    Dual-GPU secondaries that have no explicit routing_group must inherit one
    from their primary via linked_to. If the secondary's gpu_id would cause it
    to be placed in a different routing_group than its primary, that is a
    semantic inconsistency — the GPU identity of the secondary cannot fulfill
    the primary's routing_group contract.

    Raises BundleIdentityError on violation.
    """
    if _is_idle(bundle_cfg):
        return

    routing_group = bundle_cfg.get("routing_group")
    linked_to = bundle_cfg.get("linked_to", [])
    linked_from = bundle_cfg.get("linked_from", [])

    # Primary: all secondaries must declare the same routing_group as this primary
    if not linked_from and linked_to:
        # This is a primary — collect all secondaries and verify they either
        # declare the same routing_group or have no routing_group (inherit from primary)
        for sec_name in linked_to:
            sec_cfg = all_bundles.get(sec_name, {})
            if not isinstance(sec_cfg, dict):
                continue
            sec_rg = sec_cfg.get("routing_group")
            if sec_rg and sec_rg != routing_group:
                raise BundleIdentityError(
                    f"primary bundle '{bundle_name}' has routing_group "
                    f"'{routing_group}' but secondary '{sec_name}' declares "
                    f"routing_group '{sec_rg}' — all members of a dual-GPU "
                    f"exclusive chain must share the same routing_group"
                )

    # Secondary: the primary must declare the same routing_group
    if linked_from and not linked_to:
        # This is a secondary — check that the primary agrees on routing_group
        for primary_name in linked_from:
            primary_cfg = all_bundles.get(primary_name, {})
            if not isinstance(primary_cfg, dict):
                continue
            primary_rg = primary_cfg.get("routing_group")
            if routing_group and primary_rg and routing_group != primary_rg:
                raise BundleIdentityError(
                    f"secondary bundle '{bundle_name}' has routing_group "
                    f"'{routing_group}' but its primary '{primary_name}' "
                    f"declares routing_group '{primary_rg}' — "
                    f"all members of a dual-GPU exclusive chain must share "
                    f"the same routing_group"
                )


# ── Cross-bundle validators ────────────────────────────────────────────────────


def validate_claim_references(bundle_name: str, claim: str, all_bundle_names: frozenset[str]) -> None:
    """
    Validate that a single claim_resources entry names an existing bundle.

    Raises BundleIdentityError if the referenced bundle does not exist.
    """
    if claim not in all_bundle_names:
        raise BundleIdentityError(
            f"bundle '{bundle_name}' claims resource '{claim}' which does not "
            f"exist in bundles{{}}"
        )


def validate_all_claim_references(
    bundles: dict[str, dict[str, Any]]
) -> None:
    """
    Validate that every claim_resources entry in every bundle references
    an existing bundle.

    Raises BundleIdentityError on any dangling reference.
    """
    all_bundle_names = frozenset(bundles.keys())

    for bundle_name, bundle_cfg in bundles.items():
        if not isinstance(bundle_cfg, dict):
            continue
        claim_resources = bundle_cfg.get("claim_resources")
        if not isinstance(claim_resources, list):
            continue
        for claim in claim_resources:
            validate_claim_reference(bundle_name, claim, all_bundle_names)


def validate_claim_reference(
    bundle_name: str,
    claim: str,
    all_bundle_names: frozenset[str]
) -> None:
    if claim not in all_bundle_names:
        raise BundleIdentityError(
            f"bundle '{bundle_name}' claims resource '{claim}' which does not "
            f"exist in bundles{{}}"
        )


# ── Aggregate validation ───────────────────────────────────────────────────────


def validate_bundle_identities(
    bundles: dict[str, dict[str, Any]]
) -> list[str]:
    """
    Validate every bundle in bundles{} against the full identity contract.

    Returns a list of error messages (empty if all bundles are valid).
    Uses collect-all errors rather than raising on first, so callers can
    see all violations at once.

    Use validate_bundle_identities_strict() for fail-closed behavior
    (raises on any violation).
    """
    errors: list[str] = []

    for bundle_name, bundle_cfg in bundles.items():
        if not isinstance(bundle_cfg, dict):
            errors.append(f"bundle '{bundle_name}' is not a dict")
            continue

        # Skip idle bundles — they are pure identity state, not scheduling targets
        if _is_idle(bundle_cfg):
            continue

        # 1. capacity_slots (top-level entry points only)
        try:
            validate_capacity_slots(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[capacity_slots] {exc}")

        # 2. resource_group
        try:
            validate_resource_group(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[resource_group] {exc}")

        # 3. resource_role consistency
        try:
            validate_resource_role(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[resource_role] {exc}")

        # 4. resource_contract for dual-GPU participants
        try:
            validate_resource_contract(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[resource_contract] {exc}")

        # 5. primary claims completeness
        try:
            validate_primary_claims(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[primary_claims] {exc}")

        # 6. secondary claims
        try:
            validate_secondary_claims(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[secondary_claims] {exc}")

        # 7. routing_group presence on top-level non-idle bundles
        try:
            validate_routing_group_presence(bundle_name, bundle_cfg)
        except BundleIdentityError as exc:
            errors.append(f"[routing_group_presence] {exc}")

        # 8. routing_group chain consistency (dual-GPU primaries/secondaries agree)
        try:
            validate_routing_group_chain_consistency(bundle_name, bundle_cfg, bundles)
        except BundleIdentityError as exc:
            errors.append(f"[routing_group_chain] {exc}")

    # 7. Cross-bundle claim reference validity
    all_bundle_names = frozenset(bundles.keys())
    for bundle_name, bundle_cfg in bundles.items():
        if not isinstance(bundle_cfg, dict):
            continue
        claim_resources = bundle_cfg.get("claim_resources")
        if not isinstance(claim_resources, list):
            continue
        for claim in claim_resources:
            try:
                validate_claim_reference(bundle_name, claim, all_bundle_names)
            except BundleIdentityError as exc:
                errors.append(f"[claim_reference] {exc}")

    return errors


def validate_bundle_identities_strict(
    bundles: dict[str, dict[str, Any]]
) -> None:
    """
    Fail-closed variant of validate_bundle_identities().

    Raises BundleIdentityError with a joined message if any bundle violates
    the contract. Use this in the scheduler/loader hot path where silent
    defaults must never be accepted.
    """
    errors = validate_bundle_identities(bundles)
    if errors:
        raise BundleIdentityError(
            "Bundle identity contract violations:\n  " + "\n  ".join(errors)
        )


# ── Aggregate dual-GPU capacity ────────────────────────────────────────────────


def compute_aggregate_capacity(
    bundles: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """
    Compute aggregate logical capacity per resource_group.

    For exclusive dual-GPU groups (where primaries claim both GPUs atomically
    via their claim_resources), the aggregate is exactly the primary's
    capacity_slots — secondaries are NOT additive since they are activated
    together with the primary as one logical schedulable unit.

    For non-linked (standalone) bundles, capacity is additive as normal.

    Returns a dict mapping resource_group -> logical_schedulable_capacity.
    """
    by_group: dict[str, dict[str, int]] = {}

    for bundle_name, bundle_cfg in bundles.items():
        if not isinstance(bundle_cfg, dict):
            continue
        if _is_idle(bundle_cfg):
            continue

        resource_group = bundle_cfg.get("resource_group")
        if not resource_group:
            continue

        capacity_slots = bundle_cfg.get("capacity_slots")
        if not isinstance(capacity_slots, int) or capacity_slots < 1:
            continue

        if resource_group not in by_group:
            by_group[resource_group] = {"primary": 0, "secondary": 0, "total": 0}

        is_primary = _is_primary(bundle_cfg)
        is_secondary = _is_secondary(bundle_cfg)

        if is_primary:
            by_group[resource_group]["primary"] += capacity_slots
        elif is_secondary:
            by_group[resource_group]["secondary"] += capacity_slots
        else:
            # Non-linked / standalone bundle
            by_group[resource_group]["total"] += capacity_slots

    result: dict[str, int] = {}
    for rg, counts in by_group.items():
        # Exclusive dual-GPU: primary and secondary together represent ONE
        # logical schedulable unit; secondaries must NOT be added separately.
        # The logical capacity is primary + total (non-linked bundles).
        result[rg] = counts["primary"] + counts["total"]

    return result


def validate_aggregate_dual_gpu_capacity(
    bundles: dict[str, dict[str, Any]]
) -> list[str]:
    """
    Validate the dual-GPU capacity invariant: for every exclusive dual-GPU
    resource_group, the aggregate capacity must equal 1 (the primary and
    secondary are claimed together as one unit).

    Returns a list of error messages (empty if all groups are valid).
    """
    errors: list[str] = []
    by_group: dict[str, dict[str, int]] = {}

    for bundle_name, bundle_cfg in bundles.items():
        if not isinstance(bundle_cfg, dict):
            continue
        if _is_idle(bundle_cfg):
            continue

        resource_group = bundle_cfg.get("resource_group")
        if not resource_group:
            continue

        capacity_slots = bundle_cfg.get("capacity_slots", 0)
        is_primary = _is_primary(bundle_cfg)
        is_secondary = _is_secondary(bundle_cfg)

        if not is_primary and not is_secondary:
            continue  # Non-linked bundle

        if resource_group not in by_group:
            by_group[resource_group] = {"primary": 0, "secondary": 0}

        if is_primary:
            by_group[resource_group]["primary"] += capacity_slots
        elif is_secondary:
            by_group[resource_group]["secondary"] += capacity_slots

    for rg, counts in by_group.items():
        # The aggregate claim unit is 1 per resource_group
        # (primary + its secondaries = 1 logical slot)
        if counts["primary"] != counts["secondary"]:
            errors.append(
                f"resource_group '{rg}' has asymmetric dual-GPU capacity: "
                f"primary={counts['primary']} secondary={counts['secondary']}. "
                f"For exclusive dual-GPU groups, both sides must be equal "
                f"(the primary claims both GPUs atomically)."
            )

    return errors
