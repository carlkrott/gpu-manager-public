#!/usr/bin/env python3.14
"""Retire a dead provider instance and repoint the routes that named it.

The tool is dry-run by default: it reports the routes it would rewrite and the
sha256 of the current and the would-be file, and writes nothing.  ``--apply``
is the only mode that touches disk, and it refuses to run against an unsafe
target or against source bytes that changed after they were inspected.

Only names, counts, validation verdicts and hashes are ever emitted: no host,
address, token or configuration value leaves this tool.

Route references are validated through the controller's real ``ModelValidator``
so an unknown model id cannot pass by default (the validator routes unknown
ids straight through), and every surviving reference is checked to name a
member that actually exists in the registry being edited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
from types import ModuleType
from typing import Any, Mapping

#: Rule fields that name a single model member.
REFERENCE_FIELDS = ("require", "prefer", "fallback")
#: Rule fields that name a list of members.
REFERENCE_LIST_FIELDS = ("never_fallback_to",)
#: Rule fields the controller's ModelValidator consumes as task requirements.
VALIDATOR_REQUIREMENT_FIELDS = ("min_context", "require_vision", "min_complexity")
#: Capability bound the controller's validator does not enforce itself.
CAPABILITY_BOUND_FIELDS = ("max_output_tokens",)


class RetirementError(RuntimeError):
    """A refusal: nothing was written and the caller must not apply."""


class ApplyError(RuntimeError):
    """A failure during the writing phase of ``--apply``."""


def _controller_path() -> Path:
    return Path(__file__).resolve().with_name("gpu-manager.py")


_controller: ModuleType | None = None


def load_controller() -> ModuleType:
    """Load the real controller module (its loader, validator and rules)."""
    global _controller
    if _controller is None:
        path = _controller_path()
        if not path.is_file():
            raise RetirementError(f"controller not found: {path}")
        spec = importlib.util.spec_from_file_location("retire_provider_controller", path)
        if spec is None or spec.loader is None:
            raise RetirementError(f"controller is not importable: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for required in ("ModelValidator", "_load_providers_config"):
            if not hasattr(module, required):
                raise RetirementError(f"controller lacks {required}")
        _controller = module
    return _controller


def load_config(config_path: Path, controller: ModuleType | None = None) -> dict:
    """Read the config through the controller's real loader.

    ``PROVIDERS_CONFIG_PATH`` is a module global resolved at import time, so it
    is rebound before loading to make the loader read ``config_path`` rather
    than whatever the ambient environment points at.
    """
    controller = controller or load_controller()
    target = Path(config_path).resolve(strict=True)
    controller.PROVIDERS_CONFIG_PATH = str(target)
    loaded = controller._load_providers_config()
    if not isinstance(loaded, dict) or not loaded:
        raise RetirementError(f"controller loader returned no config for {target}")
    return loaded


def local_instances(config: Mapping[str, Any]) -> dict:
    """A copy of the local instance table, safe for the caller to mutate."""
    providers = config.get("providers")
    local = providers.get("local", {}) if isinstance(providers, Mapping) else {}
    instances = local.get("instances", {}) if isinstance(local, Mapping) else {}
    return dict(instances) if isinstance(instances, Mapping) else {}


def drop_instance(config: dict, name: str) -> None:
    """Remove a local instance from the config in place."""
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return
    local = providers.get("local")
    if not isinstance(local, dict):
        return
    instances = local.get("instances")
    if isinstance(instances, dict):
        instances.pop(name, None)


def known_members(config: Mapping[str, Any]) -> set:
    """Every member name the controller's validator can actually resolve."""
    members = set(local_instances(config))
    providers = config.get("providers")
    if isinstance(providers, Mapping):
        for provider_id, provider in providers.items():
            if provider_id == "local" or not isinstance(provider, Mapping):
                continue
            models = provider.get("models")
            if isinstance(models, Mapping):
                members |= {f"{provider_id}/{name}" for name in models}
    return members


def references_to(config: Mapping[str, Any], name: str) -> list:
    """Every ``(hint, field)`` pair whose member reference names ``name``."""
    found = []
    for rule in config.get("routing_rules", []) or []:
        if not isinstance(rule, Mapping):
            continue
        hint = rule.get("hint")
        for field in REFERENCE_FIELDS:
            if rule.get(field) == name:
                found.append((hint, field))
        for field in REFERENCE_LIST_FIELDS:
            values = rule.get(field)
            if isinstance(values, (list, tuple)) and name in values:
                found.append((hint, field))
    return found


def assert_no_dangling_references(config: Mapping[str, Any], known: set) -> None:
    """Refuse a bare deletion: every surviving reference must resolve."""
    dangling = []
    for rule in config.get("routing_rules", []) or []:
        if not isinstance(rule, Mapping):
            continue
        hint = rule.get("hint")
        for field in REFERENCE_FIELDS:
            target = rule.get(field)
            if isinstance(target, str) and target not in known:
                dangling.append(f"{hint}:{field}")
        for field in REFERENCE_LIST_FIELDS:
            values = rule.get(field)
            if isinstance(values, (list, tuple)):
                dangling += [f"{hint}:{field}" for v in values if v not in known]
    if dangling:
        raise RetirementError(
            "refusing bare deletion: unresolved route references "
            + ", ".join(sorted(dangling))
        )


def _requirements_for(rule: Mapping[str, Any]) -> dict:
    return {k: rule[k] for k in VALIDATOR_REQUIREMENT_FIELDS if k in rule}


def _capability_violations(
    member: str, model: Mapping[str, Any], rule: Mapping[str, Any]
) -> list:
    caps = model.get("capabilities", {}) if isinstance(model, Mapping) else {}
    caps = caps if isinstance(caps, Mapping) else {}
    limit = model.get("max_output_tokens", caps.get("max_output_tokens"))
    for field in CAPABILITY_BOUND_FIELDS:
        required = rule.get(field)
        if required is None or not isinstance(limit, (int, float)):
            continue
        if limit < required:
            return [f"{rule.get('hint')}:{field}={required}>{member}:{limit}"]
    return []


def _resolve_member(config: Mapping[str, Any], member: str) -> Mapping[str, Any] | None:
    instances = local_instances(config)
    if member in instances:
        return instances[member]
    providers = config.get("providers")
    if isinstance(providers, Mapping) and "/" in member:
        provider_id, model_name = member.split("/", 1)
        provider = providers.get(provider_id)
        if isinstance(provider, Mapping):
            models = provider.get("models")
            if isinstance(models, Mapping) and model_name in models:
                return models[model_name]
    return None


def evaluate_routes(
    config: Mapping[str, Any], controller: ModuleType | None = None, members=None
) -> list:
    """Drive the real validator over every rule's own members.

    Returns one record per ``(hint, member)`` pair the config routes to, with
    the controller's verdict.  An incompatible member surfaces here as
    ``ok=False`` instead of silently routing.
    """
    controller = controller or load_controller()
    members = known_members(config) if members is None else set(members)
    validator = controller.ModelValidator(dict(config))
    records = []
    for rule in config.get("routing_rules", []) or []:
        if not isinstance(rule, Mapping):
            continue
        hint = rule.get("hint")
        requirements = _requirements_for(rule)
        for field in REFERENCE_FIELDS:
            member = rule.get(field)
            if not isinstance(member, str) or member not in members:
                continue
            verdict = validator.validate(member, task_hint=hint, requirements=requirements)
            records.append(
                {
                    "hint": hint,
                    "field": field,
                    "member": member,
                    "ok": bool(verdict.get("ok")),
                    "action": verdict.get("action"),
                    "capability_violations": _capability_violations(
                        member, _resolve_member(config, member) or {}, rule
                    ),
                }
            )
    return records


def _rewrite(
    config: Mapping[str, Any], instance: str, replacement: str | None
) -> dict:
    """Return a new config with the instance dropped and its refs repointed.

    ``replacement=None`` performs the bare delete: the instance goes away and
    the rules keep naming it, which is exactly the state that must be refused.
    """
    candidate: dict = json.loads(json.dumps(config))
    drop_instance(candidate, instance)
    if replacement is None:
        return candidate
    for rule in candidate.get("routing_rules", []) or []:
        if not isinstance(rule, Mapping):
            continue
        for field in REFERENCE_FIELDS:
            if rule.get(field) == instance:
                rule[field] = replacement
        for field in REFERENCE_LIST_FIELDS:
            values = rule.get(field)
            if isinstance(values, list):
                rule[field] = [replacement if v == instance else v for v in values]
    return candidate


def render(config: Mapping[str, Any]) -> bytes:
    """Canonical, deterministic bytes for a config document."""
    return (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def plan_retirement(
    config: Mapping[str, Any],
    instance: str,
    replacement: str | None,
    controller: ModuleType | None = None,
) -> dict:
    """Plan the retirement without writing anything.

    Refuses a missing instance, a missing or unresolvable replacement, a
    replacement that is the instance itself, a bare deletion that would strand
    a route, and a replacement the real validator rejects for a route it would
    now serve.
    """
    controller = controller or load_controller()
    instances = local_instances(config)
    current_references = references_to(config, instance)
    if instance not in instances:
        # Idempotence: a second run over an already-retired config must report
        # "already retired", not fail.  Routes still naming the absent instance
        # is the broken bare-delete state and stays refused.
        if current_references:
            raise RetirementError(
                f"refusing to continue: {instance!r} is absent but still referenced by "
                + ", ".join(sorted(f"{h}:{f}" for h, f in current_references))
            )
        candidate: dict = json.loads(json.dumps(config))
        return {
            "instance": instance,
            "replacement": replacement,
            "rewritten": [],
            "already_retired": True,
            "candidate": candidate,
            "routes": evaluate_routes(candidate, controller),
        }
    if replacement is None:
        # Bare deletion: prove it is unsafe rather than producing a broken file.
        # The known set comes from the *stripped* config, so the retired name is
        # genuinely unknown and a rule still naming it is reported as dangling.
        stripped = _rewrite(config, instance, None)
        assert_no_dangling_references(stripped, known_members(stripped))
    if replacement == instance:
        raise RetirementError(f"replacement {replacement!r} is the retired instance itself")
    known = known_members(config)
    if replacement not in known:
        raise RetirementError(f"replacement {replacement!r} is not a resolvable member")

    candidate = _rewrite(config, instance, replacement)
    local_instances(candidate).pop(instance, None)
    assert_no_dangling_references(candidate, known)
    leftover = references_to(candidate, instance)
    if leftover:
        raise RetirementError(
            f"retired instance {instance!r} still referenced by "
            + ", ".join(sorted(f"{h}:{f}" for h, f in leftover))
        )
    records = evaluate_routes(candidate, controller, known)
    rejected = [
        r for r in records if r["member"] == replacement and not r["ok"]
    ]
    # The capability bound is checked on every route the replacement would now
    # serve, not only on the ones the validator rejected: the validator has no
    # max_output_tokens gate, so a member the validator accepts can still be
    # under a rule's bound.
    violations = [v for r in records if r["member"] == replacement
                  for v in r["capability_violations"]]
    if rejected:
        detail = ", ".join(
            f"{r['hint']}:{r['field']}->{r['member']}={r['action']}" for r in rejected
        )
        raise RetirementError(f"replacement {replacement!r} rejected by validator: {detail}")
    if violations:
        raise RetirementError(
            f"replacement {replacement!r} under bound: " + ", ".join(sorted(violations))
        )
    return {
        "instance": instance,
        "replacement": replacement,
        "rewritten": [{"hint": h, "field": f} for h, f in current_references],
        "candidate": candidate,
        "routes": records,
    }



def read_target(config_path: Path) -> bytes:
    """Read the target, refusing symlinks and non-regular files."""
    target = Path(config_path)
    if target.is_symlink():
        raise RetirementError(f"refusing symlinked config target: {target}")
    if not target.exists():
        raise RetirementError(f"config target does not exist: {target}")
    if not stat.S_ISREG(target.stat().st_mode):
        raise RetirementError(f"config target is not a regular file: {target}")
    payload = target.read_bytes()
    if not payload.strip():
        raise RetirementError(f"config target is empty: {target}")
    return payload


def backup_path(target: Path, when: float | None = None) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(when if when is not None else time.time()))
    return target.with_name(f"{target.name}.bak-{stamp}")


def write_backup(target: Path, payload: bytes, when: float | None = None) -> Path:
    """Create an exclusive 0600 copy of the current bytes next to the target."""
    destination = backup_path(target, when)
    try:
        handle = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ApplyError(f"backup already exists, refusing to overwrite: {destination}") from exc
    with os.fdopen(handle, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(destination, 0o600)
    return destination


def apply_atomically(target: Path, payload: bytes, replace=None) -> Path:
    """Write ``payload`` over ``target`` atomically, cleaning up on failure.

    ``replace`` is resolved at call time so a caller (or a test) can substitute
    the rename step and still exercise the cleanup path.
    """
    replace = replace or os.replace
    temporary = target.with_name(f".{target.name}.retire-tmp-{os.getpid()}")
    try:
        handle = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _emit(report: Mapping[str, Any], *, apply: bool) -> None:
    print(f"target: {report['target']}")
    print(f"mode: {'apply' if apply else 'dry-run'}")
    print(f"instance: {report['instance']}")
    print(f"replacement: {report['replacement'] or 'none (bare delete requested)'}")
    print(f"rewritten_routes: {len(report['rewritten'])}")
    for entry in report["rewritten"]:
        print(f"  rewrite: {entry['hint']}:{entry['field']}")
    for record in report["routes"]:
        print(
            f"  route: {record['hint']}:{record['field']} -> {record['member']} "
            f"ok={record['ok']} action={record['action']}"
        )
    print(f"sha256_current: {report['sha256_current']}")
    print(f"sha256_would_be: {report['sha256_would_be']}")
    # ``already_applied`` is the real answer in both directions: it is True only
    # when the current bytes already equal the planned bytes.  Deriving the line
    # from ``changed`` instead (the inverse flag) made this a constant "false".
    print(f"idempotent: {report['already_applied']}")
    for line in report["write"]:
        print(line)
    print(f"verdict: {report['verdict']}")


def run(args: argparse.Namespace) -> dict:
    target = Path(args.config)
    payload = read_target(target)
    current_sha = sha256_bytes(payload)
    controller = load_controller()
    config = load_config(target, controller)
    plan = plan_retirement(config, args.instance, args.replacement, controller)
    candidate_bytes = render(plan["candidate"])
    would_be = sha256_bytes(candidate_bytes)
    already_applied = current_sha == would_be
    report = {
        "target": str(target),
        "instance": args.instance,
        "replacement": args.replacement,
        "rewritten": plan["rewritten"],
        "routes": plan["routes"],
        "sha256_current": current_sha,
        "sha256_would_be": would_be,
        "changed": not already_applied,
        "already_applied": already_applied,
        "write": [],
    }
    if plan.get("already_retired"):
        report["verdict"] = "already-retired: no write needed"
        return report
    if not args.apply:
        report["verdict"] = "dry-run: no write performed"
        return report
    if args.expect_sha256 and args.expect_sha256 != current_sha:
        report["verdict"] = "refused: source bytes changed since --expect-sha256"
        raise RetirementError(report["verdict"])
    if already_applied:
        report["verdict"] = "already-retired: no write needed"
        return report
    backup = write_backup(target, payload)
    if target.read_bytes() != payload:
        backup.unlink(missing_ok=True)
        raise ApplyError("source bytes changed after backup; nothing replaced")
    apply_atomically(target, candidate_bytes)
    written = target.read_bytes()
    if written != candidate_bytes:
        raise ApplyError("replacement bytes do not match the planned bytes")
    report["write"] = [
        f"backup: {backup} mode=0600",
        f"wrote: {target} bytes={len(written)}",
    ]
    report["verdict"] = "applied"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retire_provider_instance.py",
        description=(
            "Retire a dead local provider instance and repoint every route that "
            "named it. Dry run by default: nothing is written unless --apply."
        ),
        epilog=(
            "Example dry run:\n"
            "  retire_provider_instance.py --config providers.json \\\n"
            "      --instance dead-member --replacement healthy-member\n"
            "Example apply (makes an exclusive 0600 timestamped backup first):\n"
            "  retire_provider_instance.py --config providers.json \\\n"
            "      --instance dead-member --replacement healthy-member --apply"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="providers.json to inspect and (with --apply) rewrite",
    )
    parser.add_argument(
        "--instance",
        required=True,
        help="local instance name to retire; must exist in the config",
    )
    parser.add_argument(
        "--replacement",
        default=None,
        help="existing member name every reference is repointed to; omit to "
        "prove a bare deletion is unsafe (the tool refuses it)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "write the change: 0600 timestamped backup then atomic replace. "
            "Without this flag the run is a dry run and writes nothing."
        ),
    )
    parser.add_argument(
        "--expect-sha256",
        default=None,
        help="refuse --apply unless the target still hashes to this value",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (RetirementError, ApplyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    _emit(report, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())