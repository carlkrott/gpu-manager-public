"""Behaviour of the provider-retirement editor against synthetic registries.

Every fixture here is synthetic and loopback-neutral: member names, hints and
model ids are invented, and no host, address or credential appears.  The tests
drive the *real* controller loader and the *real* ``ModelValidator`` from
``scripts/gpu-manager.py`` (loaded by path, with its module-level
``PROVIDERS_CONFIG_PATH`` rebound to the fixture) rather than re-implementing
route resolution, so a rule the validator would reject cannot pass here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import retire_provider_instance as rpi  # noqa: E402


CONTROLLER = ROOT / "scripts" / "gpu-manager.py"


# ── synthetic registry ────────────────────────────────────────────────────
# "writer-primary" is the dead member: present in the registry, named by three
# routes, and serving nothing.  "writer-standby" is healthy and clears the
# bounds the three routes carry.  "helper-tiny" fails the high-complexity gate.

DEAD = "writer-primary"
STANDBY = "writer-standby"
TINY = "helper-tiny"
CLOUD = "synthetic-cloud/planner-v2"

BOOK_WRITING = "synthetic_book_writing"
BOOK_OUTLINE = "synthetic_book_outline"
STORYBOARD = "synthetic_storyboard"


def _instance(name: str, **overrides) -> dict:
    instance = {
        "display": name,
        "context_length": 131072,
        "capabilities": {
            "max_output_tokens": 4096,
            "min_complexity": "high",
            "reasoning": True,
            "strengths": ["synthesis"],
            "vision": False,
        },
        "machine": "loopback",
        "model_file": f"/dev/null/{name}.bin",
        "port": 9200,
        "systemd_unit": f"{name}.service",
    }
    instance.update(overrides)
    return instance


def _rule(hint: str, **fields) -> dict:
    rule = {"hint": hint, "reason": f"route {hint} to the retired member", **fields}
    return rule


def _config() -> dict:
    """A registry shaped like the real defect: one dead member, three routes."""
    return {
        "providers": {
            "local": {
                "display": "loopback-local",
                "type": "local",
                "instances": {
                    DEAD: _instance(DEAD),
                    STANDBY: _instance(STANDBY, port=9201),
                    TINY: _instance(
                        TINY,
                        context_length=4096,
                        port=9202,
                        capabilities={
                            "max_output_tokens": 512,
                            "min_complexity": "low",
                            "reasoning": False,
                            "strengths": [],
                            "vision": False,
                        },
                    ),
                },
            },
            "synthetic-cloud": {
                "type": "cloud",
                "display": "synthetic",
                "models": {
                    "planner-v2": {
                        "display": "planner-v2",
                        "context_length": 200000,
                        "capabilities": {"min_complexity": "high"},
                    }
                },
            },
        },
        "routing_rules": [
            _rule(
                BOOK_WRITING,
                require=DEAD,
                min_complexity="high",
                min_context=8192,
                never_fallback_to=[TINY],
            ),
            _rule(
                BOOK_OUTLINE,
                require=DEAD,
                min_complexity="high",
                min_context=4096,
                never_fallback_to=[TINY],
            ),
            _rule(
                STORYBOARD,
                prefer=DEAD,
                fallback=CLOUD,
                max_output_tokens=2000,
                min_context=2048,
            ),
        ],
    }


def _write(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return _write(tmp_path / "config" / "providers.json", _config())


@pytest.fixture
def controller():
    return rpi.load_controller()


def _load_through_real_loader(path: Path, controller) -> dict:
    """Read the fixture back with the controller's own loader.

    ``PROVIDERS_CONFIG_PATH`` is a module global bound at import time, so it is
    rebound here before loading: without that the loader would read whatever the
    ambient environment points at instead of the fixture.
    """
    controller.PROVIDERS_CONFIG_PATH = str(path)
    loaded = controller._load_providers_config()
    assert loaded, "real loader returned nothing for the fixture"
    return loaded


def _route_target(loaded: dict, hint: str) -> str | None:
    for rule in loaded.get("routing_rules", []):
        if rule.get("hint") == hint:
            return rule.get("require") or rule.get("prefer")
    return None


# ── the defect: a dead require member that silently routes ─────────────────

def test_bare_delete_strands_the_require_rules_and_is_refused(controller):
    config = _config()
    stripped = rpi._rewrite(config, DEAD, None)

    assert DEAD not in rpi.local_instances(stripped), "instance must be gone"
    assert _route_target(stripped, BOOK_WRITING) == DEAD, "require rule must still name it"

    # The real validator does NOT catch this: an unknown id routes straight
    # through.  That is precisely why the tool validates membership itself.
    validator = controller.ModelValidator(stripped)
    assert validator.validate(DEAD, task_hint=BOOK_WRITING)["ok"] is True

    with pytest.raises(rpi.RetirementError) as excinfo:
        rpi.plan_retirement(config, DEAD, None, controller)
    message = str(excinfo.value)
    assert "bare deletion" in message
    assert BOOK_WRITING in message and BOOK_OUTLINE in message


# ── the fix: every route resolves to a real, compatible member ────────────

def test_retirement_rewrites_all_three_routes_through_real_loader_and_validator(
    target: Path, controller
):
    config = rpi.load_config(target, controller)
    plan = rpi.plan_retirement(config, DEAD, STANDBY, controller)

    rewritten = {(entry["hint"], entry["field"]) for entry in plan["rewritten"]}
    assert rewritten == {
        (BOOK_WRITING, "require"),
        (BOOK_OUTLINE, "require"),
        (STORYBOARD, "prefer"),
    }

    written = _write(target.parent / "after.json", plan["candidate"])
    loaded = _load_through_real_loader(written, controller)

    assert DEAD not in rpi.local_instances(loaded)
    assert DEAD not in json.dumps(loaded), "retired name must be unreferenced"

    validator = controller.ModelValidator(loaded)
    for hint in (BOOK_WRITING, BOOK_OUTLINE, STORYBOARD):
        member = _route_target(loaded, hint)
        assert member == STANDBY, f"{hint} must resolve to the standby member"
        rule = next(r for r in loaded["routing_rules"] if r["hint"] == hint)
        verdict = validator.validate(
            member, task_hint=hint, requirements=rpi._requirements_for(rule)
        )
        assert verdict["ok"] is True and verdict["action"] == "route", verdict

    # never_fallback_to survives, and still blocks the tiny helper.
    book = next(r for r in loaded["routing_rules"] if r["hint"] == BOOK_WRITING)
    assert book["never_fallback_to"] == [TINY]
    blocked = validator.validate(TINY, task_hint=BOOK_WRITING, requirements=rpi._requirements_for(book))
    assert blocked["ok"] is False and blocked["action"] == "upgrade"

    # prefer consistency: the fallback field is untouched by the rewrite.
    story = next(r for r in loaded["routing_rules"] if r["hint"] == STORYBOARD)
    assert story["fallback"] == CLOUD


@pytest.mark.parametrize(
    ("replacement", "reason_fragment"),
    [
        (TINY, "rejected by validator"),
        ("no-such-member", "not a resolvable member"),
        (DEAD, "is the retired instance itself"),
    ],
)
def test_incompatible_or_unknown_replacements_are_refused(controller, replacement, reason_fragment):
    config = _config()
    with pytest.raises(rpi.RetirementError) as excinfo:
        rpi.plan_retirement(config, DEAD, replacement, controller)
    assert reason_fragment in str(excinfo.value)


def test_replacement_under_a_rules_capability_bound_is_refused(target: Path, controller):
    """A member the validator accepts but that is under a rule's bound must fail.

    ``ModelValidator`` has no ``max_output_tokens`` gate, so it accepts the
    replacement; the bound is the tool's own guard and it has to fire on its
    own.  The rule is rewritten to demand more tokens than the standby member
    can produce, which is exactly the state the validator cannot see.
    """
    config = rpi.load_config(target, controller)
    story = next(r for r in config["routing_rules"] if r["hint"] == STORYBOARD)
    story["max_output_tokens"] = 65536  # standby member caps at 4096

    # The real validator cannot see the shortfall: it routes the member anyway.
    validator = controller.ModelValidator(config)
    assert validator.validate(
        STANDBY, task_hint=STORYBOARD, requirements=rpi._requirements_for(story)
    )["ok"] is True

    with pytest.raises(rpi.RetirementError) as excinfo:
        rpi.plan_retirement(config, DEAD, STANDBY, controller)
    message = str(excinfo.value)
    assert "under bound" in message
    assert f"{STORYBOARD}:max_output_tokens=65536>{STANDBY}:4096" in message

    # Lifting the bound to what the member can produce makes it acceptable.
    story["max_output_tokens"] = 4096
    plan = rpi.plan_retirement(config, DEAD, STANDBY, controller)
    assert all(r["ok"] for r in plan["routes"] if r["member"] == STANDBY)


# ── idempotence and write discipline ───────────────────────────────────────

def _args(target: Path, replacement: str | None = STANDBY, apply: bool = False):
    return argparse.Namespace(
        config=str(target),
        instance=DEAD,
        replacement=replacement,
        apply=apply,
        expect_sha256=None,
    )


def test_emit_reports_the_real_idempotence_state(target: Path, capsys):
    """The receipt's ``idempotent:`` line must track the bytes, not a constant.

    Both directions are asserted because deriving the line from the inverse
    ``changed`` flag made it print "false" even once the file was already
    retired, which is the one state the brief's idempotence receipt turns on.
    """
    rpi._emit(rpi.run(_args(target, apply=False)), apply=False)
    print()
    assert "idempotent: False" in capsys.readouterr().out

    rpi.run(_args(target, apply=True))
    capsys.readouterr()
    rpi._emit(rpi.run(_args(target, apply=False)), apply=False)
    print()
    out = capsys.readouterr().out
    assert "idempotent: True" in out, out
    assert "verdict: already-retired: no write needed" in out


def test_second_transformation_is_byte_identical(target: Path, controller):
    first = rpi.run(_args(target, apply=True))
    after_first = target.read_bytes()
    first_backup = next(p for p in target.parent.iterdir() if ".bak-" in p.name)

    second = rpi.run(_args(target, apply=True))

    assert second["verdict"] == "already-retired: no write needed"
    assert second["changed"] is False
    assert second["sha256_current"] == first["sha256_would_be"]
    assert second["sha256_current"] == hashlib.sha256(after_first).hexdigest()
    assert target.read_bytes() == after_first, "second run must not change bytes"
    assert [p for p in target.parent.iterdir() if ".bak-" in p.name] == [first_backup], (
        "an already-retired rerun must not create a second backup"
    )
    loaded = _load_through_real_loader(target, controller)
    assert DEAD not in rpi.local_instances(loaded)


def test_default_invocation_writes_nothing(target: Path):
    before = target.read_bytes()
    report = rpi.run(_args(target, apply=False))

    assert report["verdict"] == "dry-run: no write performed"
    assert report["changed"] is True
    assert len(report["rewritten"]) == 3
    assert target.read_bytes() == before, "dry run must not touch the target"
    assert [p.name for p in target.parent.iterdir()] == ["providers.json"], (
        "dry run must not leave a backup or a temporary file"
    )


def test_apply_creates_an_exclusive_0600_backup_and_replaces_atomically(target: Path):
    before = target.read_bytes()
    report = rpi.run(_args(target, apply=True))

    backups = [p for p in target.parent.iterdir() if ".bak-" in p.name]
    assert len(backups) == 1
    backup = backups[0]
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600, "backup must be owner-only"
    assert backup.read_bytes() == before, "backup must hold the pre-change bytes"

    assert report["verdict"] == "applied"
    assert report["sha256_current"] == hashlib.sha256(before).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == report["sha256_would_be"]
    assert sorted(p.name for p in target.parent.iterdir()) == [
        "providers.json",
        backup.name,
    ], "no temporary file may survive a successful apply"


def test_backup_is_never_overwritten(target: Path):
    payload = target.read_bytes()
    first = rpi.write_backup(target, payload, when=1_700_000_000)
    with pytest.raises(rpi.ApplyError, match="already exists"):
        rpi.write_backup(target, payload, when=1_700_000_000)
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert first.read_bytes() == payload


def test_failed_atomic_replace_leaves_the_target_intact_and_cleans_up(
    target: Path, monkeypatch
):
    before = target.read_bytes()

    def boom(_src, _dst):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(rpi.os, "replace", boom)
    with pytest.raises(OSError, match="synthetic replace failure"):
        rpi.apply_atomically(target, b"{}\n")

    assert target.read_bytes() == before
    leftovers = [p.name for p in target.parent.iterdir() if "retire-tmp" in p.name]
    assert leftovers == [], f"temporary files must be cleaned up, found {leftovers}"


def test_apply_refuses_when_source_bytes_changed_since_inspection(target: Path):
    before = target.read_bytes()
    args = _args(target, apply=True)
    args.expect_sha256 = "0" * 64

    with pytest.raises(rpi.RetirementError, match="source bytes changed"):
        rpi.run(args)

    assert target.read_bytes() == before
    assert [p.name for p in target.parent.iterdir()] == ["providers.json"], (
        "a refused apply must not leave a backup"
    )


# ── surface contract ───────────────────────────────────────────────────────

def test_help_documents_dry_run_by_default_and_explicit_apply():
    # argparse wraps help text across lines, so compare on whitespace collapsed.
    help_text = " ".join(rpi.build_parser().format_help().split())

    assert "--apply" in help_text
    assert "Without this flag the run is a dry run and writes nothing." in help_text
    assert "0600 timestamped backup then atomic replace" in help_text
    assert "--expect-sha256" in help_text
    assert "--replacement" in help_text
    # The parser must default to a dry run: --apply is opt-in.
    args = rpi.build_parser().parse_args(
        ["--config", "c.json", "--instance", "i", "--replacement", "r"]
    )
    assert args.apply is False


def test_target_must_be_a_real_file_not_a_symlink(target: Path, tmp_path: Path):
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(rpi.RetirementError, match="symlink"):
        rpi.read_target(link)
    with pytest.raises(rpi.RetirementError, match="does not exist"):
        rpi.read_target(tmp_path / "absent.json")


def test_controller_module_is_the_real_one():
    assert CONTROLLER.is_file()
    module = rpi.load_controller()
    assert hasattr(module, "ModelValidator") and hasattr(module, "_load_providers_config")
    # The validator's rule semantics come from the controller, not this module.
    assert module.__file__ == str(CONTROLLER)