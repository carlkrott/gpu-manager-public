"""Strict September 9 preparation contract for MiniMax Music 3.

The recovered production runner predates the final six-lane prompt contract.
This module overlays only its preparation functions inside the already-isolated
adapter child. Generation, model placement, and QC remain owned by the pinned
production sources.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


CONTRACT_ID = "minimax-music3-preparation-2026-09-09"
REQUIRED_RESEARCH_LANES = frozenset({"key", "instrument", "mixing", "mastering"})
_CAPTION_SECTIONS = (
    "Global Metadata",
    "Instrument Direction",
    "Level Plan",
    "Mix",
    "Mastering",
    "Vocal Details",
    "Arrangement",
)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping_has_strings(value: object, keys: tuple[str, ...]) -> bool:
    return isinstance(value, Mapping) and all(_nonempty(value.get(key)) for key in keys)


def classify_research_lane(value: Mapping[str, Any]) -> str | None:
    """Classify the component head first, then fall back to query attributes."""
    component = str(value.get("component") or "").strip().lower()
    fallback = " ".join(
        (
            component,
            str(value.get("query") or ""),
            " ".join(str(item) for item in (value.get("desired_attributes") or [])),
        )
    ).lower()
    # Instrument precedes harmony: "voicing" is valid instrument direction too.
    if re.search(r"\b(instrument|guitar|bass|drum|piano|synth|string|brass|woodwind)\b", component):
        return "instrument"
    if re.search(r"\b(key|harmony|harmonic|chord|tonal)\b", component):
        return "key"
    if re.search(r"\b(mix|mixing|panning|stereo|masking|depth|compression)\b", component):
        return "mixing"
    if re.search(r"\b(master|mastering|lufs|true.?peak|limiting)\b", component):
        return "mastering"
    if re.search(r"\b(instrument|guitar|bass|drum|piano|synth|register|performance)\b", fallback):
        return "instrument"
    if re.search(r"\b(key|harmony|harmonic|chord|tonal|progression)\b", fallback):
        return "key"
    if re.search(r"\b(mix|mixing|panning|stereo|masking|depth|compression)\b", fallback):
        return "mixing"
    if re.search(r"\b(master|mastering|lufs|true.?peak|limiting)\b", fallback):
        return "mastering"
    return None


def validate_expanded_brief(value: Mapping[str, Any]) -> bool:
    required_strings = ("creative_intent", "recording_style")
    if not all(_nonempty(value.get(key)) for key in required_strings):
        return False
    if not _mapping_has_strings(
        value.get("tempo_key"),
        ("key", "time_signature", "harmonic_plan", "key_change_policy"),
    ):
        return False
    bpm = value["tempo_key"].get("bpm")
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or bpm <= 0:
        return False
    instrument_direction = value.get("instrument_direction")
    if not isinstance(instrument_direction, list) or not instrument_direction:
        return False
    direction_fields = (
        "instrument", "role", "register", "performance", "relative_level",
        "pan", "depth",
    )
    if not all(
        _mapping_has_strings(item, direction_fields)
        and isinstance(item.get("sections"), list)
        and bool(item["sections"])
        for item in instrument_direction
        if isinstance(item, Mapping)
    ) or not all(isinstance(item, Mapping) for item in instrument_direction):
        return False
    levels = value.get("levels")
    if not _mapping_has_strings(levels, ("reference", "automation", "headroom")):
        return False
    relative_db = levels.get("relative_db")
    if not isinstance(relative_db, list) or not relative_db or not all(
        _mapping_has_strings(item, ("element", "level"))
        for item in relative_db
    ):
        return False
    if not _mapping_has_strings(
        value.get("mix"),
        ("stereo_field", "panning", "depth", "compression", "masking_control", "vocal_priority"),
    ):
        return False
    if not _mapping_has_strings(
        value.get("mastering"),
        ("target_lufs", "true_peak_dbtp", "dynamic_range", "limiting", "tonal_balance", "transient_preservation"),
    ):
        return False
    for key in (
        "vocals", "structure", "instrumentation", "arrangement", "effects",
        "component_reference_targets", "explicit_constraints", "inferred_choices",
    ):
        item = value.get(key)
        if not isinstance(item, (Mapping, list)) or not item:
            return False
    return True


def validate_query_plan(value: Mapping[str, Any]) -> bool:
    queries = value.get("queries")
    if not isinstance(queries, list) or not 4 <= len(queries) <= 6:
        return False
    lanes: set[str] = set()
    for item in queries:
        if not isinstance(item, Mapping):
            return False
        words = re.findall(r"[a-z0-9]+", str(item.get("query") or "").lower())
        attributes = item.get("desired_attributes")
        if (
            not _nonempty(item.get("component"))
            or not 4 <= len(words) <= 12
            or not isinstance(attributes, list)
            or not attributes
            or not all(_nonempty(attribute) for attribute in attributes)
            or item.get("audio_use") != "descriptive"
        ):
            return False
        lane = classify_research_lane(item)
        if lane:
            lanes.add(lane)
    return REQUIRED_RESEARCH_LANES <= lanes


def research_lane_summary(manifest: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    records = manifest.get("records") or []
    resolved = {
        lane
        for item in records
        if isinstance(item, Mapping)
        for lane in (classify_research_lane(item),)
        if lane
    }
    missing = REQUIRED_RESEARCH_LANES - resolved
    return sorted(resolved), sorted(missing)


def validate_final_brief(
    value: Mapping[str, Any],
    *,
    supplied_lyrics: str | None,
    duration: float | None,
    lyrics_fit,
) -> bool:
    # The recovered transport appends this internal provenance field after
    # parsing every model response; it is not an extra model-authored field.
    if set(value) - {"caption", "lyrics", "_stage"}:
        return False
    caption = value.get("caption")
    lyrics = value.get("lyrics")
    if not _nonempty(caption) or not isinstance(lyrics, str):
        return False
    if supplied_lyrics is not None and lyrics != supplied_lyrics:
        return False
    if supplied_lyrics is None and not lyrics_fit(lyrics, duration):
        return False
    words = caption.split()
    # The recovered formatter may expand a complete seven-section brief to
    # just over 300 words while still preserving every required conditioning
    # target. Keep a bounded upper limit, but do not reject that valid form.
    if not 150 <= len(words) <= 360:
        return False
    lower = caption.lower()
    positions = [lower.find(section.lower()) for section in _CAPTION_SECTIONS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        return False
    anchors = (
        r"\b\d{2,3}\s*(?:bpm|beats per minute)\b|\bbpm\s*[:=-]?\s*\d{2,3}\b",
        r"\bkey\s*[:=-]",
        r"[-+]?\d+(?:\.\d+)?\s*dB\b",
        # Accept both common target notations emitted by the text model:
        # ``-14 LUFS`` and ``Target LUFS: -14``.  They carry the same
        # conditioning target; rejecting one form made valid preparation
        # nondeterministically fail at the final-brief gate.
        r"(?:[-+]?\d+(?:\.\d+)?\s*LUFS\b|\bLUFS\s*[:=-]\s*[-+]?\d+(?:\.\d+)?)",
        r"[-+]?\d+(?:\.\d+)?\s*dBTP\b",
    )
    return all(re.search(pattern, caption, re.IGNORECASE) for pattern in anchors)


_EXPANSION_SYSTEM = """You are a music producer, recording, mix, and mastering engineer.
Expand the user's MiniMax Music 3 request into one complete JSON object. Preserve every explicit
constraint separately from inferred choices. Required keys are creative_intent, recording_style,
tempo_key, vocals, structure, instrumentation, instrument_direction, levels, arrangement, effects,
mix, mastering, component_reference_targets, explicit_constraints, and inferred_choices.
tempo_key requires bpm, key, time_signature, harmonic_plan, key_change_policy. Every
instrument_direction item requires instrument, role, register, performance, relative_level, pan,
depth, sections. levels requires reference, relative_db[{element,level}], automation, headroom.
mix requires stereo_field, panning, depth, compression, masking_control, vocal_priority.
mastering requires target_lufs, true_peak_dbtp, dynamic_range, limiting, tonal_balance,
transient_preservation. Values must be specific and non-empty.
Use these exact JSON types: creative_intent and recording_style are strings;
tempo_key.bpm is a number and its other fields are strings. instrument_direction
is an array of objects with string fields except sections, which is a non-empty
array of strings. All levels fields are strings except relative_db, which is an
array of objects containing string element and string level (include dB units).
Every mix and mastering field is a string, including numeric targets with units
such as "-14 LUFS" and "-1 dBTP". vocals is an object. structure, instrumentation,
arrangement, effects, component_reference_targets, explicit_constraints, and
inferred_choices are non-empty arrays. Return only the JSON object."""

_PLAN_SYSTEM = """Return one JSON object with a top-level "queries" array containing
4-6 distinct query objects. Do not use research_plan or research_queries as the key.
Each query object has exactly component (string), query (string),
desired_attributes (array of non-empty strings), and audio_use (the string
"descriptive"). Reserve one query
each for key/harmony, instrument direction, mixing, and mastering; use up to two adaptive queries
for the brief's vocals, recording, arrangement, effects, or genre vocabulary. Each item must have
component, a practical 4-12 word query, non-empty desired_attributes, and audio_use='descriptive'.
Research is text-only. Do not request, download, or claim analysis of reference audio."""

_FINAL_SYSTEM = """Synthesize the original request, strict expanded brief, query plan, and resolved
source snippets into exactly {caption, lyrics}. Caption must be 150-360 words with these labeled
sections in this order: Global Metadata, Instrument Direction, Level Plan, Mix, Mastering, Vocal
Details, Arrangement. Include explicit BPM, key, per-element relative dB values, pre-master
headroom, LUFS, dBTP, dynamic-range, limiting, tonal-balance, and transient-preservation targets.
These are conditioning targets, not claims about measured output. Preserve supplied lyrics exactly;
otherwise emit concise section-tagged lyrics that fit the requested duration, or [Instrumental]."""


def install_contract(minimax_module: Any) -> dict[str, Any]:
    """Install the strict preparation overlay into one isolated source module."""
    workflow = minimax_module.workflow
    original_expand = workflow._expand_brief
    original_plan = workflow._plan_references
    original_resolve = minimax_module._resolve_text_references

    def expand(prompt: str, duration: float | None, *, target: str = "ace"):
        if target != "minimax":
            return original_expand(prompt, duration, target=target)
        return workflow._structured_stage(
            "brief_expand",
            _EXPANSION_SYSTEM,
            {"prompt": prompt, "requested_duration": duration, "target": target},
            validate_expanded_brief,
            "MINIMAX_SIX_LANE_EXPANSION_INVALID",
            max_tokens=4096,
        )

    def plan(prompt: str, brief: dict[str, Any], *, target: str = "ace"):
        if target != "minimax":
            return original_plan(prompt, brief, target=target)
        return workflow._structured_stage(
            "reference_plan",
            _PLAN_SYSTEM,
            {"original_prompt": prompt, "expanded_brief": brief, "target": target},
            validate_query_plan,
            "MINIMAX_REQUIRED_RESEARCH_LANES_MISSING",
            max_tokens=3072,
        )

    def resolve(plan_value, run_dir, *, enable_web=True):
        result = dict(original_resolve(plan_value, run_dir, enable_web=enable_web))
        resolved, missing = research_lane_summary(result)
        result["required_lanes"] = sorted(REQUIRED_RESEARCH_LANES)
        result["resolved_lanes"] = resolved
        result["missing_lanes"] = missing
        if enable_web and missing:
            result["research_status"] = "insufficient"
            result.setdefault("errors", []).append(
                "required research lanes have no accepted evidence: " + ", ".join(missing)
            )
        return result

    def finalize(
        original_prompt,
        expanded,
        query_plan,
        snippet_manifest,
        duration,
        supplied_lyrics=None,
    ):
        status = snippet_manifest.get("research_status")
        if status not in {"success", "disabled"}:
            raise workflow.WorkflowError(
                "MINIMAX_RESEARCH_INCOMPLETE:" + str(status)
            )
        payload = {
            "original_prompt": original_prompt,
            "expanded_brief": expanded,
            "reference_plan": query_plan,
            "snippet_manifest": snippet_manifest,
            "requested_duration": duration,
            "supplied_lyrics": supplied_lyrics,
        }

        def validator(value):
            return validate_final_brief(
                value,
                supplied_lyrics=supplied_lyrics,
                duration=duration,
                lyrics_fit=minimax_module._auto_lyrics_fit_duration,
            )

        return workflow._structured_stage(
            "final_brief",
            _FINAL_SYSTEM,
            payload,
            validator,
            "MINIMAX_STRICT_FINAL_BRIEF_INVALID",
            max_tokens=4096,
        )

    workflow._expand_brief = expand
    workflow._plan_references = plan
    minimax_module._resolve_text_references = resolve
    minimax_module._finalize_minimax_brief = finalize
    return {
        "contract": CONTRACT_ID,
        "required_research_lanes": sorted(REQUIRED_RESEARCH_LANES),
    }


__all__ = [
    "CONTRACT_ID",
    "REQUIRED_RESEARCH_LANES",
    "classify_research_lane",
    "install_contract",
    "research_lane_summary",
    "validate_expanded_brief",
    "validate_final_brief",
    "validate_query_plan",
]
