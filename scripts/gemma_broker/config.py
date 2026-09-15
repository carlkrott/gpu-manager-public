"""Closed configuration contract for the combined Gemma broker."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


class ConfigError(ValueError):
    pass


_EXPECTED_PRIORITIES = {"interactive": 0, "normal": 10, "background": 20}


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    enabled: bool
    ordered_members: tuple[str, ...]
    priority_values: dict[str, int]
    aging_interval_seconds: float
    redis_namespace: str
    leader_ttl_seconds: float
    heartbeat_seconds: float
    freshness_seconds: float
    context_safety_margin: int
    api_wait_timeout: float
    eta_recent_window_seconds: float
    eta_sparse_window_seconds: float
    eta_min_samples: int
    lifecycle_mutation_enabled: bool
    sse_enabled: bool
    candidate_mode: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, candidate: bool) -> "BrokerConfig":
        if not isinstance(data, dict):
            raise ConfigError("CONFIG_OBJECT_REQUIRED")
        expected = {field.name for field in fields(cls)}
        if set(data) != expected:
            raise ConfigError(
                f"CONFIG_FIELDS_INVALID:missing={sorted(expected-set(data))}:unknown={sorted(set(data)-expected)}"
            )
        # The validated config is the single source of truth for mode. The
        # ``candidate`` kwarg must agree with ``data["candidate_mode"]`` so
        # call sites can never silently disagree (the prior bug was a literal
        # ``candidate=True`` hard-coded at the call site, defeating the
        # schema). Inference from the namespace is intentionally avoided.
        if type(data["candidate_mode"]) is not bool:
            raise ConfigError("BOOLEAN_REQUIRED:candidate_mode")
        if data["candidate_mode"] != candidate:
            raise ConfigError(
                f"CANDIDATE_MODE_MISMATCH:config={data['candidate_mode']}:caller={candidate}"
            )
        mode = data["candidate_mode"]
        for name in ("enabled", "lifecycle_mutation_enabled", "sse_enabled"):
            if type(data[name]) is not bool:
                raise ConfigError(f"BOOLEAN_REQUIRED:{name}")
        members = data["ordered_members"]
        if not isinstance(members, list) or any(not isinstance(item, str) for item in members):
            raise ConfigError("MEMBERS_LIST_REQUIRED")
        if len(members) != len(set(members)):
            raise ConfigError("DUPLICATE_MEMBERS")
        if not members:
            raise ConfigError("MEMBERS_NONEMPTY_REQUIRED")
        if data["priority_values"] != _EXPECTED_PRIORITIES:
            raise ConfigError("PRIORITY_VALUES_INVALID")
        positive = (
            "aging_interval_seconds",
            "leader_ttl_seconds",
            "heartbeat_seconds",
            "freshness_seconds",
            "api_wait_timeout",
            "eta_recent_window_seconds",
            "eta_sparse_window_seconds",
            "eta_min_samples",
        )
        for name in positive:
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(f"POSITIVE_NUMBER_REQUIRED:{name}")
        margin = data["context_safety_margin"]
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
            raise ConfigError("NONNEGATIVE_INTEGER_REQUIRED:context_safety_margin")
        namespace = data["redis_namespace"]
        if not isinstance(namespace, str) or not namespace:
            raise ConfigError("REDIS_NAMESPACE_REQUIRED")
        if mode and not namespace.startswith("qual:combined-gemma:"):
            raise ConfigError("CANDIDATE_REDIS_NAMESPACE_REQUIRED")
        if not mode and not namespace.startswith("prod:combined-gemma:"):
            raise ConfigError("PRODUCTION_REDIS_NAMESPACE_REQUIRED")
        if data["sse_enabled"] is not False:
            raise ConfigError("SSE_DISABLED_REQUIRED")
        if mode and data["lifecycle_mutation_enabled"] is not False:
            raise ConfigError("CANDIDATE_LIFECYCLE_MUTATION_DISABLED_REQUIRED")
        return cls(
            enabled=data["enabled"],
            ordered_members=tuple(members),
            priority_values=dict(data["priority_values"]),
            aging_interval_seconds=float(data["aging_interval_seconds"]),
            redis_namespace=namespace,
            leader_ttl_seconds=float(data["leader_ttl_seconds"]),
            heartbeat_seconds=float(data["heartbeat_seconds"]),
            freshness_seconds=float(data["freshness_seconds"]),
            context_safety_margin=margin,
            api_wait_timeout=float(data["api_wait_timeout"]),
            eta_recent_window_seconds=float(data["eta_recent_window_seconds"]),
            eta_sparse_window_seconds=float(data["eta_sparse_window_seconds"]),
            eta_min_samples=int(data["eta_min_samples"]),
            lifecycle_mutation_enabled=data["lifecycle_mutation_enabled"],
            sse_enabled=data["sse_enabled"],
            candidate_mode=mode,
        )
