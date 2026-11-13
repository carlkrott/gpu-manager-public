"""Facade-bound memory-cap and CPU/parallel policy.

Extracted verbatim from ``scripts/gpu-manager.py``.  The two classes are
leaf definitions (they read only ``Enum`` and builtins) and are shared by every
facade.  The functions read facade globals (``_TASK8A_MIB``,
``_TASK8A_UINT64_MAX`` and each other), so ``bind(facade_globals)`` rebuilds them
with ``types.FunctionType`` against the supplied facade dictionary; no
``sys.modules`` entry or shared facade pointer is kept here.
"""
from __future__ import annotations

from enum import Enum
from types import FunctionType


class _MemoryCapRejectReason(Enum):
    MAX_NOT_FINITE = "max_not_finite"
    HIGH_GEQ_MAX = "high_geq_max"
    HIGH_EXCEEDS_P99 = "high_exceeds_p99_x_1_10"
    SWAP_MAX_NOT_FINITE = "swap_max_not_finite"
    SWAP_EXCEEDS_P99 = "swap_exceeds_p99_x_1_25"
    CONFLICTING_ALIASES = "conflicting_aliases"


class _MemoryCapProvenance(dict):
    """Mapping result with attribute access for the audit's ``.valid`` form."""

    def __init__(
        self,
        valid: bool,
        reason: str,
        headroom_mib: int,
        swap_envelope_mib: int,
    ) -> None:
        super().__init__(
            valid=valid,
            reason=reason,
            headroom_mib=headroom_mib,
            swap_envelope_mib=swap_envelope_mib,
        )

    @property
    def valid(self) -> bool:
        return self["valid"]

    @property
    def reason(self) -> str:
        return self["reason"]

    @property
    def headroom_mib(self) -> int:
        return self["headroom_mib"]

    @property
    def swap_envelope_mib(self) -> int:
        return self["swap_envelope_mib"]

    def as_dict(self) -> dict:
        return dict(self)


def _finite_positive_cap(value) -> int | None:
    """Normalize one cgroup byte cap, rejecting zero/infinity/sentinels."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    integer = int(value)
    if integer <= 0 or integer >= _TASK8A_UINT64_MAX:
        return None
    return integer


def _positive_measurement(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number <= 0 or number != number or number == float("inf"):
        return None
    return number


def _memory_cap_provenance(
    memory_max_bytes=None,
    memory_high_bytes=None,
    memory_swap_max_bytes=None,
    working_set_p99_mib=None,
    working_set_swap_p99_mib=None,
    **aliases,
) -> _MemoryCapProvenance:
    """Validate finite memory caps against measured p99 envelopes.

    Both the Pythonic parameter names and the names used by ``systemctl show``
    in the Task 8A audit (``MemoryMax``, ``MemoryHigh``, ``MemorySwapMax``,
    ``p99``, ``p99_swap``) are accepted.  Unknown/missing evidence rejects.
    Supplying both spellings is permitted only when the values are identical;
    conflicting spellings are rejected rather than silently giving the alias
    precedence.
    """
    alias_pairs = {
        "MemoryMax": ("memory_max_bytes", memory_max_bytes),
        "MemoryHigh": ("memory_high_bytes", memory_high_bytes),
        "MemorySwapMax": ("memory_swap_max_bytes", memory_swap_max_bytes),
        "p99": ("working_set_p99_mib", working_set_p99_mib),
        "p99_swap": ("working_set_swap_p99_mib", working_set_swap_p99_mib),
    }
    for alias, (python_name, python_value) in alias_pairs.items():
        if alias not in aliases:
            continue
        alias_value = aliases.pop(alias)
        if python_value is not None and alias_value != python_value:
            return _MemoryCapProvenance(
                False, _MemoryCapRejectReason.CONFLICTING_ALIASES.value, 0, 0
            )
        locals_value = alias_value if python_value is None else python_value
        if python_name == "memory_max_bytes":
            memory_max_bytes = locals_value
        elif python_name == "memory_high_bytes":
            memory_high_bytes = locals_value
        elif python_name == "memory_swap_max_bytes":
            memory_swap_max_bytes = locals_value
        elif python_name == "working_set_p99_mib":
            working_set_p99_mib = locals_value
        else:
            working_set_swap_p99_mib = locals_value
    if aliases:
        return _MemoryCapProvenance(
            False, _MemoryCapRejectReason.CONFLICTING_ALIASES.value, 0, 0
        )

    maximum = _finite_positive_cap(memory_max_bytes)
    high = _finite_positive_cap(memory_high_bytes)
    if maximum is None or high is None:
        return _MemoryCapProvenance(
            False, _MemoryCapRejectReason.MAX_NOT_FINITE.value, 0, 0
        )
    headroom_mib = max(0, (maximum - high) // _TASK8A_MIB)
    if high >= maximum:
        return _MemoryCapProvenance(
            False, _MemoryCapRejectReason.HIGH_GEQ_MAX.value, headroom_mib, 0
        )

    p99 = _positive_measurement(working_set_p99_mib)
    if p99 is None or high > p99 * 1.10 * _TASK8A_MIB:
        return _MemoryCapProvenance(
            False, _MemoryCapRejectReason.HIGH_EXCEEDS_P99.value,
            headroom_mib, 0,
        )

    swap_max = _finite_positive_cap(memory_swap_max_bytes)
    if swap_max is None:
        return _MemoryCapProvenance(
            False, _MemoryCapRejectReason.SWAP_MAX_NOT_FINITE.value,
            headroom_mib, 0,
        )
    swap_p99 = _positive_measurement(working_set_swap_p99_mib)
    swap_envelope_mib = int(swap_p99 * 1.25) if swap_p99 is not None else 0
    if swap_p99 is None or swap_max > swap_p99 * 1.25 * _TASK8A_MIB:
        return _MemoryCapProvenance(
            False, _MemoryCapRejectReason.SWAP_EXCEEDS_P99.value,
            headroom_mib, swap_envelope_mib,
        )
    return _MemoryCapProvenance(True, "", headroom_mib, swap_envelope_mib)


def _normalized_unit_name(unit) -> str:
    if not isinstance(unit, str):
        return ""
    normalized = unit.strip().lower()
    return normalized[:-8] if normalized.endswith(".service") else normalized


def _is_cpu_llm_unit(unit: str) -> bool:
    return _normalized_unit_name(unit) in {"gemma-cpu", "gemma-cpu-vision"}


def _validate_production_parallel_invariant(services_config: dict) -> None:
    """Reject multi-worker generation backends in explicit strict mode.

    The worker supervisor accepts the newer ``worker_parallel`` and
    ``queue_parallel`` fields while retaining the legacy ``parallel`` field.
    Validate the same precedence before a startup or hot-reload candidate is
    promoted, but keep the guard opt-in so existing qualification configs are
    not changed implicitly.
    """
    if not isinstance(services_config, dict):
        raise ValueError("services config must be a JSON object")
    if services_config.get("strict_production_parallel") is not True:
        return
    services = services_config.get("services", {})
    if not isinstance(services, dict):
        raise ValueError("services config 'services' must be an object")
    for service_name, service_config in services.items():
        if not isinstance(service_config, dict):
            continue
        if service_config.get("type") != "generation_backend":
            continue
        effective = 1
        selected_field = "default"
        for field in ("worker_parallel", "queue_parallel", "parallel"):
            if field not in service_config or service_config[field] is None:
                continue
            value = service_config[field]
            if isinstance(value, bool):
                raise ValueError(
                    f"production parallel invariant: {service_name} has "
                    f"invalid boolean {field}={value!r}"
                )
            try:
                effective = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"production parallel invariant: {service_name} has "
                    f"invalid {field}={value!r}"
                ) from exc
            selected_field = field
            break
        if effective > 1:
            raise ValueError(
                "production parallel invariant: "
                f"{service_name} effective {selected_field}={effective} "
                "exceeds strict maximum 1"
            )


_FUNCTION_NAMES = (
    "_finite_positive_cap",
    "_positive_measurement",
    "_memory_cap_provenance",
    "_normalized_unit_name",
    "_is_cpu_llm_unit",
    "_validate_production_parallel_invariant",
)
NAMES = (
    "_MemoryCapRejectReason",
    "_MemoryCapProvenance",
    "_finite_positive_cap",
    "_positive_measurement",
    "_memory_cap_provenance",
    "_normalized_unit_name",
    "_is_cpu_llm_unit",
    "_validate_production_parallel_invariant",
)


def _rebuild(function, facade_globals):
    rebuilt = FunctionType(
        function.__code__,
        facade_globals,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebuilt.__kwdefaults__ = (
        function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    )
    rebuilt.__annotations__ = function.__annotations__.copy()
    rebuilt.__doc__ = function.__doc__
    rebuilt.__qualname__ = function.__qualname__
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt


def bind(facade_globals):
    """Return the policy objects in ``NAMES`` order, bound to a facade.

    Classes are returned as-is; functions are rebuilt so their ``__globals__``
    is exactly ``facade_globals`` and every global read resolves against that
    facade at call time.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    namespace = globals()
    return tuple(
        _rebuild(namespace[name], facade_globals)
        if name in _FUNCTION_NAMES else namespace[name]
        for name in NAMES
    )
