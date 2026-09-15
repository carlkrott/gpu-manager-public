"""Compatibility exports for the generic bundle lifecycle reducer.

Recovery decisions are owned by ``bundle_lifecycle`` at the GPU Manager
bundle layer. This module remains for existing Gemma broker imports and tests;
it contains no Gemma-specific recovery behavior.
"""
from bundle_lifecycle import (
    DesiredState,
    RecoveryAction,
    RecoveryDecision,
    RecoveryObservation,
    RecoveryPhase,
    RecoveryState,
    reduce_recovery,
)

__all__ = [
    "DesiredState",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryObservation",
    "RecoveryPhase",
    "RecoveryState",
    "reduce_recovery",
]
