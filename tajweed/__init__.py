"""
Tajweed rule detection for Phase 3 — Phoneme Alignment Engine.
Structured errors: rule_name, word, expected, detected, severity.
"""
from tajweed.rules import (
    detect_tajweed_errors,
    TajweedError,
)

__all__ = [
    "detect_tajweed_errors",
    "TajweedError",
]
