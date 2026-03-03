"""
Dataset validation: ensure audio_path and reference_text exist; check missing/corrupt/empty.
Output structured validation report.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional: try loading audio to detect corruption (can be slow)
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class ValidationReport:
    """Result of dataset validation."""
    n_total: int = 0
    n_valid: int = 0
    missing_audio_path: List[Dict[str, Any]] = field(default_factory=list)
    missing_reference: List[Dict[str, Any]] = field(default_factory=list)
    missing_files: List[Dict[str, Any]] = field(default_factory=list)
    corrupted_audio: List[Dict[str, Any]] = field(default_factory=list)
    empty_transcript: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "missing_audio_path": self.missing_audio_path,
            "missing_reference": self.missing_reference,
            "missing_files": self.missing_files,
            "corrupted_audio": self.corrupted_audio,
            "empty_transcript": self.empty_transcript,
            "errors": self.errors,
        }

    @property
    def is_valid(self) -> bool:
        return self.n_valid == self.n_total and not self.errors


def _resolve_audio_path(item: Dict[str, Any], base_dir: str) -> Optional[str]:
    raw = item.get("audio") or item.get("audio_file") or item.get("audio_path") or ""
    if not raw:
        return None
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    if base_dir:
        return os.path.join(base_dir, raw)
    return raw


def _get_reference(item: Dict[str, Any]) -> str:
    return (
        item.get("reference_text")
        or item.get("reference")
        or item.get("text_uthmani")
        or item.get("text")
        or item.get("normalized_text")
        or ""
    ).strip()


def validate_dataset(
    items: List[Dict[str, Any]],
    audio_base_dir: Optional[str] = None,
    check_audio_load: bool = False,
    sample_id_key: str = "verse_key",
) -> ValidationReport:
    """
    Validate each item has audio_path and reference_text; optionally check files exist and load.
    check_audio_load: if True, try librosa.load to detect corrupted audio (requires librosa).
    """
    report = ValidationReport()
    base_dir = audio_base_dir or os.environ.get("AUDIO_BASE_DIR", "")
    report.n_total = len(items)

    for i, item in enumerate(items):
        sid = item.get(sample_id_key) or str(i)
        entry = {"index": i, "sample_id": sid}

        # Required: reference text
        ref = _get_reference(item)
        if not ref:
            report.missing_reference.append(entry)
            continue

        # Required: audio path (if dataset is audio-based)
        audio_path = _resolve_audio_path(item, base_dir)
        if not audio_path:
            # Allow items that only have reference + hypothesis (e.g. for metrics-only benchmark)
            report.n_valid += 1
            continue

        if not os.path.isfile(audio_path):
            report.missing_files.append({**entry, "path": audio_path})
            continue

        if check_audio_load and HAS_LIBROSA:
            try:
                y, _ = librosa.load(audio_path, sr=16000, duration=1.0)
                if len(y) == 0:
                    report.corrupted_audio.append({**entry, "reason": "empty_audio"})
                    continue
            except Exception as e:
                report.corrupted_audio.append({**entry, "reason": str(e)})
                continue

        # Optional: if hypothesis/transcript present, check not empty when ref is non-empty
        hyp = (item.get("hypothesis") or item.get("transcript") or "").strip()
        if ref and hyp == "" and (item.get("hypothesis") is not None or item.get("transcript") is not None):
            report.empty_transcript.append(entry)

        report.n_valid += 1

    return report
