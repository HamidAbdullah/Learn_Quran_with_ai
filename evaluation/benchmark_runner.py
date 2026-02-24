"""
Benchmark runner: run evaluation on a dataset and output structured reports.
"""
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Callable

from evaluation.asr_metrics import compute_wer_cer_distribution, EvaluationReport

logger = logging.getLogger(__name__)


def _get_reference(item: Dict[str, Any]) -> str:
    return (
        (item.get("reference") or item.get("text_uthmani") or item.get("text") or item.get("normalized_text") or "")
    ).strip()


def _get_hypothesis(item: Dict[str, Any]) -> str:
    return (item.get("hypothesis") or item.get("transcript") or "").strip()


def run_benchmark(
    dataset_path: str,
    audio_base_dir: Optional[str] = None,
    run_asr_fn: Optional[Callable[[str, str], str]] = None,
    limit: Optional[int] = None,
    reference_key: str = "reference",
    hypothesis_key: str = "hypothesis",
    sample_id_key: str = "verse_key",
) -> EvaluationReport:
    """
    Load dataset JSON, optionally run ASR per item, compute WER/CER distribution.
    run_asr_fn: (audio_path, reference) -> hypothesis. If None, hypotheses must be in dataset.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of items")
    items = data[:limit] if limit else data
    base_dir = audio_base_dir or os.environ.get("AUDIO_BASE_DIR", "")

    samples = []
    for i, item in enumerate(items):
        ref = _get_reference(item)
        if not ref:
            continue
        hyp = _get_hypothesis(item)
        if not hyp and run_asr_fn:
            audio_path = item.get("audio") or item.get("audio_file") or ""
            if audio_path and not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            if audio_path and os.path.isfile(audio_path):
                try:
                    hyp = run_asr_fn(audio_path, ref)
                except Exception as e:
                    logger.warning("ASR failed for %s: %s", item.get(sample_id_key, i), e)
        if ref:
            sid = item.get(sample_id_key) or str(i)
            samples.append({
                reference_key: ref,
                hypothesis_key: hyp or "",
                sample_id_key: sid,
            })

    return compute_wer_cer_distribution(
        samples,
        reference_key=reference_key,
        hypothesis_key=hypothesis_key,
        sample_id_key=sample_id_key,
    )


def write_report(report: EvaluationReport, output_path: str) -> None:
    """Write evaluation report to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Wrote evaluation report to %s", output_path)
