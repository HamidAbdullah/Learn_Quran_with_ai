"""
ASR metrics with distribution tracking: mean, median, percentile, worst cases.
Structured evaluation reports and failure/high-error logging.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.metrics import wer, cer

logger = logging.getLogger(__name__)

# Thresholds for logging
WER_HIGH_ERROR_THRESHOLD = 0.5  # log verses with WER >= 50%
CER_HIGH_ERROR_THRESHOLD = 0.4
WER_FAILURE_THRESHOLD = 1.0    # treat as failure (e.g. empty or completely wrong)


@dataclass
class SampleResult:
    """Single sample: reference, hypothesis, WER, CER, and optional id."""
    ref: str
    hyp: str
    wer: float
    cer: float
    sample_id: Optional[str] = None  # verse_key or index


@dataclass
class EvaluationReport:
    """Structured report: distribution stats, worst cases, failure/high-error lists."""
    n_samples: int = 0
    wer_mean: float = 0.0
    wer_median: float = 0.0
    wer_p95: float = 0.0
    cer_mean: float = 0.0
    cer_median: float = 0.0
    cer_p95: float = 0.0
    worst_wer: List[Dict[str, Any]] = field(default_factory=list)   # worst 10 by WER
    worst_cer: List[Dict[str, Any]] = field(default_factory=list)   # worst 10 by CER
    failure_verses: List[Dict[str, Any]] = field(default_factory=list)   # WER >= 1.0 or empty
    high_error_verses: List[Dict[str, Any]] = field(default_factory=list)  # WER >= 0.5 or CER >= 0.4
    all_wer: List[float] = field(default_factory=list)
    all_cer: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "wer_mean": round(self.wer_mean, 4),
            "wer_median": round(self.wer_median, 4),
            "wer_p95": round(self.wer_p95, 4),
            "cer_mean": round(self.cer_mean, 4),
            "cer_median": round(self.cer_median, 4),
            "cer_p95": round(self.cer_p95, 4),
            "worst_wer": self.worst_wer,
            "worst_cer": self.worst_cer,
            "failure_verses": self.failure_verses,
            "high_error_verses": self.high_error_verses,
        }


def _percentile(sorted_values: List[float], p: float) -> float:
    """p in [0, 100]. Returns percentile value."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f]) if c != f else sorted_values[f]


def compute_wer_cer_distribution(
    samples: List[Dict[str, Any]],
    reference_key: str = "reference",
    hypothesis_key: str = "hypothesis",
    sample_id_key: str = "verse_key",
    high_wer_threshold: float = WER_HIGH_ERROR_THRESHOLD,
    high_cer_threshold: float = CER_HIGH_ERROR_THRESHOLD,
    failure_wer_threshold: float = WER_FAILURE_THRESHOLD,
    worst_n: int = 10,
) -> EvaluationReport:
    """
    Compute WER/CER for each sample and build distribution report.
    samples: list of {"reference": ref, "hypothesis": hyp, "verse_key": optional id}.
    """
    report = EvaluationReport()
    results: List[SampleResult] = []

    for i, item in enumerate(samples):
        ref = item.get(reference_key) or item.get("text_uthmani") or item.get("text") or ""
        hyp = item.get(hypothesis_key) or item.get("transcript") or item.get("hypothesis") or ""
        sid = item.get(sample_id_key) or str(i)
        if not ref.strip():
            continue
        w = wer(ref, hyp)
        c = cer(ref, hyp)
        results.append(SampleResult(ref=ref, hyp=hyp, wer=w, cer=c, sample_id=sid))

    if not results:
        return report

    report.n_samples = len(results)
    report.all_wer = [r.wer for r in results]
    report.all_cer = [r.cer for r in results]

    sorted_wer = sorted(report.all_wer)
    sorted_cer = sorted(report.all_cer)
    report.wer_mean = sum(report.all_wer) / report.n_samples
    report.wer_median = _percentile(sorted_wer, 50)
    report.wer_p95 = _percentile(sorted_wer, 95)
    report.cer_mean = sum(report.all_cer) / report.n_samples
    report.cer_median = _percentile(sorted_cer, 50)
    report.cer_p95 = _percentile(sorted_cer, 95)

    # Worst N by WER
    by_wer = sorted(results, key=lambda x: -x.wer)
    report.worst_wer = [
        {"sample_id": r.sample_id, "wer": round(r.wer, 4), "cer": round(r.cer, 4), "ref": r.ref[:80], "hyp": r.hyp[:80]}
        for r in by_wer[:worst_n]
    ]

    by_cer = sorted(results, key=lambda x: -x.cer)
    report.worst_cer = [
        {"sample_id": r.sample_id, "wer": round(r.wer, 4), "cer": round(r.cer, 4), "ref": r.ref[:80], "hyp": r.hyp[:80]}
        for r in by_cer[:worst_n]
    ]

    # Failure verses (WER >= 1.0 or effectively empty)
    report.failure_verses = [
        {"sample_id": r.sample_id, "wer": round(r.wer, 4), "cer": round(r.cer, 4)}
        for r in results if r.wer >= failure_wer_threshold
    ]
    for r in report.failure_verses:
        logger.warning("Failure verse: %s WER=%.2f", r["sample_id"], r["wer"])

    # High error
    report.high_error_verses = [
        {"sample_id": r.sample_id, "wer": round(r.wer, 4), "cer": round(r.cer, 4)}
        for r in results if r.wer >= high_wer_threshold or r.cer >= high_cer_threshold
    ]
    for r in report.high_error_verses:
        logger.info("High error recitation: %s WER=%.2f CER=%.2f", r["sample_id"], r["wer"], r["cer"])

    return report


def get_worst_cases(
    samples: List[Dict[str, Any]],
    reference_key: str = "reference",
    hypothesis_key: str = "hypothesis",
    sample_id_key: str = "verse_key",
    worst_n: int = 10,
    by: str = "wer",
) -> List[Dict[str, Any]]:
    """
    Return worst N cases by WER or CER.
    by: "wer" | "cer"
    """
    results = []
    for i, item in enumerate(samples):
        ref = item.get(reference_key) or item.get("text_uthmani") or item.get("text") or ""
        hyp = item.get(hypothesis_key) or item.get("transcript") or item.get("hypothesis") or ""
        if not ref.strip():
            continue
        w = wer(ref, hyp)
        c = cer(ref, hyp)
        sid = item.get(sample_id_key) or str(i)
        results.append({"sample_id": sid, "wer": w, "cer": c, "ref": ref, "hyp": hyp})
    key = "wer" if by == "wer" else "cer"
    results.sort(key=lambda x: -x[key])
    return [
        {"sample_id": r["sample_id"], "wer": round(r["wer"], 4), "cer": round(r["cer"], 4), "ref": r["ref"][:80], "hyp": r["hyp"][:80]}
        for r in results[:worst_n]
    ]
