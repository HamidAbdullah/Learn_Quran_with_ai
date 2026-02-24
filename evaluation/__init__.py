"""
Evaluation intelligence layer: metrics distribution, benchmark runner, dataset validation.
"""
from evaluation.asr_metrics import (
    compute_wer_cer_distribution,
    EvaluationReport,
    get_worst_cases,
)
from evaluation.benchmark_runner import run_benchmark, write_report
from evaluation.dataset_validation import validate_dataset, ValidationReport

__all__ = [
    "compute_wer_cer_distribution",
    "EvaluationReport",
    "get_worst_cases",
    "run_benchmark",
    "write_report",
    "validate_dataset",
    "ValidationReport",
]
