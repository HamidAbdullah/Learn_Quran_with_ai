"""
Dataset validation tests: required fields (audio_path, reference_text), missing files, empty transcripts.
"""
import json
import os
import tempfile
import unittest

from evaluation.dataset_validation import validate_dataset, ValidationReport
from evaluation.asr_metrics import compute_wer_cer_distribution, get_worst_cases, EvaluationReport


class TestValidateDatasetRequiredFields(unittest.TestCase):
    def test_valid_item_has_reference_and_audio_path(self):
        items = [
            {"verse_key": "1:1", "text_uthmani": "بسم الله", "audio": "path/to/001001.mp3"},
        ]
        report = validate_dataset(items, audio_base_dir="/base")
        self.assertEqual(report.n_total, 1)
        self.assertEqual(len(report.missing_files), 1)
        self.assertIn("001001", report.missing_files[0]["path"])

    def test_missing_reference(self):
        items = [
            {"verse_key": "1:1", "audio": "x.mp3"},
        ]
        report = validate_dataset(items)
        self.assertEqual(len(report.missing_reference), 1)

    def test_reference_from_text_uthmani(self):
        items = [
            {"verse_key": "1:1", "text_uthmani": "بسم الله الرحمن الرحيم"},
        ]
        report = validate_dataset(items)
        self.assertEqual(report.n_valid, 1)
        self.assertEqual(report.n_total, 1)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            items = [
                {"verse_key": "1:1", "text_uthmani": "بسم الله", "audio": "nonexistent.mp3"},
            ]
            report = validate_dataset(items, audio_base_dir=d)
            self.assertEqual(len(report.missing_files), 1)


class TestValidateDatasetReport(unittest.TestCase):
    def test_report_to_dict(self):
        report = ValidationReport(n_total=5, n_valid=3)
        report.missing_reference.append({"sample_id": "1:1"})
        d = report.to_dict()
        self.assertEqual(d["n_total"], 5)
        self.assertEqual(d["n_valid"], 3)
        self.assertEqual(len(d["missing_reference"]), 1)


class TestAsrMetricsDistribution(unittest.TestCase):
    def test_wer_cer_distribution(self):
        samples = [
            {"reference": "بسم الله الرحمن الرحيم", "hypothesis": "بسم الله الرحمن الرحيم", "verse_key": "1:1"},
            {"reference": "الحمد لله", "hypothesis": "الحمد كله", "verse_key": "1:2"},
        ]
        report = compute_wer_cer_distribution(samples)
        self.assertEqual(report.n_samples, 2)
        self.assertEqual(report.wer_mean, (0.0 + 0.5) / 2)
        self.assertEqual(len(report.worst_wer), 2)

    def test_get_worst_cases(self):
        samples = [
            {"reference": "بسم الله", "hypothesis": "بسم الله", "verse_key": "1:1"},
            {"reference": "الحمد لله رب", "hypothesis": "الحمد", "verse_key": "1:2"},
        ]
        worst = get_worst_cases(samples, worst_n=1, by="wer")
        self.assertEqual(len(worst), 1)
        self.assertEqual(worst[0]["sample_id"], "1:2")


if __name__ == "__main__":
    unittest.main()
