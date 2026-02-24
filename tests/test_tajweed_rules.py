"""
Unit tests for Phase 3 Tajweed rule detection (tajweed/rules.py).

Covers detect_tajweed_errors and tajweed_rule_accuracy with mock alignment results.
Run: python -m pytest tests/test_tajweed_rules.py -v
"""

import unittest
from tajweed.rules import detect_tajweed_errors, tajweed_rule_accuracy


class TestDetectTajweedErrorsMadd(unittest.TestCase):
    """Madd (long vowel) errors in alignment -> structured error."""

    def test_madd_substitution_in_positions(self):
        alignment_result = {
            "phoneme_errors": [],
            "tajweed_error_positions": [
                {
                    "ref_idx": 1,
                    "word_index": 0,
                    "type": "substitution",
                    "ref_phoneme": "aa",
                    "hyp_phoneme": "a",
                    "word": "فاعل",
                }
            ],
            "word_boundaries": [(0, 2), (2, 5)],
        }
        errors = detect_tajweed_errors(alignment_result, reference_words=["فاعل", "كبير"])
        madd_errors = [e for e in errors if e["rule_name"] == "Madd"]
        self.assertGreaterEqual(len(madd_errors), 1)
        self.assertEqual(madd_errors[0]["expected"], "long vowel (aa)")
        self.assertIn("severity", madd_errors[0])
        self.assertIn(madd_errors[0]["severity"], ("high", "medium", "low"))


class TestDetectTajweedErrorsGhunnah(unittest.TestCase):
    """Ghunnah (nasalization) errors."""

    def test_ghunnah_deletion_in_positions(self):
        alignment_result = {
            "phoneme_errors": [],
            "tajweed_error_positions": [
                {
                    "ref_idx": 2,
                    "word_index": 0,
                    "type": "deletion",
                    "ref_phoneme": "n_G",
                    "hyp_phoneme": None,
                    "word": "من",
                }
            ],
            "word_boundaries": [(0, 3)],
        }
        errors = detect_tajweed_errors(alignment_result, reference_words=["من"])
        ghunnah_errors = [e for e in errors if e["rule_name"] == "Ghunnah"]
        self.assertGreaterEqual(len(ghunnah_errors), 1)
        self.assertIn("nasalization", ghunnah_errors[0]["expected"])


class TestDetectTajweedErrorsQalqalah(unittest.TestCase):
    """Qalqalah (bounce) errors."""

    def test_qalqalah_substitution_in_positions(self):
        alignment_result = {
            "phoneme_errors": [],
            "tajweed_error_positions": [
                {
                    "ref_idx": 1,
                    "word_index": 0,
                    "type": "substitution",
                    "ref_phoneme": "q_Q",
                    "hyp_phoneme": "k",
                    "word": "قل",
                }
            ],
            "word_boundaries": [(0, 2)],
        }
        errors = detect_tajweed_errors(alignment_result, reference_words=["قل"])
        qalqalah_errors = [e for e in errors if e["rule_name"] == "Qalqalah"]
        self.assertGreaterEqual(len(qalqalah_errors), 1)
        self.assertIn("bounce", qalqalah_errors[0]["expected"].lower() or "q" in qalqalah_errors[0]["expected"])


class TestDetectTajweedErrorsHeavyLight(unittest.TestCase):
    """Heavy/Light letter errors."""

    def test_heavy_light_substitution_in_positions(self):
        alignment_result = {
            "phoneme_errors": [],
            "tajweed_error_positions": [
                {
                    "ref_idx": 0,
                    "word_index": 0,
                    "type": "substitution",
                    "ref_phoneme": "s_H",
                    "hyp_phoneme": "s",
                    "word": "صد",
                }
            ],
            "word_boundaries": [(0, 2)],
        }
        errors = detect_tajweed_errors(alignment_result, reference_words=["صد"])
        heavy_errors = [e for e in errors if e["rule_name"] == "Heavy_Light"]
        self.assertGreaterEqual(len(heavy_errors), 1)
        self.assertIn("heavy", heavy_errors[0]["expected"].lower())


class TestTajweedRuleAccuracy(unittest.TestCase):
    """tajweed_rule_accuracy from alignment result."""

    def test_empty_positions_high_accuracy(self):
        alignment_result = {
            "tajweed_error_positions": [],
            "reference_length": 20,
        }
        acc = tajweed_rule_accuracy(alignment_result)
        self.assertGreater(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_with_errors_lower_accuracy(self):
        alignment_result = {
            "tajweed_error_positions": [
                {"ref_phoneme": "aa", "ref_idx": 1},
                {"ref_phoneme": "n_G", "ref_idx": 3},
            ],
            "reference_length": 24,
        }
        acc = tajweed_rule_accuracy(alignment_result)
        self.assertLess(acc, 1.0)
        self.assertGreaterEqual(acc, 0.0)

    def test_zero_reference_returns_one(self):
        alignment_result = {"tajweed_error_positions": [], "reference_length": 0}
        acc = tajweed_rule_accuracy(alignment_result)
        self.assertEqual(acc, 1.0)


class TestTajweedErrorStructure(unittest.TestCase):
    """Each error has rule_name, word, expected, detected, severity."""

    def test_error_keys_present(self):
        alignment_result = {
            "phoneme_errors": [],
            "tajweed_error_positions": [
                {"ref_idx": 0, "word_index": 0, "type": "substitution", "ref_phoneme": "aa", "hyp_phoneme": "a", "word": "ا"}
            ],
            "word_boundaries": [(0, 1)],
        }
        errors = detect_tajweed_errors(alignment_result, reference_words=["ا"])
        for e in errors:
            self.assertIn("rule_name", e)
            self.assertIn("expected", e)
            self.assertIn("detected", e)
            self.assertIn("severity", e)
            self.assertIn(e["severity"], ("high", "medium", "low"))
