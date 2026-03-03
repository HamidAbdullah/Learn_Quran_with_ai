"""
Tests for ASR quality score (confidence module).
"""
import unittest
from core.confidence import get_asr_quality_score
from core.metrics import wer


class TestGetAsrQualityScore(unittest.TestCase):
    def test_perfect_match(self):
        ref = "بسم الله الرحمن الرحيم"
        hyp = "بسم الله الرحمن الرحيم"
        out = get_asr_quality_score(hyp, ref)
        self.assertEqual(out["edit_distance_score"], 1.0)
        self.assertEqual(out["match_score"], 1.0)
        self.assertGreaterEqual(out["final_score"], 0.9)

    def test_no_logits_uses_edit_distance(self):
        ref = "الحمد لله"
        hyp = "الحمد كله"
        out = get_asr_quality_score(hyp, ref)
        self.assertLess(out["edit_distance_score"], 1.0)
        self.assertGreaterEqual(out["edit_distance_score"], 0.0)
        self.assertGreaterEqual(out["final_score"], 0.0)
        self.assertLessEqual(out["final_score"], 1.0)

    def test_empty_hypothesis(self):
        ref = "بسم الله"
        out = get_asr_quality_score("", ref)
        self.assertEqual(out["edit_distance_score"], 0.0)
        self.assertEqual(out["match_score"], 0.0)

    def test_keys_present(self):
        out = get_asr_quality_score("بسم", "بسم الله")
        for key in ("final_score", "token_confidence", "edit_distance_score", "match_score"):
            self.assertIn(key, out)
            self.assertIsInstance(out[key], (int, float))


if __name__ == "__main__":
    unittest.main()
