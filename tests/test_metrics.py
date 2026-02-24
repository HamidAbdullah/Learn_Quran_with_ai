"""
Unit tests for WER and CER metrics (Phase 2).
No audio or models required.
"""
import unittest
from core.normalization import normalize_arabic
from core.metrics import wer, cer, wer_cer


class TestWER(unittest.TestCase):
    def test_perfect_match(self):
        ref = "بسم الله الرحمن الرحيم"
        hyp = "بسم الله الرحمن الرحيم"
        self.assertEqual(wer(ref, hyp), 0.0)

    def test_normalized_match(self):
        ref = "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیمِ"
        hyp = "بسم الله الرحمن الرحيم"
        self.assertEqual(wer(ref, hyp), 0.0)

    def test_one_word_wrong(self):
        ref = "الحمد لله رب"
        hyp = "الحمد لله العالم"
        # رب -> العالم: one substitution
        w = wer(ref, hyp)
        self.assertGreater(w, 0)
        self.assertLessEqual(w, 1.0)

    def test_one_word_deletion(self):
        ref = "بسم الله الرحمن الرحيم"
        hyp = "بسم الله الرحيم"
        w = wer(ref, hyp)
        self.assertGreater(w, 0)
        self.assertLessEqual(w, 1.0)

    def test_empty_reference(self):
        self.assertEqual(wer("", "anything"), 0.0)
        self.assertEqual(wer("", ""), 0.0)

    def test_empty_hypothesis(self):
        ref = "بسم الله"
        w = wer(ref, "")
        self.assertGreater(w, 0)
        self.assertLessEqual(w, 1.0)


class TestCER(unittest.TestCase):
    def test_perfect_match(self):
        ref = "بسم الله"
        hyp = "بسم الله"
        self.assertEqual(cer(ref, hyp), 0.0)

    def test_one_char_diff(self):
        ref = "الله"
        hyp = "اللاه"  # ه vs ة would normalize to same; use different letter
        c = cer(ref, hyp)
        self.assertGreater(c, 0)
        self.assertLess(c, 1.0)

    def test_spaces_ignored(self):
        ref = "بسم الله الرحمن"
        hyp = "بسم الله الرحمن"
        self.assertEqual(cer(ref, hyp, remove_spaces=True), 0.0)


class TestWERCER(unittest.TestCase):
    def test_wer_cer_both(self):
        ref = "بسم الله الرحمن الرحيم"
        hyp = "بسم الله الرحمن الرحيم"
        w, c = wer_cer(ref, hyp)
        self.assertEqual(w, 0.0)
        self.assertEqual(c, 0.0)

    def test_wer_cer_mismatch(self):
        ref = "الحمد لله"
        hyp = "الحمد كله"
        w, c = wer_cer(ref, hyp)
        self.assertGreater(w, 0)
        self.assertGreater(c, 0)


if __name__ == "__main__":
    unittest.main()
