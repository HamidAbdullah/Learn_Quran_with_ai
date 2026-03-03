"""
Unit tests for Phase 3 phoneme extraction (core/phoneme.py).

Covers arabic_to_phonemes and verse_to_phoneme_sequence: diacritics, Madd,
Qalqalah, Ghunnah, heavy letters, word boundaries. No audio required.
Run: python -m pytest tests/test_phoneme.py -v
"""

import unittest
from core.phoneme import (
    arabic_to_phonemes,
    verse_to_phoneme_sequence,
    get_qalqalah_phonemes,
    get_heavy_phonemes,
    get_ghunnah_phonemes,
)


class TestArabicToPhonemes(unittest.TestCase):
    """Single-word phoneme conversion."""

    def test_bismillah_word(self):
        # بسم (with diacritics: بِسْمِ) -> b, i, s, m, i
        word = "بِسْمِ"
        out = arabic_to_phonemes(word)
        self.assertIsInstance(out, list)
        self.assertIn("b", out)
        self.assertIn("s", out)
        self.assertIn("m", out)
        self.assertTrue(any(v in out for v in ("i", "a")))

    def test_word_without_diacritics(self):
        # بسم without harakat still yields consonants
        word = "بسم"
        out = arabic_to_phonemes(word)
        self.assertGreater(len(out), 0)
        self.assertIn("b", out)
        self.assertIn("s", out)
        self.assertIn("m", out)

    def test_madd_alif(self):
        # فَاعِل pattern: fatha + alif -> aa
        word = "فَاعِل"
        out = arabic_to_phonemes(word)
        self.assertIn("aa", out, "Madd (long a) should appear as 'aa'")

    def test_qalqalah_tag(self):
        # ق with explicit sukun (قْ) should get _Q (qalqalah)
        word = "بَقْ"  # ba then qaf with sukun
        out = arabic_to_phonemes(word)
        qalqalah_phonemes = get_qalqalah_phonemes()
        self.assertTrue(
            any(p in out for p in qalqalah_phonemes),
            f"Expected a Qalqalah-tagged phoneme in {out}",
        )

    def test_heavy_letter(self):
        # ص (sad) -> s_H
        word = "صَد"
        out = arabic_to_phonemes(word)
        heavy = get_heavy_phonemes()
        self.assertTrue(
            any(p in out for p in heavy),
            f"Expected heavy phoneme in {out}",
        )

    def test_empty_input(self):
        self.assertEqual(arabic_to_phonemes(""), [])
        self.assertEqual(arabic_to_phonemes("   "), [])


class TestVerseToPhonemeSequence(unittest.TestCase):
    """Full-verse phoneme sequence and word boundaries."""

    def test_multi_word_verse(self):
        text = "بِسْمِ اللّٰهِ الرَّحْمٰنِ"
        seq = verse_to_phoneme_sequence(text)
        self.assertIn("phoneme_sequence", seq)
        self.assertIn("word_boundaries", seq)
        self.assertIn("words", seq)
        self.assertIn("phonemes_per_word", seq)
        self.assertEqual(len(seq["words"]), len(seq["word_boundaries"]))
        self.assertEqual(len(seq["phoneme_sequence"]), sum(len(pw) for pw in seq["phonemes_per_word"]))

    def test_word_boundaries_contiguous(self):
        text = "الله الرحمن"
        seq = verse_to_phoneme_sequence(text)
        boundaries = seq["word_boundaries"]
        for i in range(len(boundaries) - 1):
            self.assertEqual(boundaries[i][1], boundaries[i + 1][0], "Boundaries should be contiguous")

    def test_phonemes_per_word_lengths(self):
        text = "الحمد لله"
        seq = verse_to_phoneme_sequence(text)
        flat = seq["phoneme_sequence"]
        reconstructed = []
        for start, end in seq["word_boundaries"]:
            reconstructed.extend(flat[start:end])
        self.assertEqual(flat, reconstructed)

    def test_empty_verse(self):
        seq = verse_to_phoneme_sequence("")
        self.assertEqual(seq["phoneme_sequence"], [])
        self.assertEqual(seq["word_boundaries"], [])
        self.assertEqual(seq["words"], [])
        self.assertEqual(seq["phonemes_per_word"], [])


class TestPhonemeHelpers(unittest.TestCase):
    """Helper functions for rule detection."""

    def test_qalqalah_list(self):
        q = get_qalqalah_phonemes()
        self.assertIsInstance(q, list)
        self.assertTrue(all("_Q" in p for p in q))

    def test_heavy_list(self):
        h = get_heavy_phonemes()
        self.assertIsInstance(h, list)
        self.assertTrue(all("_H" in p or p == "q" for p in h))

    def test_ghunnah_list(self):
        g = get_ghunnah_phonemes()
        self.assertIn("n_G", g)
        self.assertIn("m_G", g)
