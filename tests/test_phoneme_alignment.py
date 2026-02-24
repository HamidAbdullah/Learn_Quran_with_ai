"""
Unit tests for Phase 3 phoneme alignment (alignment/phoneme_alignment.py).

Covers align_phoneme_sequences: Levenshtein alignment, phoneme_accuracy,
phoneme_errors, tajweed_error_positions. Uses mock phoneme sequences.
Run: python -m pytest tests/test_phoneme_alignment.py -v
"""

import unittest
from alignment.phoneme_alignment import align_phoneme_sequences


class TestAlignPhonemeSequencesPerfect(unittest.TestCase):
    """Identical reference and hypothesis -> 1.0 accuracy, no errors."""

    def test_identical_sequences(self):
        ref = ["b", "i", "s", "m", "i"]
        hyp = ["b", "i", "s", "m", "i"]
        result = align_phoneme_sequences(ref, hyp)
        self.assertEqual(result["phoneme_accuracy"], 1.0)
        self.assertEqual(len(result["phoneme_errors"]), 0)
        self.assertEqual(result["num_substitutions"], 0)
        self.assertEqual(result["num_deletions"], 0)
        self.assertEqual(result["num_insertions"], 0)
        self.assertEqual(result["reference_length"], 5)
        self.assertEqual(result["hypothesis_length"], 5)

    def test_empty_reference(self):
        result = align_phoneme_sequences([], ["a", "b"])
        self.assertEqual(result["phoneme_accuracy"], 0.0)
        self.assertEqual(result["reference_length"], 0)
        self.assertEqual(result["hypothesis_length"], 2)

    def test_empty_both(self):
        result = align_phoneme_sequences([], [])
        self.assertEqual(result["phoneme_accuracy"], 1.0)


class TestAlignPhonemeSequencesSubstitution(unittest.TestCase):
    """One substitution -> accuracy < 1, one substitution error."""

    def test_one_substitution(self):
        ref = ["b", "i", "s", "m", "i"]
        hyp = ["b", "i", "s", "n", "i"]  # m -> n
        result = align_phoneme_sequences(ref, hyp)
        self.assertLess(result["phoneme_accuracy"], 1.0)
        self.assertEqual(result["num_substitutions"], 1)
        self.assertEqual(result["num_deletions"], 0)
        self.assertEqual(result["num_insertions"], 0)
        subs = [e for e in result["phoneme_errors"] if e["type"] == "substitution"]
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["ref_phoneme"], "m")
        self.assertEqual(subs[0]["hyp_phoneme"], "n")


class TestAlignPhonemeSequencesDeletion(unittest.TestCase):
    """One deletion -> one deletion error."""

    def test_one_deletion(self):
        ref = ["b", "i", "s", "m", "i"]
        hyp = ["b", "i", "s", "i"]  # m missing
        result = align_phoneme_sequences(ref, hyp)
        self.assertLess(result["phoneme_accuracy"], 1.0)
        self.assertEqual(result["num_deletions"], 1)
        dels = [e for e in result["phoneme_errors"] if e["type"] == "deletion"]
        self.assertEqual(len(dels), 1)
        self.assertEqual(dels[0]["ref_phoneme"], "m")
        self.assertIsNone(dels[0]["hyp_phoneme"])


class TestAlignPhonemeSequencesInsertion(unittest.TestCase):
    """One insertion -> one insertion error."""

    def test_one_insertion(self):
        ref = ["b", "i", "s", "i"]
        hyp = ["b", "i", "s", "m", "i"]  # extra m
        result = align_phoneme_sequences(ref, hyp)
        self.assertLess(result["phoneme_accuracy"], 1.0)
        self.assertEqual(result["num_insertions"], 1)
        ins = [e for e in result["phoneme_errors"] if e["type"] == "insertion"]
        self.assertEqual(len(ins), 1)
        self.assertIsNone(ins[0]["ref_phoneme"])
        self.assertEqual(ins[0]["hyp_phoneme"], "m")


class TestTajweedErrorPositions(unittest.TestCase):
    """tajweed_error_positions populated when ref/hyp has Tajweed-relevant phonemes."""

    def test_tajweed_position_on_substitution(self):
        # Madd aa in ref, wrong in hyp
        ref = ["b", "aa", "s", "m"]
        hyp = ["b", "a", "s", "m"]  # short a instead of aa
        boundaries = [(0, 1), (1, 2), (2, 3), (3, 4)]
        result = align_phoneme_sequences(ref, hyp, word_boundaries=boundaries, reference_words=["ب", "ا", "س", "م"])
        self.assertIn("tajweed_error_positions", result)
        # aa is Madd -> should appear in tajweed_error_positions
        tajweed = result["tajweed_error_positions"]
        self.assertGreaterEqual(len(tajweed), 1)
        self.assertTrue(
            any(
                t.get("ref_phoneme") == "aa" or t.get("hyp_phoneme") == "aa"
                for t in tajweed
            ),
            f"Expected Madd (aa) in tajweed_error_positions: {tajweed}",
        )

    def test_word_boundaries_preserved(self):
        ref = ["a", "b", "c"]
        hyp = ["a", "b", "c"]
        boundaries = [(0, 1), (1, 2), (2, 3)]
        result = align_phoneme_sequences(ref, hyp, word_boundaries=boundaries)
        self.assertEqual(result.get("word_boundaries"), boundaries)
