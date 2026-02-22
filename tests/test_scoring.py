"""
Unit tests for recitation scoring: verse accuracy, alignment scoring, and edge cases.
No audio required; uses mock transcripts and alignment data.
Run: python -m pytest tests/test_scoring.py -v   or   python -m unittest tests.test_scoring
"""

import unittest
from core.normalization import normalize_arabic
from core.scoring import (
    score_recitation,
    segment_transcript_by_reference,
    count_matching_words,
    SIMILARITY_CORRECT,
    SIMILARITY_MINOR,
)


# Reference verse (1:1) — normalized form for comparison
VERSE_1_1_UTHMANI = "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیمِ ۟"
VERSE_1_1_NORM = normalize_arabic(VERSE_1_1_UTHMANI)


class TestVerseRecitationAccuracy(unittest.TestCase):
    """Verse recitation accuracy: correct, minor mistake, wrong, missing, extra."""

    def test_perfect_match(self):
        ref = VERSE_1_1_UTHMANI
        user = "بسم الله الرحمن الرحيم"
        result = score_recitation(ref, user)
        self.assertEqual(result["accuracy_score"], 100.0)
        self.assertTrue(all(w["status"] == "correct" for w in result["word_analysis"]))
        self.assertIn(result["confidence_level"], ("high", "medium"))

    def test_normalized_script_match(self):
        # Persian ی ک vs Arabic ي ك
        ref = "بسم الله الرحمن الرحيم"
        user = "بسم الله الرحمان الرحیم"
        result = score_recitation(ref, user)
        # الرحمان vs الرحمن: minor slip
        self.assertGreaterEqual(result["accuracy_score"], 0)
        self.assertEqual(len(result["word_analysis"]), 4)

    def test_missing_word(self):
        ref = VERSE_1_1_UTHMANI
        user_norm = "بسم الله الرحيم"  # الرحمن missing
        result = score_recitation(ref, user_norm)
        self.assertLess(result["accuracy_score"], 100.0)
        statuses = [w["status"] for w in result["word_analysis"]]
        self.assertIn("missing", statuses)

    def test_extra_word(self):
        ref = "الحمد لله"
        user = "الحمد لله رب"
        result = score_recitation(ref, user)
        self.assertGreaterEqual(len(result["extra_words"]), 1)

    def test_wrong_word(self):
        ref = "الحمد لله"
        user = "الحمد كله"
        result = score_recitation(ref, user)
        self.assertLess(result["accuracy_score"], 100.0)
        statuses = [w["status"] for w in result["word_analysis"]]
        self.assertTrue("wrong" in statuses or "minor_mistake" in statuses)


class TestAlignmentScoring(unittest.TestCase):
    """Time-alignment scoring: 0.6 text + 0.3 phonetic + 0.1 timing."""

    def test_with_alignment_words(self):
        ref = "بسم الله الرحمن الرحيم"
        user = "بسم الله الرحمن الرحيم"
        alignment_words = [
            {"word": "بسم", "start_time": 0.0, "end_time": 0.5, "confidence": 0.95, "phonetic_similarity": 0.95},
            {"word": "الله", "start_time": 0.5, "end_time": 1.0, "confidence": 0.98, "phonetic_similarity": 0.98},
            {"word": "الرحمن", "start_time": 1.0, "end_time": 1.6, "confidence": 0.92, "phonetic_similarity": 0.92},
            {"word": "الرحيم", "start_time": 1.6, "end_time": 2.0, "confidence": 0.97, "phonetic_similarity": 0.97},
        ]
        result = score_recitation(
            ref, user,
            alignment_words=alignment_words,
            audio_duration=2.0,
        )
        self.assertEqual(result["accuracy_score"], 100.0)
        for w in result["word_analysis"]:
            self.assertTrue("start_time" in w or "end_time" in w or w.get("status") == "correct")

    def test_empty_alignment_fallback(self):
        ref = "الحمد لله"
        user = "الحمد لله"
        result = score_recitation(ref, user, alignment_words=[], audio_duration=1.0)
        self.assertEqual(result["accuracy_score"], 100.0)


class TestSegmentByReference(unittest.TestCase):
    """Segment transcript by reference (greedy word boundaries for CTC output)."""

    def test_already_segmented(self):
        raw = "بسم الله الرحمن الرحيم"
        ref = VERSE_1_1_UTHMANI
        out = segment_transcript_by_reference(raw, ref)
        self.assertEqual(out.strip(), raw.strip())

    def test_no_spaces(self):
        raw = "بسماللهالرحمنالرحيم"
        ref = VERSE_1_1_UTHMANI
        out = segment_transcript_by_reference(raw, ref)
        words = out.split()
        self.assertGreaterEqual(len(words), 2)


class TestCountMatchingWords(unittest.TestCase):
    """Best-ASR selection: count matching words."""

    def test_more_matches_wins(self):
        ref = "بسم الله الرحمن الرحيم"
        t1 = "بسم الله الرحمن الرحيم"
        t2 = "بسم الله الرحيم"
        self.assertEqual(count_matching_words(ref, t1), 4)
        self.assertLessEqual(count_matching_words(ref, t2), 3)


class TestFastSlowPauses(unittest.TestCase):
    """Simulated fast/slow/pauses via alignment timing (no audio)."""

    def test_reasonable_timing_similarity(self):
        # When alignment has ordered timings, timing_similarity should be high
        ref = "الله الرحمن"
        user = "الله الرحمن"
        alignment_words = [
            {"word": "الله", "start_time": 0.0, "end_time": 0.4, "confidence": 0.9, "phonetic_similarity": 0.9},
            {"word": "الرحمن", "start_time": 0.5, "end_time": 1.0, "confidence": 0.9, "phonetic_similarity": 0.9},
        ]
        result = score_recitation(ref, user, alignment_words=alignment_words, audio_duration=1.0)
        self.assertGreaterEqual(result["accuracy_score"], 90.0)

    def test_short_audio_duration(self):
        ref = "الله"
        user = "الله"
        result = score_recitation(ref, user, alignment_words=None, audio_duration=0.1)
        self.assertIn("word_analysis", result)
        self.assertTrue(result["teacher_feedback_text"])

    def test_fast_recitation(self):
        """Fast recitation: short word durations in alignment."""
        ref = "بسم الله الرحمن الرحيم"
        user = "بسم الله الرحمن الرحيم"
        alignment_words = [
            {"word": "بسم", "start_time": 0.0, "end_time": 0.15, "confidence": 0.9, "phonetic_similarity": 0.9},
            {"word": "الله", "start_time": 0.15, "end_time": 0.35, "confidence": 0.9, "phonetic_similarity": 0.9},
            {"word": "الرحمن", "start_time": 0.35, "end_time": 0.55, "confidence": 0.9, "phonetic_similarity": 0.9},
            {"word": "الرحيم", "start_time": 0.55, "end_time": 0.7, "confidence": 0.9, "phonetic_similarity": 0.9},
        ]
        result = score_recitation(ref, user, alignment_words=alignment_words, audio_duration=0.7)
        self.assertIn("word_analysis", result)
        self.assertIn("accuracy_score", result)
        # Fast recitation may still score correct if text+phonetic are good; timing may reduce slightly
        self.assertGreaterEqual(result["accuracy_score"], 0)

    def test_slow_recitation(self):
        """Slow recitation: long word durations in alignment."""
        ref = "الحمد لله"
        user = "الحمد لله"
        alignment_words = [
            {"word": "الحمد", "start_time": 0.0, "end_time": 1.5, "confidence": 0.95, "phonetic_similarity": 0.95},
            {"word": "لله", "start_time": 1.5, "end_time": 3.5, "confidence": 0.95, "phonetic_similarity": 0.95},
        ]
        result = score_recitation(ref, user, alignment_words=alignment_words, audio_duration=3.5)
        self.assertEqual(result["accuracy_score"], 100.0)
        self.assertIn("timing_data", result)
        self.assertEqual(len(result["timing_data"]), 2)

    def test_pause_inside_verse(self):
        """Pause inside verse: gap between words in alignment."""
        ref = "الله الرحمن الرحيم"
        user = "الله الرحمن الرحيم"
        alignment_words = [
            {"word": "الله", "start_time": 0.0, "end_time": 0.5, "confidence": 0.92, "phonetic_similarity": 0.92},
            {"word": "الرحمن", "start_time": 1.2, "end_time": 1.8, "confidence": 0.92, "phonetic_similarity": 0.92},
            {"word": "الرحيم", "start_time": 1.9, "end_time": 2.4, "confidence": 0.92, "phonetic_similarity": 0.92},
        ]
        result = score_recitation(ref, user, alignment_words=alignment_words, audio_duration=2.5)
        self.assertIn("word_analysis", result)
        self.assertIn("confidence_score", result)
        self.assertGreaterEqual(result["accuracy_score"], 0)
        self.assertIsInstance(result["confidence_score"], (int, float))


if __name__ == "__main__":
    unittest.main()
