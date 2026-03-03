"""
Unit tests for dual ASR selection logic (Phase 2).
Uses mocks; no real Whisper/Wav2Vec2 load.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from core.asr import run_dual_asr
from core.normalization import normalize_arabic


# Reference verse (1:1)
REF_1_1 = "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیْمِ ۟"


def _make_whisper_model(transcribe_return):
    m = MagicMock()
    m.transcribe = MagicMock(return_value=transcribe_return)
    return m


def _make_phonetic_analyzer(transcript=None, logits=None, confidence=0.9):
    a = MagicMock()
    if transcript is not None and logits is not None:
        a.run_forward = MagicMock(return_value=(transcript, logits))
    else:
        a.run_forward = MagicMock(return_value=(None, None))
    a.transcribe = MagicMock(return_value=transcript or "")
    a.analyze_alignment_from_logits = MagicMock(
        return_value={"transcription": transcript or "", "confidence_avg": confidence, "frames": []}
    )
    a.analyze_alignment = MagicMock(
        return_value={"transcription": transcript or "", "confidence_avg": confidence, "frames": []}
    )
    return a


class TestASRSelection(unittest.TestCase):
    def test_whisper_only(self):
        whisper = _make_whisper_model({"text": "بسم الله الرحمن الرحيم", "no_speech_prob": 0.0})
        phonetic = _make_phonetic_analyzer(transcript=None, logits=None)
        out = run_dual_asr(
            whisper_model=whisper,
            phonetic_analyzer=phonetic,
            reference_text=REF_1_1,
            audio_path="/tmp/x.wav",
            audio_y_sr=(np.zeros(16000 * 2), 16000),
        )
        self.assertEqual(out["selected_source"], "whisper")
        self.assertEqual(out["selected_transcript"], "بسم الله الرحمن الرحيم")

    def test_wav2vec_only(self):
        whisper = _make_whisper_model({"text": "", "no_speech_prob": 0.99})
        phonetic = _make_phonetic_analyzer(
            transcript="بسم الله الرحمن الرحيم",
            logits=MagicMock(),
            confidence=0.92,
        )
        out = run_dual_asr(
            whisper_model=whisper,
            phonetic_analyzer=phonetic,
            reference_text=REF_1_1,
            audio_path="/tmp/x.wav",
            audio_y_sr=(np.zeros(16000 * 2), 16000),
        )
        self.assertEqual(out["selected_source"], "wav2vec")
        self.assertIn("الرحمن", out["selected_transcript"])

    def test_selection_by_match_count(self):
        # Wav2Vec matches all 4 words; Whisper misses one → wav2vec should win
        # Selection can be "match_count" (fallback) or "weighted_similarity" (new path)
        whisper = _make_whisper_model({"text": "بسم الله الرحيم", "no_speech_prob": 0.0})
        phonetic = _make_phonetic_analyzer(
            transcript="بسم الله الرحمن الرحيم",
            logits=MagicMock(),
            confidence=0.85,
        )
        out = run_dual_asr(
            whisper_model=whisper,
            phonetic_analyzer=phonetic,
            reference_text=REF_1_1,
            audio_path="/tmp/x.wav",
            audio_y_sr=(np.zeros(16000 * 2), 16000),
        )
        self.assertEqual(out["selected_source"], "wav2vec")
        self.assertIn(out["selection_reason"], ("match_count", "weighted_similarity"))

    def test_selection_by_confidence_on_tie(self):
        # Same transcript from both; wav2vec has higher confidence → wav2vec should win
        # Selection can be "confidence_tie" (fallback) or "weighted_similarity" (new path)
        same = "بسم الله الرحمن الرحيم"
        whisper = _make_whisper_model({"text": same, "no_speech_prob": 0.2})
        phonetic = _make_phonetic_analyzer(transcript=same, logits=MagicMock(), confidence=0.95)
        out = run_dual_asr(
            whisper_model=whisper,
            phonetic_analyzer=phonetic,
            reference_text=REF_1_1,
            audio_path="/tmp/x.wav",
            audio_y_sr=(np.zeros(16000 * 2), 16000),
        )
        self.assertEqual(out["selected_source"], "wav2vec")
        self.assertIn(out["selection_reason"], ("confidence_tie", "weighted_similarity"))

    def test_asr_result_keys(self):
        whisper = _make_whisper_model({"text": "بسم الله", "no_speech_prob": 0.0})
        phonetic = _make_phonetic_analyzer(transcript="بسم الله", logits=MagicMock(), confidence=0.9)
        out = run_dual_asr(
            whisper_model=whisper,
            phonetic_analyzer=phonetic,
            reference_text=REF_1_1,
            audio_path="/tmp/x.wav",
            audio_y_sr=(np.zeros(16000 * 2), 16000),
        )
        for key in (
            "selected_transcript",
            "selected_source",
            "selection_reason",
            "transcript_whisper",
            "transcript_wav2vec",
            "confidence_whisper",
            "confidence_wav2vec",
            "match_count_whisper",
            "match_count_wav2vec",
        ):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
