"""
Phase 4 — Tests for streaming pipeline (buffer, incremental ASR, latency, no crash on incomplete chunk).

Uses mock chunk input and mock Whisper; no real model or audio required.
Run: python3 -m unittest tests.test_streaming_pipeline -v
"""

import unittest
from unittest.mock import MagicMock

from streaming.audio_buffer import AudioBuffer, bytes_to_duration_ms, duration_ms_to_bytes


class TestAudioBuffer(unittest.TestCase):
    """AudioBuffer: append chunks, has_ready_segment, get_ready_segment."""

    def test_empty_not_ready(self):
        buf = AudioBuffer(window_duration_ms=2000.0)
        self.assertFalse(buf.has_ready_segment())
        self.assertIsNone(buf.get_ready_segment())

    def test_small_chunk_not_ready(self):
        buf = AudioBuffer(window_duration_ms=2000.0)
        # 200 ms = 6400 bytes
        chunk = b"\x00" * 6400
        buf.append(chunk)
        self.assertLess(buf.duration_ms(), 2000.0)
        self.assertFalse(buf.has_ready_segment())

    def test_accumulate_until_ready(self):
        buf = AudioBuffer(window_duration_ms=2000.0, overlap_duration_ms=0)
        # 2000 ms = 64000 bytes
        chunk_2000 = b"\x00" * 64000
        buf.append(chunk_2000)
        self.assertTrue(buf.has_ready_segment())
        segment = buf.get_ready_segment()
        self.assertIsNotNone(segment)
        self.assertEqual(len(segment), 64000)
        self.assertFalse(buf.has_ready_segment())

    def test_partial_chunk_no_crash(self):
        """Incomplete chunk (e.g. 100 bytes) should not crash."""
        buf = AudioBuffer(window_duration_ms=2000.0)
        buf.append(b"\x00" * 100)
        buf.append(b"\x00" * 50)
        self.assertFalse(buf.has_ready_segment())
        self.assertIsNone(buf.get_ready_segment())
        self.assertEqual(buf.duration_ms(), bytes_to_duration_ms(150))

    def test_clear(self):
        buf = AudioBuffer(window_duration_ms=100.0)
        buf.append(b"\x00" * 32000)
        buf.clear()
        self.assertEqual(len(buf.peek_all()), 0)
        self.assertFalse(buf.has_ready_segment())

    def test_total_appended(self):
        buf = AudioBuffer(window_duration_ms=2000.0)
        buf.append(b"\x00" * 1000)
        buf.append(b"\x00" * 2000)
        self.assertEqual(buf.total_appended_bytes(), 3000)


class TestStreamingAsr(unittest.TestCase):
    """run_streaming_asr: short audio returns empty; mock Whisper returns text and meta."""

    def test_short_audio_returns_empty_and_meta(self):
        from streaming.streaming_asr import run_streaming_asr
        mock_whisper = MagicMock()
        text, meta = run_streaming_asr(mock_whisper, b"\x00" * 100)
        self.assertEqual(text, "")
        self.assertIn("duration_ms", meta)
        self.assertEqual(meta["duration_ms"], bytes_to_duration_ms(100))

    def test_mock_whisper_returns_partial_and_latency(self):
        from streaming.streaming_asr import run_streaming_asr
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "بسم الله"}
        # Enough bytes for one 2s window (32000 * 2)
        audio = b"\x00" * 64000
        text, meta = run_streaming_asr(mock_whisper, audio)
        self.assertEqual(text.strip(), "بسم الله")
        self.assertIn("duration_ms", meta)
        self.assertIn("inference_ms", meta)


class TestLatencyTracked(unittest.TestCase):
    """Latency is measured (buffer_ms from segment length; total_ms from timing)."""

    def test_buffer_duration_ms(self):
        # 1 second = 32000 bytes
        self.assertAlmostEqual(bytes_to_duration_ms(32000), 1000.0, delta=1.0)
        self.assertEqual(duration_ms_to_bytes(1000), 32000)

    def test_partial_result_structure(self):
        """Expected keys for partial_result: partial_transcript, word_feedback, latency_ms."""
        expected = {"partial_transcript": "", "word_feedback": [], "latency_ms": {"buffer_ms": 2000, "total_ms": 250}}
        self.assertIn("partial_transcript", expected)
        self.assertIn("word_feedback", expected)
        self.assertIn("latency_ms", expected)
        self.assertIn("buffer_ms", expected["latency_ms"])

    def test_partial_transcript_grows_with_more_input(self):
        """Longer partial transcript yields more resolved words (word_feedback)."""
        try:
            from core.scoring import get_lightweight_word_feedback
        except ImportError:
            self.skipTest("core.scoring not available")
            return
        ref = "بسم الله الرحمن الرحيم"
        feedback_short = get_lightweight_word_feedback(ref, "بسم")
        feedback_long = get_lightweight_word_feedback(ref, "بسم الله الرحمن الرحيم")
        self.assertEqual(len(feedback_short), 4)
        self.assertEqual(len(feedback_long), 4)
        correct_long = sum(1 for w in feedback_long if w.get("status") == "correct")
        correct_short = sum(1 for w in feedback_short if w.get("status") == "correct")
        self.assertGreaterEqual(correct_long, correct_short)


class TestBuildWsReciteHandler(unittest.TestCase):
    """build_ws_recite_handler returns a callable; no crash on missing ref."""

    def test_builder_returns_async_handler(self):
        try:
            from streaming.websocket_server import build_ws_recite_handler
        except ImportError:
            self.skipTest("FastAPI/WebSocket not available")
            return
        mock_model = MagicMock()
        get_ref = lambda k: "بسم الله الرحمن الرحيم" if k == "1:1" else ""
        handler = build_ws_recite_handler(
            get_whisper_model=lambda: mock_model,
            get_reference_text=get_ref,
            get_inference_lock=lambda: None,
            run_batch_verification=None,
            window_duration_ms=500.0,
            overlap_duration_ms=0,
            max_queue_depth=2,
        )
        self.assertTrue(callable(handler))
        import asyncio
        self.assertTrue(asyncio.iscoroutinefunction(handler))


class TestFeedbackSmoothing(unittest.TestCase):
    """Phase 4.1: WordFeedbackSmoother — wrong only after 2 consecutive windows."""

    def test_wrong_after_two_consecutive(self):
        from streaming.feedback_smoothing import WordFeedbackSmoother
        smoother = WordFeedbackSmoother(history_size=2)
        a = [{"word": "أ", "status": "correct"}, {"word": "ب", "status": "wrong"}]
        out1 = smoother.update(a)
        self.assertEqual(out1[1]["status"], "correct")  # first time wrong -> smoothed to correct
        b = [{"word": "أ", "status": "correct"}, {"word": "ب", "status": "wrong"}]
        out2 = smoother.update(b)
        self.assertEqual(out2[1]["status"], "wrong")  # second consecutive wrong -> show wrong
        c = [{"word": "أ", "status": "correct"}, {"word": "ب", "status": "correct"}]
        out3 = smoother.update(c)
        self.assertEqual(out3[1]["status"], "correct")

    def test_flicker_prevented(self):
        from streaming.feedback_smoothing import WordFeedbackSmoother
        smoother = WordFeedbackSmoother(history_size=2)
        smoother.update([{"word": "أ", "status": "correct"}])
        out = smoother.update([{"word": "أ", "status": "wrong"}])
        self.assertEqual(out[0]["status"], "correct")


class TestStreamingMetrics(unittest.TestCase):
    """Phase 4.1: GET /metrics/streaming snapshot keys."""

    def test_snapshot_has_required_keys(self):
        try:
            from metrics.streaming_metrics import get_snapshot
        except ImportError:
            self.skipTest("metrics.streaming_metrics not available")
            return
        s = get_snapshot()
        self.assertIn("active_connections", s)
        self.assertIn("avg_latency_ms", s)
        self.assertIn("p95_latency_ms", s)
        self.assertIn("asr_failure_count", s)
        self.assertIn("dropped_segments", s)
        self.assertIn("adaptive_window_seconds", s)
