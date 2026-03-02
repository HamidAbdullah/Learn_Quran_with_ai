#!/usr/bin/env python3
"""
Benchmark WER and CER for Quran ASR (Phase 2).

Usage:
  # Compute WER/CER from a dataset JSON that has "text_uthmani" and optional "hypothesis" or "transcript"
  python scripts/benchmark_wer_cer.py dataset/fatiha_dataset.json

  # With audio: run full ASR and then compute metrics (requires models; set AUDIO_BASE_DIR if paths are relative)
  AUDIO_BASE_DIR=/path/to/audio python scripts/benchmark_wer_cer.py dataset/fatiha_dataset.json --run-asr

  # Use training_dataset.json format (audio path, text, normalized_text)
  python scripts/benchmark_wer_cer.py dataset/training_dataset.json --limit 20

Expects JSON:
  - List of {"text_uthmani": "...", "audio": "path/or/url", ...} or
  - List of {"text": "...", "normalized_text": "...", "audio": "...", ...}
  - Optional "hypothesis" or "transcript" for precomputed ASR (no model run).
"""
import argparse
import json
import os
import sys
import warnings

# Suppress Whisper FP16-on-CPU warning (expected when not using GPU)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import wer, cer
from core.normalization import normalize_arabic


def load_dataset(path: str, limit: int = None):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Use a real path to a JSON file, e.g.:\n"
            "  dataset/training_dataset.json   (create via scripts/generate_training_dataset.py)\n"
            "  quran_with_audio.json           (if you have it, with audio_file + text_uthmani)\n"
            "Example: python scripts/benchmark_wer_cer.py dataset/training_dataset.json --run-asr --limit 20 -o report.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of items")
    items = data[:limit] if limit else data
    return items


def get_reference(item: dict) -> str:
    return (
        item.get("text_uthmani")
        or item.get("text")
        or item.get("normalized_text")
        or ""
    ).strip()


def get_hypothesis(item: dict) -> str:
    return (item.get("hypothesis") or item.get("transcript") or "").strip()


def run_asr_for_item(audio_path: str, reference: str):
    """Run dual ASR (beam search, Quran Wav2Vec2, Whisper small). Loads audio once to WAV so we decode MP3 only once and can skip bad files."""
    import tempfile
    try:
        import librosa
        import soundfile as sf
        # Decode audio once (if MP3 is malformed we fail here and skip; avoids decoding twice)
        try:
            y, sr = librosa.load(audio_path, sr=16000)
        except Exception:
            print(f"  [skip] Could not load audio: {audio_path}", file=sys.stderr)
            return ""
        if y is None or len(y) == 0:
            return ""
        # Temp WAV so both Whisper and Wav2Vec2 use the same decoded audio (no second MP3 read)
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(fd)
            sf.write(wav_path, y, sr)
            import main as app
            from core.asr import run_dual_asr
            whisper_beam_size = getattr(app, "WHISPER_BEAM_SIZE", 5)
            asr_result = run_dual_asr(
                whisper_model=app.whisper_model,
                phonetic_analyzer=app.phonetic_analyzer,
                reference_text=reference,
                audio_path=wav_path,
                audio_y_sr=(y, sr),
                whisper_beam_size=whisper_beam_size,
            )
            return asr_result["selected_transcript"]
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass
    except Exception as e:
        print(f"ASR error for {audio_path}: {e}", file=sys.stderr)
        return ""


def resolve_audio_path(item: dict, audio_base_dir: str = None) -> str:
    raw = item.get("audio") or item.get("audio_file") or ""
    if not raw:
        return ""
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    if audio_base_dir:
        return os.path.join(audio_base_dir, raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description="Benchmark WER/CER for Quran ASR")
    parser.add_argument("dataset", help="Path to dataset JSON (list of {text_uthmani, audio?, hypothesis?})")
    parser.add_argument("--run-asr", action="store_true", help="Run ASR for each item (requires models)")
    parser.add_argument("--audio-base-dir", default=None, help="Base directory for relative audio paths")
    parser.add_argument("--limit", type=int, default=None, help="Max number of items to process")
    parser.add_argument("--output", "-o", default=None, help="Write report JSON to file")
    parser.add_argument("--self-test", action="store_true", help="Use reference as hypothesis (expect 0 WER/CER)")
    args = parser.parse_args()

    base_dir = args.audio_base_dir or os.environ.get("AUDIO_BASE_DIR", "")
    items = load_dataset(args.dataset, args.limit)

    indexed = []  # (verse_key, wer, cer)
    for i, item in enumerate(items):
        ref = get_reference(item)
        if not ref:
            continue
        hyp = get_hypothesis(item)
        if args.self_test:
            hyp = ref
        elif not hyp and args.run_asr:
            audio_path = resolve_audio_path(item, base_dir)
            if audio_path and os.path.isfile(audio_path):
                hyp = run_asr_for_item(audio_path, ref)
        if not hyp:
            continue
        w = wer(ref, hyp)
        c = cer(ref, hyp)
        key = item.get("verse_key") or item.get("audio") or str(i)
        indexed.append((key, w, c))
        print(f"  {key}: WER={w:.4f} CER={c:.4f}")

    if not indexed:
        print("No items with reference and hypothesis. Add 'hypothesis' to JSON or use --run-asr with valid audio paths.")
        return 1
    n = len(indexed)
    avg_wer = sum(x[1] for x in indexed) / n
    avg_cer = sum(x[2] for x in indexed) / n
    worst_10 = sorted(indexed, key=lambda x: -x[1])[:10]
    best_10 = sorted(indexed, key=lambda x: x[1])[:10]

    print(f"\n{'='*60}")
    print("ASR BENCHMARK REPORT (Quran high-accuracy mode)")
    print(f"{'='*60}")
    print(f"Processed: {n} items")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"\n--- Worst 10 verses (by WER) ---")
    for sid, w, c in worst_10:
        print(f"  {sid}: WER={w:.4f} CER={c:.4f}")
    print(f"\n--- Best 10 verses (by WER) ---")
    for sid, w, c in best_10:
        print(f"  {sid}: WER={w:.4f} CER={c:.4f}")
    print(f"{'='*60}\n")

    if args.output:
        report = {
            "n_samples": n,
            "avg_wer": round(avg_wer, 4),
            "avg_cer": round(avg_cer, 4),
            "worst_10_by_wer": [{"verse_key": s, "wer": round(w, 4), "cer": round(c, 4)} for s, w, c in worst_10],
            "best_10_by_wer": [{"verse_key": s, "wer": round(w, 4), "cer": round(c, 4)} for s, w, c in best_10],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
