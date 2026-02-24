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

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import wer, cer
from core.normalization import normalize_arabic


def load_dataset(path: str, limit: int = None):
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
    """Run dual ASR and return selected transcript. Requires app context (whisper_model, phonetic_analyzer)."""
    try:
        import main as app
        from core.asr import run_dual_asr
        asr_result = run_dual_asr(
            whisper_model=app.whisper_model,
            phonetic_analyzer=app.phonetic_analyzer,
            reference_text=reference,
            audio_path=audio_path,
            audio_y_sr=None,
        )
        return asr_result["selected_transcript"]
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
    parser.add_argument("--self-test", action="store_true", help="Use reference as hypothesis (expect 0 WER/CER)")
    args = parser.parse_args()

    base_dir = args.audio_base_dir or os.environ.get("AUDIO_BASE_DIR", "")
    items = load_dataset(args.dataset, args.limit)

    wers, cers = [], []
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
        wers.append(w)
        cers.append(c)
        key = item.get("verse_key") or item.get("audio") or str(i)
        print(f"  {key}: WER={w:.4f} CER={c:.4f}")

    if not wers:
        print("No items with reference and hypothesis. Add 'hypothesis' to JSON or use --run-asr with valid audio paths.")
        return 1
    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)
    print(f"\nProcessed {len(wers)} items.")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
