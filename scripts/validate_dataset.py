#!/usr/bin/env python3
"""
Validate a dataset JSON: check audio_path, reference_text, missing files, corrupted audio, empty transcripts.
Output: validation report JSON.

Usage:
  python scripts/validate_dataset.py dataset/fatiha_dataset.json
  python scripts/validate_dataset.py dataset/training_dataset.json --check-audio --limit 100
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.dataset_validation import validate_dataset, ValidationReport


def main():
    parser = argparse.ArgumentParser(description="Validate dataset JSON")
    parser.add_argument("dataset", help="Path to dataset JSON")
    parser.add_argument("--audio-base-dir", default=None, help="Base dir for relative audio paths")
    parser.add_argument("--check-audio", action="store_true", help="Try loading audio to detect corruption")
    parser.add_argument("--limit", type=int, default=None, help="Max items to validate")
    parser.add_argument("--output", "-o", default=None, help="Write report to JSON file")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Error: dataset must be a list", file=sys.stderr)
        return 1
    items = data[:args.limit] if args.limit else data

    report = validate_dataset(
        items,
        audio_base_dir=args.audio_base_dir or os.environ.get("AUDIO_BASE_DIR", ""),
        check_audio_load=args.check_audio,
    )

    out = report.to_dict()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
