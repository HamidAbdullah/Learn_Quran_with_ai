#!/usr/bin/env python3
"""
Test Phase 3 pipeline without audio: phoneme extraction → alignment → tajweed detection → scoring.

Requires: pip install -r requirements.txt (so rapidfuzz and core deps are available).
Run from project root:
  python3 scripts/test_phase3_pipeline.py

Shows phoneme_accuracy, phoneme_aware_score, and tajweed_errors from score_recitation.
"""

import sys

# Ensure project root is on path
sys.path.insert(0, ".")


def main():
    from core.phoneme import verse_to_phoneme_sequence
    from alignment.phoneme_alignment import align_phoneme_sequences
    from tajweed.rules import detect_tajweed_errors, tajweed_rule_accuracy
    from core.scoring import score_recitation

    # Reference (with diacritics) vs user transcript (e.g. normalized or with a small error)
    ref = "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیمِ"
    hyp = "بسم الله الرحمن الرحيم"  # perfect text match

    print("Reference:", ref)
    print("Hypothesis:", hyp)
    print()

    r = verse_to_phoneme_sequence(ref)
    h = verse_to_phoneme_sequence(hyp)
    print("Reference phonemes (first 20):", r["phoneme_sequence"][:20])
    print("Hypothesis phonemes (first 20):", h["phoneme_sequence"][:20])
    print()

    align = align_phoneme_sequences(
        r["phoneme_sequence"],
        h["phoneme_sequence"],
        word_boundaries=r["word_boundaries"],
        reference_words=r["words"],
    )
    print("Phoneme accuracy:", align["phoneme_accuracy"])
    print("Substitutions:", align["num_substitutions"], "Deletions:", align["num_deletions"], "Insertions:", align["num_insertions"])
    print()

    errs = detect_tajweed_errors(align, reference_words=r["words"])
    tacc = tajweed_rule_accuracy(align)
    print("Tajweed rule accuracy:", tacc)
    print("Tajweed errors count:", len(errs))
    for e in errs[:5]:
        print("  -", e.get("rule_name"), "|", e.get("word"), "|", e.get("expected"), "|", e.get("detected"))
    if len(errs) > 5:
        print("  ...")
    print()

    result = score_recitation(
        ref,
        hyp,
        phoneme_accuracy=align["phoneme_accuracy"],
        tajweed_rule_accuracy=tacc,
        tajweed_errors=errs,
    )
    print("score_recitation result (Phase 3 keys):")
    print("  accuracy_score:", result.get("accuracy_score"))
    print("  phoneme_accuracy:", result.get("phoneme_accuracy"))
    print("  tajweed_rule_accuracy:", result.get("tajweed_rule_accuracy"))
    print("  phoneme_aware_score:", result.get("phoneme_aware_score"))
    print("  tajweed_errors in result:", len(result.get("tajweed_errors", [])))
    print()
    print("Phase 3 pipeline OK.")


if __name__ == "__main__":
    main()
