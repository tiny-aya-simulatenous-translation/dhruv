#!/usr/bin/env python3
"""
Fetch one English–Hindi parallel speech pair from FLEURS for the overfit experiment.

FLEURS has n-way parallel sentences: the same sentence in many languages (including
en_us and hi_in). We load one pair by index and save as source (English) and
target (Hindi) WAVs at 24 kHz for DualCodec.

Usage (run from DualCodec repo root):
  pip install 'datasets<4.0.0' soundfile   # FLEURS uses legacy script; datasets 4.x dropped support
  python scripts/fetch_en_hi_audio.py [--output_dir data/en_hi_sample] [--index 0]
  python scripts/fetch_en_hi_audio.py --sentence_id 123   # or pick by FLEURS sentence id
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Fetch one En–Hi parallel pair from FLEURS")
    parser.add_argument("--output_dir", type=str, default="data/en_hi_sample", help="Where to save source/target WAVs")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="FLEURS split")
    parser.add_argument("--index", type=int, default=None, help="Nth English sample (0-based); use to try different pairs")
    parser.add_argument("--sentence_id", type=int, default=None, help="FLEURS sentence id (from a previous run); overrides --index")
    parser.add_argument("--source_gain", type=float, default=1.0, help="Multiply source (English) volume by this (e.g. 2.0 to make louder)")
    parser.add_argument("--normalize_peak", type=float, default=None, help="Scale source so peak is this (e.g. 0.95); overrides --source_gain for source")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise SystemExit(f"Install: pip install datasets soundfile\n{e}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLEURS en_us and hi_in (same-sentence parallel)...")
    try:
        en = load_dataset("google/fleurs", "en_us", split=args.split, trust_remote_code=True)
        hi = load_dataset("google/fleurs", "hi_in", split=args.split, trust_remote_code=True)
    except RuntimeError as e:
        if "no longer supported" in str(e) and "script" in str(e).lower():
            raise SystemExit(
                "FLEURS uses a legacy dataset script; Hugging Face datasets 4.x removed support.\n"
                "Install an older version:  pip install 'datasets<4.0.0'\n"
                "Then run this script again."
            ) from e
        raise

    hi_by_id = {row["id"]: row for row in hi}
    en_by_id = {row["id"]: row for row in en}

    if args.sentence_id is not None:
        sentence_id = args.sentence_id
        if sentence_id not in en_by_id:
            raise SystemExit(f"No English sample with id={sentence_id} in {args.split}.")
        if sentence_id not in hi_by_id:
            raise SystemExit(f"No Hindi sample with id={sentence_id} in {args.split}.")
        en_row = en_by_id[sentence_id]
        hi_row = hi_by_id[sentence_id]
    else:
        idx = args.index if args.index is not None else 0
        if idx >= len(en):
            raise SystemExit(f"Index {idx} out of range (en_us {args.split} has {len(en)} samples). Try --index 0 to {len(en)-1}.")
        en_row = en[idx]
        sentence_id = en_row["id"]
        if sentence_id not in hi_by_id:
            raise SystemExit(f"No Hindi sample with id={sentence_id}. Try another --index.")
        hi_row = hi_by_id[sentence_id]

    # Audio: dataset gives dict with "array" and "sampling_rate"
    en_audio = en_row["audio"]
    hi_audio = hi_row["audio"]
    en_sr = en_audio["sampling_rate"]
    hi_sr = hi_audio["sampling_rate"]
    en_arr = np.array(en_audio["array"], dtype=np.float32)
    hi_arr = np.array(hi_audio["array"], dtype=np.float32)

    # Resample to 24 kHz for DualCodec (simple linear interpolation if needed)
    def resample_24k(x, orig_sr):
        if orig_sr == 24000:
            return x
        duration = len(x) / orig_sr
        new_len = int(24000 * duration)
        indices = np.linspace(0, len(x) - 1, new_len)
        return np.interp(indices, np.arange(len(x)), x).astype(np.float32)

    en_24 = resample_24k(en_arr, en_sr)
    hi_24 = resample_24k(hi_arr, hi_sr)

    # Boost source volume if requested
    if args.normalize_peak is not None:
        peak = np.abs(en_24).max()
        if peak > 1e-8:
            en_24 = en_24 * (args.normalize_peak / peak)
        en_24 = np.clip(en_24, -1.0, 1.0).astype(np.float32)
    elif args.source_gain != 1.0:
        en_24 = (en_24 * args.source_gain).astype(np.float32)
        en_24 = np.clip(en_24, -1.0, 1.0).astype(np.float32)

    source_path = out_dir / "source_en.wav"
    target_path = out_dir / "target_hi.wav"
    sf.write(str(source_path), en_24, 24000)
    sf.write(str(target_path), hi_24, 24000)

    print(f"Saved English (source): {source_path}")
    print(f"Saved Hindi (target):   {target_path}")
    print(f"Parallel sentence id:  {sentence_id}")
    print(f"English: {en_row.get('transcription', '')[:80]}...")
    print(f"Hindi:   {hi_row.get('transcription', '')[:80]}...")
    print()
    print("Run overfit script:")
    print(f"  python scripts/overfit_s2s_one_sample.py \\")
    print(f"    --source_wav {source_path} \\")
    print(f"    --target_wav {target_path} \\")
    print(f"    --out_dir overfit_out")


if __name__ == "__main__":
    main()
