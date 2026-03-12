#!/usr/bin/env python3
"""
Step 2: Standardize raw audio files.

For each sample in the raw manifest:
  1. Load audio (supports mp3, wav, flac, ogg)
  2. Convert to mono
  3. Resample to 24kHz
  4. Trim leading/trailing silence
  5. Loudness normalize to -23 LUFS
  6. Duration gate: skip if < 3s or > 45s
  7. Peak-clip to [-1.0, 1.0]
  8. Save as 24kHz 16-bit mono WAV

Usage:
  python standardize.py --config config.yaml
  python standardize.py --config config.yaml --num_workers 16
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import yaml
from tqdm import tqdm

from utils.audio import (
    get_duration,
    load_audio,
    loudness_normalize,
    resample_audio,
    to_mono,
    trim_silence,
)
from utils.manifest import (
    append_manifest,
    get_processed_ids,
    read_manifest,
)


def process_one_sample(
    entry: dict,
    target_sr: int,
    target_lufs: float,
    min_dur: float,
    max_dur: float,
    trim_db: float,
    do_trim: bool,
    output_dir: Path,
) -> dict | None:
    """Process a single sample. Returns updated manifest entry or None if skipped."""
    utt_id = entry["utt_id"]
    audio_path = entry["audio_path"]

    try:
        waveform, sr = load_audio(audio_path)
    except Exception as e:
        print(f"[SKIP] {utt_id}: failed to load ({e})")
        return None

    waveform = to_mono(waveform)
    waveform = resample_audio(waveform, sr, target_sr)

    if do_trim:
        waveform = trim_silence(waveform, target_sr, db_threshold=trim_db)

    dur = get_duration(waveform, target_sr)
    if dur < min_dur:
        return None
    if dur > max_dur:
        return None

    waveform = loudness_normalize(waveform, target_sr, target_lufs)

    # Peak-clip safety
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak * 0.99

    out_path = output_dir / f"{utt_id}.wav"
    sf.write(str(out_path), waveform.squeeze(0).numpy(), target_sr, subtype="PCM_16")

    return {
        "utt_id": utt_id,
        "audio_path": str(out_path),
        "duration_s": round(dur, 3),
        "transcript": entry.get("transcript", ""),
        "source_dataset": entry["source_dataset"],
        "original_sr": entry.get("original_sr", sr),
        "sample_rate": target_sr,
    }


def main():
    parser = argparse.ArgumentParser(description="Standardize raw audio")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    target_sr = config["sample_rate"]
    target_lufs = config["loudness_lufs"]
    min_dur = config["min_duration_s"]
    max_dur = config["max_duration_s"]
    trim_db = config.get("trim_db_threshold", 30)
    do_trim = config.get("trim_silence", True)
    num_workers = args.num_workers or config.get("num_workers", 8)

    raw_manifest_path = Path(config["paths"]["raw_manifest"])
    std_manifest_path = Path(config["paths"]["standardized_manifest"])
    output_dir = Path(config["paths"]["standardized_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_entries = read_manifest(raw_manifest_path)
    already_done = get_processed_ids(std_manifest_path)
    to_process = [e for e in raw_entries if e["utt_id"] not in already_done]

    print(f"Raw samples: {len(raw_entries)}")
    print(f"Already standardized: {len(already_done)}")
    print(f"To process: {len(to_process)}")
    print(f"Settings: {target_sr}Hz, {target_lufs} LUFS, duration [{min_dur}-{max_dur}]s")

    accepted = 0
    skipped = 0

    if num_workers <= 1:
        for entry in tqdm(to_process, desc="Standardizing"):
            result = process_one_sample(
                entry, target_sr, target_lufs, min_dur, max_dur, trim_db, do_trim, output_dir
            )
            if result:
                append_manifest(std_manifest_path, result)
                accepted += 1
            else:
                skipped += 1
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for entry in to_process:
                fut = executor.submit(
                    process_one_sample,
                    entry, target_sr, target_lufs, min_dur, max_dur, trim_db, do_trim, output_dir,
                )
                futures[fut] = entry["utt_id"]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Standardizing"):
                try:
                    result = fut.result()
                    if result:
                        append_manifest(std_manifest_path, result)
                        accepted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"[ERROR] {futures[fut]}: {e}")
                    skipped += 1

    total_entries = read_manifest(std_manifest_path)
    total_hours = sum(e["duration_s"] for e in total_entries) / 3600
    print(f"\nStandardization complete.")
    print(f"  Accepted: {accepted}, Skipped: {skipped}")
    print(f"  Total standardized samples: {len(total_entries)}")
    print(f"  Total hours: {total_hours:.1f}")


if __name__ == "__main__":
    main()
