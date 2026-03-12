#!/usr/bin/env python3
"""
Step 1: Download Hindi speech datasets and produce a unified raw manifest.

Supports:
  - Mozilla Common Voice Hindi (HuggingFace)
  - IndicVoices-R Hindi (HuggingFace)
  - OpenSLR Hindi (direct download)
  - FLEURS Hindi (HuggingFace, requires datasets<4.0.0)

Usage:
  python download.py --config config.yaml --datasets common_voice_hi indicvoices_r openslr_hindi
  python download.py --config config.yaml --datasets fleurs_hi
  python download.py --config config.yaml --datasets fleurs_hi --max_samples 500
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf
import yaml
from tqdm import tqdm

from utils.manifest import append_manifest, get_processed_ids, write_manifest


def download_common_voice_hi(config: dict, output_dir: Path, manifest_path: Path, max_samples: int = None):
    """Download Common Voice Hindi via HuggingFace datasets library."""
    from datasets import load_dataset

    ds_config = config["datasets"]["common_voice_hi"]
    dataset_dir = output_dir / "common_voice_hi"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    already_done = get_processed_ids(manifest_path)
    print(f"[common_voice_hi] Already processed: {len(already_done)} samples")

    total_added = 0
    for split in ds_config["splits"]:
        print(f"[common_voice_hi] Loading split: {split}")
        ds = load_dataset(
            ds_config["name"],
            "hi",
            split=split,
            trust_remote_code=True,
        )

        for i, row in enumerate(tqdm(ds, desc=f"common_voice_hi/{split}")):
            if max_samples is not None and total_added >= max_samples:
                print(f"[common_voice_hi] Reached --max_samples={max_samples}, stopping.")
                return

            utt_id = f"cv_hi_{split}_{i:06d}"
            if utt_id in already_done:
                continue

            audio = row[ds_config["audio_column"]]
            audio_array = audio["array"]
            sr = audio["sampling_rate"]
            transcript = row.get(ds_config["text_column"], "")

            out_path = dataset_dir / f"{utt_id}.wav"
            sf.write(str(out_path), audio_array, sr)

            entry = {
                "utt_id": utt_id,
                "audio_path": str(out_path),
                "duration_s": round(len(audio_array) / sr, 3),
                "transcript": transcript or "",
                "source_dataset": "common_voice_hi",
                "original_sr": sr,
                "split_source": split,
            }
            append_manifest(manifest_path, entry)
            total_added += 1

    print(f"[common_voice_hi] Done. Added {total_added} samples.")


def download_indicvoices_r(config: dict, output_dir: Path, manifest_path: Path, max_samples: int = None):
    """Download IndicVoices-R Hindi subset via HuggingFace datasets library."""
    from datasets import load_dataset

    ds_config = config["datasets"]["indicvoices_r"]
    dataset_dir = output_dir / "indicvoices_r"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    already_done = get_processed_ids(manifest_path)
    print(f"[indicvoices_r] Already processed: {len(already_done)} samples")

    total_added = 0
    for split in ds_config["splits"]:
        print(f"[indicvoices_r] Loading split: {split} (Hindi only)")
        try:
            ds = load_dataset(
                ds_config["name"],
                "hi",
                split=split,
                trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                ds_config["name"],
                split=split,
                trust_remote_code=True,
            )
            ds = ds.filter(lambda x: x.get("language", "") == "hi")

        for i, row in enumerate(tqdm(ds, desc=f"indicvoices_r/{split}")):
            if max_samples is not None and total_added >= max_samples:
                print(f"[indicvoices_r] Reached --max_samples={max_samples}, stopping.")
                return

            utt_id = f"ivr_hi_{split}_{i:06d}"
            if utt_id in already_done:
                continue

            audio = row[ds_config["audio_column"]]
            audio_array = audio["array"]
            sr = audio["sampling_rate"]
            transcript = row.get(ds_config["text_column"], "")

            out_path = dataset_dir / f"{utt_id}.wav"
            sf.write(str(out_path), audio_array, sr)

            entry = {
                "utt_id": utt_id,
                "audio_path": str(out_path),
                "duration_s": round(len(audio_array) / sr, 3),
                "transcript": transcript or "",
                "source_dataset": "indicvoices_r",
                "original_sr": sr,
                "split_source": split,
            }
            append_manifest(manifest_path, entry)
            total_added += 1

    print(f"[indicvoices_r] Done. Added {total_added} samples.")


def download_openslr_hindi(config: dict, output_dir: Path, manifest_path: Path, max_samples: int = None):
    """
    Download OpenSLR Hindi dataset (resource 103 or similar).

    OpenSLR typically provides tar.gz archives with WAV files and a
    transcription file. Adjust the URL and parsing based on the actual
    resource structure.
    """
    dataset_dir = output_dir / "openslr_hindi"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    already_done = get_processed_ids(manifest_path)
    print(f"[openslr_hindi] Already processed: {len(already_done)} samples")

    # OpenSLR resource 103: Hindi multi-speaker TTS data
    # Adjust URL to the specific resource you want
    base_url = "https://www.openslr.org/resources/103"
    tar_filename = "hindi_train.tar.gz"
    tar_path = dataset_dir / tar_filename

    if not tar_path.exists():
        url = f"{base_url}/{tar_filename}"
        print(f"[openslr_hindi] Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, str(tar_path))
        except Exception as e:
            print(f"[openslr_hindi] Download failed: {e}")
            print("[openslr_hindi] Please manually download from https://www.openslr.org/103/")
            print(f"[openslr_hindi] Place the tar.gz file at: {tar_path}")
            return

    extract_dir = dataset_dir / "extracted"
    if not extract_dir.exists():
        print("[openslr_hindi] Extracting archive...")
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(path=str(extract_dir))

    # Find all wav files and associated transcripts
    wav_files = sorted(extract_dir.rglob("*.wav"))
    transcript_files = list(extract_dir.rglob("*.txt")) + list(extract_dir.rglob("*.tsv"))

    # Try to build a transcript map from available metadata
    transcript_map = {}
    for tf in transcript_files:
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t") if "\t" in line else line.strip().split("|")
                if len(parts) >= 2:
                    file_id = Path(parts[0]).stem
                    transcript_map[file_id] = parts[-1]

    print(f"[openslr_hindi] Found {len(wav_files)} WAV files, {len(transcript_map)} transcripts")

    total_added = 0
    for i, wav_path in enumerate(tqdm(wav_files, desc="openslr_hindi")):
        if max_samples is not None and total_added >= max_samples:
            print(f"[openslr_hindi] Reached --max_samples={max_samples}, stopping.")
            break

        utt_id = f"oslr_hi_{i:06d}"
        if utt_id in already_done:
            continue

        info = sf.info(str(wav_path))
        duration_s = info.duration
        sr = info.samplerate
        transcript = transcript_map.get(wav_path.stem, "")

        # Copy/link to our directory structure
        out_path = dataset_dir / f"{utt_id}.wav"
        if not out_path.exists():
            os.link(str(wav_path), str(out_path))

        entry = {
            "utt_id": utt_id,
            "audio_path": str(out_path),
            "duration_s": round(duration_s, 3),
            "transcript": transcript,
            "source_dataset": "openslr_hindi",
            "original_sr": sr,
            "split_source": "train",
        }
        append_manifest(manifest_path, entry)
        total_added += 1

    print(f"[openslr_hindi] Done. Added {total_added} samples.")


def download_fleurs_hi(config: dict, output_dir: Path, manifest_path: Path, max_samples: int = None):
    """
    Download FLEURS Hindi (hi_in) via HuggingFace datasets library.

    FLEURS uses a legacy dataset script and requires datasets<4.0.0.
    All splits combined yield ~10 hours of 16kHz Hindi speech.
    """
    from datasets import load_dataset

    ds_config = config["datasets"]["fleurs_hi"]
    dataset_dir = output_dir / "fleurs_hi"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    already_done = get_processed_ids(manifest_path)
    print(f"[fleurs_hi] Already processed: {len(already_done)} samples")

    total_added = 0
    for split in ds_config["splits"]:
        print(f"[fleurs_hi] Loading split: {split}")
        try:
            ds = load_dataset(
                ds_config["name"],
                ds_config["language"],
                split=split,
                trust_remote_code=True,
            )
        except RuntimeError as e:
            if "no longer supported" in str(e) and "script" in str(e).lower():
                print(
                    "[fleurs_hi] ERROR: FLEURS uses a legacy dataset script.\n"
                    "  Install: pip install 'datasets<4.0.0'\n"
                    "  Then run this script again."
                )
                return
            raise

        for i, row in enumerate(tqdm(ds, desc=f"fleurs_hi/{split}")):
            if max_samples is not None and total_added >= max_samples:
                print(f"[fleurs_hi] Reached --max_samples={max_samples}, stopping.")
                return

            utt_id = f"fleurs_hi_{split}_{i:06d}"
            if utt_id in already_done:
                continue

            audio = row[ds_config["audio_column"]]
            audio_array = audio["array"]
            sr = audio["sampling_rate"]
            transcript = row.get(ds_config["text_column"], "")

            out_path = dataset_dir / f"{utt_id}.wav"
            sf.write(str(out_path), audio_array, sr)

            entry = {
                "utt_id": utt_id,
                "audio_path": str(out_path),
                "duration_s": round(len(audio_array) / sr, 3),
                "transcript": transcript or "",
                "source_dataset": "fleurs_hi",
                "original_sr": sr,
                "split_source": split,
            }
            append_manifest(manifest_path, entry)
            total_added += 1

    print(f"[fleurs_hi] Done. Added {total_added} samples.")


DATASET_HANDLERS = {
    "common_voice_hi": download_common_voice_hi,
    "indicvoices_r": download_indicvoices_r,
    "openslr_hindi": download_openslr_hindi,
    "fleurs_hi": download_fleurs_hi,
}


def main():
    parser = argparse.ArgumentParser(description="Download Hindi speech datasets")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_HANDLERS.keys()),
        choices=list(DATASET_HANDLERS.keys()),
        help="Which datasets to download",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to download per dataset (useful for trial runs)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["paths"]["raw_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["paths"]["raw_manifest"])

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Downloading: {ds_name}")
        if args.max_samples:
            print(f"  (capped at {args.max_samples} samples)")
        print(f"{'='*60}")
        handler = DATASET_HANDLERS[ds_name]
        handler(config, output_dir, manifest_path, max_samples=args.max_samples)

    from utils.manifest import read_manifest

    entries = read_manifest(manifest_path)
    total_hours = sum(e["duration_s"] for e in entries) / 3600
    print(f"\nTotal samples: {len(entries)}")
    print(f"Total hours (raw): {total_hours:.1f}")
    for ds in args.datasets:
        ds_entries = [e for e in entries if e["source_dataset"] == ds]
        ds_hours = sum(e["duration_s"] for e in ds_entries) / 3600
        print(f"  {ds}: {len(ds_entries)} samples, {ds_hours:.1f} hours")


if __name__ == "__main__":
    main()
