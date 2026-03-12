#!/usr/bin/env python3
"""
Step 5: Package the filtered dataset for HuggingFace.

Reads the final manifest, creates a train/val split, converts to
HuggingFace datasets format with Audio feature, and optionally pushes
to the Hub.

Usage:
  python package.py --config config.yaml
  python package.py --config config.yaml --push --hub_repo "your-org/hindi-speech-24k"
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml

from utils.manifest import read_manifest


def stratified_split(entries: list[dict], val_ratio: float = 0.05, seed: int = 42):
    """Split entries into train/val, stratified by source_dataset."""
    rng = random.Random(seed)

    by_dataset = defaultdict(list)
    for e in entries:
        by_dataset[e["source_dataset"]].append(e)

    train, val = [], []
    for ds_name, ds_entries in by_dataset.items():
        rng.shuffle(ds_entries)
        n_val = max(1, int(len(ds_entries) * val_ratio))
        val.extend(ds_entries[:n_val])
        train.extend(ds_entries[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def entries_to_parquet(entries: list[dict], output_path: Path):
    """Save manifest entries as a Parquet file with metadata columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = {
        "audio_path": [e["audio_path"] for e in entries],
        "utt_id": [e["utt_id"] for e in entries],
        "duration_s": [e["duration_s"] for e in entries],
        "transcript": [e.get("transcript", "") for e in entries],
        "source_dataset": [e["source_dataset"] for e in entries],
        "dnsmos_ovrl": [e.get("dnsmos_ovrl", 0.0) for e in entries],
        "dnsmos_sig": [e.get("dnsmos_sig", 0.0) for e in entries],
        "dnsmos_bak": [e.get("dnsmos_bak", 0.0) for e in entries],
    }
    table = pa.table(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))


def build_hf_dataset(entries: list[dict], split_name: str):
    """Convert manifest entries to a HuggingFace Dataset with Audio feature."""
    from datasets import Audio, Dataset

    records = []
    for e in entries:
        records.append({
            "audio": e["audio_path"],
            "utt_id": e["utt_id"],
            "duration_s": e["duration_s"],
            "transcript": e.get("transcript", ""),
            "source_dataset": e["source_dataset"],
            "dnsmos_ovrl": e.get("dnsmos_ovrl", 0.0),
            "dnsmos_sig": e.get("dnsmos_sig", 0.0),
            "dnsmos_bak": e.get("dnsmos_bak", 0.0),
        })

    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=24000))
    return ds


def main():
    parser = argparse.ArgumentParser(description="Package dataset for HuggingFace")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub_repo", type=str, default=None,
                        help="HuggingFace repo ID (e.g., 'your-org/hindi-speech-24k')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save dataset locally to this directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    final_manifest_path = Path(config["paths"]["final_manifest"])
    entries = read_manifest(final_manifest_path)

    if not entries:
        print("No entries in final manifest. Run filter_dedup.py first.")
        return

    print(f"Final manifest: {len(entries)} samples, "
          f"{sum(e['duration_s'] for e in entries) / 3600:.1f} hours")

    # Split
    train_entries, val_entries = stratified_split(entries, val_ratio=args.val_ratio)
    print(f"Train: {len(train_entries)} samples, "
          f"{sum(e['duration_s'] for e in train_entries) / 3600:.1f} hrs")
    print(f"Val:   {len(val_entries)} samples, "
          f"{sum(e['duration_s'] for e in val_entries) / 3600:.1f} hrs")

    output_dir = Path(args.output_dir or str(Path(config["paths"]["final_dir"]) / "hf_dataset"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split manifests as JSONL
    from utils.manifest import write_manifest

    write_manifest(output_dir / "train.jsonl", train_entries)
    write_manifest(output_dir / "validation.jsonl", val_entries)

    # Save as Parquet (always works, no dill/pickle dependency)
    print("\nSaving Parquet files...")
    entries_to_parquet(train_entries, output_dir / "train.parquet")
    entries_to_parquet(val_entries, output_dir / "validation.parquet")
    print(f"  {output_dir / 'train.parquet'}")
    print(f"  {output_dir / 'validation.parquet'}")

    # Try HF Dataset format (may fail on Python 3.14 due to dill/pickle incompatibility)
    try:
        print("\nBuilding HuggingFace Dataset objects...")
        train_ds = build_hf_dataset(train_entries, "train")
        val_ds = build_hf_dataset(val_entries, "validation")

        from datasets import DatasetDict

        dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})
        print(dataset_dict)

        hf_dir = output_dir / "arrow"
        print(f"Saving HF Arrow format to {hf_dir}...")
        dataset_dict.save_to_disk(str(hf_dir))
        print("Saved.")

        if args.push:
            if not args.hub_repo:
                print("ERROR: --hub_repo is required when using --push")
                return
            print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
            dataset_dict.push_to_hub(args.hub_repo, private=True)
            print(f"Done! Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")

    except Exception as e:
        print(f"\n[WARN] HF Dataset creation failed ({type(e).__name__}: {e})")
        print("  This is a known dill/pickle issue on Python 3.14.")
        print("  Parquet + JSONL files were saved successfully -- use those instead.")
        print("  To create HF Arrow datasets, re-run with Python 3.12 or 3.13.")

    print(f"\nPackaging complete. Output directory: {output_dir}")
    print(f"  train.jsonl / train.parquet: {len(train_entries)} samples")
    print(f"  validation.jsonl / validation.parquet: {len(val_entries)} samples")


if __name__ == "__main__":
    main()
