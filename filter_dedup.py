#!/usr/bin/env python3
"""
Step 4: Filter by DNSMOS thresholds and deduplicate by transcript.

Reads the scored manifest, applies quality gates, removes duplicate
transcripts (keeping the highest-DNSMOS sample), and writes the final manifest.

Usage:
  python filter_dedup.py --config config.yaml
  python filter_dedup.py --config config.yaml --ovrl_threshold 3.2 --bak_threshold 3.0
"""

import argparse
import unicodedata
import re
from pathlib import Path

import yaml

from utils.manifest import read_manifest, write_manifest


def normalize_transcript(text: str) -> str:
    """
    Normalize transcript for deduplication:
    lowercase, strip punctuation, collapse whitespace, unicode normalize.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text


def main():
    parser = argparse.ArgumentParser(description="Filter and deduplicate scored manifest")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ovrl_threshold", type=float, default=None,
                        help="Override DNSMOS OVRL threshold from config")
    parser.add_argument("--bak_threshold", type=float, default=None,
                        help="Override DNSMOS BAK threshold from config")
    parser.add_argument("--skip_dedup", action="store_true",
                        help="Skip transcript deduplication")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ovrl_thresh = args.ovrl_threshold or config.get("dnsmos_ovrl_threshold", 3.0)
    bak_thresh = args.bak_threshold or config.get("dnsmos_bak_threshold", 2.5)

    scored_manifest_path = Path(config["paths"]["scored_manifest"])
    final_manifest_path = Path(config["paths"]["final_manifest"])
    final_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = read_manifest(scored_manifest_path)
    print(f"Input samples: {len(entries)}")
    print(f"Input hours: {sum(e['duration_s'] for e in entries) / 3600:.1f}")

    # Step 1: Quality threshold filter
    filtered = []
    rejected_quality = 0
    for e in entries:
        ovrl = e.get("dnsmos_ovrl", 0)
        bak = e.get("dnsmos_bak", 0)
        if ovrl < ovrl_thresh:
            rejected_quality += 1
            continue
        if bak < bak_thresh:
            rejected_quality += 1
            continue
        filtered.append(e)

    print(f"\nAfter quality filter (OVRL >= {ovrl_thresh}, BAK >= {bak_thresh}):")
    print(f"  Kept: {len(filtered)}, Rejected: {rejected_quality}")
    print(f"  Hours: {sum(e['duration_s'] for e in filtered) / 3600:.1f}")

    # Step 2: Transcript deduplication
    if not args.skip_dedup:
        # Group by normalized transcript, keep highest OVRL per group
        transcript_groups: dict[str, list[dict]] = {}
        no_transcript = []

        for e in filtered:
            transcript = e.get("transcript", "").strip()
            if not transcript:
                no_transcript.append(e)
                continue
            key = normalize_transcript(transcript)
            if not key:
                no_transcript.append(e)
                continue
            transcript_groups.setdefault(key, []).append(e)

        deduped = list(no_transcript)  # keep all samples without transcripts
        duplicates_removed = 0
        for key, group in transcript_groups.items():
            best = max(group, key=lambda x: x.get("dnsmos_ovrl", 0))
            deduped.append(best)
            duplicates_removed += len(group) - 1

        print(f"\nAfter transcript deduplication:")
        print(f"  Unique transcripts: {len(transcript_groups)}")
        print(f"  Duplicates removed: {duplicates_removed}")
        print(f"  Samples without transcript (kept as-is): {len(no_transcript)}")
        print(f"  Final count: {len(deduped)}")
        print(f"  Hours: {sum(e['duration_s'] for e in deduped) / 3600:.1f}")

        final = deduped
    else:
        print("\nSkipping deduplication.")
        final = filtered

    # Sort by dataset then utt_id for reproducibility
    final.sort(key=lambda e: (e["source_dataset"], e["utt_id"]))

    # Write final manifest
    write_manifest(final_manifest_path, final)
    print(f"\nFinal manifest written to: {final_manifest_path}")
    print(f"  Total samples: {len(final)}")
    total_hours = sum(e["duration_s"] for e in final) / 3600
    print(f"  Total hours: {total_hours:.1f}")

    # Per-dataset breakdown
    from collections import Counter

    ds_counts = Counter(e["source_dataset"] for e in final)
    for ds, count in sorted(ds_counts.items()):
        ds_hours = sum(e["duration_s"] for e in final if e["source_dataset"] == ds) / 3600
        print(f"    {ds}: {count} samples, {ds_hours:.1f} hrs")


if __name__ == "__main__":
    main()
