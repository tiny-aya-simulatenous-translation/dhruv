#!/usr/bin/env python3
"""
Step 3: Compute DNSMOS quality scores for all standardized audio.

Runs the DNSMOS P.835 model (ONNX, CPU) on each sample and appends
OVRL, SIG, BAK scores to the manifest.

Usage:
  python score_quality.py --config config.yaml
  python score_quality.py --config config.yaml --num_workers 8
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from utils.dnsmos import DNSMOSScorer, download_dnsmos_model
from utils.manifest import (
    append_manifest,
    get_processed_ids,
    read_manifest,
)


# Global scorer for worker processes
_scorer = None


def _init_scorer(model_dir: str):
    global _scorer
    _scorer = DNSMOSScorer(model_dir=model_dir)  # model already downloaded by main


def _score_one(entry: dict) -> dict | None:
    """Score a single sample. Returns entry with DNSMOS scores added."""
    global _scorer
    try:
        scores = _scorer.score_file(entry["audio_path"])
        return {**entry, **scores}
    except Exception as e:
        print(f"[ERROR] {entry['utt_id']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compute DNSMOS quality scores")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    num_workers = args.num_workers or config.get("num_workers", 8)
    model_dir = config["dnsmos"]["onnx_dir"]

    std_manifest_path = Path(config["paths"]["standardized_manifest"])
    scored_manifest_path = Path(config["paths"]["scored_manifest"])
    scored_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Download model once in main process before spawning workers
    print("Ensuring DNSMOS model is downloaded...")
    download_dnsmos_model(model_dir)

    std_entries = read_manifest(std_manifest_path)
    already_done = get_processed_ids(scored_manifest_path)
    to_process = [e for e in std_entries if e["utt_id"] not in already_done]

    print(f"Standardized samples: {len(std_entries)}")
    print(f"Already scored: {len(already_done)}")
    print(f"To score: {len(to_process)}")

    scored = 0
    failed = 0

    if num_workers <= 1:
        _init_scorer(model_dir)
        for entry in tqdm(to_process, desc="Scoring DNSMOS"):
            result = _score_one(entry)
            if result:
                append_manifest(scored_manifest_path, result)
                scored += 1
            else:
                failed += 1
    else:
        futures = {}
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_scorer,
            initargs=(model_dir,),
        ) as executor:
            for entry in to_process:
                fut = executor.submit(_score_one, entry)
                futures[fut] = entry["utt_id"]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring DNSMOS"):
                try:
                    result = fut.result()
                    if result:
                        append_manifest(scored_manifest_path, result)
                        scored += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[ERROR] {futures[fut]}: {e}")
                    failed += 1

    all_scored = read_manifest(scored_manifest_path)
    print(f"\nScoring complete.")
    print(f"  Scored: {scored}, Failed: {failed}")
    print(f"  Total scored samples: {len(all_scored)}")

    # Quick stats
    if all_scored:
        ovrl_scores = [e["dnsmos_ovrl"] for e in all_scored if "dnsmos_ovrl" in e]
        if ovrl_scores:
            import numpy as np

            arr = np.array(ovrl_scores)
            print(f"  DNSMOS OVRL: mean={arr.mean():.2f}, std={arr.std():.2f}, "
                  f"min={arr.min():.2f}, max={arr.max():.2f}")


if __name__ == "__main__":
    main()
