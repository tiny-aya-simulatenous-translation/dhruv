#!/usr/bin/env python3
"""
Standalone TTFAT (Time to First Audio Token) measurement.

Requires instantiating actual codec wrappers. Add your codec below.

Usage:
    python run_ttfat.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from benchmark import run_ttfat


def build_codec_wrappers(config: dict) -> dict:
    """
    Instantiate codec wrappers for TTFAT measurement.

    Each team member should add their codec here. Comment out codecs
    that aren't available in the current environment.
    """
    wrappers = {}

    # --- DualCodec (Dhruv) ---
    # Uncomment and set paths when running DualCodec TTFAT:
    #
    # from codec_wrappers.dualcodec_wrapper import DualCodecWrapper
    # wrappers["dualcodec"] = DualCodecWrapper(
    #     checkpoint_path="path/to/checkpoint",
    #     dualcodec_repo="path/to/DualCodec",
    # )

    # --- Mimi (Mayank) ---
    # from codec_wrappers.mimi_wrapper import MimiWrapper
    # wrappers["mimi"] = MimiWrapper(...)

    # --- Kanade (Pranav) ---
    # from codec_wrappers.kanade_wrapper import KanadeWrapper
    # wrappers["kanade"] = KanadeWrapper(...)

    # --- BiCodec (Ananya) ---
    # from codec_wrappers.bicodec_wrapper import BiCodecWrapper
    # wrappers["bicodec"] = BiCodecWrapper(...)

    return wrappers


def main():
    parser = argparse.ArgumentParser(description="TTFAT measurement")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    wrappers = build_codec_wrappers(config)
    if not wrappers:
        print("No codec wrappers configured. Edit build_codec_wrappers() in this file.")
        return

    results = run_ttfat(config, wrappers)

    output_dir = Path(config.get("output_dir", "benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ttfat_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTTFAT results saved to {output_dir / 'ttfat_results.json'}")


if __name__ == "__main__":
    main()
