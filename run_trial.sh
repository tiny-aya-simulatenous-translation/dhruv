#!/usr/bin/env bash
# Trial run: full pipeline on FLEURS Hindi (~10 hours)
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Usage:
#   bash run_trial.sh
#
# Estimated time: ~15-25 min on a Mac (CPU-only)
# Estimated disk: ~4 GB peak

set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo " Step 1: Download FLEURS Hindi"
echo "=========================================="
python3 download.py --config config.yaml --datasets fleurs_hi

echo ""
echo "=========================================="
echo " Step 2: Standardize (24kHz, -23 LUFS)"
echo "=========================================="
python3 standardize.py --config config.yaml --num_workers 4

echo ""
echo "=========================================="
echo " Step 3: Score DNSMOS quality"
echo "=========================================="
python3 score_quality.py --config config.yaml --num_workers 4

echo ""
echo "=========================================="
echo " Step 4: Filter + Deduplicate"
echo "=========================================="
python3 filter_dedup.py --config config.yaml

echo ""
echo "=========================================="
echo " Step 5: Package for HuggingFace"
echo "=========================================="
python3 package.py --config config.yaml

echo ""
echo "=========================================="
echo " Trial complete!"
echo "=========================================="
echo "Output manifest: data/final/manifest.jsonl"
echo "HF dataset:      data/final/hf_dataset/"
echo ""
echo "Next: open notebooks/quality_analysis.ipynb to inspect score distributions."
