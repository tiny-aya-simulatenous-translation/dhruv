#!/usr/bin/env bash
#
# DualCodec finetuning on Turkish audio data.
#
# Prerequisites:
#   1. pip install -e ".[tts]"   (from the DualCodec repo root)
#   2. huggingface-cli login     (if dataset is private)
#
# Usage:
#   bash scripts/train_turkish.sh <hf_dataset_name>
#
# Examples:
#   bash scripts/train_turkish.sh tiny-aya-translate/tr-subset-v0.1
#
# The script will:
#   1. Download pretrained DualCodec + discriminator weights (if not present)
#   2. Download Wav2Vec2-BERT 2.0 (if not present)
#   3. Launch finetuning on a single GPU
#
set -euo pipefail
cd "$(dirname "$0")/.."   # cd to DualCodec repo root

HF_DATASET="${1:?Usage: bash scripts/train_turkish.sh <hf_dataset_name>}"

# Strip full HuggingFace URL to just org/dataset if user passed a URL
HF_DATASET="${HF_DATASET#https://huggingface.co/datasets/}"

# ──────────────────────────────────────────────
# 0. Make sure Python can find the dualcodec package
# ──────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# ──────────────────────────────────────────────
# 1. Download pretrained models (idempotent)
# ──────────────────────────────────────────────

if [ ! -d "w2v-bert-2.0" ]; then
    echo ">>> Downloading facebook/w2v-bert-2.0 ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/w2v-bert-2.0', local_dir='w2v-bert-2.0')
"
fi

if [ ! -f "dualcodec_ckpts/discriminator_dualcodec_25hz_16384_1024.safetensors" ]; then
    echo ">>> Downloading amphion/dualcodec (with discriminator weights) ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('amphion/dualcodec', local_dir='dualcodec_ckpts', revision='a4243540cfb149e38c82dc80dfa5c83d5e0af2a9')
"
fi

# ──────────────────────────────────────────────
# 2. Launch training
# ──────────────────────────────────────────────

echo ">>> Starting DualCodec Turkish finetuning ..."
echo "    HF Dataset: ${HF_DATASET}"
echo ""

python3 -m accelerate.commands.launch --num_processes 1 train.py \
    --config-name=dualcodec_ft_turkish \
    machine.hf_dataset_name="${HF_DATASET}" \
    data.segment_speech.segment_length=76800
