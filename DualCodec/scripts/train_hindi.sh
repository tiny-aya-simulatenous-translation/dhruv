#!/usr/bin/env bash
#
# DualCodec finetuning on a HuggingFace audio dataset.
#
# Prerequisites:
#   1. pip install -e ".[tts]"   (from the DualCodec repo root)
#   2. huggingface-cli login     (if dataset is private)
#
# Usage:
#   bash scripts/train_hindi.sh <hf_dataset_name>
#
# Examples:
#   bash scripts/train_hindi.sh Pranavz/hinglish-casual
#   bash scripts/train_hindi.sh rumik-ai/hinglish-casual-003
#
# The script will:
#   1. Download pretrained DualCodec + discriminator weights (if not present)
#   2. Download Wav2Vec2-BERT 2.0 (if not present)
#   3. Launch finetuning on a single GPU
#
set -euo pipefail
cd "$(dirname "$0")/.."   # cd to DualCodec repo root

HF_DATASET="${1:?Usage: bash scripts/train_hindi.sh <hf_dataset_name>}"

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

echo ">>> Starting DualCodec finetuning ..."
echo "    HF Dataset: ${HF_DATASET}"
echo ""

# Default: finetune 25Hz model on a single H100 GPU.
#
# To switch to 12Hz model, add these overrides:
#   model=dualcodec_12hz_16384_4096_8vq
#   trainer.args.model_1_name=dualcodec_12hz_16384_4096.safetensors
#   trainer.args.model_2_name=discriminator_dualcodec_12hz_16384_4096.safetensors
#   trainer.exp_name=dualcodec_ft_hindi_12hz
#
# To use a local JSONL manifest instead of HuggingFace:
#   --config-name=dualcodec_ft_hindi
#   data=hindi_jsonl_static_batch
#   machine.manifest_path=/path/to/train.jsonl

accelerate launch --num_processes 1 train.py \
    --config-name=dualcodec_ft_hindi \
    machine.hf_dataset_name="${HF_DATASET}" \
    trainer.batch_size=6 \
    data.segment_speech.segment_length=76800
