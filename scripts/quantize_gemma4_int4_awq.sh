#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize_int4.py \
    --model gemma4_31b \
    --algorithm awq \
    --export-dir ./output/Gemma-4-31B-it-INT4-AWQ \
    --calib-config configs/calib_gemma4_31b.toml \
    --cpu-capacity 200GiB
