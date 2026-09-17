#!/bin/bash
# NVFP4 for TheDrummer/Behemoth-R1-123B-v2, all linears. ~65 GiB variant.
#
# Scope: NVFP4 W4A4 on q/k/v + o_proj + gate/up/down. BF16 on embeddings,
# lm_head, norms, and the KV cache. See Behemoth-123B_v2_R1.md.
#
# This replaces the q_proj-only variant, which exports cleanly but cannot be
# served: vLLM fuses q/k/v into one layer and rejects mixed precision across its
# shards. Quantizing all three keeps the fused group uniform.
#
# Identical calibration set and method to the 86 GiB run, so a KLD comparison
# between the exports isolates the effect of quantizing attention and nothing
# else. Neither prior amax file is reusable: q/k/v had no quantizers in the
# 86 GiB run, so calibration must run again in full.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

SRC=/media/fmodels/TheDrummer/Behemoth-R1-123B-v2
WORK=/media/fmodels2/working_Model-Opt/Behemoth-R1-123B-v2-nvfp4-qkv
FINAL=/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4-qkv
CALIB=data/text/behemoth_r1_123b_calib

for len in 4096 8192; do
    if [ ! -f "${CALIB}_${len}.jsonl" ]; then
        echo "Missing ${CALIB}_${len}.jsonl — build the calibration set first:"
        echo "  python tools/build_calib_from_yaml.py \\"
        echo "      --yaml data/behemoth_r1_123b_calib.yaml \\"
        echo "      --output ${CALIB}.jsonl --think-tag think"
        exit 1
    fi
done

# Outside $WORK on purpose. $WORK is renamed to $FINAL on success, so anything
# written inside it is published with the model -- and amaxes are calibration
# state, not weights. Keeping them here also means they survive the rename and
# can be found again, which is the whole basis of the shortcut in 6.4b.
AMAX=/media/fmodels2/working_Model-Opt/amax/behemoth_r1_123b_qkv.safetensors

mkdir -p "$WORK" "$(dirname "$FINAL")" "$(dirname "$AMAX")"

python quantize.py \
    --model behemoth_r1_123b_qkv \
    --model-id "$SRC" \
    --export-dir "$WORK" \
    --calib-config configs/calib_behemoth_r1_123b.toml \
    --batch-tokens 32768 \
    --save-amax "$AMAX"

# quantize.py checkpoints amaxes into --export-dir every few batches. That is
# scratch state and must not ship inside the model.
if [ -f "$WORK/amax_checkpoint.safetensors" ]; then
    mv "$WORK/amax_checkpoint.safetensors" \
       "$(dirname "$AMAX")/behemoth_r1_123b_qkv_checkpoint.safetensors"
fi

if [ -e "$FINAL" ]; then
    echo "$FINAL already exists. Export left in $WORK; move it yourself."
    exit 1
fi
mv "$WORK" "$FINAL"
echo "Done: $FINAL"
