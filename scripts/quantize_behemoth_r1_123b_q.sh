#!/bin/bash
# DO NOT RUN THIS. It produces a checkpoint vLLM cannot load.
#
# NVFP4 W4A4 on q_proj + o_proj + gate/up/down, BF16 on k/v. That splits the
# fused QKV layer across two precisions, and vLLM requires one precision per
# fused layer:
#
#   ValueError: Detected some but not all shards of
#   model.layers.0.self_attn.qkv_proj are quantized.
#
# The export is otherwise correct and passes every per-module check, which is
# exactly what makes it a trap. Use quantize_behemoth_r1_123b_qkv.sh instead --
# all of q/k/v in NVFP4, 65 GiB, and it loads. Kept here so the mistake stays
# documented rather than repeatable. See Behemoth-123B_v2_R1.md, "Fused layers
# constrain the scope".
#
# The one useful output was its amax file: k/v amaxes are derivable from it
# exactly, with no recalibration. See 6.4b and tools/synth_kv_amax.py.
set -e
echo "This scope cannot be served by vLLM. See the comment above." >&2
exit 1

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

SRC=/media/fmodels/TheDrummer/Behemoth-R1-123B-v2
WORK=/media/fmodels2/working_Model-Opt/Behemoth-R1-123B-v2-nvfp4-q
FINAL=/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4-q
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
# state, not weights. Keeping them here also means they survive the rename.
AMAX=/media/fmodels2/working_Model-Opt/amax/behemoth_r1_123b_q.safetensors

mkdir -p "$WORK" "$(dirname "$FINAL")" "$(dirname "$AMAX")"

python quantize.py \
    --model behemoth_r1_123b_q \
    --model-id "$SRC" \
    --export-dir "$WORK" \
    --calib-config configs/calib_behemoth_r1_123b.toml \
    --batch-tokens 32768 \
    --save-amax "$AMAX"

# quantize.py checkpoints amaxes into --export-dir every few batches. That is
# scratch state and must not ship inside the model.
if [ -f "$WORK/amax_checkpoint.safetensors" ]; then
    mv "$WORK/amax_checkpoint.safetensors" \
       "$(dirname "$AMAX")/behemoth_r1_123b_q_checkpoint.safetensors"
fi

if [ -e "$FINAL" ]; then
    echo "$FINAL already exists. Export left in $WORK; move it yourself."
    exit 1
fi
mv "$WORK" "$FINAL"
echo "Done: $FINAL"
