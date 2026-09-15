#!/bin/bash
# NVFP4 for TheDrummer/Behemoth-R1-123B-v2 on 4x RTX PRO 6000 Blackwell (SM120).
#
# Scope: NVFP4 W4A4 on gate/up/down + o_proj. BF16 on q/k/v, embeddings,
# lm_head, and the KV cache. See Behemoth-123B_v2_R1.md.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Without this, piping this script into tee or a log file makes Python
# block-buffer stdout at 8 KB. tqdm writes to stderr and keeps appearing, so
# the run looks hung after "Loading weights" while every print sits in the
# buffer -- for a 2,138-batch run the progress lines may never flush.
export PYTHONUNBUFFERED=1

SRC=/media/fmodels/TheDrummer/Behemoth-R1-123B-v2
WORK=/media/fmodels2/working_Model-Opt/Behemoth-R1-123B-v2-nvfp4
FINAL=/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4
CALIB=data/text/behemoth_r1_123b_calib

# Two files: 14,704 samples at 4096 and 1,200 long-form at 8192.
for len in 4096 8192; do
    if [ ! -f "${CALIB}_${len}.jsonl" ]; then
        echo "Missing ${CALIB}_${len}.jsonl — build the calibration set first:"
        echo "  python tools/build_calib_from_yaml.py \\"
        echo "      --yaml data/behemoth_r1_123b_calib.yaml \\"
        echo "      --output ${CALIB}.jsonl --think-tag think"
        exit 1
    fi
done

mkdir -p "$WORK" "$(dirname "$FINAL")"

# 229 GB BF16 into 384 GB of VRAM, so no --streaming: accelerate holds the
# whole model and calibration runs at full speed.
# No --floor-amaxes: that flag only patches sparse MoE expert amaxes.
python quantize.py \
    --model behemoth_r1_123b \
    --model-id "$SRC" \
    --export-dir "$WORK" \
    --calib-config configs/calib_behemoth_r1_123b.toml \
    --batch-tokens 32768 \
    --save-amax "$WORK/amax.safetensors"

# Both paths live on /media/fmodels2, so this is a rename, not a copy.
# Guard the re-run case: mv into an existing dir would nest instead of replace.
if [ -e "$FINAL" ]; then
    echo "$FINAL already exists. Export left in $WORK; move it yourself."
    exit 1
fi
mv "$WORK" "$FINAL"
echo "Done: $FINAL"
