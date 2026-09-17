#!/usr/bin/env python3
"""Extend a q_proj-scope amax file to cover k_proj and v_proj, without calibrating.

The q_proj-only export cannot be served -- vLLM fuses q/k/v into one layer and
rejects mixed precision across its shards -- so the scope has to widen to all of
q/k/v. Recalibrating for that would be another full pass over 16k samples, and
it is unnecessary, because both amaxes k/v need are exactly recoverable from the
run that already finished:

  input_quantizer   q_proj, k_proj and v_proj all read the *same* tensor, the
                    output of input_layernorm. A max-calibrated amax over the
                    same data and the same tensor is the same number. This is an
                    identity, not an approximation.

  weight_quantizer  Weight amax is an absmax over stored weights. It never
                    depended on calibration data at all, and is read straight
                    out of the BF16 source checkpoint.

Feed the result to quantize.py as --resume-amax with --resume-batch past the
batch count, and calibration is skipped entirely.

Nothing here is guessed: tensor shape conventions are read off the q_proj
entries in the input file, and the tool refuses to write if it cannot confirm
them or if any expected key is missing.

Usage:
    python tools/synth_kv_amax.py --amax WORK/amax.safetensors \\
        --model /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \\
        --output WORK/amax_qkv.safetensors

    python tools/synth_kv_amax.py --amax WORK/amax.safetensors --inspect
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

QUANTIZERS = ("input_quantizer", "weight_quantizer")
KEY_RE = re.compile(r"^(?P<layer>.*\.layers\.(?P<idx>\d+))\.(?P<mod>.*?)\.(?P<q>\w+_quantizer)$")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--amax", required=True, help="amax.safetensors from the q_proj run.")
ap.add_argument("--model", help="BF16 source dir, for k/v weight amaxes.")
ap.add_argument("--output", help="Where to write the extended amax file.")
ap.add_argument("--inspect", action="store_true",
                help="Report the file's structure and exit without writing.")
args = ap.parse_args()

saved = load_file(args.amax)
print(f"Loaded {len(saved)} amax tensors from {args.amax}")

# ---------------------------------------------------------------------------
# What is actually in the file.
by_module = defaultdict(dict)
unparsed = []
for key, val in saved.items():
    m = KEY_RE.match(key)
    if not m:
        unparsed.append(key)
        continue
    by_module[(int(m["idx"]), m["mod"])][m["q"]] = val

kinds = defaultdict(list)
for (idx, mod), quants in by_module.items():
    kinds[mod.split(".")[-1]].append((idx, quants))

print("\n=== Structure ===")
for kind in sorted(kinds):
    entries = kinds[kind]
    quant_names = sorted({q for _, qs in entries for q in qs})
    shapes = {q: tuple(qs[q].shape) for _, qs in entries[:1] for q in qs}
    print(f"  {kind:<12} {len(entries):>4} layer(s)  {quant_names}")
    print(f"               shapes {shapes}")
if unparsed:
    print(f"  unparsed keys: {len(unparsed)}  e.g. {unparsed[:3]}")

if args.inspect:
    raise SystemExit(0)

if not args.model or not args.output:
    raise SystemExit("--model and --output are required unless --inspect is given.")

# ---------------------------------------------------------------------------
# Locate the q_proj entries that k/v will be derived from.
q_modules = {idx: (mod, quants) for (idx, mod), quants in by_module.items()
             if mod.endswith("q_proj")}
if not q_modules:
    raise SystemExit("No q_proj quantizers in this amax file. It is not from a "
                     "q_proj-scope run, so there is nothing to extend.")

FAIL = []
for idx, (mod, quants) in sorted(q_modules.items()):
    missing = [q for q in QUANTIZERS if q not in quants]
    if missing:
        FAIL.append(f"layer {idx} q_proj is missing {missing}")
for (idx, mod) in by_module:
    if mod.endswith(("k_proj", "v_proj")):
        FAIL.append(f"layer {idx} {mod} already has amaxes; refusing to overwrite")
if FAIL:
    for f in FAIL[:10]:
        print(f"  FAIL  {f}")
    raise SystemExit(f"\n{len(FAIL)} problem(s). Nothing written.")

print(f"\nq_proj amaxes found for {len(q_modules)} layer(s).")

# ---------------------------------------------------------------------------
# How is a weight amax laid out? Read it off q_proj rather than assuming. NVFP4
# double-quantizes: a per-tensor amax sets weight_scale_2, while the per-16-block
# FP8 scales are derived from the weight at quantize time and never live here.
# A per-output-channel layout is handled too, in case the preset changes.
ref = next(iter(q_modules.values()))[1]["weight_quantizer"]
if ref.numel() == 1:
    reduce_dim, layout = None, "per-tensor"
elif ref.dim() == 2 and ref.shape[1] == 1:
    reduce_dim, layout = 1, "per-output-channel"
else:
    raise SystemExit(f"Unrecognised weight amax layout {tuple(ref.shape)}. "
                     f"Refusing to guess; nothing written.")
print(f"Weight amax layout: {layout}  shape {tuple(ref.shape)}  dtype {ref.dtype}")


def weight_amax(w):
    a = w.to(torch.float32).abs()
    if reduce_dim is None:
        return a.max().reshape(ref.shape)
    return a.amax(dim=reduce_dim, keepdim=True)


# ---------------------------------------------------------------------------
root = Path(args.model)
index = root / "model.safetensors.index.json"
if index.exists():
    weight_map = json.loads(index.read_text())["weight_map"]
else:
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors found in {root}")
    weight_map = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            weight_map.update({k: shard.name for k in f.keys()})

wanted = {}
for idx, (q_mod, _) in sorted(q_modules.items()):
    for proj in ("k_proj", "v_proj"):
        mod = q_mod[: -len("q_proj")] + proj
        key = f"model.layers.{idx}.{mod}.weight"
        if key not in weight_map:
            raise SystemExit(f"{key} not present in the source checkpoint. "
                             f"Nothing written.")
        wanted[(idx, mod)] = key

print(f"\nReading {len(wanted)} k/v weight tensors from {len(set(weight_map[k] for k in wanted.values()))} shard(s)...")

by_shard = defaultdict(list)
for target, key in wanted.items():
    by_shard[weight_map[key]].append((target, key))

new = {}
for shard, items in sorted(by_shard.items()):
    with safe_open(root / shard, framework="pt") as f:
        for (idx, mod), key in items:
            new[f"model.layers.{idx}.{mod}.weight_quantizer"] = weight_amax(f.get_tensor(key))
    print(f"  {shard}  {len(items)} tensor(s)")

# The identity this whole tool rests on: one input tensor, three projections.
for idx, (q_mod, quants) in sorted(q_modules.items()):
    src = quants["input_quantizer"]
    for proj in ("k_proj", "v_proj"):
        mod = q_mod[: -len("q_proj")] + proj
        new[f"model.layers.{idx}.{mod}.input_quantizer"] = src.clone()

# ---------------------------------------------------------------------------
out = dict(saved)
out.update(new)
bad = {k: v for k, v in new.items()
       if not torch.isfinite(v).all() or (v == 0).all()}
if bad:
    for k in list(bad)[:5]:
        print(f"  FAIL  {k} is zero or non-finite")
    raise SystemExit(f"\n{len(bad)} synthesized amax(es) unusable. Nothing written.")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
save_file({k: (v.reshape(1) if v.dim() == 0 else v) for k, v in out.items()}, args.output)

print(f"\n=== Written ===")
print(f"  {args.amax}")
print(f"    existing                 {len(saved)}")
print(f"    synthesized k/v weight   {len(wanted)}")
print(f"    synthesized k/v input    {len(new) - len(wanted)}")
print(f"    total                    {len(out)}")
print(f"  -> {args.output}")
print(f"\nNext: quantize with --model behemoth_r1_123b_qkv --resume-amax "
      f"{args.output} --resume-batch 999999 (any number past the batch count "
      f"skips calibration; every amax is already present).")
