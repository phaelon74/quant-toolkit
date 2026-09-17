#!/usr/bin/env python3
"""Check exported NVFP4 scales against the BF16 source, per projection kind.

Written to answer one question: when an export verifies structurally but scores
like noise, are its scales actually right? The structural verifier only proves
scales are finite and non-zero, which a badly wrong scale also is.

The test is constant-free. For a healthy export, weight_scale_2 is a fixed
multiple of the weight's absolute maximum -- amax / (6 * 448) in ModelOpt's NVFP4
double quantization -- so the ratio weight_scale_2 / absmax(source weight) must
be the *same* for every projection. It does not matter what the constant is. Any
projection whose ratio departs from the others has a scale derived differently
from the rest, which is the failure that synthesized or restored amaxes cause.

  --compare  Diff weight_scale_2 against a second export for every module they
             share. Two runs over the same weights and the same amaxes must
             agree bit for bit; a difference localises which modules a resumed
             amax file actually changed.

Usage:
    python tools/audit_nvfp4_scales.py EXPORT_DIR \\
        --source /media/fmodels/TheDrummer/Behemoth-R1-123B-v2

    python tools/audit_nvfp4_scales.py EXPORT_DIR \\
        --source SRC --compare OTHER_EXPORT_DIR
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FUSED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"),
         "gate_up_proj": ("gate_proj", "up_proj")}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("export_dir")
ap.add_argument("--source", required=True, help="BF16 source model directory.")
ap.add_argument("--compare", help="A second export to diff weight_scale_2 against.")
ap.add_argument("--layers", default="0,22,44,66,87",
                help="Comma-separated layer indices to sample.")
args = ap.parse_args()

layers = [int(x) for x in args.layers.split(",")]


def index_of(root):
    root = Path(root)
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        return {k: root / v for k, v in json.loads(idx.read_text())["weight_map"].items()}
    out = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            out.update({k: shard for k in f.keys()})
    return out


def fetch(index, keys):
    """Read a set of tensors, opening each shard once."""
    by_shard = defaultdict(list)
    for k in keys:
        if k in index:
            by_shard[index[k]].append(k)
    out = {}
    for shard, ks in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


exp_idx = index_of(args.export_dir)
src_idx = index_of(args.source)

paths = {(i, kind): f"model.layers.{i}.self_attn.{kind}"
         if kind.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
         else f"model.layers.{i}.mlp.{kind}"
         for i in layers for kind in KINDS}

scales = fetch(exp_idx, [f"{p}.weight_scale_2" for p in paths.values()]
               + [f"{p}.input_scale" for p in paths.values()])
weights = fetch(src_idx, [f"{p}.weight" for p in paths.values()])

# ---------------------------------------------------------------------------
print("=== weight_scale_2 versus source absmax ===")
print(f"{'kind':<12}{'layer':>6}{'weight_scale_2':>17}{'src absmax':>14}{'ratio':>13}")
ratios = defaultdict(list)
missing = []
for (i, kind), p in sorted(paths.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    ws2, w = scales.get(f"{p}.weight_scale_2"), weights.get(f"{p}.weight")
    if ws2 is None or w is None:
        missing.append(p)
        continue
    ws2 = float(ws2.float().reshape(-1)[0])
    absmax = float(w.to(torch.float32).abs().max())
    r = ws2 / absmax if absmax else float("nan")
    ratios[kind].append(r)
    print(f"{kind:<12}{i:>6}{ws2:>17.6g}{absmax:>14.6g}{r:>13.6g}")

if missing:
    print(f"\n  missing from one side: {len(missing)}  e.g. {missing[:3]}")

# ---------------------------------------------------------------------------
print("\n=== ratio consistency (this is the test) ===")
medians = {k: statistics.median(v) for k, v in ratios.items() if v}
if not medians:
    raise SystemExit("No ratios computed; nothing to conclude.")
overall = statistics.median(medians.values())
print(f"{'kind':<12}{'median ratio':>15}{'vs all-kind median':>22}")
suspect = []
for kind in KINDS:
    if kind not in medians:
        continue
    rel = medians[kind] / overall
    flag = "" if 0.98 <= rel <= 1.02 else "   <-- INCONSISTENT"
    if flag:
        suspect.append((kind, rel))
    print(f"{kind:<12}{medians[kind]:>15.6g}{rel:>21.4f}x{flag}")

print(f"\n  all-kind median ratio        {overall:.6g}")
print(f"  implied constant             {1 / overall:.1f}   (ModelOpt NVFP4 uses 6*448 = 2688)")

# ---------------------------------------------------------------------------
print("\n=== input_scale, and fused-group sharing ===")
for fused, members in FUSED.items():
    print(f"  {fused}")
    for i in layers:
        vals = {}
        for m in members:
            t = scales.get(f"{paths[(i, m)]}.input_scale")
            if t is not None:
                vals[m] = float(t.float().reshape(-1)[0])
        if not vals:
            continue
        spread = (max(vals.values()) - min(vals.values())) / max(vals.values())
        note = "identical" if spread < 1e-6 else f"spread {spread:.2%}"
        print(f"    layer {i:<3} " + "  ".join(f"{m.split('_')[0]}={v:.6g}"
                                              for m, v in vals.items()) + f"   {note}")

# ---------------------------------------------------------------------------
if args.compare:
    print(f"\n=== weight_scale_2 diff vs {args.compare} ===")
    other_idx = index_of(args.compare)
    other = fetch(other_idx, [f"{p}.weight_scale_2" for p in paths.values()])
    same, diff, absent = [], [], []
    for (i, kind), p in sorted(paths.items()):
        a, b = scales.get(f"{p}.weight_scale_2"), other.get(f"{p}.weight_scale_2")
        if b is None:
            absent.append(kind)
            continue
        av, bv = float(a.float().reshape(-1)[0]), float(b.float().reshape(-1)[0])
        (same if av == bv else diff).append((kind, i, av, bv))
    print(f"  identical                    {len(same)}")
    print(f"  different                    {len(diff)}")
    print(f"  absent from the other export {len(absent)}"
          + (f"   ({sorted(set(absent))})" if absent else ""))
    for kind, i, av, bv in diff[:12]:
        print(f"    {kind:<11} layer {i:<3} {av:.6g}  vs  {bv:.6g}   "
              f"({bv / av if av else float('nan'):.4f}x)")

# ---------------------------------------------------------------------------
print("\n=== verdict ===")
if suspect:
    print("  Scales are NOT internally consistent. These kinds derive their")
    print("  weight_scale_2 differently from the rest of the model:")
    for kind, rel in suspect:
        print(f"    {kind:<12} {rel:.4f}x the others")
    print("  That is a broken export, not a quantization-quality result.")
else:
    print("  Every projection shares one weight_scale_2 / absmax ratio, so the")
    print("  scales are internally consistent and the amaxes behaved. A bad")
    print("  score is then about quantization or serving, not scale derivation.")
