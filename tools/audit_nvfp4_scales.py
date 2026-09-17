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

MEMBER_OF = {m: g for g, ms in FUSED.items() for m in ms}

# ---------------------------------------------------------------------------
# The basis a projection's weight_scale_2 should be derived from. For a fused
# group that is the largest absmax in the group, not the projection's own: the
# group shares one scale at serve time, so quantize.py ties their amaxes to the
# max. Measuring each projection against its own absmax makes a correctly tied
# group look wrong and an incorrectly untied one look right, which is exactly
# backwards.
absmax = {}
for (i, kind), p in paths.items():
    w = weights.get(f"{p}.weight")
    if w is not None:
        absmax[(i, kind)] = float(w.to(torch.float32).abs().max())

basis = {}
for (i, kind) in absmax:
    group = MEMBER_OF.get(kind)
    if group:
        peers = [absmax[(i, m)] for m in FUSED[group] if (i, m) in absmax]
        basis[(i, kind)] = max(peers) if peers else absmax[(i, kind)]
    else:
        basis[(i, kind)] = absmax[(i, kind)]

print("=== weight_scale_2 versus the basis it should derive from ===")
print("  (fused groups share one scale, so their basis is the group's max absmax)")
print(f"\n{'kind':<12}{'layer':>6}{'weight_scale_2':>17}{'own absmax':>13}"
      f"{'basis':>13}{'ratio':>13}")
ratios = defaultdict(list)
missing = []
for (i, kind), p in sorted(paths.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    ws2 = scales.get(f"{p}.weight_scale_2")
    if ws2 is None or (i, kind) not in absmax:
        missing.append(p)
        continue
    ws2 = float(ws2.float().reshape(-1)[0])
    b = basis[(i, kind)]
    r = ws2 / b if b else float("nan")
    ratios[kind].append(r)
    mark = "" if abs(absmax[(i, kind)] - b) < 1e-12 else "  (tied up)"
    print(f"{kind:<12}{i:>6}{ws2:>17.6g}{absmax[(i, kind)]:>13.6g}"
          f"{b:>13.6g}{r:>13.6g}{mark}")

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
def group_spread(kind_of_scale):
    """Per fused group and layer, how far apart its members' scales are."""
    rows, untied = [], defaultdict(int)
    for fused, members in FUSED.items():
        for i in layers:
            vals = {}
            for m in members:
                t = scales.get(f"{paths[(i, m)]}.{kind_of_scale}")
                if t is not None:
                    vals[m] = float(t.float().reshape(-1)[0])
            if len(vals) < 2:
                continue
            hi = max(vals.values())
            spread = (hi - min(vals.values())) / hi if hi else 0.0
            if spread > 1e-6:
                untied[fused] += 1
            rows.append((fused, i, vals, spread))
    return rows, untied


print("\n=== fused groups must share weight_scale_2 ===")
print("  vLLM serves each group from one concatenated tensor with one scale.")
ws2_rows, ws2_untied = group_spread("weight_scale_2")
for fused, i, vals, spread in ws2_rows:
    note = "tied" if spread <= 1e-6 else f"UNTIED  spread {spread:.1%}"
    print(f"  {fused:<14} layer {i:<3} "
          + "  ".join(f"{m.split('_')[0]}={v:.4g}" for m, v in vals.items())
          + f"   {note}")

print("\n=== input_scale sharing (informational) ===")
for fused, i, vals, spread in group_spread("input_scale")[0]:
    note = "identical" if spread <= 1e-6 else f"spread {spread:.2%}"
    print(f"  {fused:<14} layer {i:<3} "
          + "  ".join(f"{m.split('_')[0]}={v:.4g}" for m, v in vals.items())
          + f"   {note}")

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
if ws2_untied:
    for fused, n in ws2_untied.items():
        print(f"  {fused} is UNTIED in {n} of {len(layers)} sampled layer(s).")
    print("\n  This is fatal and silent. The group is concatenated into one")
    print("  tensor served with one weight_scale_2, so the shards that lost the")
    print("  max are dequantized against a scale that is wrong for them by")
    print("  whatever factor their amaxes differed by. Nothing in the export")
    print("  looks malformed; the model just produces noise.")
    print("\n  Fix at quantization time, not by patching the export: quantize.py")
    print("  ties each fused group's weight amaxes to their max before export.")
    print("  Re-run with --resume-amax; the amaxes themselves are fine.")
elif suspect:
    print("  Fused groups are tied, but these kinds derive weight_scale_2 from")
    print("  something other than the basis every other projection uses:")
    for kind, rel in suspect:
        print(f"    {kind:<12} {rel:.4f}x the others")
else:
    print("  Every projection derives weight_scale_2 from the same basis at the")
    print("  same ratio, and every fused group shares one scale. Scale")
    print("  derivation is sound; a bad score is about something else.")
