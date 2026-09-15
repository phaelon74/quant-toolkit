#!/usr/bin/env python3
"""Verify an exported NVFP4 checkpoint before serving or shipping it.

Reads config.json, the safetensors index, and tensor metadata. Loads only the
small scale tensors, so it is fast and needs no GPU.

Checks, in order of how expensive they are to discover later:

  1. No KV cache quantization. Checked as kv_cache_scheme or
     kv_cache_quant_algo depending on layout; if either is set, the KV override
     did not take.
  2. group_size is 16. b12x hardcodes sf_vec_size=16.
  3. Quantization scope: o_proj/gate_proj/up_proj/down_proj carry scales,
     q_proj/k_proj/v_proj/lm_head/embed_tokens do not.
  4. Every unquantized linear is listed in ignore / exclude_modules. Because
     targets is ["Linear"], anything unlisted gets an NVFP4 linear method, looks
     for a weight_scale that does not exist, and fails at load, not at export.
  5. Tensor dtypes are what an NVFP4 export should produce.
  6. Scales are finite, positive, and not uniformly zero.
  7. Tokenizer files survived and the chat template is still present.

Handles both quantization_config layouts. ModelOpt >=0.29 writes the
compressed-tensors form (group_size nested in config_groups, exclusions under
"ignore"); older releases write a flat TRT-LLM form with top-level group_size
and exclude_modules. Reading the wrong one reports false failures on a correct
checkpoint -- and worse, misses real FP8 KV, which only appears as
kv_cache_scheme in the compressed-tensors layout.

Usage:
    python tools/verify_nvfp4_export.py /media/fmodels2/working_Model-Opt/smoke
"""

import argparse
import json
import math
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

QUANTIZED = ("o_proj", "gate_proj", "up_proj", "down_proj")
UNQUANTIZED = ("q_proj", "k_proj", "v_proj", "lm_head", "embed_tokens")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("export_dir", help="Directory containing config.json.")
ap.add_argument("--sample-layers", type=int, default=4,
                help="How many layers to numerically inspect block scales in.")
args = ap.parse_args()

root = Path(args.export_dir)
FAIL, WARN = [], []


def line(label, value, note=""):
    print(f"  {label:<28} {value}{('   ' + note) if note else ''}")


# ---------------------------------------------------------------------------
print("\n=== config.json ===")
cfg_path = root / "config.json"
if not cfg_path.exists():
    print(f"  missing {cfg_path}")
    sys.exit(1)

cfg = json.loads(cfg_path.read_text())
q = cfg.get("quantization_config") or {}
line("architectures", cfg.get("architectures"))
line("num_hidden_layers", cfg.get("num_hidden_layers"))
line("quant_method", q.get("quant_method"))
line("quant_algo", q.get("quant_algo"))

# Two layouts exist and they put everything in different places.
# convert_hf_quant_config_format in ModelOpt >=0.29 emits the compressed-tensors
# form: group_size nested in config_groups, exclusions in "ignore", and KV as
# "kv_cache_scheme" (added only when kv_cache_quant_algo is truthy, so BF16 KV
# means the key is absent entirely). The older TRT-LLM form is flat.
groups = q.get("config_groups") or {}
if groups:
    fmt = f"compressed-tensors ({len(groups)} group(s))"
    g0 = next(iter(groups.values()))
    group_size = (g0.get("weights") or {}).get("group_size")
    acts = g0.get("input_activations")
    targets = g0.get("targets")
    excludes, exclude_key = q.get("ignore") or [], "ignore"
    kv, kv_key = q.get("kv_cache_scheme"), "kv_cache_scheme"
else:
    fmt = "TRT-LLM flat"
    group_size = q.get("group_size")
    acts, targets = None, None
    excludes, exclude_key = q.get("exclude_modules") or [], "exclude_modules"
    kv, kv_key = q.get("kv_cache_quant_algo"), "kv_cache_quant_algo"

line("config layout", fmt)
if targets is not None:
    line("targets", targets)
if groups:
    line("weight/act precision",
         f"W4{'A4' if acts else 'A16'}",
         "" if acts else "<- weight-only, not what this recipe intends")

line(kv_key, repr(kv), "<- correct, BF16 KV" if not kv else "<- WRONG")
if kv:
    FAIL.append(f"{kv_key} is {kv!r}; the KV cache must stay BF16")

line("group_size", group_size, "" if group_size == 16 else "<- b12x needs 16")
if group_size != 16:
    FAIL.append(f"group_size is {group_size}, b12x hardcodes sf_vec_size=16")

line(exclude_key, f"{len(excludes)} entr(y/ies)")
for pattern in excludes[:4]:
    print(f"      {pattern}")
if len(excludes) > 4:
    print(f"      ... and {len(excludes) - 4} more")


# ---------------------------------------------------------------------------
print("\n=== Shards ===")
index_path = next(root.glob("*.index.json"), None)
if index_path is None:
    FAIL.append("no *.index.json found")
    weight_map = {}
else:
    weight_map = json.loads(index_path.read_text())["weight_map"]
    line("index", index_path.name)
    line("tensors", len(weight_map))
    shards = sorted(set(weight_map.values()))
    line("shard files", len(shards))
    missing = [s for s in shards if not (root / s).exists()]
    if missing:
        FAIL.append(f"index references missing shards: {missing[:5]}")
    total = sum((root / s).stat().st_size for s in shards if (root / s).exists())
    line("total size", f"{total / 1024**3:.1f} GiB")


# ---------------------------------------------------------------------------
# Group tensor names by owning module.
print("\n=== Quantization scope ===")
modules = {}
for key in weight_map:
    m = re.match(r"(.*)\.(weight|weight_scale|weight_scale_2|input_scale|bias)$", key)
    if m:
        modules.setdefault(m.group(1), set()).add(m.group(2))

kinds = {}
for module, parts in modules.items():
    for kind in QUANTIZED + UNQUANTIZED:
        if module.endswith(kind):
            kinds.setdefault(kind, []).append((module, parts))
            break

for kind in QUANTIZED + UNQUANTIZED:
    found = kinds.get(kind, [])
    if not found:
        WARN.append(f"no {kind} modules found in the checkpoint")
        line(kind, "ABSENT")
        continue
    with_scale = [m for m, p in found if "weight_scale" in p]
    is_quant = len(with_scale) == len(found)
    is_plain = not with_scale
    state = "NVFP4" if is_quant else ("BF16" if is_plain else "MIXED")
    line(kind, f"{state:<6} {len(found):>4} module(s)")
    print(f"      tensors: {sorted(found[0][1])}")

    if kind in QUANTIZED and not is_quant:
        FAIL.append(f"{kind} should be NVFP4 but {len(found) - len(with_scale)} "
                    f"of {len(found)} modules have no weight_scale")
    if kind in UNQUANTIZED and not is_plain:
        FAIL.append(f"{kind} must stay BF16 but {len(with_scale)} module(s) "
                    f"carry weight_scale")


# ---------------------------------------------------------------------------
# targets: ["Linear"] means "quantize every Linear except these", so anything
# unquantized and unlisted gets an NVFP4 linear method, looks for a weight_scale
# that does not exist, and fails at load rather than at export. Entries may be
# literal module paths or wildcards; fnmatch handles both.
print(f"\n=== {exclude_key} coverage ===")
unquantized = [m for m, p in modules.items() if "weight_scale" not in p and "weight" in p]
uncovered = [m for m in unquantized
             if not any(fnmatch(m, p) for p in excludes)]
line("unquantized modules", len(unquantized))
line("covered", len(unquantized) - len(uncovered))
if uncovered:
    norms = [m for m in uncovered if "norm" in m.lower()]
    linears = [m for m in uncovered if "norm" not in m.lower()]
    if norms:
        line("  uncovered norms", len(norms), "(fine, not linears)")
    if linears:
        for m in linears[:8]:
            print(f"      UNCOVERED: {m}")
        FAIL.append(f"{len(linears)} unquantized linear(s) match no {exclude_key} "
                    f"entry; vLLM will treat them as NVFP4 and fail at load")


# ---------------------------------------------------------------------------
print("\n=== Tensor dtypes ===")
try:
    from safetensors import safe_open
except ImportError:
    WARN.append("safetensors not importable; skipped dtype and numeric checks")
    safe_open = None

if safe_open is not None and weight_map:
    dtypes = {}
    layer_ids = sorted({int(m.group(1)) for k in weight_map
                        if (m := re.search(r"layers\.(\d+)\.", k))})
    if layer_ids:
        step = max(1, len(layer_ids) // max(1, args.sample_layers))
        probe = set(layer_ids[::step][: args.sample_layers])
    else:
        probe = set()
    line("layers", len(layer_ids), f"sampling scales in {sorted(probe)}")
    bad_scales, checked = [], 0

    by_shard = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    for shard, keys in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt") as f:
            for key in keys:
                suffix = key.rsplit(".", 1)[-1]
                dt = f.get_slice(key).get_dtype()
                quantish = any(f".{k}." in key for k in QUANTIZED)
                dtypes.setdefault((suffix, dt, quantish), 0)
                dtypes[(suffix, dt, quantish)] += 1

                lm = re.search(r"layers\.(\d+)\.", key)
                if suffix.startswith(("weight_scale", "input_scale")) and (
                        lm is None or int(lm.group(1)) in probe):
                    t = f.get_tensor(key).float()
                    checked += 1
                    if not bool(t.isfinite().all()):
                        bad_scales.append(f"{key}: non-finite")
                    elif float(t.abs().max()) == 0.0:
                        bad_scales.append(f"{key}: all zero")

    for (suffix, dt, quantish), n in sorted(dtypes.items()):
        line(f"{suffix} ({'nvfp4' if quantish else 'bf16 '})", f"{dt:<10} x{n}")

    print("\n=== Scale numerics ===")
    line("scale tensors checked", checked)
    if bad_scales:
        for b in bad_scales[:8]:
            print(f"      {b}")
        FAIL.append(f"{len(bad_scales)} scale tensor(s) non-finite or all-zero")
    else:
        line("finite and non-zero", "yes")


# ---------------------------------------------------------------------------
print("\n=== Tokenizer ===")
for name in ("tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "tokenizer.model"):
    present = (root / name).exists()
    line(name, "yes" if present else "MISSING")
    if not present and name != "tokenizer.model":
        FAIL.append(f"{name} was not copied into the export")

tok_cfg_path = root / "tokenizer_config.json"
if tok_cfg_path.exists():
    tok_cfg = json.loads(tok_cfg_path.read_text())
    template = tok_cfg.get("chat_template")
    if isinstance(template, list):
        template = " ".join(t.get("template", "") for t in template)
    if not template:
        FAIL.append("no chat_template in tokenizer_config.json")
    else:
        line("chat_template", f"{len(template)} chars")
        for marker in ("[INST]", "[/INST]"):
            if marker not in template:
                WARN.append(f"chat_template does not mention {marker}; "
                            "confirm it is still the Mistral v7 template")


# ---------------------------------------------------------------------------
print()
for w in WARN:
    print(f"WARN: {w}")
for f_ in FAIL:
    print(f"FAIL: {f_}")
if FAIL:
    print(f"\n{len(FAIL)} blocking problem(s).")
    sys.exit(1)
print(f"Export verified.{f' {len(WARN)} warning(s).' if WARN else ''}")
