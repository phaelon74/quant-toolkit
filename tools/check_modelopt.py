#!/usr/bin/env python3
"""Verify the environment before starting a long quantization run.

Checks versions, GPU compute capability, which ModelOpt features the installed
build actually exposes, and that the private export symbols this toolkit
imports still exist. Exits non-zero if anything required is missing.

Usage:
    python tools/check_modelopt.py
"""

import importlib
import sys

FAIL = []
WARN = []


def line(label, value, note=""):
    print(f"  {label:<26} {value}{('  ' + note) if note else ''}")


def probe(module_name, *symbols):
    """Return the list of symbols missing from a module ('*' = module itself)."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        return [f"{module_name} ({type(exc).__name__})"]
    return [s for s in symbols if not hasattr(mod, s)]


# ---------------------------------------------------------------------------
print("\n=== Python / Torch ===")
print(f"  {'python':<26} {sys.version.split()[0]}")
if sys.version_info < (3, 12):
    FAIL.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.12")

try:
    import torch

    line("torch", torch.__version__)
    line("torch cuda", torch.version.cuda)
    line("cuda available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        FAIL.append("torch.cuda.is_available() is False")
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    line("gpu count", n)
    caps = set()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        cap = f"{p.major}.{p.minor}"
        caps.add(cap)
        line(f"  gpu{i}", f"{p.name}", f"sm_{p.major}{p.minor}, {p.total_memory / 1024**3:.0f} GiB")
    if caps and not any(int(c.split(".")[0]) >= 10 for c in caps):
        WARN.append(
            f"No Blackwell GPU (found sm {sorted(caps)}). Calibration and export "
            "still work; NVFP4 *inference* needs Blackwell."
        )
except ImportError:
    FAIL.append("torch is not installed")

try:
    import transformers

    line("transformers", transformers.__version__)
except ImportError:
    FAIL.append("transformers is not installed")


# ---------------------------------------------------------------------------
print("\n=== ModelOpt ===")
try:
    import modelopt
    import modelopt.torch.quantization as mtq

    line("nvidia-modelopt", getattr(modelopt, "__version__", "unknown"))
except ImportError as exc:
    print(f"  !! import failed: {exc}")
    FAIL.append("nvidia-modelopt is not installed or not importable")
    print()
    for f in FAIL:
        print(f"FAIL: {f}")
    print("\nNothing else can be checked until the venv is complete. "
          "See Install.md section 5.")
    sys.exit(1)

# quant_cfg layout. ModelOpt 0.44 switched from a pattern-keyed dict to an
# ordered list of QuantizerCfgEntry. quantize.py handles both, but knowing
# which one you have makes any override bug far easier to read.
base = getattr(mtq, "NVFP4_DEFAULT_CFG", None)
if base is None:
    FAIL.append("mtq.NVFP4_DEFAULT_CFG is missing")
else:
    layout = "list (>=0.44)" if isinstance(base["quant_cfg"], list) else "dict (<=0.43)"
    line("quant_cfg layout", layout)

presets = sorted(n for n in dir(mtq) if n.startswith(("NVFP4_", "MXFP4_")))
line("NVFP4 presets", len(presets))
for name in ("NVFP4_DEFAULT_CFG", "NVFP4_MLP_ONLY_CFG", "NVFP4_OMLP_ONLY_CFG"):
    present = hasattr(mtq, name)
    line(f"  {name}", "yes" if present else "MISSING")
# Behemoth's recipe starts from the omlp preset.
if not hasattr(mtq, "NVFP4_OMLP_ONLY_CFG"):
    FAIL.append(
        "NVFP4_OMLP_ONLY_CFG missing (added in ModelOpt 0.43). Either upgrade, or "
        "set base_quant_cfg='NVFP4_DEFAULT_CFG' in models/behemoth_r1_123b.py."
    )


# ---------------------------------------------------------------------------
print("\n=== Calibration algorithms ===")
CALIB = {
    "max": ("modelopt.torch.quantization.model_calib", "max_calibrate"),
    "mse": ("modelopt.torch.quantization.model_calib", "mse_calibrate"),
    "local_hessian": ("modelopt.torch.quantization.model_calib", "local_hessian_calibrate"),
    "nvfp4_act_headroom": (
        "modelopt.torch.quantization.config",
        "NVFP4ActHeadroomCalibConfig",
    ),
}
available = []
for method, (module_name, symbol) in CALIB.items():
    ok = not probe(module_name, symbol)
    available.append(method) if ok else None
    line(f"  --calib-method {method}", "yes" if ok else "not in this build")

if "max" not in available:
    FAIL.append("max calibration is unavailable, which should be impossible")
if "mse" not in available:
    WARN.append(
        "configs/calib_behemoth_r1_123b.toml defaults to method = \"mse\". "
        "Change it to \"max\"."
    )
if "nvfp4_act_headroom" not in available:
    print("\n  note: nvfp4_act_headroom is absent, which is expected on 0.46.0.")
    print("        Keep method = \"mse\" in the calibration TOML.")

quantile_ok = not probe("modelopt.torch.quantization.calib.quantile", "save_quantile_data")
line("  --calib-method quantile", "yes (patched build)" if quantile_ok else "no (expected)")


# ---------------------------------------------------------------------------
# export_hf.py imports several internal and one private symbol. These move
# between releases, and they fail at export time -- after calibration.
print("\n=== Export symbols used by export_hf.py ===")
EXPORT = {
    "modelopt.torch.export.convert_hf_config": ("convert_hf_quant_config_format",),
    "modelopt.torch.export.layer_utils": (
        "get_expert_linear_names",
        "is_moe",
        "set_expert_quantizer_amax",
    ),
    "modelopt.torch.export.quant_utils": ("get_quant_config", "postprocess_state_dict"),
    "modelopt.torch.export.unified_export_hf": (
        "_process_quantized_modules",
        "requantize_resmooth_fused_llm_layers",
    ),
}
for module_name, symbols in EXPORT.items():
    missing = probe(module_name, *symbols)
    short = module_name.rsplit(".", 1)[-1]
    line(f"  {short}", "ok" if not missing else f"MISSING {missing}")
    if missing:
        FAIL.append(f"{module_name}: missing {missing}")


# ---------------------------------------------------------------------------
print("\n=== This repo ===")
try:
    from models import AVAILABLE_MODELS

    line("registered models", len(AVAILABLE_MODELS))
    line("  behemoth_r1_123b", "yes" if "behemoth_r1_123b" in AVAILABLE_MODELS else "MISSING")
    if "behemoth_r1_123b" not in AVAILABLE_MODELS:
        FAIL.append("behemoth_r1_123b not registered in models/__init__.py")
except Exception as exc:
    FAIL.append(f"cannot import models package: {type(exc).__name__}: {exc}")

for name, mod in (("datasets", "datasets"), ("pyyaml", "yaml")):
    line(f"  {name}", "yes" if not probe(mod) else "not installed (tools only)")


# ---------------------------------------------------------------------------
print()
for w in WARN:
    print(f"WARN: {w}")
for f in FAIL:
    print(f"FAIL: {f}")

if FAIL:
    print(f"\n{len(FAIL)} blocking problem(s). See Install.md section 12.")
    sys.exit(1)
print("All required checks passed." + (f" {len(WARN)} warning(s)." if WARN else ""))
