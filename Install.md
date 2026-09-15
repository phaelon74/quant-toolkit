# Install

Setup for **post-training NVFP4 quantization** on a fresh Linux GPU machine.

> **Run** `python tools/check_modelopt.py` **after installing.** It verifies versions, GPU compute capability, which ModelOpt calibration algorithms your build actually has, and that the private export symbols this toolkit imports still exist. Everything in §7 is checked automatically.

## Contents

1. [What you are installing](#1-what-you-are-installing)
2. [Hardware](#2-hardware)
3. [System prerequisites](#3-system-prerequisites)
4. [Clone the repo (Git LFS)](#4-clone-the-repo-git-lfs)
5. [Create the venv](#5-create-the-venv)
6. [Optional extras](#6-optional-extras)
7. [ModelOpt version compatibility — read this](#7-modelopt-version-compatibility--read-this)
8. [Verify the install](#8-verify-the-install)
9. [Caches, paths, environment](#9-caches-paths-environment)
10. [How this toolkit uses ModelOpt](#10-how-this-toolkit-uses-modelopt)
11. [Serving venv (vLLM on SM120)](#11-serving-venv-vllm-on-sm120)
12. [Troubleshooting](#12-troubleshooting)
13. [Next](#13-next)

---



## 1. What you are installing

Post-training NVFP4 quantization with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt) driven by large JSONL calibration sets.

Do this on **Linux x86_64 with NVIDIA GPUs**. A Windows checkout is fine for editing; it is not a runtime for 123B-class PTQ.


| Piece                  | Pin                                  | Why exactly this                                                                                                              |
| ---------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Python                 | **3.12**                             | `requires-python = ">=3.12"`. ModelOpt's own docs call 3.12 the production default; 3.10 support is being dropped.            |
| PyTorch + torchvision  | **CUDA 13.0 wheels** (`cu130` index) | CUDA 13 covers `sm_120` (RTX PRO 6000 Blackwell). Getting a default-PyPI CPU wheel is the single most common install mistake. |
| nvidia-modelopt | `==0.46.0`                           | Newest tagged release. Was previously an unpinned git URL, which is a moving target — see §7. No `[torch]` extra: 0.46.0 does not define one and the torch subpackage is in the base distribution. |
| transformers           | `==5.5.3`                            | Exact pin in `pyproject.toml`. ModelOpt 0.46's floor is 4.57.                                                                 |
| Calibration JSONL      | Git LFS under `data/`                | `data/**/*.jsonl` is LFS in `.gitattributes`.                                                                                 |


> `pyproject.toml` previously read `nvidia-modelopt[torch] @ git+https://github.com/NVIDIA/TensorRT-Model-Optimizer.git`. Unpinned git HEAD means two installs a week apart get different libraries. It is now a version pin. §7 explains what breaks if you move off it.



## 2. Hardware



### The reference box for this repo's Behemoth recipe

4× RTX PRO 6000 Blackwell Workstation (96 GB each, **384 GB total**), 768 GB system RAM, ~2 TB disk, PCIe Gen5, **no NVLink**.

That configuration matters in three ways:

- **229 GB of BF16 weights fit in 384 GB of VRAM**, so `device_map="auto"` holds the whole model and you do **not** pass `--streaming`.
- These are **SM120** (compute capability 12.0), *not* SM100 like B200. Different kernel family — see §11.
- `device_map="auto"` is pipeline parallelism, so the four cards give you capacity, not 4× throughput. ModelOpt supports true data-parallel calibration, but this toolkit does not wire it up. Details in [Behemoth-123B_v2_R1.md §6.1](Behemoth-123B_v2_R1.md).



### Minimum to install and smoke-test

Linux x86_64, a driver new enough for the cu130 wheels (`nvidia-smi` works), ~25 GB disk for the venv, network access to PyPI and `download.pytorch.org`.

### Sizing a real job

PTQ loads the **BF16** model, runs calibration forwards, then exports. Disk and RAM are dominated by the source checkpoint, not the venv.


| Resource                      | Dense 123B (Behemoth)  | Huge MoE (GLM-5 / Qwen3.5-397B) |
| ----------------------------- | ---------------------- | ------------------------------- |
| Source checkpoint             | ~229 GB BF16           | often 0.5–1.5 TB                |
| NVFP4 export (MLP + `o_proj`) | ~92 GB                 | depends on the model            |
| Peak disk during a run        | source + export + amax | same                            |


**Two ways to fit the weights:**

- `device_map="auto"` (faster, the default): needs combined GPU memory for all BF16 weights plus activations. 4× 96 GB works for 123B.
- `--streaming` (this repo's `StreamingModelLoader`): GPU 0 is execution-only; other GPUs, CPU RAM, and disk hold layers. Runs 123B on **one 80 GB GPU** given ~200 GB+ CPU RAM and fast NVMe, but it re-transfers layer weights every batch and is much slower. Only use it when the model genuinely does not fit.

Activation batch size is `batch_tokens // max_len`. Default `--batch-tokens` is 131072, which is too aggressive for a dense 123B — start at **32768**.

### Blackwell is not one target


| Family                           | Examples                              | Compute           | Tensor-core path                                       |
| -------------------------------- | ------------------------------------- | ----------------- | ------------------------------------------------------ |
| Datacenter Blackwell             | B100, B200, GB200                     | **SM100**         | TMEM + `tcgen05`                                       |
| Workstation / consumer Blackwell | **RTX PRO 6000**, RTX 5090, DGX Spark | **SM120 / SM121** | warp-level MMA (`b12x` / `sparkinfer`, via FlashInfer) |


NVFP4 **inference** needs Blackwell. Calibration and HF export do **not** — you can quantize on Hopper/Ada and serve on Blackwell later. Record your exact compute capability (`tools/check_modelopt.py` prints it) before assuming a kernel path exists.

## 3. System prerequisites

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs python3.12 python3.12-venv python3.12-dev \
    build-essential curl ca-certificates
git lfs install
```

RHEL / Rocky:

```bash
sudo dnf install -y git git-lfs python3.12 python3.12-devel gcc gcc-c++ make curl
git lfs install
```

Check the driver and capability:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

For 4× RTX PRO 6000 you want `compute_cap` = **12.0** on every row. You need a CUDA 13-capable driver, but **not** a full local CUDA toolkit — the PyTorch cu130 wheels bundle their own runtime.

## 4. Clone the repo (Git LFS)

```bash
git clone <this-repo-url> quant-toolkit
cd quant-toolkit
git lfs pull
```

Confirm the calibration files are real JSONL and not LFS pointer stubs:

```bash
head -c 60 data/text/diverse_calib.jsonl; echo
wc -l data/text/*.jsonl
```

A pointer stub is ~130 bytes and begins `version https://git-lfs.github.com/spec/v1`. If you see that, run `git lfs install && git lfs pull` again.


| File                                           | Samples | Size   |
| ---------------------------------------------- | ------- | ------ |
| `data/text/diverse_calib.jsonl`                | 16329   | ~65 MB |
| `data/text/agentic_coding_calib_v3.jsonl`      | 15264   | ~52 MB |
| `data/text/agentic_coding_calib_generic.jsonl` | 23458   | ~94 MB |
| `data/text/deep_calib.jsonl`                   | 819     | ~15 MB |


These are the repo's stock sets, and they are coding-heavy. The Behemoth recipe does not use them — it builds its own set from `data/behemoth_r1_123b_calib.yaml`, which needs `datasets` and `pyyaml` from §6.

## 5. Create the venv



### With uv (recommended)

`pyproject.toml` is written for [uv](https://docs.astral.sh/uv/): Python 3.12, Torch from the official CUDA 13.0 index.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd quant-toolkit
uv venv --python 3.12 .venv
source .venv/bin/activate

# Torch and torchvision from the CUDA 13.0 index, explicitly.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Everything else from PyPI.
uv pip install -e .
```

`uv.lock` is gitignored, so use `uv pip install -e .`, not `uv sync`.

**Why two commands.** `[tool.uv.sources]` routes `torch` and `torchvision` to the `pytorch-cu130` index, but `tool.uv.sources` is a *project-mode* mechanism and `uv pip install` is the pip-compatible interface, which does not reliably apply it. Installing Torch first makes the CUDA build deterministic; the second command then sees `torch>=2.10` already satisfied and leaves it alone.

The `pytorch-cu130` index in `pyproject.toml` is marked `explicit = true`, which confines it to packages that name it. This is load-bearing — see §12 if you hit a `setuptools` resolution error.

**What `-e .` actually installs.** Only the `models` package, declared via `[tool.setuptools] packages = ["models"]`. This repo is an application, not a library: `quantize.py`, `export_hf.py`, `streaming_loader.py` and the rest are top-level scripts you run from the repo root, resolved through the working directory rather than installed. The editable `models` package is still worth having, because `tools/check_modelopt.py` does `from models import AVAILABLE_MODELS` and, when run as `python tools/check_modelopt.py`, gets `tools/` on `sys.path` rather than the repo root.

The first command also pulls `setuptools==78.1.0` from the PyTorch mirror, because `--index-url` replaces PyPI for that one invocation. The second command upgrades it to the `>=80` that ModelOpt requires. Confirm with `uv pip show setuptools` if a later step complains.

The second command pulls `nvidia-modelopt==0.46.0`, `transformers==5.5.3`, `accelerate`, `safetensors`, `sentencepiece`, `protobuf`, `pillow`, and `requests`, and upgrades `setuptools` to 84.x.

### With pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Torch from the cu130 index FIRST, or you may get a CPU build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

Verify you did not get a CPU wheel before going further:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.version.cuda` must be **13.x** and `is_available()` must be `True`. If it says `None` / `False`, reinstall Torch from the cu130 index.

## 6. Optional extras

Not in `pyproject.toml`, but needed by specific tools:

```bash
# Build calibration JSONL from Hugging Face datasets.
# Required for the Behemoth recipe (tools/build_calib_from_yaml.py).
uv pip install datasets pyyaml

# Faster Hugging Face downloads
uv pip install "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

# KV-scale live-server helper (tools/kv_calib_requests.py)
uv pip install aiohttp
```

You can skip `datasets` entirely if you only use the JSONL already committed under `data/text/`.

## 7. ModelOpt version compatibility — read this

This toolkit reaches into ModelOpt internals, so the version genuinely matters. Four things to know.

### 7.1 `quant_cfg` changed shape in 0.44

**ModelOpt 0.44 (2026-05-14) turned** `quant_cfg` **from a pattern-keyed dict into an ordered list of** `QuantizerCfgEntry`, and rebuilt every shipped preset — including `NVFP4_DEFAULT_CFG` — in the new form. Entries apply in list order and later entries override earlier ones.

The old dict form is still accepted *as input* (with a `DeprecationWarning`), but the presets you read back are lists. So this pattern, which the repo used everywhere:

```python
qcfg["quant_cfg"][pattern] = override   # TypeError on 0.44+
```

raises `TypeError: list indices must be integers or slices, not str`.

`quantize.py` now handles both layouts through `apply_quant_overrides()`, appending entries on new builds and assigning keys on old ones. Appending gives the same precedence the dict assignment had. `tools/check_modelopt.py` prints which layout you have.

### 7.2 Feature availability by release


| Feature                                          | Added                       | In 0.46.0?                                                |
| ------------------------------------------------ | --------------------------- | --------------------------------------------------------- |
| `NVFP4_OMLP_ONLY_CFG` / `NVFP4_EXPERTS_ONLY_CFG` | 0.43                        | Yes — Behemoth's recipe starts from `NVFP4_OMLP_ONLY_CFG` |
| List-form `quant_cfg`                            | 0.44                        | Yes (handled)                                             |
| `mse` / `local_hessian` FP8 block-scale sweep    | ≤0.45                       | Yes — the recipe's default `method = "mse"`               |
| `nvfp4_act_headroom`                             | not in the 0.46.0 changelog | **Probably not.** Present on `main`. The probe tells you. |
| `calib.quantile` (P2 streaming quantiles)        | —                           | **No. Never upstream.**                                   |




### 7.3 There is no quantile calibrator upstream

Several configs in `configs/` set `method = "quantile"`. Upstream ModelOpt validates `calibrator` against `["max", "histogram"]` only, and `modelopt.torch.quantization.calib.quantile` does not exist, so `--calib-method quantile` and `--save-quantiles` fail on any stock install. Those configs assume a patched fork.

For NVFP4, the useful upstream options are:


| Method               | What it does                                                                                                                        | Availability                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `max`                | Plain amax. Always works.                                                                                                           | Everywhere                    |
| `mse`                | Max-calibrates, then sweeps the 126 candidate FP8-E4M3 block scales per NVFP4 block. Triton-accelerated.                            | 0.46.0 — **the default here** |
| `local_hessian`      | Same sweep, minimizing Hessian-weighted error. ~34× faster than the reference sweep on `main`.                                      | 0.46.0                        |
| `nvfp4_act_headroom` | Clips *activation* global scales above `upper_percentile` so one outlier block can't drive every other block's FP8 scale subnormal. | newer than 0.46.0             |




### 7.4 `export_hf.py` imports private symbols

`export_hf.py` is a fork of ModelOpt's `unified_export_hf.py` and imports internals, including one private name:

```python
from modelopt.torch.export.unified_export_hf import (
    _process_quantized_modules,          # private
    requantize_resmooth_fused_llm_layers,
)
```

These can move between releases, and they fail at **export** time — after you have already spent days calibrating. This is the strongest reason to pin. `tools/check_modelopt.py` probes every one of them up front; run it before any long job.

### 7.5 If you want to change the pin

Edit `pyproject.toml`, reinstall, then re-run the probe:

```bash
uv pip install -e . --reinstall-package nvidia-modelopt
python tools/check_modelopt.py
```

Do not go below 0.43 (you lose `NVFP4_OMLP_ONLY_CFG`, and older releases predate the transformers 5.x export work). If you move to `main` to get `nvfp4_act_headroom`, expect to re-verify §7.4 and pin a commit SHA rather than a branch:

```toml
"nvidia-modelopt @ git+https://github.com/NVIDIA/Model-Optimizer.git@<commit-sha>",
```



## 8. Verify the install

```bash
source .venv/bin/activate
python tools/check_modelopt.py
```

Expected on the reference box:

```text
=== Python / Torch ===
  python                     3.12.x
  torch                      2.1x.x+cu130
  torch cuda                 13.0
  cuda available             True
  gpu count                  4
    gpu0                     NVIDIA RTX PRO 6000 Blackwell...  sm_120, 96 GiB
    ...
  transformers               5.5.3

=== ModelOpt ===
  nvidia-modelopt            0.46.0
  quant_cfg layout           list (>=0.44)
    NVFP4_OMLP_ONLY_CFG      yes

=== Calibration algorithms ===
    --calib-method max       yes
    --calib-method mse       yes
    --calib-method quantile  no (expected)

=== Export symbols used by export_hf.py ===
    unified_export_hf        ok
```

Exit code 0 means you are clear to run.

**Warnings you can ignore.** ModelOpt emits `FutureWarning: torch.jit.script is deprecated` on import — that is internal to ModelOpt and harmless. If you see `warning: The package nvidia-modelopt==0.46.0 does not have an extra named torch` during install, your `pyproject.toml` predates the fix that dropped the `[torch]` extra; the install still succeeds either way.

Then a CLI smoke check:

```bash
python quantize.py --help
```



## 9. Caches, paths, environment



### Hugging Face cache on a large disk

A 123B BF16 download is ~229 GB. Never leave that on a small home partition:

```bash
export HF_HOME=/media/fmodels2/hf
export HF_HUB_CACHE=/media/fmodels2/hf/hub
mkdir -p "$HF_HUB_CACHE"
```

Put these in `~/.bashrc` so downloads and quantization agree on one location. If your model is already on disk (the Behemoth recipe passes `--model-id /media/fmodels/...`), the cache is barely used.

Behemoth is **not** gated. Some models (Mistral base, certain Gemma/Qwen) need `huggingface-cli login`.

### Runtime environment variables

Every `scripts/quantize_*.sh` activates `.venv` and sets:

```bash
export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

`expandable_segments` matters: PTQ allocates and frees large activation tensors in a repeating pattern, which fragments the caching allocator badly without it.

`quantize.py` also takes `--cpu-capacity` (default `200GiB`) as the streaming loader's CPU packing budget. It is ignored unless `--streaming` is set.

## 10. How this toolkit uses ModelOpt

1. Load a model adapter from `models/` (`--model`), which supplies the HF id, loader flags, the base preset, and quantizer overrides.
2. Build calibration batches from a TOML (`--calib-config`) or a single JSONL, applying each model's chat template.
3. Deep-copy the adapter's base preset — `NVFP4_DEFAULT_CFG` by default, `NVFP4_OMLP_ONLY_CFG` for Behemoth.
4. Apply the adapter's overrides via `apply_quant_overrides()`, which is layout-agnostic (§7.1).
5. Run `mtq.quantize(model, qcfg, forward_loop)` with `max`, `mse`, `local_hessian`, or `nvfp4_act_headroom`.
6. Export a HF checkpoint through `export_hf.py`, layer by layer, so CPU RAM never holds the whole state dict.

**Why sample count matters:** NVFP4 *weight* block scales are essentially data-free, but *activation / input* scales are fitted from the forward loop. This repo's results come from thousands of in-template samples rather than the 128–512 CNN/DailyMail default in upstream examples. Streaming to disk instead of holding everything in VRAM is what makes those counts affordable.

**On layer scope:** NVIDIA recommends restricting NVFP4 to MLP (`nvfp4_mlp_only`), MoE experts (`nvfp4_experts_only`), or MLP + `o_proj` (`nvfp4_omlp_only`) for PTQ accuracy, keeping the sensitive attention QKV projections in higher precision. `models/base.py`'s shared `COMMON_QUANT_OVERRIDES` approximates `mlp_only` by disabling all `*self_attn`* quantizers **and enables an FP8 KV cache**. Behemoth deliberately does not inherit it — it starts from the real `omlp_only` preset and keeps the KV cache in BF16. Check what your adapter actually does before assuming.

## 11. Serving venv (vLLM on SM120)

**Install vLLM in a separate venv.** vLLM, ModelOpt, and FlashInfer each pin torch, `transformers`, and CUDA-side packages, and they will fight in one environment. Quantization and serving are separate jobs on separate schedules — keep them separate.

```bash
python3.12 -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate
pip install --upgrade pip
pip install vllm
```

SM120 needs explicit configuration, because the fallback is Marlin W4A16 — which dequantizes FP4 to FP16 and gives up much of the point:

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_FORCE_SM=120f
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x
```

Requirements and traps:

- **A recent vLLM.** Older builds gate NVFP4 kernels behind `is_device_capability_family(100)`, which is `False` for SM120, so they refuse to load or emit garbage.
- **FlashInfer ≥ 0.6.9** for the `b12x` backend. Kernels are JIT-compiled on first use and cached, so the first request is slow.
- `nvidia-cutlass-dsl` **is version-sensitive.** vLLM's SM120 integration pinned `4.4.2` (4.5.0 emitted bad PTX for SM121); `sparkinfer` currently asks for `4.6.0`. If you get PTX or JIT errors, this is the first thing to bisect.
- **Dense models are the good case.** The NVFP4 *MoE* grouped-GEMM path has been broken on SM120. Behemoth is dense and uses the `mm_fp4` dense path, which is what `b12x` was built for.
- **Confirm the kernel.** Check startup logs for the selected NVFP4 linear kernel. If it names Marlin or `FLASHINFER_CUTLASS`, fix the environment before benchmarking.

Serving command and flags: [Behemoth-123B_v2_R1.md §7](Behemoth-123B_v2_R1.md).

## 12. Troubleshooting



### Install-time


| Symptom                                                          | Cause and fix                                                                                                           |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `only setuptools<=78.1.0 is available and nvidia-modelopt==0.46.0 depends on setuptools>=80` | The `pytorch-cu130` index is missing `explicit = true`. uv found `setuptools` mirrored on `download.pytorch.org` and, to block dependency-confusion attacks, refuses to consider PyPI for it. **Fix:** add `explicit = true` to the `[[tool.uv.index]]` block; the current `pyproject.toml` has it. Do *not* use `--index-strategy unsafe-best-match` — that weakens the protection for every package to work around one misscoped index. |
| Any other `No solution found` naming a package PyTorch mirrors (`jinja2`, `networkx`, `requests`, …) | Same root cause and same fix as above. |
| `Multiple top-level packages discovered in a flat-layout: ['data', 'models', 'configs']` | `pyproject.toml` was missing both a `[build-system]` table and an explicit package list, so setuptools fell back to `build_meta:__legacy__` and tried to auto-discover packages. It found three candidate directories and refused to guess. **Fix:** the current `pyproject.toml` declares `[build-system]` with `setuptools>=80` and `[tool.setuptools] packages = ["models"]`. |
| `No module named 'modelopt'`                                     | venv not activated, or `pip install -e .` failed. Re-run and read the resolver output.                                  |
| `torch.cuda.is_available()` is `False`, `version.cuda` is `None` | CPU-only wheel. Install Torch from the cu130 index **before** `pip install -e .` (§5).                                  |
| Resolver conflict on `transformers`                              | ModelOpt 0.46's floor is 4.57 and the repo pins `==5.5.3`. Do not loosen both at once; change one and re-run the probe. |
| `CUDA capability` / driver too old                               | Upgrade the NVIDIA driver to a CUDA 13-capable release.                                                                 |
| Calibration JSONL is ~130 bytes                                  | `git lfs install && git lfs pull`.                                                                                      |
| `No module named 'yaml'` / `'datasets'`                          | `uv pip install datasets pyyaml` (§6).                                                                                  |




### Run-time


| Symptom                                                     | Cause and fix                                                                                                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TypeError: list indices must be integers` near `quant_cfg` | Pre-fix `quantize.py` against ModelOpt ≥ 0.44. Pull this repo's current `quantize.py` (§7.1).                                                          |
| `NVFP4_OMLP_ONLY_CFG is not available`                      | ModelOpt < 0.43. Upgrade, or set `base_quant_cfg="NVFP4_DEFAULT_CFG"` in the adapter.                                                                  |
| `No module named ...calib.quantile`                         | Expected. Use `--calib-method max` or `mse` (§7.3).                                                                                                    |
| Validation error on `method = "nvfp4_act_headroom"`         | Not in your build. Use `mse`. The probe reports this.                                                                                                  |
| `Unknown model config`                                      | Adapter not registered in `models/__init__.py`.                                                                                                        |
| `huggingface_hub` 401                                       | `huggingface-cli login` for gated models.                                                                                                              |
| OOM during calibration                                      | Lower `--batch-tokens` (try 16384). Only add `--streaming` if the model genuinely does not fit; raise `--cpu-capacity` only if you truly have the RAM. |
| ImportError from `modelopt.torch.export.*` at export time   | Version drift in the internals from §7.4. Run the probe; pin back to 0.46.0.                                                                           |
| KV cache exported as FP8 when you wanted BF16               | The adapter inherited `COMMON_QUANT_OVERRIDES`, which enables FP8 KV. See §10.                                                                         |




### Diagnosing anything else

```bash
python tools/check_modelopt.py          # environment and feature probe
python quantize.py --help               # current CLI surface
python -c "import modelopt; print(modelopt.__version__)"
```

Always reproduce on a few hundred samples before a multi-day run. The Behemoth guide has a smoke-run recipe that exercises load → calibrate → **export** in hours, which is the only way to catch export-time problems early.

## 13. Next

- Behemoth-R1-123B-v2 NVFP4, end to end: [Behemoth-123B_v2_R1.md](Behemoth-123B_v2_R1.md)

