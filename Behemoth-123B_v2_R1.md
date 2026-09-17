# Behemoth-R1-123B-v2 → NVFP4

Recipe for [TheDrummer/Behemoth-R1-123B-v2](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2) on 4× RTX PRO 6000 Blackwell, served with vLLM.

Machine setup: [Install.md](Install.md).

| | |
| --- | --- |
| Source (BF16, 229 GB) | `/media/fmodels/TheDrummer/Behemoth-R1-123B-v2` |
| Working dir | `/media/fmodels2/working_Model-Opt/Behemoth-R1-123B-v2-nvfp4` |
| Final checkpoint | `/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4` |
| Expected output size | **~92 GB** |
| Scope | NVFP4 W4A4 on `gate/up/down` + `o_proj`; BF16 `q/k/v`, embeddings, `lm_head`, **KV cache** |

## 1. Corrections to the first pass

Six things in the earlier version were wrong for this target and are now fixed.

| Issue | Was | Now |
| --- | --- | --- |
| **KV cache** | FP8. `COMMON_QUANT_OVERRIDES` enables `*[kv]_bmm_quantizer` and the adapter never overrode it. | BF16. Adapter no longer inherits `COMMON_QUANT_OVERRIDES` and disables every attention BMM/softmax quantizer. |
| **Would not run at all** | `qcfg["quant_cfg"][pattern] = override` | ModelOpt 0.44 made `quant_cfg` an ordered **list**, so that line raised `TypeError` on every current release. `quantize.py` now applies overrides through a layout-agnostic helper. |
| **Calibration algorithm** | `method = "quantile"` | `mse`. There is **no quantile calibrator in upstream ModelOpt** — `calibrator` is validated against `["max", "histogram"]` and `modelopt.torch.quantization.calib.quantile` does not exist. That config would have crashed. |
| **Layer scope plumbing** | `NVFP4_DEFAULT_CFG` + hand-rolled `*self_attn*` disables | Starts from NVIDIA's own `NVFP4_OMLP_ONLY_CFG` preset, which *is* the MLP + `o_proj` scope. |
| **`o_proj`** | BF16 | NVFP4. It was excluded only as a side effect of the blanket `*self_attn*` disable, costing 10.8% of the model for no measured accuracy benefit. |
| **Loader** | `--streaming` | Off. 229 GB fits in 384 GB of VRAM. |

The adapter overrides `get_all_quant_overrides()` rather than merging into `COMMON_QUANT_OVERRIDES`. Merging would leave both `*self_attn*weight_quantizer: disable` and an `o_proj: enable` in the same `quant_cfg` and rely on later-pattern-wins ordering — a contract not worth depending on when the alternative is an explicit list.

Version requirements and the environment probe are in [Install.md §7](Install.md).

## 2. Layer scope

### The model

Dense `MistralForCausalLM`, 122.61B BF16 params, 88 layers, hidden 12288, MLP 28672, GQA **96 query heads / 8 KV heads**, head_dim 128, 131072 context, `sliding_window: null`, `rope_theta` 1e6, vocab 32768, **untied** embeddings.

Per-layer parameter counts drive everything below:

| Tensor | Params / layer | Total (×88) | Share of model |
| --- | ---: | ---: | ---: |
| `gate_proj` + `up_proj` + `down_proj` | 1,056,964,608 | 93.01 B | **75.9%** |
| `o_proj` | 150,994,944 | 13.29 B | **10.8%** |
| `q_proj` | 150,994,944 | 13.29 B | 10.8% |
| `k_proj` + `v_proj` | 25,165,824 | 2.21 B | 1.8% |
| `embed_tokens` + `lm_head` | — | 0.81 B | 0.7% |

### What gets quantized

**NVFP4 W4A4:** `gate_proj`, `up_proj`, `down_proj`, `o_proj` — 106.30 B params, **86.7% of the model**.

**BF16:** `q_proj`, `k_proj`, `v_proj`, `embed_tokens`, `lm_head`, all norms, and the KV cache — 16.31 B params.

This is NVIDIA's `nvfp4_omlp_only` scope, and the adapter now gets it by starting from their actual preset rather than reconstructing it:

```python
base_quant_cfg="NVFP4_OMLP_ONLY_CFG",
```

Their PTQ guidance is to start at the narrowest activation-quantized scope and widen only as far as your target requires: `mlp_only` → `omlp_only` → `default`, describing `omlp_only` as *"a middle ground… adds the o_proj GEMM (often safe) without quantizing the more sensitive q/k/v projections."* The explicit q/k/v disables in the adapter are redundant against that preset and kept as insurance.

A second adapter, `behemoth_r1_123b_qkv`, adds all of q/k/v to that scope, leaving only embeddings and `lm_head` in BF16. It is built up from `NVFP4_DEFAULT_CFG` by subtracting those two, **not** from `NVFP4_OMLP_ONLY_CFG` with q/k/v re-enabled — that preset never creates attention quantizers, so no override can switch them back on. Starting from the everything-preset and subtracting is the only construction that reliably widens the scope.

There is also `behemoth_r1_123b_q`, which quantizes `q_proj` alone. **It exports correctly and cannot be served**; see "Fused layers constrain the scope" below. It is left in the tree so the mistake stays documented rather than repeatable.

### Why this split and not another

The [FP4 inference sensitivity analysis](https://arxiv.org/html/2603.08747v1) measures activation outlier ratio (max ÷ P99.9) per component. Higher is harder to quantize:

| Component | Outlier ratio | Verdict here |
| --- | ---: | --- |
| `down_proj` | **80–334×** | Worst in the model — and it is inside the MLP we must quantize. This is what the calibration algorithm in §4 exists to handle. |
| `up_proj` / `gate_proj` | 5.5–11× | Fine. |
| `v_proj` | 2.9–4.8× | Moderate, and only 1.1 B params. Not worth it. |
| `q_proj` | 2.9–4.7× | Moderate, and 10.8% of params. Skipped by default, but it is 29% of the exported file — quantized in the `behemoth_r1_123b_qkv` variant so the cost can be measured rather than assumed. It cannot be quantized without k/v — see "Fused layers constrain the scope". |
| `k_proj` | ~2.9–4.9× | Second-least sensitive, but only 1.1 B params. Nothing to gain. |
| `o_proj` | **2.6–3.6×** | **Lowest in the model.** 10.8% of params, free speed. Quantize it. |

Two notes on the evidence. First, [NVFP4 pretraining work](https://arxiv.org/html/2602.02047) reaches a different conclusion on `o_proj`/`v_proj`, ranking them most sensitive once normalized per parameter, and NVIDIA's *pretraining* recipe also excludes the last 4 layers. That is a training-stability result; for PTQ inference the inference-side measurement plus NVIDIA's own `omlp_only` PTQ preset are the better guide. Second, none of NVIDIA's PTQ presets exclude layers by index, so this recipe does not either — but see §8 if evals regress.

Why `k_proj`/`v_proj` are especially pointless to quantize here: GQA with 8 KV heads makes them 12.6 M params each versus 151 M for `q_proj`/`o_proj`. They are 1.8% of the model combined.

### Size at each scope

NVFP4 stores 4 bits per weight plus one FP8-E4M3 scale per 16-element block = 4.5 bits = 0.5625 bytes/param.

| Scope | NVFP4 | BF16 | Total | vs BF16 | Servable |
| --- | ---: | ---: | ---: | ---: | :---: |
| `mlp_only` (first pass) | 93.0 B | 29.6 B | 111.5 GB | 2.06× | yes |
| **`omlp_only` (`behemoth_r1_123b`)** | 106.3 B | 16.3 B | **92.4 GB** | **2.48×** | yes |
| `omlp_only` + `q_proj` (`behemoth_r1_123b_q`) | 119.6 B | 3.0 B | 73.3 GB | 3.13× | **no — see below** |
| **all linears (`behemoth_r1_123b_qkv`)** | 121.8 B | 0.8 B | **70.1 GB** | **3.27×** | yes |
| all linears, q/k/v weight-only (`..._qkv_a16`) | 121.8 B | 0.8 B | 70.1 GB | 3.27× | yes, untested |

Two rows are confirmed by measurement rather than estimated: `omlp_only` exported at 86.1 GiB (92.41 GB decimal) and the `q_proj` variant at 68.3 GiB (73.3 GB). The predictions were exact, so treat the remaining row as reliable — all-linear should land at 65.3 GiB.

Adding `o_proj` to the first-pass recipe saves 19 GB and moves 10.8% more of the model onto FP4 tensor cores.

### Fused layers constrain the scope, and they are not negotiable

**vLLM merges `q_proj`, `k_proj` and `v_proj` into one `QKVParallelLinear`, and `gate_proj` with `up_proj` into one `MergedColumnParallelLinear`. Every shard of a fused layer must have the same precision.** Mixing them raises, at load time:

```
ValueError: Detected some but not all shards of model.layers.0.self_attn.qkv_proj
are quantized. All shards of fused layers to have the same precision.
```

This is checked in `is_layer_skipped`, which compares each fused member against the checkpoint's `ignore` list. ModelOpt has no such notion, so it will happily calibrate, quantize and export a mixed group — the file is internally consistent and passes every per-module check, and is still unloadable.

The consequence is that for attention there are exactly **two** legal choices, all of q/k/v in BF16 or all of them in NVFP4, and nothing in between. `behemoth_r1_123b_q` violates this and is retained only as a documented dead end. `verify_nvfp4_export.py` now checks fused-group uniformity, so this fails in seconds rather than after a 15-hour run.

Note why the two working scopes were never at risk: `omlp_only` quantizes `gate_proj` and `up_proj` together and leaves all of q/k/v alone, so both groups are uniform by accident of the preset's design.

### NVFP4 weights do not require NVFP4 activations

The rule above is about **quantized versus not quantized**, not about W4A4 versus weight-only. NVFP4 with BF16 activations is a real configuration, this repo already uses it — `models/base.py` disables `*self_attn*input_quantizer` while leaving weight quantizers on, and `glm5_1` and `qwen3_5_moe` mix the two per module — and it opens a scope the earlier framing missed.

`behemoth_r1_123b_qkv_a16` drops only the q/k/v input quantizers. Every shard of the fused layer carries the same scheme, so it is legal, and it removes the reason k/v were unwanted in the first place: those two have the widest activation range per parameter in the model, and it was never their *weights* that were the concern. This quantizes the weights, which is where the bytes are, and leaves their activations in BF16. Disk cost is nil — weight bytes are identical, only the `input_scale` scalars disappear.

What it costs is math throughput. Blackwell's FP4 tensor cores need both operands in FP4 (§3), so these GEMMs fall back to dequantize-and-BF16 and keep only the bandwidth win. q/k/v are ~12% of linear FLOPs per layer: close to free during bandwidth-bound decode, real during prefill. `o_proj` stays W4A4 — its input is the attention output, a different tensor, and the 86 GiB run already measured that as cheap.

**A fused group must share the activation scheme too, and this failure is quieter than the precision one.** If `q_proj` were W4A4 and k/v weight-only, every shard is quantized, so vLLM's own guard says nothing while the concatenated tensor is served by a single kernel that can only apply one scheme. `verify_nvfp4_export.py` reports `NVFP4/A4`, `NVFP4/A16` or `NVFP4/A-MIXED` per projection and fails a group that mixes them.

Two things to establish before trusting an export from this config, neither of which is settled:

1. That ModelOpt 0.46 describes a mixed W4A4/weight-only checkpoint in a form vLLM reads correctly. It plausibly needs two `config_groups`, and the exporter has only been observed emitting one here.
2. That the prefill regression is smaller than the accuracy gain.

The cheap part is producing it: weight amaxes are unchanged from the `qkv` run and the q/k/v activation amaxes simply go unused, so the same amax file works with `--resume-amax` and the run is hours rather than days.

### Why the file is not a quarter of BF16

The obvious expectation is 245.2 GB ÷ 4 = 61.3 GB, and `omlp_only` lands 51% above it. Two effects compound:

| Component | Params | Bytes/param | Size |
| --- | ---: | ---: | ---: |
| NVFP4 weights | 106.30 B | 0.5 | 53.15 GB |
| NVFP4 block scales | — | 0.0625 | 6.64 GB |
| `q_proj` BF16 | 13.29 B | 2 | **26.58 GB** |
| `k_proj` + `v_proj` BF16 | 2.21 B | 2 | 4.43 GB |
| `embed_tokens` + `lm_head` BF16 | 0.81 B | 2 | 1.61 GB |

First, **86.7% of parameters are 4-bit but only 65% of the bytes.** BF16 costs 3.6× more per parameter than NVFP4-with-scales, so the 13.3% left out becomes 35% of the file. Second, the block scales are not optional overhead you can tune away: one FP8 byte per 16 weights is 12.5% on top of every quantized tensor, and `group_size` must stay 16 because b12x hardcodes `sf_vec_size=16` (§3).

A literal quarter of BF16 is unreachable in this format. With embeddings and `lm_head` in BF16 — which they should be — **~70 GB is the floor at any scope.**

### The small variant, and why it is about latency rather than disk

Attention is 24 GB of BF16, and GQA makes the distribution counterintuitive: "keep QKV in BF16" sounds cheap, but with 8 KV heads `k_proj` and `v_proj` are 12.6 M params each while `q_proj` is 151 M — the same size as `o_proj`, which we already quantize. The 26.6 GB QKV bill is 92% Q.

That arithmetic is what made a Q-only variant look attractive: 19.1 of the 24 GB for the cheapest third of the risk, leaving k/v — the widest activation range per parameter — in BF16 for only 4.4 GB. The fused-layer rule above makes it unbuildable, so `behemoth_r1_123b_qkv` takes all of q/k/v and reaches 70.1 GB. Including k/v is not a choice being defended on its merits; it is the price of a loadable checkpoint, and it happens to cost only 3.2 GB more of the model's most sensitive weights.

Disk is not the reason to do any of this. At 86 GiB across four cards you use 21.6 GiB per GPU and have ~62 GiB of KV cache each; there is no pressure. The reason is §3's finding that TP4 on SM120 pays 176 PCIe all-reduces per token with every fast path disabled. **At 65.3 GiB the model loads on a single 96 GB card**, which removes tensor parallelism entirely and lets you run four independent replicas. At 86 GiB that is not possible.

This is a hypothesis about a quality/latency trade, so it is not the default. `behemoth_r1_123b_qkv` exists to be measured against `behemoth_r1_123b` — see §6.6.

### KV cache in BF16 costs you very little here

GQA with 8 KV heads: 88 layers × 2 × 8 heads × 128 dim × 2 bytes = **352 KiB/token**, so a full 131,072-token sequence is **44 GiB**. With 92 GB of weights across 384 GB, you have roughly 290 GB for KV. BF16 KV is comfortable on this box — the "never quantize KV" rule is close to free at this architecture.

## 3. B12X / SparkInfer — and why it decides the whole plan

[`sparkinfer`, formerly and still API-named `b12x`](https://github.com/local-inference-lab/sparkinfer), is an **SM120/SM121 CuTe DSL kernel library** explicitly targeting DGX Spark, RTX 5090, and **RTX PRO 6000 Blackwell**. That is your hardware. The name has gone `b12x` → `sparkinfer` → back to `b12x` in integration identifiers; the repo README says "sparkinfer (formerly b12x)" while FlashInfer and vLLM still expose it as `b12x`.

### How it works

RTX PRO 6000 Blackwell is **compute capability 12.0 (SM120)**, not SM100 like B200. It has FP4 tensor cores but **no TMEM, no `tcgen05`, no 2-CTA instructions, no multi-cluster**. The datacenter NVFP4 kernels are built on exactly those features, so they do not port.

b12x rewrites the block-scaled GEMM for that target: **warp-level MMA** (`MmaMXF4NVF4Op`, atom m16n8k64, atom_layout (4,2,1)), 256 MMA + 32 DMA threads, `PipelineTmaAsync`, cluster shape always (1,1,1), adaptive tile sizing to keep SM utilization up on small-M decode shapes.

Three consequences for how we quantize:

1. **NVFP4 only, block size 16.** The kernel hardcodes `sf_vec_size=16` with FP4-E2M1 operands and FP8-E4M3 scales. MXFP4 (block 32) is not supported. `NVFP4_DEFAULT_CFG` produces exactly this layout — so the format is right, and MXFP4 would have been a dead end.
2. **ModelOpt export is a first-class input.** b12x's weight planner takes `source_format="modelopt_nvfp4"` directly. This toolkit's exporter is the correct producer.
3. **Only NVFP4 layers get the fast path.** Anything left in BF16 runs an ordinary BF16 GEMM. This is the entire reason for moving `o_proj` into scope: at `mlp_only`, 23.5% of the model's GEMM work never touches an FP4 tensor core.

### The SM120 trap you need to know about

On this exact hardware, the **MoE** NVFP4 path has been badly broken. A published run of `Qwen3.5-397B-A17B-NVFP4` on 4× RTX PRO 6000 found native CUTLASS NVFP4 MoE producing **garbage output**, and the best working configuration was a **Marlin W4A16 fallback at 50.5 tok/s** — Marlin dequantizes FP4 to FP16 and runs a standard GEMM, giving up roughly half the theoretical throughput. Expert parallelism over PCIe collapsed to 1.4–2.6 tok/s. vLLM also had `is_device_capability_family(100)` checks that simply did not recognise SM120.

**Behemoth is dense, so none of the grouped-GEMM MoE breakage applies.** You are on the `mm_fp4` / `FlashInferB12xNvFp4LinearKernel` dense path, which is the part b12x was actually built for. That is a genuine advantage of quantizing a dense 123B rather than a big MoE on this box — but it does mean you should explicitly verify at serve time that you got the b12x kernel and not a Marlin fallback (§7).

## 4. Calibration algorithm

`configs/calib_behemoth_r1_123b.toml` uses `method = "mse"`, which is the best option present in the pinned ModelOpt 0.46.0. It max-calibrates everything, then refines each NVFP4 weight block by sweeping the 126 valid FP8-E4M3 block-scale candidates instead of taking the plain max. Triton-accelerated.

```toml
[calibration]
method = "mse"
```

### What we would rather use, and why

The hard problem is `down_proj`: its input activations have a max/P99.9 ratio up to 334×. Under `max` calibration the per-tensor global scale is set by that lone outlier and — in ModelOpt's own words — "chasing a lone outlier pushes every other block's FP8 block scale below subnormal."

`nvfp4_act_headroom` fixes exactly that, anchoring the activation global scale low in the FP8 range and clipping above `upper_percentile` (default 99.99). But it does not appear in the 0.46.0 changelog and is likely `main`-only. `mse` does **not** address it — `mse` refines *weight* scales and leaves activation scales max-based.

So run the probe. If it reports `nvfp4_act_headroom` as available, prefer it:

```toml
[calibration]
method = "nvfp4_act_headroom"
upper_percentile = 99.99
rho = 16384.0
weight_scale_method = "mse"
```

That combination gives clipped activation scales *and* swept weight scales. Guaranteed-safe fallback is `method = "max"`. Never set `method = "quantile"` — see [Install.md §7.3](Install.md).

### One caveat before a long run

`--resume-amax` reads `_calibrator._calib_amax`, which non-`max` calibrators may not expose, so resume may silently not work under `mse` or `nvfp4_act_headroom`. Establish that in the smoke run (§6.3) rather than discovering it 30 hours in.

## 5. Calibration data

Your point stands: the shipped mix (16,329 diverse + 15,264 coding + 819 long-context + 23,458 generic coding) is wrong for a creative-writing/RP finetune. Coding was the majority of it.

New spec: **`data/behemoth_r1_123b_calib.yaml`** — 16,000 samples across 39 sources, at exactly the split you asked for. A real build collects 16,000/16,000 and keeps **15,904** after truncation and the short-sample filter.

| Bucket | Samples | Share | Leading sources |
| --- | ---: | ---: | --- |
| Creative writing | 9,600 | 60% | Opus-WritingPrompts, euclaise/writingprompts, ChatGPT-4o-Writing-Prompts, gutenberg3, nopm_claude_writing, writingPromptAug, Prosemaxx-Adventure, bookcorpusopen, movie scripts, poetry |
| — of which long-form | 1,200 | — | bookcorpusopen, movie scripts, run at **8192** rather than 4096 |
| Roleplay | 3,200 | 20% | Sonnet3.5-Charcard-Roleplay, stheno-filtered, kalo-opus-no-refusal, Roleplay-Anime-Characters, Kinomaxx-VanillaBackrooms |
| General reasoning | 1,600 | 10% | OpenThoughts-114k, OpenMathReasoning (cot), NuminaMath-CoT, OpenScienceReasoning-2, theory-of-mind, physical-reasoning |
| General breadth | 1,600 | 10% | ultrachat, dolly, no_robots, neuralmagic/calibration, philosophy, SocraticChat, HelpSteer, medical, legal, finance, multilingual |

Sources are drawn from your own specs in `data/examples/`, but the `columns` and `formatter` values there are **not** all correct, so they were re-verified against the Hub with `tools/check_calib_spec.py`. Three entries had column names that do not exist (silently yielding 0 samples), two exceeded their split size, one was a script-based repo `datasets` v4 cannot load, and `storytracer/US-PD-Books` turned out to hold no text at all — only metadata and an archive.org URL, so it produced 400 book *titles*. Run the validator after any edit to the YAML:

```bash
python tools/check_calib_spec.py --yaml data/behemoth_r1_123b_calib.yaml
```

The book and screenplay sources carry a per-entry `max_seq_length: 8192`. `quantize.py` sets `max_len` per `[[dataset]]`, so the builder emits one JSONL per distinct length and the TOML references both. Sample counts and the 60/20/10/10 split are untouched — only the token budget shifts, from a 65.5M ceiling to 70.5M.

Code is deliberately **100 samples (0.6%)** — enough to keep 2411's latent coding range represented, not enough to bias scales. Set `nvidia/OpenCodeInstruct` to `num_samples: 0` for a zero-code run.

### Reasoning is planned for, explicitly

Two things matter because the model will run with a `<think>` prefill:

1. **PTQ only ever sees prefill.** `quantize.py` calls the model with `use_cache=False` and never generates. So a `<think>` prefill at serve time is only in-distribution if reasoning-tagged text was in the calibration set to begin with. The 10% reasoning slice is there for that.
2. **Tag mismatch is real.** OpenThoughts and friends use `<|begin_of_thought|>` / `<|end_of_thought|>`, not `<think>`. The builder's `--think-tag think` rewrites those markers (and `<reasoning>`, `<thinking>`, and the `begin_of_solution` wrappers) to `<think>…</think>` so calibration sees the tokens the model will actually emit.

### There was no builder for these YAMLs

`tools/build_calib_dataset.py` only has a hardcoded `--mode coding|diverse` plan and cannot read the `calibration_set` schema. Nothing in the repo consumed `data/examples/*.yaml`. Added **`tools/build_calib_from_yaml.py`**, which implements the four formatters (`prompt_answer`, `sharegpt`, `chat_completion`, `raw_text`), honours `subset`/`split`/`streaming`/`num_samples`, shuffles by seed, and emits the `{"messages": [...]}` JSONL that `quantize.py` expects.

It also **drops `tool` and `function` roles**. This is not optional: Behemoth's Mistral v7 template calls `raise_exception('Only user, system and assistant roles are supported!')`, so tool-role samples would silently fall through `quantize.py`'s exception handler into out-of-template concatenated text.

### Chat template

```jinja
{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% elif message['role'] == 'system' %}{{ '[SYSTEM_PROMPT] ' + message['content'] + '[/SYSTEM_PROMPT]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% endif %}{% endfor %}
```

Mistral v7 non-Tekken. `quantize.py` uses `tokenizer.apply_chat_template`, which is correct — the card warns that wrong whitespace degrades the model, so never hand-roll `[INST]`.

## 6. Runbook

### 6.1 Does it go faster with more GPUs?

Two separate answers, because ModelOpt and *this toolkit* differ.

**ModelOpt does support true parallel multi-GPU PTQ.** `max_calibrate()` defaults to `distributed_sync=True` and MAX-all-reduces each quantizer's amax across the data-parallel group; `mse_calibrate` and `local_hessian_calibrate` take the same flag. On top of that, `hf_ptq.py --use_fsdp2` shards decoder layers with FSDP2 under `torchrun`, and `parallel_load_and_prepare_fsdp2` round-robins layers across ranks so each rank reads only `model_size / world_size` from disk. That is genuine data parallelism — all four GPUs computing different batches at once.

**This toolkit does not use any of it.** `quantize.py` is a single-process script built on `AutoModelForCausalLM.from_pretrained(device_map="auto")` or its own streaming loader. There is no `torchrun` entry point, no process group, no FSDP2. So as written, four GPUs give you **capacity, not throughput**: `device_map="auto"` is naive pipeline parallelism — layers split across devices and executed in sequence, one GPU computing while the other three idle.

So on this toolkit:

- **The one big win is fitting.** 229 GB into 384 GB means you can drop `--streaming`. The streaming loader shuttles every layer's weights to GPU 0 and back for *every batch*; avoiding that is worth multiples, not percentages. Going from 2 GPUs (must stream) to 4 (fits) is the largest single speed decision here.
- **Past fitting, extra cards add nothing.** A 5th and 6th GPU would not speed up calibration. No NVLink is fine, since pipeline-parallel PTQ only passes activations between stages.

**If you want the ~4× data-parallel speedup**, there are two routes, and both cost something:

| Route | Gets you | Costs you |
| --- | --- | --- |
| NVIDIA's `hf_ptq.py --use_fsdp2` under `torchrun --nproc_per_node=4` | Working DP calibration today | This repo's 16k JSONL pipeline, amax checkpoint/resume, and streaming export. `hf_ptq.py` drives its own dataset handling and recipe format. |
| Add FSDP2 + `torchrun` to `quantize.py` | Keeps this repo's pipeline | Real work: process group setup, per-rank dataset sharding, an FSDP-wrapped forward loop, and — the hard part — a DTensor-aware full-state-dict gather in the streaming exporter. |

Two caveats that blunt the win. NVIDIA's own note: *"FSDP2 is designed for training workloads and may result in longer calibration and export times."* And under data parallelism, `nvfp4_act_headroom` combines per-rank scales with a MAX all-reduce, giving *"the largest per-rank headroom scale rather than the scale implied by pooling every rank's per-block distribution"* — so DP mildly degrades exactly the percentile calibration we chose in §4.

**Budget real time for the single-process path.** 14,800 samples at 4096 plus 1,200 at 8192, with `--batch-tokens 32768`, is roughly 2,150 forward passes of a 123B model one-GPU-at-a-time, plus fake-quant overhead and an amax checkpoint after every batch. Expect **multiple days**. Measure the first 20 batches and extrapolate before committing the whole run.

### 6.2 Build the calibration JSONL

```bash
source .venv/bin/activate
uv pip install datasets pyyaml      # not in pyproject; tool-only deps

python tools/build_calib_from_yaml.py \
    --yaml data/behemoth_r1_123b_calib.yaml \
    --output data/text/behemoth_r1_123b_calib.jsonl \
    --think-tag think
```

This writes `..._4096.jsonl` (14,704) and `..._8192.jsonl` (1,200), and prints the matching `[[dataset]]` blocks plus a bucket-mix report. A healthy run reports `Collected 16000/16000; kept 15904`, with the mix within a tenth of a point of 60/20/10/10.

Watch for two things in that output. A `<< SHORT` marker means a source could not fill its quota — run `tools/check_calib_spec.py` to find out why. And the truncation count (~1,800) is expected, not a warning: it is mostly the long-form slice being cut to the character budget, which is the intended behaviour.

Then check what you got:

```bash
wc -l data/text/behemoth_r1_123b_calib_*.jsonl
python - <<'PY'
import glob, json
from collections import Counter
for path in sorted(glob.glob("data/text/behemoth_r1_123b_calib_*.jsonl")):
    n, think, roles = 0, 0, Counter()
    for line in open(path, encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        n += 1
        roles.update(m["role"] for m in msgs)
        if any("<think>" in m["content"] for m in msgs):
            think += 1
    print(path, "| samples", n, "| with <think>", think, "| roles", dict(roles))
PY
```

`roles` must contain only `system`, `user`, `assistant` — anything else means Behemoth's Mistral v7 template will raise during calibration. The `<think>` count should be a few thousand, coming from the reasoning slice.

If you land far short of 16,000, raise `num_samples` on sources the validator confirms have headroom rather than adding new dataset IDs.

At `--batch-tokens 32768` this plan becomes **2,138 batches**: 1,838 at batch 8 / maxlen 4096, and 300 at batch 4 / maxlen 8192. That number is worth writing down — it is what the progress counter divides by.

### 6.3 Smoke run first

Confirm the environment, then validate the config on a few hundred samples. Do not start a multi-day job on either untested.

```bash
python tools/check_modelopt.py
```

That must exit 0, and it tells you whether `mse` and `nvfp4_act_headroom` exist in your build. Then copy the TOML, point it at a small head of each JSONL, and drop to `method = "max"` so the first pass tests the pipeline rather than the algorithm:

```bash
head -n 448 data/text/behemoth_r1_123b_calib_4096.jsonl > data/text/behemoth_smoke_4096.jsonl
head -n  64 data/text/behemoth_r1_123b_calib_8192.jsonl > data/text/behemoth_smoke_8192.jsonl
sed -e 's/behemoth_r1_123b_calib_/behemoth_smoke_/' \
    -e 's/^method = .*/method = "max"/' \
    -e '/weight_scale_method/d' \
    configs/calib_behemoth_r1_123b.toml > configs/calib_behemoth_smoke.toml

python -u quantize.py \
    --model behemoth_r1_123b \
    --model-id /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --export-dir /media/fmodels2/working_Model-Opt/smoke \
    --calib-config configs/calib_behemoth_smoke.toml \
    --batch-tokens 32768 2>&1 | tee /media/fmodels2/working_Model-Opt/smoke.log
```

> **`-u` is not optional when piping.** Piped stdout is block-buffered at 8 KB, while tqdm writes to stderr and keeps updating — so the run looks hung after `Loading weights: 100%` while the dtype table and every `Batch i/N` line sit unflushed. The whole smoke run emits under 2 KB, so without `-u` you may see nothing until it exits.

Two other things that look like faults but are not. `nvidia-smi` will show low, spiky per-GPU utilization: `device_map="auto"` gives **pipeline** parallelism, so one GPU computes while the other three wait. And host RSS stays near 3 GB because the weights live in VRAM, not system RAM.

That is **72 batches** (56 + 16), about 3.4% of the full run's 2,138, and it proves model load, chat templating, quantizer placement, calibration **and export** — instead of finding an export bug on day three. Verify the result with §6.5 before going further; a smoke export with the wrong layer scope is the cheapest possible place to catch that.

Then re-run the same smoke config with the algorithm you actually intend to use, this time saving amaxes, to settle the `--resume-amax` question from §4 before it matters:

```bash
sed -i 's/^method = .*/method = "mse"/' configs/calib_behemoth_smoke.toml

mkdir -p /media/fmodels2/working_Model-Opt/smoke_mse
python -u quantize.py \
    --model behemoth_r1_123b \
    --model-id /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --export-dir /media/fmodels2/working_Model-Opt/smoke_mse \
    --calib-config configs/calib_behemoth_smoke.toml \
    --batch-tokens 32768 \
    --save-amax /media/fmodels2/working_Model-Opt/smoke_mse/amax.safetensors
```

Two things to confirm: ModelOpt accepts the `mse` algorithm dict, and `amax.safetensors` is written and non-empty. Note that `mse`'s weight sweep runs *after* all batches and its cost scales with parameter count, not sample count — so whatever it adds here, it adds to the full run too.

### 6.4 Full run

```bash
chmod +x scripts/quantize_behemoth_r1_123b.sh
./scripts/quantize_behemoth_r1_123b.sh
```

That script exports to the working dir and then `mv`s to the final path (same filesystem, so it is a rename, not a copy). Explicitly:

```bash
python quantize.py \
    --model behemoth_r1_123b \
    --model-id /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --export-dir /media/fmodels2/working_Model-Opt/Behemoth-R1-123B-v2-nvfp4 \
    --calib-config configs/calib_behemoth_r1_123b.toml \
    --batch-tokens 32768 \
    --save-amax /media/fmodels2/working_Model-Opt/amax/behemoth_r1_123b.safetensors
```

Note the amax path is **not** under the export directory. The export directory gets renamed to the final model path on success, so anything written inside it is published with the weights — and amaxes are calibration state, not part of the checkpoint.

No `--streaming` (fits in VRAM). No `--floor-amaxes` (MoE-only). No `--save-quantiles` (quantile path only).

Raise `--batch-tokens` to 65536 if the first batches leave headroom; drop to 16384 if a long batch OOMs.

Disk: 229 GB source + ~92 GB export + a few GB of amax on 2 TB. Fine.

For the smaller variant, `scripts/quantize_behemoth_r1_123b_qkv.sh` runs the same command with `--model behemoth_r1_123b_qkv` and its own working and final paths. The calibration TOML is shared deliberately: identical data and method mean a KLD comparison between the two exports isolates the effect of quantizing attention and nothing else.

The amax file from the 86 GiB run is not reusable — q/k/v had no quantizers then. **The amax file from the unservable `q_proj` run is**, which is the one good thing to come out of that mistake; see §6.4b.

### 6.4b Reaching the qkv scope without recalibrating

Widening from `q_proj` to all of q/k/v needs two amaxes per layer per projection, and both are exactly recoverable from the run that already finished. Nothing is estimated:

- **Input amax.** `q_proj`, `k_proj` and `v_proj` all read the same tensor, the output of `input_layernorm`. A max-calibrated amax over the same data and the same tensor is the same number. This is an identity, not an approximation.
- **Weight amax.** An absmax over stored weights. It never depended on calibration data, and comes straight out of the BF16 source.

So the 16k-sample pass can be skipped entirely:

The amax file is **inside the published model directory**, not the working directory. The scripts wrote `--save-amax "$WORK/amax.safetensors"` and then renamed `$WORK` to `$FINAL` on success, so calibration state was carried into the model. Both scripts now write amaxes to `working_Model-Opt/amax/` instead, and move `amax_checkpoint.safetensors` out of the export directory before the rename — but the file from the run that already happened is at the old location:

```bash
python tools/synth_kv_amax.py \
    --amax /media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4-q/amax.safetensors \
    --model /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --output /media/fmodels2/working_Model-Opt/amax/behemoth_r1_123b_qkv.safetensors

python -u quantize.py \
    --model behemoth_r1_123b_qkv \
    --model-id /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --export-dir /media/fmodels2/working_Model-Opt/Behemoth-R1-123B-v2-nvfp4-qkv \
    --calib-config configs/calib_behemoth_r1_123b.toml \
    --batch-tokens 32768 \
    --resume-amax /media/fmodels2/working_Model-Opt/amax/behemoth_r1_123b_qkv.safetensors \
    --resume-batch 999999
```

`--resume-batch` past the batch count skips every forward pass, leaving load, weight quantization and export — a couple of hours instead of fifteen. The tool reads ModelOpt's amax shape conventions off the `q_proj` entries rather than assuming them, and refuses to write if a key is missing, a layout is unrecognised, or the file already carries k/v amaxes.

Three log lines decide whether this worked. All of them must be right before the export is trusted, because a restore that silently misses quantizers produces a checkpoint that verifies clean and is quietly wrong:

| Line | Expected |
| --- | --- |
| `Restored N/1232 calibrator amaxes` | **1232/1232.** Anything less means module names did not match. |
| `Summary: N nonzero, N zero, N NaN` | **1232 nonzero, 0 zero, 0 NaN**, preceded by `(dense model: ...)`. |
| `Saved N amax values` | **1232**, with no `WARNING` lines after it. |

The middle one needed a fix to be worth reading. That diagnostic only walked MoE expert `ModuleList`s, so on a dense model like this one it counted nothing and printed `0 nonzero, 0 zero, 0 NaN` regardless of the actual state — a passing-looking result that meant nothing. It now falls back to every quantizer in the model when no expert lists are found.

Note that `method = "mse"` still does its full weight-scale sweep on this run. That is the point: the sweep is data-free, it is where NVFP4 weight quality comes from, and skipping calibration does not skip it. Expect the run to cost model load, a pointless but harmless tokenization pass over the calibration JSONL, the MSE sweep, and the export.

Expected log landmarks:

1. Calibration plan — 2 datasets: batch 8 @ maxlen 4096, batch 4 @ maxlen 8192
2. Dtype distribution before quant — essentially all BF16, ~245 GB
3. `Batch i/N` with an amax checkpoint after each
4. `Tied gate/up weight_quantizer amax for 88` — one pair per layer, dense not experts
5. Streaming export of shards, `model-inputscales.safetensors`, `config.json`, tokenizer files

### 6.5 Verify the export

```bash
python tools/verify_nvfp4_export.py \
    /media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4
```

It reads `config.json`, the index, and tensor *metadata*, loading only the small scale tensors — seconds, no GPU. It must exit 0. What it asserts:

- **No KV cache quantization.** BF16 KV means the key is absent entirely.
- **`group_size` 16** — b12x hardcodes `sf_vec_size=16`.
- **Scope**: 88 modules each of `o_proj`/`gate_proj`/`up_proj`/`down_proj` carry `weight` + `weight_scale` + `weight_scale_2` + `input_scale`; all `q_proj`/`k_proj`/`v_proj` plus `lm_head` and `embed_tokens` carry a lone BF16 `weight`.
- **Every unquantized linear is listed in the exclusion list.** Since `targets` is `["Linear"]`, anything unlisted gets an NVFP4 linear method, hunts for a `weight_scale` that does not exist, and fails at *load*, not at export.
- **Dtypes**: `U8` packed weights, `F8_E4M3` block scales, `F32` global and input scales.
- **Scales** finite and not uniformly zero; tokenizer files and chat template present.

### Two `quantization_config` layouts

Do not hand-check this with `q.get("group_size")`. ModelOpt ≥ 0.29 emits the **compressed-tensors** layout, where `group_size` is nested in `config_groups.group_0.weights`, exclusions live under `ignore`, and KV quantization would appear as `kv_cache_scheme`. Older releases emit a flat TRT-LLM layout with top-level `group_size` and `exclude_modules`. On 0.46.0 you get the former:

```json
{
  "config_groups": {"group_0": {
      "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
      "weights":           {"num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
  "ignore": ["lm_head", "model.embed_tokens", "model.layers.0.self_attn.q_proj", "..."],
  "quant_algo": "NVFP4",
  "quant_method": "modelopt"
}
```

Reading the flat keys against this reports false failures on a perfectly good checkpoint — and, far worse, silently misses real FP8 KV, which only ever surfaces as `kv_cache_scheme` here. `tools/verify_nvfp4_export.py` handles both.

For this recipe `ignore` should hold exactly **266** entries: 88 × 3 for `q/k/v_proj`, plus `lm_head` and `model.embed_tokens`. Note these are literal module paths, not wildcards — vLLM `fnmatch`es them, and an exact string matches itself. It returns `UnquantizedLinearMethod` for those, so a partially quantized NVFP4 checkpoint is a supported configuration — **provided the exclusions never split a fused layer.** `ignore` is exactly the list `is_layer_skipped` consults when it decides a fused group disagrees with itself.

Pass `--scope` to match the export, or the tool will correctly fail it. All three scopes are numerically distinguishable, which is the cheapest way to confirm the config took:

| | `omlp` | `omlp-q` | `all-linear` |
| --- | ---: | ---: | ---: |
| NVFP4 modules | 352 | 440 | 616 |
| Index tensors | 1851 | 2115 | 2643 |
| `ignore` entries | 266 | 178 | 2 |
| Amax entries | 704 | 880 | 1232 |
| Total size | 86.1 GiB | 68.3 GiB | ~65.3 GiB |
| Loads in vLLM | yes | **no** | yes |

The `omlp` and `omlp-q` columns are measured. The verifier's **Fused layer groups** section is the one that matters before committing a day to a run: it reports `qkv_proj` and `gate_up_proj` as uniform or `MIXED PRECISION`, and fails on the latter with the exact `ValueError` vLLM would raise hours later.

### 6.6 Measuring the two variants against BF16

Perplexity on its own is not enough to choose between these. A quant can hold PPL flat while reshuffling the distribution below the argmax, and that reshuffling is exactly what degrades long-form prose. `tools/kld_eval.py` reports KL divergence against the BF16 model, plus PPL and top-1 agreement.

It is teacher-forced and deterministic, and sends **token IDs rather than text** so both runs score byte-identical positions — no tokenizer drift can slip in between reference and candidate. It refuses to compare two runs whose prompt IDs differ. Rows with a `messages` field go through the chat template, so the scored tokens are the ones the model sees when served; scoring raw text would measure a distribution nobody uses.

**Build the held-out set first.** `data/behemoth_r1_123b_eval.yaml` mirrors the calibration mix at 60/20/10/10 with 500 samples. Both a new seed *and* `--exclude` are required: a different seed reshuffles, but can still draw rows the calibration run already used, and measuring drift on data the scales were fitted to understates it.

```bash
python tools/build_calib_from_yaml.py \
    --yaml data/behemoth_r1_123b_eval.yaml \
    --output data/text/behemoth_r1_123b_eval.jsonl \
    --think-tag think \
    --exclude data/text/behemoth_r1_123b_calib_4096.jsonl \
              data/text/behemoth_r1_123b_calib_8192.jsonl
```

`--exclude` matches on a hash of each sample's first 1024 characters, not the whole body. Samples already on disk were think-normalized and truncated to a character budget, so a full-body hash would never match its own earlier copy; the head survives both transforms.

**Serve each model in turn and collect.** `--max-logprobs 64` is not optional — vLLM defaults to 20, which is too coarse for a stable KLD tail, and the tool warns if the candidate's top-k fails to cover 98% of the reference's probability mass.

```bash
vllm serve /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --served-model-name behemoth-bf16 \
    --tensor-parallel-size 4 --max-model-len 8192 \
    --max-logprobs 64 --gpu-memory-utilization 0.90

python tools/kld_eval.py collect \
    --model behemoth-bf16 \
    --tokenizer /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --texts data/text/behemoth_r1_123b_eval_8192.jsonl \
    --out /media/fmodels2/working_Model-Opt/kld/bf16.npz \
    --seq-len 1024 --max-seqs 256 -k 64
```

Repeat for each NVFP4 export with the same `--texts`, `--seq-len` and `-k`. Use `--tensor-parallel-size 4` for the quantized runs too, even though the 65 GiB variant would serve faster at TP1: matching the all-reduce order removes the last confound, and 262k tokens of prefill finishes in minutes at any TP size.

```bash
python tools/kld_eval.py compare \
    /media/fmodels2/working_Model-Opt/kld/bf16.npz \
    /media/fmodels2/working_Model-Opt/kld/nvfp4_omlp_q.npz
```

**Read p99, not the mean.** The mean is dominated by the overwhelming majority of positions where the model is confident and quantization changes nothing. p99 is the tail where the quant changed its mind, and for creative writing that tail is what you feel. Rough bands for the mean: below 0.01 excellent, 0.01–0.05 good, 0.05–0.15 noticeable, above 0.15 expect visible loss in long generations.

Reported KLD is a floor. When a reference token falls outside the candidate's top-k, the tool assigns it the candidate's smallest observed logprob — the most generous available bound — so the true divergence can only be larger than reported, never smaller.

### Measured baseline: `omlp` versus BF16

261,888 positions, 256 windows of 1024 tokens, held-out set, `-k 64` with 99.757% of reference mass covered.

| Metric | Value |
| --- | ---: |
| Perplexity | 4.8029 → 4.8752 (**+1.51%**) |
| top-1 agreement, all positions | 94.10% |
| top-1 agreement, ref p > 0.5 | 99.19% |
| **top-1 agreement, ref p > 0.9** | **99.93%** |
| KLD mean / median / p99 | 0.0198 / 0.0085 / 0.1587 |

| Domain | KLD mean | KLD p99 | top-1 | top-1 p>0.9 |
| --- | ---: | ---: | ---: | ---: |
| `creative_writing` | 0.0216 | 0.1805 | 93.51% | 99.89% |
| `roleplay` | 0.0195 | 0.1300 | 93.84% | 99.97% |
| `reasoning` | 0.0156 | 0.1237 | 95.72% | 99.98% |
| `breadth` | 0.0115 | 0.0998 | 96.38% | 100.00% |

Two conclusions worth carrying forward.

**The aggregate KLD overstates the damage.** Reasoning diverges 28% less than creative writing on the mean and 31% less at p99, yet creative writing supplies 66% of the positions and drags the overall figure up. Creative writing is intrinsically higher-entropy — more continuations are genuinely valid, so identical logit perturbation produces larger KL — so part of that gap is the domain rather than the quantization.

**Almost every argmax flip is at a position BF16 was unsure about.** Decomposing the confidence bands, roughly 91% of the ~15,400 flips occur where the reference was itself under 50% confident. Where BF16 was near-certain, the quant disagrees about 64 times in 91,200 decisions. Reasoning at 99.98% and breadth at 100.00% mean essentially no confident decision changed outside prose. Read the p > 0.9 row and ignore the headline 94.10%, which mostly counts interchangeable word choice.

This is the bar for the `q_proj` variant, and it shows the headroom: 99.85% near-certain agreement would be roughly 130 flips instead of 64 — imperceptible in practice. A variant that holds above ~99.8% should be taken for the single-GPU serving it unlocks.

## 7. Serving on vLLM / SM120

Load the final directory as a local model. Two things need explicit attention on SM120.

**Force the b12x GEMM backend and tell FlashInfer the arch.** SM120 is not auto-detected everywhere, and the fallback is Marlin W4A16, which dequantizes to FP16 and gives up much of the FP4 win.

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_FORCE_SM=120f
# Do NOT set VLLM_NVFP4_GEMM_BACKEND. It is not a recognised variable on 0.29.x;
# vLLM logs "Unknown vLLM environment variable detected" and ignores it. Kernel
# selection is KernelConfig.linear_backend, default 'auto'. See Install.md §11.

vllm serve /media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4 \
    --served-model-name Behemoth-R1-123B-v2-NVFP4 \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --enable-prefix-caching
```

**Do not pass `--kv-cache-dtype fp8`.** Default `auto` gives BF16 KV for a BF16-KV model. That is the whole point of §2.

Notes on the flags: TP=4 across all four cards; avoid expert parallelism entirely (irrelevant for a dense model, and catastrophic on PCIe anyway). `--max-model-len 65536` is a starting point — 131072 is available but a single full-length sequence is 44 GiB of BF16 KV, so size it to your actual concurrency. Requires a recent vLLM with the SM120 capability-family fixes and FlashInfer ≥ 0.6.9; `nvidia-cutlass-dsl` is version-sensitive on this path.

**Confirm the kernel.** Grep the startup log for `for NVFP4 GEMM`. A verified smoke serve on this hardware reported:

```
INFO [__init__.py:1180] Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
INFO [core.py:123] ... quantization=modelopt_fp4 ... kv_cache_dtype=auto
INFO [compilation.py:336] Enabled custom fusions: act_quant
```

That is a real FP4 tensor-core path, not the Marlin W4A16 dequant fallback, and `act_quant` fusion confirms W4A4 activation quantization is live. Only `Marlin` in that line indicates a problem. `kv_cache_dtype=auto` resolves to BF16 here, which is what you want.

**The real bottleneck on this box is PCIe, not the kernel.** At TP4 with no NVLink, vLLM disables SymmMem, FlashInfer all-reduce, *and* custom all-reduce, leaving PYNCCL. An 88-layer dense model needs two all-reduces per layer, so 176 PCIe round-trips per token. Measured single-request decode was ~43 tok/s versus a bandwidth roofline near 77. If you want latency, benchmark `--tensor-parallel-size 2` against TP4 — the 86 GiB checkpoint fits in two cards, and a smaller all-reduce group often wins.

At serve time, keep the Mistral v7 template and prefill `<think>` after `[/INST]` for the reasoning phase, exactly as the model card describes. Never substitute a Tekken template or a v3 template without `[SYSTEM_PROMPT]`.

## 8. If accuracy regresses

In order of what to try:

1. **Drop `o_proj` back to BF16** — one-line change in `models/behemoth_r1_123b.py` (add `*o_proj*` disables), reverting to `mlp_only`. Costs 19 GB and some speed. This is the single highest-leverage accuracy knob.
2. **Lower `upper_percentile`** toward 99.9, or raise toward 100 (100 = no clipping, literal observed max). Both directions are defensible; measure.
3. **Try `weight_scale_method = "local_hessian"`** instead of `mse`.
4. **Exclude the last 4 layers** — NVIDIA's *pretraining* recipe does this. For 88 layers, add `*layers.8[4-7].mlp*` and `*layers.8[4-7].self_attn.o_proj*` disables. Costs ~7 GB. Not standard for PTQ; treat as a diagnostic.
5. **More creative-writing samples** — raise `num_samples` on the top CW sources before adding new categories.
6. **Never** re-enable KV quantization, and never enable `q/k/v`.

## 9. Do not copy these from the other recipes

| Flag / setting | Behemoth? |
| --- | --- |
| `--floor-amaxes` | No — patches sparse MoE expert amaxes only |
| `--streaming` | No — model fits in 384 GB |
| `--save-quantiles` / `method = "quantile"` | No — not in upstream ModelOpt |
| `COMMON_QUANT_OVERRIDES` | No — adapter overrides `get_all_quant_overrides()` |
| `trust_remote_code` | No |
| `vqa_calib.jsonl` / multimodal | No — text-only |
| `extra_mtp_prefixes` | No — no MTP in this checkpoint |
| `transformers_compat` | No — MiniMax only |
| `--kv-cache-dtype fp8` at serve | No — BF16 KV, always |
