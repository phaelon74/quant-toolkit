"""INT4 W4A16 weight-only quantization (PTQ / AWQ / GPTQ-lite) via NVIDIA ModelOpt.

Parallel to quantize.py (NVFP4); does not share NVFP4-specific calibration or export.
"""
import argparse
import copy
import gc
import json
import os
import tomllib

import torch
from transformers import AutoTokenizer
import modelopt.torch.quantization as mtq
import logging

from models import load_config, AVAILABLE_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS,
                    help="Model config to use.")
parser.add_argument("--model-id", default=None,
                    help="Override the default HuggingFace model ID or local path.")
parser.add_argument("--export-dir", required=True)
parser.add_argument("--algorithm", required=True, choices=["ptq", "awq", "gptq"],
                    help="INT4 W4A16: ptq (max), awq (INT4_AWQ_CFG), gptq (ModelOpt gptq_lite).")
parser.add_argument("--calib-config", default=None,
                    help="TOML file describing calibration datasets and parameters.")
parser.add_argument("--data-dir", default="data",
                    help="Base directory for relative dataset paths in the TOML.")
parser.add_argument("--calib-jsonl", default=None,
                    help="Single calibration JSONL (shorthand; ignored if --calib-config given).")
parser.add_argument("--calib-limit", type=int, default=192)
parser.add_argument("--batch-size", type=int, default=48)
parser.add_argument("--batch-tokens", type=int, default=128 * 1024,
                    help="Token budget for auto-computing batch_size per dataset.")
parser.add_argument("--max-len", type=int, default=4096)
parser.add_argument("--cpu-capacity", type=str, default="200GiB")
parser.add_argument("--skip-export", action="store_true",
                    help="Skip HF export after quantization.")
parser.add_argument("--streaming", action="store_true", default=None,
                    help="Force streaming loader. Default: use model config.")
args = parser.parse_args()


def load_calib_datasets(args):
    """Return list of dicts with keys: path, limit, batch_size, max_len."""
    if args.calib_config:
        with open(args.calib_config, "rb") as f:
            toml_cfg = tomllib.load(f)
        datasets = toml_cfg.get("dataset", [])
        if not datasets:
            parser.error(f"No [[dataset]] entries in {args.calib_config}")
        batch_tokens = args.batch_tokens
        for i, ds in enumerate(datasets):
            if "path" not in ds:
                parser.error(f"dataset[{i}] missing 'path' in {args.calib_config}")
            if not os.path.isabs(ds["path"]):
                ds["path"] = os.path.join(args.data_dir, ds["path"])
            ds.setdefault("max_len", 4096)
            if "batch_size" not in ds:
                ds["batch_size"] = max(1, batch_tokens // ds["max_len"])
        return datasets

    if not args.calib_jsonl:
        parser.error("Provide either --calib-config or --calib-jsonl")
    return [{
        "path": args.calib_jsonl,
        "limit": args.calib_limit,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
    }]


calib_datasets = load_calib_datasets(args)
print(f"\nCalibration plan: {len(calib_datasets)} dataset(s)")
for i, ds in enumerate(calib_datasets):
    lim = ds.get("limit", "all")
    print(f"  [{i+1}] {ds['path']}  (limit={lim}, batch={ds['batch_size']}, maxlen={ds['max_len']})")


cfg = load_config(args.model)
MODEL_ID = args.model_id or cfg.model_id
TRUST_REMOTE = cfg.trust_remote_code
use_streaming = args.streaming if args.streaming is not None else cfg.streaming

cfg.register_moe()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

has_mm = any(ds.get("multimodal", False) for ds in calib_datasets)
processor = None
if has_mm:
    from transformers import AutoProcessor
    from PIL import Image
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
    print(f"Loaded multimodal processor for {MODEL_ID}")


def _parse_gib(s):
    s = s.strip()
    for suffix in ("GiB", "GB", "gib", "gb"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)])
    return float(s)


model_cls = cfg.get_model_cls()

if use_streaming:
    from streaming_loader import StreamingModelLoader

    print(f"Loading model from {MODEL_ID} with streaming loader...")
    loader = StreamingModelLoader(
        model_id=MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=TRUST_REMOTE,
        cpu_capacity_gib=_parse_gib(args.cpu_capacity),
    )
    model = loader.load_model(model_cls=model_cls)
else:
    from transformers import AutoModelForCausalLM

    print(f"Loading model from {MODEL_ID} onto GPUs...")
    loader = None
    cls = model_cls or AutoModelForCausalLM
    model = cls.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=TRUST_REMOTE,
        device_map="auto",
    )

print(f"\n{'='*60}")
print("Data type distribution BEFORE quantization:")
dtype_stats = {}
for name, param in model.named_parameters():
    dtype = str(param.dtype)
    if dtype not in dtype_stats:
        dtype_stats[dtype] = {"count": 0, "size_bytes": 0}
    dtype_stats[dtype]["count"] += 1
    dtype_stats[dtype]["size_bytes"] += param.numel() * param.element_size()

for dtype, stats in sorted(dtype_stats.items()):
    print(f"  {dtype:<20} {stats['count']:>6} tensors, {stats['size_bytes']/1e9:>8.2f} GB")
print(f"{'='*60}")


def _apply_chat_template(messages):
    tmpl = processor if processor is not None else tokenizer
    return tmpl.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def iter_prompts(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            j = json.loads(line)
            if "messages" in j:
                try:
                    yield _apply_chat_template(j["messages"])
                except Exception:
                    texts = [m["content"] for m in j["messages"] if m.get("role") == "user"]
                    if texts:
                        yield " ".join(texts)
            elif "prompt" in j or "text" in j:
                yield j.get("prompt") or j.get("text")


def _tokenize_batch(texts, max_len):
    if processor is not None:
        batch = processor(
            text=texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        if "pixel_values" not in batch:
            seq_len = batch["input_ids"].shape[1]
            batch["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand_as(batch["input_ids"])
        if "mm_token_type_ids" not in batch and "input_ids" in batch:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch
    return tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len)


def build_batches(prompts, max_len, batch_size):
    buf = []
    for p in prompts:
        buf.append(p)
        if len(buf) == batch_size:
            yield _tokenize_batch(buf, max_len)
            buf = []
    if buf:
        yield _tokenize_batch(buf, max_len)


def iter_mm_samples(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            j = json.loads(line)
            messages = j.get("messages", [])
            images = []
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, str):
                    continue
                for part in content:
                    if part.get("type") == "image":
                        img_path = part.get("image", "")
                        if img_path:
                            images.append(Image.open(img_path).convert("RGB"))
            if images:
                yield messages, images


def build_mm_batches(samples, max_len, batch_size):
    buf_texts = []
    buf_images = []
    for messages, images in samples:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        buf_texts.append(text)
        buf_images.extend(images)
        if len(buf_texts) == batch_size:
            yield processor(
                text=buf_texts, images=buf_images, padding=True,
                truncation=True, max_length=max_len, return_tensors="pt",
            )
            buf_texts = []
            buf_images = []
    if buf_texts:
        yield processor(
            text=buf_texts, images=buf_images, padding=True,
            truncation=True, max_length=max_len, return_tensors="pt",
        )


all_batches = []
for ds_idx, ds in enumerate(calib_datasets):
    if ds.get("multimodal", False):
        ds_batches = list(build_mm_batches(
            iter_mm_samples(ds["path"], limit=ds.get("limit")),
            max_len=ds["max_len"],
            batch_size=ds["batch_size"],
        ))
    else:
        ds_batches = list(build_batches(
            iter_prompts(ds["path"], limit=ds.get("limit")),
            max_len=ds["max_len"],
            batch_size=ds["batch_size"],
        ))
    print(f"  Dataset [{ds_idx+1}]: {len(ds_batches)} batches")
    all_batches.extend((ds_idx, b) for b in ds_batches)

print(f"  Total: {len(all_batches)} batches across {len(calib_datasets)} dataset(s)")

model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

os.makedirs(args.export_dir, exist_ok=True)


def forward_loop(m):
    input_device = next(m.parameters()).device
    print(f"\nCalibration: {len(all_batches)} batches across {len(calib_datasets)} dataset(s)...")
    cur_ds = -1
    for i, (ds_idx, batch) in enumerate(all_batches, 1):
        if ds_idx != cur_ds:
            cur_ds = ds_idx
            ds = calib_datasets[ds_idx]
            print(f"\n  --- Dataset [{ds_idx+1}]: {os.path.basename(ds['path'])} "
                  f"(batch={ds['batch_size']}, maxlen={ds['max_len']}) ---")
        print(f"  Batch {i}/{len(all_batches)}...")
        kwargs = {
            k: v.to(input_device, non_blocking=True)
            for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        if "mm_token_type_ids" not in kwargs and "input_ids" in kwargs:
            kwargs["mm_token_type_ids"] = torch.zeros_like(kwargs["input_ids"])
        with torch.no_grad():
            m(**kwargs, use_cache=False)
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()
    print("Calibration complete.")


# COMMON_QUANT_OVERRIDES patterns that must NOT carry over to INT4 W4A16:
#  *self_attn*weight_quantizer  -- NVFP4 disables attention weights (MLP-only recipe);
#                                  INT4 W4A16 quantizes ALL linear weights.
#  *[kv]_bmm_quantizer          -- FP8 KV cache; irrelevant for weight-only INT4.
_NVFP4_ONLY_PATTERNS = frozenset({
    "*self_attn*weight_quantizer",
    "*[kv]_bmm_quantizer",
})


def _build_int4_ptq_cfg():
    return {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"enable": False},
            "*output_quantizer": {"enable": False},
            "*q_bmm_quantizer": {"enable": False},
            "*softmax_quantizer": {"enable": False},
            "*[kv]_bmm_quantizer": {"enable": False},
        },
        "algorithm": "max",
    }


def _build_int4_awq_cfg():
    cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    try:
        qc = cfg.get("quant_cfg")
        if not isinstance(qc, dict):
            return cfg
        wq = qc.get("*weight_quantizer")
        if isinstance(wq, dict) and "block_sizes" in wq:
            bs = wq["block_sizes"]
            if isinstance(bs, list) and len(bs) > 0:
                bs[-1] = 128
            elif isinstance(bs, dict):
                bs[-1] = 128
    except (TypeError, KeyError, IndexError):
        pass
    return cfg


def _build_int4_gptq_cfg():
    return {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"enable": False},
            "*output_quantizer": {"enable": False},
            "*q_bmm_quantizer": {"enable": False},
            "*softmax_quantizer": {"enable": False},
            "*[kv]_bmm_quantizer": {"enable": False},
        },
        "algorithm": {"method": "gptq_lite", "sequential": True},
    }


_ALGO_BUILDERS = {
    "ptq": _build_int4_ptq_cfg,
    "awq": _build_int4_awq_cfg,
    "gptq": _build_int4_gptq_cfg,
}

qcfg = _ALGO_BUILDERS[args.algorithm]()
for pattern, override in cfg.get_all_quant_overrides().items():
    if pattern in _NVFP4_ONLY_PATTERNS:
        continue
    qcfg["quant_cfg"][pattern] = override

print(f"\nQuantizing with INT4 W4A16 (model={args.model}, algorithm={args.algorithm})...")
model = mtq.quantize(model, qcfg, forward_loop)
print(f"{'='*60}")

if args.skip_export:
    print("\nSkipping export (--skip-export).")
else:
    from export_hf_int4 import export_hf

    print("\nExporting quantized model to HF format...")
    prepare_fn = loader.prepare_export if loader is not None else None
    export_hf(
        model,
        export_dir=args.export_dir,
        prepare_fn=prepare_fn,
        extra_mtp_prefixes=cfg.extra_mtp_prefixes,
    )
    print(f"Quantized model exported to {args.export_dir}")
