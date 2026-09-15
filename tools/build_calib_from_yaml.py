#!/usr/bin/env python3
"""Build a calibration JSONL from a declarative YAML spec.

Reads the `calibration_set` schema used by the files in data/examples/ and emits
the {"messages": [...]} JSONL that quantize.py consumes.

Usage:
    python tools/build_calib_from_yaml.py \
        --yaml data/behemoth_r1_123b_calib.yaml \
        --output data/text/behemoth_r1_123b_calib.jsonl \
        --think-tag think

Supported per-dataset keys: dataset, subset, split, columns, formatter,
num_samples, streaming. Supported formatters: prompt_answer, sharegpt,
chat_completion, raw_text.
"""

import argparse
import importlib.util
import json
import os
import random
import re

import yaml
from datasets import load_dataset

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--yaml", required=True, help="Calibration spec YAML.")
ap.add_argument("--output", required=True, help="Output JSONL path.")
ap.add_argument("--think-tag", default=None,
                help="Normalize reasoning-trace markers to this tag (e.g. 'think'). "
                     "Off by default.")
ap.add_argument("--chars-per-token", type=int, default=8,
                help="Character budget per token of max_seq_length. Samples over "
                     "budget are truncated, not dropped. 8 is deliberately "
                     "generous so a truncated sample still fills max_len.")
ap.add_argument("--min-chars", type=int, default=48,
                help="Drop samples whose total content is under this.")
ap.add_argument("--seed", type=int, default=None,
                help="Override the seed in the YAML.")
args = ap.parse_args()


# ---------------------------------------------------------------------------
# Cleaning / normalization.
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Reasoning-trace markers used by the upstream CoT datasets. Rewritten to the
# tag the target model actually emits so calibration sees in-distribution text.
_THOUGHT_OPEN_RE = re.compile(
    r"<\|?\s*(?:begin_of_thought|begin_thought|thinking|thought|reasoning)\s*\|?>",
    re.IGNORECASE,
)
_THOUGHT_CLOSE_RE = re.compile(
    r"</\|?\s*(?:end_of_thought|end_thought|thinking|thought|reasoning)\s*\|?>"
    r"|<\|\s*end_of_thought\s*\|>",
    re.IGNORECASE,
)
_SOLUTION_RE = re.compile(
    r"</?\|?\s*(?:begin_of_solution|end_of_solution|solution)\s*\|?>",
    re.IGNORECASE,
)

# Behemoth's Mistral v7 template raises on anything outside these roles.
_ROLE_MAP = {
    "system": "system",
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "ai": "assistant",
    "model": "assistant", "chatgpt": "assistant", "bot": "assistant",
}


def clean(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.replace("\ufeff", "").replace("\ufffe", "")
    text = _CONTROL_RE.sub("", text)
    return text.strip()


def normalize_think(text: str, tag: str) -> str:
    text = _THOUGHT_OPEN_RE.sub(f"<{tag}>", text)
    text = _THOUGHT_CLOSE_RE.sub(f"</{tag}>", text)
    return _SOLUTION_RE.sub("", text)


def coerce_messages(raw) -> list | None:
    """Normalize any conversation-ish structure into system/user/assistant turns."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    out = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or turn.get("speaker") or ""
        role = _ROLE_MAP.get(str(role).strip().lower())
        content = turn.get("content")
        if content is None:
            content = turn.get("value", turn.get("text", ""))
        # Multimodal-style content lists: keep the text parts.
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        content = clean(content)
        if role and content:
            out.append({"role": role, "content": content})
    return out or None


def hub_status() -> list:
    """Describe hub auth and transfer settings, so a silent 401 is visible."""
    out = []
    try:
        from huggingface_hub import get_token, whoami
    except ImportError:
        return ["hf auth: huggingface_hub not installed"]

    if not get_token():
        out.append("hf auth: anonymous — gated sources will be skipped, and 40 "
                   "repos may hit the rate limit")
    else:
        try:
            out.append(f"hf auth: {whoami()['name']}")
        except Exception as exc:
            out.append(f"hf auth: token present but rejected ({type(exc).__name__})")

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        ok = importlib.util.find_spec("hf_transfer") is not None
        out.append("hf_transfer: enabled" if ok
                   else "hf_transfer: HF_HUB_ENABLE_HF_TRANSFER=1 but package missing")
    return out


def truncate_messages(msgs: list, budget: int) -> list:
    """Trim message contents to a total character budget, keeping the head."""
    out, used = [], 0
    for m in msgs:
        if used >= budget:
            break
        content = m["content"][: budget - used]
        if content:
            out.append({"role": m["role"], "content": content})
            used += len(content)
    return out


def _column_values(example, columns):
    return [example.get(c) for c in columns] if columns else []


# ---------------------------------------------------------------------------
# Formatters.
# ---------------------------------------------------------------------------

def fmt_prompt_answer(example, columns):
    vals = _column_values(example, columns)
    prompt = clean(vals[0] if vals else "")
    answer = clean(vals[1]) if len(vals) > 1 else ""
    if not prompt:
        return None
    msgs = [{"role": "user", "content": prompt}]
    if answer:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def fmt_sharegpt(example, columns):
    for col in (columns or []):
        got = coerce_messages(example.get(col))
        if got:
            return got
    for key in ("conversations", "conversation", "messages", "chat"):
        got = coerce_messages(example.get(key))
        if got:
            return got
    return None


def fmt_chat_completion(example, columns):
    # Used loosely in the example specs: sometimes a real messages list,
    # sometimes a pair of scalar columns, sometimes a list of bare strings.
    got = fmt_sharegpt(example, columns)
    if got:
        return got

    vals = _column_values(example, columns)
    if vals and isinstance(vals[0], list):
        turns = [clean(v) for v in vals[0] if clean(v)]
        if turns:
            return [
                {"role": "user" if i % 2 == 0 else "assistant", "content": t}
                for i, t in enumerate(turns)
            ]
    return fmt_prompt_answer(example, columns)


def fmt_raw_text(example, columns):
    vals = [clean(v) for v in _column_values(example, columns)]
    vals = [v for v in vals if v]
    if not vals:
        return None
    # Convention in the example specs: [text] or [content, title].
    body = max(vals, key=len)
    return [{"role": "user", "content": body}]


FORMATTERS = {
    "prompt_answer": fmt_prompt_answer,
    "sharegpt": fmt_sharegpt,
    "chat_completion": fmt_chat_completion,
    "raw_text": fmt_raw_text,
}


# ---------------------------------------------------------------------------
# Collection.
# ---------------------------------------------------------------------------

def collect(entry, seed, default_len):
    name = entry["dataset"]
    group_len = int(entry.get("max_seq_length", default_len))
    bucket = entry.get("bucket", "unlabelled")
    split = entry.get("split", "train")
    subset = entry.get("subset")
    columns = entry.get("columns") or []
    want = int(entry.get("num_samples", 0))
    formatter = FORMATTERS.get(entry.get("formatter", "sharegpt"))

    if want <= 0 or formatter is None:
        return []

    label = f"{name}" + (f"[{subset}]" if subset else "") + f"/{split}"
    try:
        ds = load_dataset(name, subset, split=split,
                          streaming=bool(entry.get("streaming", False)))
        if entry.get("streaming", False):
            ds = ds.shuffle(seed=seed, buffer_size=10000)
        else:
            ds = ds.shuffle(seed=seed)
    except Exception as exc:
        print(f"  !! skip {label}: {exc}")
        return []

    picked = []
    seen = 0
    for example in ds:
        seen += 1
        # Give up on a stubborn source rather than scanning a whole shard.
        if seen > max(20000, want * 50):
            break
        try:
            msgs = formatter(example, columns)
        except Exception:
            continue
        if not msgs or not any(m["role"] == "user" for m in msgs):
            continue
        picked.append({"messages": msgs, "_len": group_len, "_bucket": bucket})
        if len(picked) >= want:
            break

    short = "" if len(picked) >= want else "   << SHORT"
    print(f"  {label}: {len(picked)}/{want} @ {group_len}{short}")
    return picked


def main():
    with open(args.yaml, encoding="utf-8") as f:
        spec = yaml.safe_load(f)["calibration_set"]

    seed = args.seed if args.seed is not None else spec.get("seed", 42)
    random.seed(seed)

    entries = spec.get("datasets", [])
    planned = sum(int(e.get("num_samples", 0)) for e in entries)
    default_len = int(spec.get("max_seq_length", 4096))
    print(f"Spec: {args.yaml}")
    print(f"  {len(entries)} sources, {planned} samples planned, seed={seed}")
    print(f"  default max_seq_length={default_len}")
    for note in hub_status():
        print(f"  {note}")
    print()

    samples = []
    for entry in entries:
        samples.extend(collect(entry, seed, default_len))

    if spec.get("shuffle", True):
        random.shuffle(samples)

    groups, buckets, truncated, dropped_short = {}, {}, 0, 0
    for sample in samples:
        msgs = sample["messages"]
        if args.think_tag:
            for m in msgs:
                if m["role"] == "assistant":
                    m["content"] = normalize_think(m["content"], args.think_tag)

        total = sum(len(m["content"]) for m in msgs)
        if total < args.min_chars:
            dropped_short += 1
            continue

        # Over budget is truncated, never dropped. The tokenizer cuts at max_len
        # anyway, and dropping would discard precisely the book- and
        # script-length samples the long-form sources exist to provide.
        budget = sample["_len"] * args.chars_per_token
        if total > budget:
            msgs = truncate_messages(msgs, budget)
            truncated += 1
            if not any(m["role"] == "user" and m["content"] for m in msgs):
                dropped_short += 1
                continue

        groups.setdefault(sample["_len"], []).append({"messages": msgs})
        buckets[sample["_bucket"]] = buckets.get(sample["_bucket"], 0) + 1

    kept = sum(len(v) for v in groups.values())
    print(f"\nCollected {len(samples)}/{planned}; kept {kept} "
          f"({truncated} truncated to budget, {dropped_short} dropped as too short).")

    if buckets:
        print("\nBucket mix:")
        for name, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<20} {count:>6}  {100 * count / kept:5.1f}%")

    # One file per max_seq_length: quantize.py sets max_len per [[dataset]].
    stem = re.sub(r"\.jsonl$", "", args.output)
    print("\nPaste into the calibration TOML:\n")
    for group_len in sorted(groups):
        path = f"{stem}_{group_len}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in groups[group_len]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  # {len(groups[group_len])} samples")
        print(f"  [[dataset]]")
        print(f'  path = "{path.split("data/", 1)[-1]}"')
        print(f"  max_len = {group_len}\n")


if __name__ == "__main__":
    main()
