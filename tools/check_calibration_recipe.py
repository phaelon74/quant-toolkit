"""
Calibration recipe preflight for the Qwen3.8-27B W8A8 build.

Run this BEFORE the quantization job. It:
  - resolves every dataset in the recipe against HuggingFace
  - checks the requested columns exist
  - applies the recipe's formatter and renders through the model chat template
  - reports samples collected vs requested, per source
  - reports token-length statistics at your target seqlen
  - writes a JSONL cache so the real build uses exactly this data
    (pass it to the quantizer with --calib-jsonl)

Exit codes: 0 all sources healthy, 1 some sources failed or came up short.

Usage:
  python check_calibration_recipe.py calibrate_software_engineer.yaml \
      --tokenizer /models/Qwen3.8-27B --seqlen 8192 --out calib_swe_8k.jsonl
"""
import argparse
import importlib.util
import json
import os
import random
import statistics
import sys

QUANT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Qwen3.8-27B_int8-w8a8.py")


def load_formatters():
    """Reuse the formatter implementations from the quantization script."""
    spec = importlib.util.spec_from_file_location("w8a8_quant", QUANT_SCRIPT)
    if spec is None or not os.path.isfile(QUANT_SCRIPT):
        sys.exit(f"ERROR: cannot find {QUANT_SCRIPT}; keep both scripts in one directory.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("--tokenizer", required=True,
                    help="Model dir, for the chat template and token counts.")
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--out", default="calibration_cache.jsonl")
    ap.add_argument("--sample-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool-factor", type=int, default=4,
                    help="Rows fetched per requested sample before filtering.")
    ap.add_argument("--no-cache", action="store_true", help="Check only; write nothing.")
    args = ap.parse_args()

    import yaml
    from datasets import load_dataset
    from transformers import AutoTokenizer

    quant = load_formatters()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    with open(args.recipe, encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh)["calibration_set"]
    specs = recipe["datasets"]
    seed = recipe.get("seed", args.seed)
    recipe_len = recipe.get("max_seq_length")

    print(f"recipe        : {os.path.basename(args.recipe)}")
    print(f"sources       : {len(specs)}")
    print(f"recipe seqlen : {recipe_len}  ->  building at {args.seqlen}")
    print(f"tokenizer     : {args.tokenizer}")
    print()
    header = f"{'status':<7} {'got/want':<11} {'tokens avg/max':<16} {'dataset':<52} note"
    print(header)
    print("-" * len(header))

    collected, failures, short = [], [], []
    per_source = []

    for spec in specs:
        name = spec["dataset"]
        want = max(1, int(spec.get("num_samples", 8) * args.sample_scale))
        note = ""
        texts = []
        try:
            kwargs = {"split": spec.get("split", "train")}
            if spec.get("subset"):
                kwargs["name"] = spec["subset"]
            if spec.get("data_files"):
                kwargs["data_files"] = spec["data_files"]
            if spec.get("streaming"):
                kwargs["streaming"] = True
            ds = load_dataset(name, **kwargs)

            if spec.get("streaming"):
                rows = []
                for row in ds:
                    rows.append(row)
                    if len(rows) >= want * args.pool_factor:
                        break
            else:
                pool = min(len(ds), want * args.pool_factor)
                rows = list(ds.shuffle(seed=seed).select(range(pool)))

            if rows:
                missing = [c for c in (spec.get("columns") or []) if c not in rows[0]]
                if missing:
                    note = f"missing columns {missing}; available: {sorted(rows[0])[:6]}"

            for row in rows:
                text = quant.format_row(row, spec, tokenizer)
                if text and len(text.strip()) > 32:
                    texts.append(text)
                if len(texts) >= want:
                    break

            if not texts:
                failures.append((name, note or "formatter produced nothing"))
                status = "FAIL"
            elif len(texts) < want:
                short.append((name, f"{len(texts)}/{want}"))
                status = "SHORT"
            else:
                status = "ok"
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {str(exc)[:100]}"))
            status = "FAIL"
            note = note or f"{type(exc).__name__}: {str(exc)[:60]}"

        lengths = [len(tokenizer(t, truncation=True, max_length=args.seqlen).input_ids)
                   for t in texts]
        avg = int(statistics.mean(lengths)) if lengths else 0
        mx = max(lengths) if lengths else 0
        print(f"{status:<7} {f'{len(texts)}/{want}':<11} {f'{avg}/{mx}':<16} {name[:52]:<52} {note}")

        per_source.append({"dataset": name, "got": len(texts), "want": want,
                           "avg_tokens": avg, "max_tokens": mx, "status": status})
        collected.extend({"text": t, "source": name} for t in texts)

    if recipe.get("shuffle", True):
        random.Random(seed).shuffle(collected)

    lengths = [s["avg_tokens"] for s in per_source if s["avg_tokens"]]
    all_lengths = [len(tokenizer(c["text"], truncation=True,
                                 max_length=args.seqlen).input_ids) for c in collected]
    requested = sum(s["want"] for s in per_source)

    print()
    print(f"collected {len(collected)} / {requested} requested samples "
          f"from {len(specs) - len(failures)} / {len(specs)} sources")
    if all_lengths:
        all_lengths.sort()
        def pct(p):
            return all_lengths[min(len(all_lengths) - 1, int(len(all_lengths) * p))]
        print(f"token lengths : min {all_lengths[0]}, p50 {pct(0.5)}, p90 {pct(0.9)}, "
              f"max {all_lengths[-1]}, total {sum(all_lengths):,}")
        at_cap = sum(1 for x in all_lengths if x >= args.seqlen)
        long_enough = sum(1 for x in all_lengths if x >= 2048)
        print(f"              : {at_cap} samples hit the {args.seqlen} cap, "
              f"{long_enough} are >= 2048 tokens")
        if long_enough < len(all_lengths) * 0.1:
            print("WARNING: under 10% of samples exceed 2048 tokens. Long-context "
                  "activation ranges will be under-represented for the recurrent "
                  "layers; consider adding long-document sources.")

    if failures:
        print(f"\n{len(failures)} source(s) FAILED:")
        for name, why in failures:
            print(f"  {name}: {why}")
    if short:
        print(f"\n{len(short)} source(s) came up SHORT:")
        for name, ratio in short:
            print(f"  {name}: {ratio}")

    if not args.no_cache and collected:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in collected:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(collected)} samples -> {args.out}")
        print(f"build with: --calib-jsonl {args.out} --seqlen {args.seqlen}")

    healthy = not failures and not short
    print("\n=== PREFLIGHT: " + ("PASS" if healthy else "ISSUES FOUND") + " ===")
    if not healthy:
        print("The cache is still usable; decide whether the missing sources matter "
              "or substitute replacements in the recipe.")
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
