"""
Build a held-out eval set for tools/kld_eval.py from the calibration recipe.

Guarantees the eval text is disjoint from the calibration cache (by exact text
hash), and attaches the `bucket` label kld_eval.py's per-domain breakdown wants.

  # 1. collect a larger pool with a different seed
  python check_calibration_recipe.py calibrate_qwen38_general_v2.yaml \
      --tokenizer /media/netmodels/Qwen/Qwen3.8-27B --seqlen 8192 \
      --seed 1234 --sample-scale 2.0 --out pool_seed1234.jsonl

  # 2. subtract the calibration set and label the remainder
  python make_kld_eval_set.py pool_seed1234.jsonl \
      --exclude calib_general_8k.jsonl \
      --out qwen38_eval_8192.jsonl --per-bucket 64
"""
import argparse
import collections
import hashlib
import json
import random
import sys

BUCKETS = [
    ("code", ("code", "humaneval", "stack", "kubernetes", "CodeArena", "Nemotron-Comp")),
    ("tool_use", ("tool_use", "hermes", "calibration", "Nemotron-Post")),
    ("reasoning", ("Science", "Math", "Numina", "MegaScience", "Platypus", "camel",
                   "physical-reasoning", "theory-of-mind")),
    ("professional", ("Lawyer", "Medical", "finance", "business", "yahoo", "chatgpt-prompts")),
    ("long_doc", ("pubmed", "medium-articles", "paul_graham", "TvTroper")),
    ("creative", ("WritingPrompts", "writingprompts", "claude_writing")),
    ("chat", ("ultrachat", "no_robots", "dolly", "HelpSteer", "Socratic")),
    ("multilingual", ("Multilingual",)),
]


def bucket_of(source):
    for name, needles in BUCKETS:
        if any(n.lower() in source.lower() for n in needles):
            return name
    return "breadth"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def digest(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", help="JSONL from check_calibration_recipe.py (different seed)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="calibration JSONL(s) to subtract; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=64)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    banned = set()
    for path in args.exclude:
        rows = load(path)
        banned |= {digest(r["text"]) for r in rows}
        print(f"excluding {len(rows)} rows from {path}")

    pool = load(args.pool)
    kept, dropped_overlap, dropped_short = [], 0, 0
    seen = set()
    for row in pool:
        text = row["text"]
        h = digest(text)
        if h in banned:
            dropped_overlap += 1
            continue
        if h in seen:
            continue
        if len(text) < args.min_chars:
            dropped_short += 1
            continue
        seen.add(h)
        kept.append({"text": text, "bucket": bucket_of(row.get("source", "")),
                     "source": row.get("source", "")})

    print(f"pool {len(pool)} -> kept {len(kept)} "
          f"(overlap {dropped_overlap}, too short {dropped_short})")

    by_bucket = collections.defaultdict(list)
    for row in kept:
        by_bucket[row["bucket"]].append(row)

    rng = random.Random(args.seed)
    out = []
    for bucket, rows in sorted(by_bucket.items()):
        rng.shuffle(rows)
        take = rows[: args.per_bucket]
        out.extend(take)
        print(f"  {bucket:<14} {len(take):>4} of {len(rows)} available")

    if not out:
        sys.exit("ERROR: nothing left after exclusion; use a different --seed for the pool")
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out)} eval rows -> {args.out}")
    print("verify disjointness:")
    print(f"  python -c \"import json,hashlib;"
          f"d=lambda p:{{hashlib.sha1(json.loads(l)['text'].strip().encode()).hexdigest() "
          f"for l in open(p) if l.strip()}};"
          f"print('overlap:', len(d('{args.out}') & d('{args.exclude[0] if args.exclude else ''}')))\"")


if __name__ == "__main__":
    main()
