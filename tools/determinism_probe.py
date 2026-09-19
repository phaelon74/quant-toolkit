#!/usr/bin/env python3
"""Check whether a served model reproduces its own output byte for byte.

Batch invariance is about outputs not changing when *other* requests share the
batch. This asks the narrower question that actually matters when every request
is sent alone: given an identical body and seed, does the server return an
identical completion twice in a row? If yes, a --concurrency 1 evaluation run is
reproducible and batch-invariant kernels buy nothing. If no, some kernel on the
path is accumulating non-deterministically and no scheduling discipline will fix
it.

The prompt is padded to a configurable token budget on purpose. A short prompt
exercises neither chunked prefill nor long-context attention, which is where the
longform harness spends most of its time, so a short probe can pass while the
real run still drifts.
"""

import argparse
import hashlib
import random
import sys

import requests

WORDS = ("harbour lantern gravel quiet mercy iron thistle vellum shutter cinder "
         "orchard tallow bramble signal marrow pewter fathom lichen ember drift").split()


def filler(tokens, seed=7):
    """Roughly `tokens` tokens of varied prose-shaped text.

    Varied rather than repeated so prefix caching and dedup cannot collapse it
    into something unrepresentative of a real long context.
    """
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(int(tokens * 0.75))]
    return " ".join(words)


def served_root(args):
    """The filesystem path the server actually has open for this model name.

    A served name is just a label, so the same name can point at a different
    checkpoint after any re-serve. vLLM reports the real path in the model
    card's `root`, which is the only way to tell from the client side that you
    are scoring the file you think you are.
    """
    try:
        cards = requests.get(f"{args.base_url}/models", timeout=30).json()["data"]
    except Exception as exc:
        return f"<could not query /v1/models: {exc}>"
    for card in cards:
        if card.get("id") == args.model:
            return card.get("root") or "<no root reported>"
    return f"<{args.model} not served; served: " \
           f"{', '.join(repr(c.get('id')) for c in cards)}>"


def once(args, body):
    resp = requests.post(f"{args.base_url}/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {args.api_key}"},
                         timeout=args.timeout)
    if resp.status_code != 200:
        sys.exit(f"{resp.status_code} from server: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True, help="Model name as served.")
    p.add_argument("--trials", type=int, default=3,
                   help="Distinct prompts to test; each is sent twice.")
    p.add_argument("--prompt-tokens", type=int, default=24000,
                   help="Pad the prompt to about this many tokens.")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-p", type=float, default=0.1)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    print(f"model {args.model}, {args.trials} trials, "
          f"~{args.prompt_tokens} prompt tokens, {args.max_tokens} generated")
    print(f"served from {served_root(args)}")

    mismatches = 0
    sample = None
    for trial in range(args.trials):
        prompt = (f"{filler(args.prompt_tokens, seed=trial)}\n\n"
                  "Ignore the word list above. Write the opening of a story "
                  "about a lighthouse keeper who receives an unsigned letter.")
        body = {"model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "min_p": args.min_p,
                "seed": 1000 + trial}
        a, b = once(args, body), once(args, body)
        sample = sample or a
        ha = hashlib.sha256(a.encode()).hexdigest()[:12]
        hb = hashlib.sha256(b.encode()).hexdigest()[:12]
        ok = ha == hb
        mismatches += not ok
        note = ""
        if not ok:
            common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                          min(len(a), len(b)))
            note = f"  diverged at char {common} of {min(len(a), len(b))}"
        print(f"  trial {trial}: {'OK  ' if ok else 'DIFF'} {ha} {hb}{note}")

    # A broken export is perfectly reproducible, so the hashes above say nothing
    # about whether the checkpoint works. Print prose and read it.
    if sample:
        print("\n--- first 400 chars of output; this must read as English ---")
        print(sample[:400].replace("\n", " "))

    print()
    if mismatches:
        print(f"NOT reproducible ({mismatches}/{args.trials} differed).")
        print("Single-stream runs will not be bitwise repeatable on this build.")
    else:
        print(f"Reproducible across {args.trials}/{args.trials} trials.")
        print("A --concurrency 1 run is repeatable; batch-invariant kernels add nothing.")


if __name__ == "__main__":
    main()
