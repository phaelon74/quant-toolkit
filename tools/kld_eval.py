#!/usr/bin/env python3
"""Measure how far a quantized checkpoint's output distribution drifts from BF16.

KL divergence against the unquantized model is the metric that actually
separates two quant recipes. Perplexity alone hides damage: a quant can keep
PPL flat while reshuffling everything below the argmax, which is exactly what
degrades long-form generation. So this reports both, plus top-1 agreement.

Teacher-forced and deterministic. Prompts are sent as token IDs, not text, so
both runs score byte-identical positions and no tokenizer drift can creep in
between the BF16 reference and the candidate.

Two phases, because the BF16 model and the candidate rarely fit at once:

    collect   score a corpus against one served model, save logprobs to .npz
    compare   read two .npz files, report PPL, KLD, and agreement

The server must be started with --max-logprobs >= the -k used here; vLLM's
default of 20 is too coarse for a stable KLD tail.

Usage:
    # 1. serve BF16, collect the reference
    python tools/kld_eval.py collect \
        --base-url http://localhost:8000/v1 --model behemoth-bf16 \
        --tokenizer /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
        --texts data/eval/behemoth_holdout.jsonl \
        --out /media/fmodels2/working_Model-Opt/kld/bf16.npz

    # 2. serve the NVFP4 export, collect the candidate (same --texts, same -k)
    python tools/kld_eval.py collect ... --out .../nvfp4_omlp.npz

    # 3. compare
    python tools/kld_eval.py compare .../bf16.npz .../nvfp4_omlp.npz
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PAD_ID = -1


def line(label, value, note=""):
    print(f"  {label:<30} {value}{('   ' + note) if note else ''}")


# ---------------------------------------------------------------------------
# collect


def load_windows(texts_path, tokenizer_dir, seq_len, max_seqs):
    """Tokenize the corpus into fixed-length windows of exactly seq_len tokens.

    Fixed length keeps every position equally weighted in the averages and
    makes the two runs trivially comparable. Short documents are dropped rather
    than padded; padding would score meaningless positions.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)

    windows, rejected = [], [0]
    with open(texts_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            # "messages" rows are rendered through the chat template so the
            # scored tokens match what the model sees when actually served;
            # scoring raw text would measure a distribution nobody uses.
            if row.get("messages"):
                # Mistral v7 rejects non-alternating roles, which ShareGPT
                # sources do contain. Skip those rows instead of aborting a run
                # that is otherwise fine.
                try:
                    ids = tok.apply_chat_template(row["messages"], tokenize=True)
                except Exception:
                    rejected[0] += 1
                    continue
            elif row.get("text"):
                ids = tok(row["text"], add_special_tokens=False)["input_ids"]
            else:
                continue
            for start in range(0, len(ids) - seq_len + 1, seq_len):
                windows.append(ids[start:start + seq_len])
                if len(windows) >= max_seqs:
                    return windows, rejected[0]
    return windows, rejected[0]


def score_window(session, args, ids):
    """Return (actual logprobs, top-k ids, top-k logprobs) for one window.

    Position 0 has no prediction, so every array is seq_len-1 long.
    """
    k = args.k
    resp = session.post(
        args.base_url.rstrip("/") + "/completions",
        # Token IDs, not text: the server must score exactly the positions we
        # tokenized, or the two runs are not comparable.
        json={"model": args.model, "prompt": ids, "max_tokens": 1,
              "temperature": 0.0, "echo": False, "prompt_logprobs": k},
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=args.timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} from server: {resp.text[:400]}")

    plp = resp.json()["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise RuntimeError(
            "server returned no prompt_logprobs; it is likely too old or was "
            "started without --max-logprobs")

    n = len(ids) - 1
    actual = np.zeros(n, dtype=np.float32)
    top_ids = np.full((n, k), PAD_ID, dtype=np.int32)
    top_lp = np.full((n, k), -np.inf, dtype=np.float32)

    for pos in range(1, len(ids)):
        entry = plp[pos] or {}
        # Keys arrive as ints from the python client and strings over raw JSON.
        pairs = sorted(((int(t), v["logprob"] if isinstance(v, dict) else v.logprob)
                        for t, v in entry.items()),
                       key=lambda p: -p[1])
        i = pos - 1
        want = ids[pos]
        for tid, lp in pairs:
            if tid == want:
                actual[i] = lp
                break
        for slot, (tid, lp) in enumerate(pairs[:k]):
            top_ids[i, slot] = tid
            top_lp[i, slot] = lp

    return actual, top_ids, top_lp


def cmd_collect(args):
    import requests

    print("\n=== corpus ===")
    windows, rejected = load_windows(args.texts, args.tokenizer, args.seq_len,
                                     args.max_seqs)
    if not windows:
        sys.exit(f"no window of {args.seq_len} tokens found in {args.texts}")
    line("texts", args.texts)
    line("windows", f"{len(windows)} x {args.seq_len} tokens")
    line("scored positions", len(windows) * (args.seq_len - 1))
    if rejected:
        line("rejected by chat template", rejected, "non-alternating roles")

    session = requests.Session()

    print("\n=== scoring ===")
    line("endpoint", args.base_url)
    line("model", args.model)
    line("top-k", args.k)

    results = [None] * len(windows)
    done = 0

    def run(i):
        return i, score_window(session, args, windows[i])

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for i, out in pool.map(run, range(len(windows))):
            results[i] = out
            done += 1
            if done % max(1, len(windows) // 20) == 0 or done == len(windows):
                print(f"  {done}/{len(windows)} windows", flush=True)

    actual = np.concatenate([r[0] for r in results])
    top_ids = np.concatenate([r[1] for r in results])
    top_lp = np.concatenate([r[2] for r in results])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        actual_lp=actual,
        top_ids=top_ids,
        top_lp=top_lp,
        prompt_ids=np.asarray(windows, dtype=np.int32),
        meta=json.dumps({"model": args.model, "seq_len": args.seq_len,
                         "k": args.k, "texts": str(args.texts)}),
    )

    print("\n=== result ===")
    line("perplexity", f"{float(np.exp(-actual.mean())):.4f}")
    line("saved", str(out))


# ---------------------------------------------------------------------------
# compare


def logsumexp(a, axis):
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak, axis=axis) + np.log(
        np.exp(a - peak).sum(axis=axis))


def cmd_compare(args):
    ref = np.load(args.reference, allow_pickle=False)
    cand = np.load(args.candidate, allow_pickle=False)
    ref_meta = json.loads(str(ref["meta"]))
    cand_meta = json.loads(str(cand["meta"]))

    print("\n=== inputs ===")
    line("reference", f"{ref_meta['model']}", str(args.reference))
    line("candidate", f"{cand_meta['model']}", str(args.candidate))

    if not np.array_equal(ref["prompt_ids"], cand["prompt_ids"]):
        sys.exit("prompt token IDs differ; the two runs did not score the same "
                 "positions and the comparison would be meaningless")
    line("positions", len(ref["actual_lp"]), "identical prompts")

    ref_ids, ref_lp = ref["top_ids"], ref["top_lp"]
    cand_ids, cand_lp = cand["top_ids"], cand["top_lp"]

    # Mask padded slots so they contribute no probability mass.
    ref_lp = np.where(ref_ids == PAD_ID, -np.inf, ref_lp)
    cand_lp = np.where(cand_ids == PAD_ID, -np.inf, cand_lp)

    n = len(ref_lp)
    kld = np.empty(n, dtype=np.float64)
    covered = np.empty(n, dtype=np.float64)

    # Chunked: the id-matching broadcast is [chunk, k, k] and would otherwise
    # allocate tens of GB at corpus scale.
    step = 4096
    for s in range(0, n, step):
        e = min(s + step, n)
        r_ids, r_lp = ref_ids[s:e], ref_lp[s:e].astype(np.float64)
        c_ids, c_lp = cand_ids[s:e], cand_lp[s:e].astype(np.float64)

        match = r_ids[:, :, None] == c_ids[:, None, :]
        found = match.any(axis=2)
        slot = match.argmax(axis=2)
        q_lp = np.take_along_axis(c_lp, slot, axis=1)

        # A reference token outside the candidate's top-k gets the candidate's
        # smallest observed logprob. That is the most generous available bound,
        # so a reported KLD is a floor, never an exaggeration.
        floor = np.min(np.where(np.isfinite(c_lp), c_lp, np.inf), axis=1)
        q_lp = np.where(found, q_lp, floor[:, None])
        q_lp = np.where(np.isfinite(r_lp), q_lp, -np.inf)

        log_p = r_lp - logsumexp(r_lp, axis=1)[:, None]
        log_q = q_lp - logsumexp(q_lp, axis=1)[:, None]
        p = np.exp(log_p)

        terms = np.where(p > 0, p * (log_p - log_q), 0.0)
        kld[s:e] = terms.sum(axis=1)
        covered[s:e] = np.where(found, p, 0.0).sum(axis=1)

    ref_top1 = ref_ids[np.arange(n), np.argmax(ref_lp, axis=1)]
    cand_top1 = cand_ids[np.arange(n), np.argmax(cand_lp, axis=1)]
    agree = float((ref_top1 == cand_top1).mean())

    ref_ppl = float(np.exp(-ref["actual_lp"].mean()))
    cand_ppl = float(np.exp(-cand["actual_lp"].mean()))

    print("\n=== perplexity ===")
    line("reference", f"{ref_ppl:.4f}")
    line("candidate", f"{cand_ppl:.4f}")
    line("ratio", f"{cand_ppl / ref_ppl:.4f}", f"{(cand_ppl / ref_ppl - 1) * 100:+.2f}%")

    print("\n=== KL divergence (nats) ===")
    line("mean", f"{kld.mean():.6f}")
    line("median", f"{np.median(kld):.6f}")
    line("p90", f"{np.percentile(kld, 90):.6f}")
    line("p99", f"{np.percentile(kld, 99):.6f}")
    line("max", f"{kld.max():.6f}")

    print("\n=== agreement ===")
    line("top-1 match", f"{agree * 100:.2f}%")
    line("ref mass inside cand top-k", f"{covered.mean() * 100:.3f}%",
         "low values mean -k was too small")

    print("\n=== reading ===")
    print("  mean KLD  <0.01 excellent, 0.01-0.05 good, 0.05-0.15 noticeable,")
    print("            >0.15 expect visible quality loss in long generations.")
    print("  p99 matters more than the mean for creative writing: it is the")
    print("  tail of positions where the quant changed its mind.")

    if covered.mean() < 0.98:
        print("\n  WARNING: candidate top-k covers only "
              f"{covered.mean() * 100:.2f}% of reference mass; re-collect with "
              "a larger -k for a trustworthy KLD.")


# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)

c = sub.add_parser("collect", help="Score a corpus against one served model.")
c.add_argument("--base-url", default="http://localhost:8000/v1")
c.add_argument("--api-key", default="EMPTY")
c.add_argument("--model", required=True, help="Model name as served.")
c.add_argument("--tokenizer", required=True,
               help="Path used to tokenize locally. Both runs must use the same one.")
c.add_argument("--texts", required=True,
               help="JSONL with a 'messages' or 'text' field. Use held-out data, "
                    "not the calibration set.")
c.add_argument("--out", required=True, help="Destination .npz.")
c.add_argument("--seq-len", type=int, default=1024)
c.add_argument("--max-seqs", type=int, default=256)
c.add_argument("-k", type=int, default=64,
               help="Top-k logprobs per position. Server needs --max-logprobs >= this.")
c.add_argument("--concurrency", type=int, default=8)
c.add_argument("--timeout", type=int, default=600)
c.set_defaults(func=cmd_collect)

m = sub.add_parser("compare", help="Compare two collected .npz files.")
m.add_argument("reference", help=".npz from the BF16 model.")
m.add_argument("candidate", help=".npz from the quantized model.")
m.set_defaults(func=cmd_compare)

args = ap.parse_args()
args.func(args)
