# KLD and confident top-1 agreement

What `tools/kld_eval.py` measures, and why the number we actually decide on is
**confident top-1 agreement** rather than KL divergence.

The question this tool answers is narrow on purpose: *when the BF16 weights are
certain about the next token, does the quantized checkpoint make the same call?*
Not "is this model good" — that is what benchmarks are for, and they cannot
resolve differences this small. The reference is always the exact BF16
checkpoint that was quantized, never an upstream base, so every number here is a
checkpoint compared to its own source.

## Two phases

The BF16 model and the candidate rarely fit on the same GPUs at once, so scoring
is split. `collect` talks to a served model and saves top-k logprobs to `.npz`;
`compare` and `report` are pure CPU work on those files and need no server.

```bash
# 1. serve BF16 with --max-logprobs >= k, then collect the reference
python tools/kld_eval.py collect \
    --model behemoth-bf16 \
    --tokenizer /media/fmodels/TheDrummer/Behemoth-R1-123B-v2 \
    --texts data/text/behemoth_r1_123b_eval_8192.jsonl \
    --out /media/fmodels2/working_Model-Opt/kld/bf16.npz \
    --seq-len 1024 --max-seqs 256 -k 64

# 2. serve the quant, same --texts, same --tokenizer, same -k
python tools/kld_eval.py collect --model behemoth-nvfp4-qkv ... \
    --out /media/fmodels2/working_Model-Opt/kld/nvfp4_qkv.npz

# 3. detailed view of one candidate, including the per-domain breakdown
python tools/kld_eval.py compare .../bf16.npz .../nvfp4_qkv.npz

# 4. rank every candidate against the reference
python tools/kld_eval.py report .../bf16.npz .../nvfp4_omlp.npz .../nvfp4_qkv.npz
```

Scoring is **teacher-forced and deterministic**. Prompts go to the server as
token IDs rather than text, so both runs score byte-identical positions and no
tokenizer drift can creep in. No sampling is involved anywhere, so the same two
`.npz` files always produce the same report.

## Why perplexity and KLD are not enough

**Perplexity hides reshuffling.** A quant can hold PPL nearly flat while
reordering everything below the argmax. PPL only looks at the probability
assigned to the one token that actually came next; it is blind to what happened
to the rest of the distribution, which is what shapes sampled generation.

**KLD is entropy-confounded.** It measures how far the whole distribution moved,
in nats, which makes it excellent for one job: catching a structurally broken
export. But its magnitude scales with how wide the reference distribution was to
begin with. Creative writing is inherently higher-entropy than arithmetic, so
some of its KLD is intrinsic rather than damage, and a raw KLD comparison across
domains rewards whichever domain the model was already confident about. It also
saturates for the comparison we actually care about — two good exports of the
same weights sit at 0.0198 and 0.0230 nats, which is a real difference in the
numbers and an unreadable one in practice.

So KLD is the early-warning system, not the decision metric.

## Confident top-1 agreement

Top-1 agreement asks the direct question: at each position, does the candidate's
argmax match the reference's argmax? But a single agreement number conflates two
completely different events:

- BF16 was split 51/49 between "walked" and "strode", and the quant picked the
  other one. **This costs you nothing.** It is interchangeable word choice.
- BF16 was 97% certain of a token, and the quant picked something else. **This
  is capability loss** — a decision the unquantized model considered settled.

So agreement is stratified by the reference's own confidence, and the tiers are
read bottom-up. On the 87 GiB `omlp` export:

| tier | agreement | positions |
|---|---|---|
| all positions | 94.10% | 261,888 |
| reference p > 0.5 | 99.19% | 168,632 |
| reference p > 0.9 | **99.93%** | 91,200 |

That top-to-bottom spread is the whole argument. Ranking on the all-positions
number would have made this export look mediocre, when in fact nearly all of its
6% disagreement is the harmless kind.

`flips / 10k` expresses the same thing as a rate over *all* scored positions
rather than a percentage of a subset, because subset sizes differ between
candidates and percentages of different denominators are not comparable. It also
maps onto something physical: confident decisions changed per ten thousand tokens
written.

## What we measured

Reference: BF16 at PPL 4.8029, 261,888 positions from 256 windows of 1,024
tokens, top-k 64, held-out corpus disjoint from calibration.

| candidate | on disk | conf. agree | flips/10k | all-pos | Δ PPL | KLD mean | entropy |
|---|---|---|---|---|---|---|---|
| NVFP4 `omlp` | 87 GB | 99.930% | 2.44 | 94.10% | +1.51% | 0.019825 | −0.24% |
| NVFP4 `qkv` | 66 GB | 99.927% | 2.56 | 93.57% | +1.76% | 0.023046 | +0.43% |

In absolute counts: of 91,200 positions where BF16 was more than 90% certain,
`omlp` changed about **64** decisions and `qkv` changed about **67**. Moving all
of q/k/v from BF16 into NVFP4 cost three extra confident flips across a quarter
of a million tokens and bought 21 GB.

For contrast, the first `qkv` export — structurally valid, but with untied
`q/k/v` weight amaxes inside the fused `qkv_proj` — scored **PPL 16,329 and
1.388% confident agreement**. Both metrics scream at that, which is the point:
KLD and PPL are what catch a broken export, and they catch it instantly.

## Per-domain, because "intelligence loss" is not one number

`compare` breaks every metric down by the `bucket` label carried in the eval
JSONL. An aggregate cannot separate lost reasoning from lost prose style; this
can. From the `omlp` run:

| domain | KLD mean | KLD p99 | top-1 | top-1 where ref p>0.9 |
|---|---|---|---|---|
| breadth | 0.011474 | 0.099816 | 96.38% | 100.00% |
| creative_writing | 0.021551 | 0.180514 | 93.51% | 99.89% |
| reasoning | 0.015584 | 0.123689 | 95.72% | 99.98% |
| roleplay | 0.019474 | 0.130029 | 93.84% | 99.97% |

The two top-1 columns are the fairer cross-domain read, for the entropy reason
above. Creative writing is the weakest domain on both measures and `breadth` is
untouched at 100.00%.

## Two things agreement cannot see

**Spurious confidence.** Agreement is one-directional. It catches the candidate
failing to reproduce the reference's confidence, but not the candidate becoming
confident where the reference was uncertain. A quant that is several times
sharper than BF16 at high-entropy positions can still score high agreement while
producing flatter, more repetitive prose. That is why the report carries an
`entropy` column — negative means the candidate narrowed.

**Error compounding.** This is the structural limit, and it is not fixable with
more data. Every position is scored given the *reference's* prefix, so the
candidate is put back on the rails 261,888 times and never allowed to drift. It
cannot see the failure where a slightly-wrong choice at token 500 leads somewhere
worse by token 5,000.

We have a measured example of exactly this. `qkv` scored its per-position entropy
**0.43% wider** than BF16 — teacher-forced, it looked marginally *more* diverse.
Left to generate 15,000-word stories on its own, it produced narrower and more
repetitive prose than BF16 on every measure: repeated 8-gram rate 1.909% → 2.185%,
MATTR-500 0.5428 → 0.5380, MTLD 101.5 → 97.0, invented person-names per 10k words
28.8 → 55.0, and detectable repetition loops in 8 of 12 stories against BF16's 5
— a strict superset, with no premise where the quant held together and BF16 did
not.

Which is the honest summary of the relationship between the two harnesses:

- **Agreement** is deterministic, n = 261,888, and *primary*. Rank on it.
- **Longform** (`tools/longform_ab.py`) is sampled, n = 12 premises, high
  variance, and answers only the one question teacher-forcing structurally
  cannot: does anything degenerate over thousands of tokens? Twelve sampled
  stories cannot overturn 261,888 deterministic comparisons, but they can reveal
  a failure mode those comparisons are blind to. The 5-versus-8 loop difference
  above has an exact two-sided McNemar p of 0.25 — directionally clean,
  statistically underpowered.

## Reading the output

| signal | reading |
|---|---|
| conf. agree | the decision metric. >99.5% is indistinguishable in practice |
| flips/10k | the same thing, physical. Single digits is fine |
| all-pos top-1 | mostly interchangeable word choice. Do not rank on it |
| entropy | negative = narrowed = flatter, more repetitive prose |
| mean KLD | <0.01 excellent, 0.01–0.05 good, >0.15 expect visible loss |
| KLD p99 | matters more than the mean for prose: the tail where the quant changed its mind |
| ref mass in cand top-k | below 98% means `-k` was too small and the KLD tail is unreliable |

Because a reference token outside the candidate's top-k is assigned the
candidate's smallest observed logprob — the most generous available bound — a
reported KLD is always a floor, never an exaggeration.

## Gotchas

- The server needs `--max-logprobs >= -k`. vLLM's default of 20 is too coarse
  for a stable KLD tail; we use 64.
- Use **held-out** text. `tools/build_calib_from_yaml.py --exclude` exists to
  make the eval set provably disjoint from the calibration set.
- Both `collect` runs must use the same `--texts`, `--tokenizer`, `--seq-len`,
  `--max-seqs` and `-k`. `compare` prints the position count for both files;
  if they differ, something drifted.
- Serve every model with **identical** flags, including tensor-parallel size.
  TP changes all-reduce ordering and therefore the numerics.
- Confirm *which file* the server has open, not just the served name. A served
  name is a label and can point anywhere after a re-serve;
  `tools/determinism_probe.py` prints the path from the model card's `root`.
  We once scored a good checkpoint and then generated from a broken one under
  the same name, and lost six GPU-hours to it.
