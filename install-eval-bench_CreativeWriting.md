# Creative Writing Evaluation Bench

Everything needed to judge whether NVFP4 quantization cost Behemoth-R1-123B-v2
anything that matters, for a model whose job is prose, character work and
roleplay.

Three checkpoints are in play, and every tool here should be run against all
three or none:

| Label | Path | Size |
| --- | --- | ---: |
| `bf16` | `/media/fmodels/TheDrummer/Behemoth-R1-123B-v2` | 229 GB |
| `nvfp4-omlp` | `/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4` | 86 GiB |
| `nvfp4-omlp-q` | `/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4-q` | ~68 GiB |

---

## 1. Read this before installing anything

**The measurement you want is a difference between three near-identical models,
not a leaderboard position.** That single fact should drive every choice below,
and it is where most quantization evaluations go wrong.

LLM-judge benchmarks were built to separate GPT-4 from Mistral-7B — gaps of
hundreds of Elo. You are looking for a gap that may be a few points, sitting
underneath judge noise that is comfortably larger. Running Creative Writing v3
once per checkpoint and comparing the three scores is very likely to measure
nothing but variance, and it will look like a real result.

Three consequences:

1. **Run the deterministic, judge-free metrics first** (§4). They cost nothing,
   have no variance, and detect the specific failure modes quantization actually
   causes. If NVFP4 broke something, repetition and lexical collapse show it
   before any judge does.
2. **When you do use a judge, compare outputs pairwise on identical prompts with
   identical seeds**, not by scoring each model in isolation. Paired comparison
   cancels prompt difficulty, which is the dominant variance term.
3. **Budget iterations for statistical power, not coverage.** Three iterations
   over 32 prompts is a leaderboard protocol. For an A/B you want more samples
   on fewer prompts.

`tools/kld_eval.py` already gave you the cheapest capability signal there is:
the near-certain top-1 agreement rate. Everything here is for the questions it
cannot answer — whether the prose is still *good*, not just similar.

---

## 2. Serving the models identically

Any difference in sampling configuration between runs invalidates the whole
comparison. Both EQ-Bench harnesses generate at **`temperature=0.7`, `min_p=0.1`**
by default; leave that alone and change nothing else between checkpoints.

```bash
# In the vLLM venv. One checkpoint at a time -- BF16 alone needs all four cards.
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_FORCE_SM=120f

vllm serve /media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/nvfp4 \
    --served-model-name behemoth-nvfp4-omlp \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

`--max-model-len 32768` is not optional for the longform bench: it writes eight
chapters of roughly 1,000 words each while carrying the plan and character
profiles in context, and a 8192 window will truncate it into incoherence.

Two things to verify on the first run:

- **`min_p` reaches vLLM.** It is not an OpenAI-standard field. vLLM accepts it,
  but if a harness sends it somewhere that rejects unknown keys you will get
  either a 400 or, worse, silent greedy-ish sampling. Confirm with a manual
  `curl` including `"min_p": 0.1` before launching a paid run.
- **Serve the tokenizer from the checkpoint itself**, not a separate path, so the
  chat template applied is the one shipped in the export.

Use a distinct `--served-model-name` per checkpoint. Every harness records it,
and mixing runs under one name silently pools results from different models.

---

## 3. Venv and the judge

Keep this separate from both the quantization and vLLM environments.

```bash
python -m venv ~/venvs/evalbench
source ~/venvs/evalbench/bin/activate
pip install --upgrade pip
```

### The judge decision

Every rubric benchmark here needs a judge model stronger than the model under
test, reached over an OpenAI-compatible API. For a 123B creative-writing model
that means a frontier API model; a local judge is not credible here and would
also be competing for the same GPUs.

| Judge | Used by | Note |
| --- | --- | --- |
| `claude-sonnet-4-6` | Creative Writing v3 | Required for leaderboard-comparable Elo |
| `anthropic/claude-sonnet-4` | Longform Writing | Repo default |
| `claude-opus-4-6` | EQ-Bench 3 | What the public leaderboard uses |

You need an Anthropic key, or an OpenRouter key which covers all of them behind
one endpoint and is simpler for three harnesses. Set a spend cap before you
start — a longform run judges eight chapters plus the assembled novella, per
iteration, per checkpoint, and three checkpoints multiplies everything.

If you ever want to validate a cheaper judge rather than trust it,
**Judgemark** ([eqbench.com/judgemark-v2.html](https://eqbench.com/judgemark-v2.html))
measures how well a given judge model discriminates writing quality at all. Do
not swap in a cheap judge without it.

---

## 4. Tier 1 — deterministic, free, run these first

No judge, no API spend, no variance. These directly target how NVFP4 damage
manifests: narrowed logit distributions producing flatter, more repetitive,
lower-entropy prose.

### 4.1 Slop Score

```bash
git clone https://github.com/sam-paech/slop-score.git
cd slop-score && pip install -r requirements.txt
```

Measures over-represented words and phrases against a reference distribution,
plus **MATTR-500** (moving-average type-token ratio) for lexical diversity.
Leaderboard at [eqbench.com/slop-score.html](https://eqbench.com/slop-score.html).

The reason this is Tier 1 for a *quantization* comparison: if NVFP4 flattened
the tail of the distribution, the model reaches for stock phrasing more often
and MATTR-500 drops. That is a direct, unambiguous, judge-free signal of exactly
the damage you are looking for.

### 4.2 Slop Forensics

```bash
git clone https://github.com/sam-paech/slop-forensics.git
```

Stylometric analysis producing a per-model slop profile of over-represented
n-grams. Useful comparatively: generate the same 200 completions from each
checkpoint and diff the profiles. If the 68 GiB variant develops phrase
attractors the 86 GiB one lacks, that shows up here as a concrete list of
phrases rather than a score.

### 4.3 Repetition and degeneration at length

Quantization damage compounds autoregressively, so the failure you care about
appears at 4,000 tokens, not 400. Generate long continuations from each
checkpoint at fixed seed and measure:

- distinct-n (n = 1–4) and self-BLEU across samples
- MTLD or vocd-D lexical diversity
- verbatim n-gram repetition rate within a single generation
- whether any generation enters a repetition loop at all (a binary that matters
  more than any score — one looping checkpoint is disqualified)

The Creative Writing v3 harness computes slop and repetition metrics directly
from generated text with no LLM grading involved, so you get these alongside its
judged output for free.

### 4.4 Distribution fidelity — already done

`tools/kld_eval.py` covers this. Re-read the **near-certain top-1 agreement**
row and the per-domain table rather than the aggregate KLD. See
`Behemoth-123B_v2_R1.md` §6.6.

---

## 5. Tier 2 — LLM-judged writing quality

### 5.1 Longform Writing Bench — the most relevant one here

[github.com/EQ-bench/longform-writing-bench](https://github.com/EQ-bench/longform-writing-bench)

**Start with this one, not Creative Writing v3.** It is the only benchmark
listed here whose structure matches how quantization actually fails: it runs 13
sequential generation steps — five for planning and character profiles, then
eight chapters of ~1,000 words — and judges narrative consistency across the
whole thing. Damage that compounds over long context is precisely what a 4-bit
model risks, and precisely what a 500-word prompt response cannot reveal.

```bash
git clone https://github.com/EQ-bench/longform-writing-bench.git
cd longform-writing-bench
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

| Key | Value |
| --- | --- |
| `TEST_API_URL` | `http://localhost:8000/v1` |
| `TEST_API_KEY` | `EMPTY` |
| `JUDGE_API_URL` | your judge endpoint |
| `JUDGE_API_KEY` | your judge key |
| `REQUEST_TIMEOUT` | raise well above the `300` default — 1,000-word chapters from a 123B are slow |

```bash
python3 longform_writing_bench.py \
    --test-model  "behemoth-nvfp4-omlp" \
    --judge-model "anthropic/claude-sonnet-4" \
    --runs-file   "results/longform_bench_runs.json" \
    --run-id      "nvfp4_omlp" \
    --threads     8 \
    --iterations  1
```

Chapters are scored on a 0–20 rubric, the assembled story is judged separately,
and the result carries a 95% CI from 500 bootstrap resamples. **Read the CI, not
the mean.** If the intervals for two checkpoints overlap, you have not measured
a difference — and at `--iterations 1` they very likely will.

Runs are resumable with atomic writes, so re-issuing the same command after a
crash skips completed steps. Use a distinct `--run-id` per checkpoint and keep
one shared `--runs-file` so results sit side by side.

### 5.2 Creative Writing v3

[github.com/EQ-bench/creative-writing-bench](https://github.com/EQ-bench/creative-writing-bench)

32 prompts × 3 iterations = 96 items, scored against a rubric and then placed by
sparse pairwise matchups into a Glicko-2 Elo.

```bash
git clone https://github.com/EQ-bench/creative-writing-bench.git
cd creative-writing-bench
pip install -r requirements.txt
# or: pip install requests python-dotenv numpy scipy tqdm glicko2 nltk joblib
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict')"
cp .env.example .env
```

```bash
python3 creative_writing_bench.py \
    --test-model "behemoth-nvfp4-omlp" \
    --judge-model "claude-sonnet-4-6" \
    --runs-file "creative_bench_runs.json" \
    --creative-prompts-file "data/creative_writing_prompts_v3.json" \
    --run-id "nvfp4_omlp" \
    --threads 8 \
    --iterations 3
```

Prompts cover humour, romance, spatial awareness and unusual perspectives, and
the harness includes explicit mitigation for length, position, verbosity and
poetic-incoherence judge biases.

**Two cautions specific to your use case.** Keep the bundled
`creative_bench_runs.json`, since Elo is computed relative to the historical
models inside it — but for comparing your three checkpoints, **use the raw
rubric scores rather than the Elo**. Elo is normalized by anchoring reference
models and adds pairwise-sampling noise on top of rubric noise; for an A/B
between near-identical models that is noise you don't need. Second, `--threads
500` in the upstream README assumes a hosted API. Against one vLLM instance,
keep it near your actual concurrency or you will just build a queue.

### 5.3 EQ-Bench 3 — roleplay and emotional intelligence

[github.com/EQ-bench/eqbench3](https://github.com/EQ-bench/eqbench3)

A multi-turn benchmark assessing active EQ, interpersonal skill, psychological
insight and analytical depth through challenging roleplays. Given that roleplay
is 20% of your calibration mix and a core use case for this finetune, this
covers a capability the writing benchmarks miss: **sustaining a character across
turns** rather than producing one good passage.

Multi-turn also means errors compound across turns, making it a second
long-horizon probe alongside the longform bench.

---

## 6. Tier 3 — objective and complementary probes

### 6.1 Creative Story-Writing Benchmark

[github.com/lechmazur/writing](https://github.com/lechmazur/writing)

Requires the model to integrate **10 mandatory story elements** — characters,
objects, core concepts, attributes, motivations — into a short story, graded
across 16 rubric parts.

Worth running because constraint satisfaction is more objective than "is this
good prose." Whether ten required elements are present and coherently used is
closer to a checkable fact than a matter of taste, which makes it less
judge-noise-prone than open-ended quality scoring. It also probes instruction
following under creative load, a capability quantization can degrade
independently of prose quality.

### 6.2 Divergent Thinking

[github.com/lechmazur/divergent](https://github.com/lechmazur/divergent)

Generates 25 unique unconnected words from a given letter. A narrow, cheap probe
of output diversity — and directly sensitive to a distribution that has been
flattened by quantization. Low cost, quick to run, sharp signal for this
specific failure mode.

### 6.3 Refusal drift

Not a published benchmark, but worth building as a fixed prompt set. Behemoth is
an uncensored RP finetune, and quantization perturbs logits enough to shift
refusal boundaries in either direction. A model that starts refusing scenarios
the BF16 original handled has lost something real that no writing-quality
benchmark will report. Run a fixed set of in-character prompts through all three
checkpoints and diff the refusal rate.

### 6.4 Position bias, if you build your own pairwise judging

[github.com/lechmazur/position_bias](https://github.com/lechmazur/position_bias)

Documents how strongly judges favour one slot in a pairwise comparison. If you
write your own A/B harness — which §7 recommends — you must randomize
presentation order and ideally score each pair both ways. Skipping this produces
a confident, entirely artificial winner.

---

## 7. The comparison that will actually resolve your question

Given how small the expected differences are, this is what I would spend the
budget on rather than three independent leaderboard runs.

**Build a paired A/B.** Take 50 prompts spanning creative writing and roleplay.
Generate from all three checkpoints at identical sampling settings and identical
seeds. Present outputs to the judge in randomized pairs, blind, both orderings,
asking only which is better and by how much. Aggregate as win rate with a
binomial confidence interval.

This works where the leaderboard protocols struggle because prompt difficulty —
the largest variance component — cancels within each pair. It is also much
cheaper: one judge call per pair rather than a full rubric per item, plus Elo
machinery you don't need.

**Suggested order of work:**

1. Tier 1 on all three checkpoints. Free, deterministic. If repetition or
   lexical diversity has visibly degraded, you have your answer and can stop.
2. Longform Writing Bench, `--iterations 1`, all three. Confirms nothing is
   structurally broken over long context.
3. Paired A/B on 50 prompts, `nvfp4-omlp` versus `nvfp4-omlp-q`. This is the
   decision you actually need to make; BF16 is the sanity anchor, not the
   candidate.
4. EQ-Bench 3 only if roleplay specifically looks suspect.
5. Creative Writing v3 last, and mainly if you want a public number to cite.

**Expected outcome, stated in advance so it can be wrong:** given a mean KLD of
0.0198 and reasoning diverging *less* than prose, I expect Tier 1 and the
longform bench to find no significant difference between the two NVFP4
variants, and the paired A/B to land near 50% with an interval spanning it. If
that happens it is a real result — it means take the 68 GiB checkpoint and the
single-GPU serving that comes with it.

---

## 8. Pitfalls

**Judge noise exceeds your effect size.** The single most likely failure mode.
Always report intervals; treat overlapping intervals as "no difference measured"
rather than a ranking.

**Elo is the wrong instrument for an A/B.** It normalizes against anchor models
and adds sampling noise. Use raw rubric scores or direct pairwise win rates.

**Sampling drift between runs.** Confirm `temperature`, `min_p`, `seed`,
`max_tokens` and the chat template are byte-identical across checkpoints.
Serving one model with a truncated `--max-model-len` is an easy way to
manufacture a large fake difference.

**Optimal sampling may itself have shifted.** NVFP4 changes the logit
distribution, so the temperature that suited BF16 is not automatically ideal for
a quantized variant. Comparing both at BF16-tuned settings is the fair
comparison and the right default — but if a variant looks weak, a short
temperature sweep before concluding is worthwhile.

**Long generations are slow at TP4.** The PCIe all-reduce bottleneck
(`Behemoth-123B_v2_R1.md` §3) makes the longform bench substantially slower than
throughput numbers suggest. Budget hours, and remember TP2 is faster for the 86
GiB checkpoint if you are not comparing timing.

**Contamination.** Do not evaluate on anything in
`data/text/behemoth_r1_123b_calib_*.jsonl`. These benchmarks ship their own
prompts, so this mainly applies if you assemble your own A/B set — reuse
`data/behemoth_r1_123b_eval.yaml`'s exclusion mechanism.

---

## 9. Component summary

| Component | Judge? | Cost | Measures |
| --- | --- | --- | --- |
| [`kld_eval.py`](tools/kld_eval.py) | no | free | Distribution fidelity, decision agreement |
| [slop-score](https://github.com/sam-paech/slop-score) | no | free | Slop, MATTR-500 lexical diversity |
| [slop-forensics](https://github.com/sam-paech/slop-forensics) | no | free | Over-represented n-gram profile |
| [longform-writing-bench](https://github.com/EQ-bench/longform-writing-bench) | yes | high | Long-context narrative coherence |
| [creative-writing-bench](https://github.com/EQ-bench/creative-writing-bench) | yes | medium | Rubric prose quality, Elo |
| [eqbench3](https://github.com/EQ-bench/eqbench3) | yes | medium | Multi-turn roleplay, EQ |
| [lechmazur/writing](https://github.com/lechmazur/writing) | yes | medium | Constraint satisfaction in fiction |
| [lechmazur/divergent](https://github.com/lechmazur/divergent) | yes | low | Output diversity |
| [Judgemark](https://eqbench.com/judgemark-v2.html) | — | low | Whether your judge can discriminate at all |
| [position_bias](https://github.com/lechmazur/position_bias) | — | — | Reference for building pairwise judging |
| Refusal drift set | no | free | Behavioural shift on an uncensored finetune |

Related reading: [Antislop framework](https://arxiv.org/pdf/2510.15061) and
[auto-antislop](https://github.com/sam-paech/auto-antislop), if slop turns out to
be a problem worth actively suppressing rather than only measuring.
