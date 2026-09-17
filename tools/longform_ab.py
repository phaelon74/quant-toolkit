#!/usr/bin/env python3
"""Long-form A/B harness: does a quantized checkpoint still write well?

Answers what tools/kld_eval.py structurally cannot. That tool is teacher-forced
-- every position is scored against BF16's context, so it measures single
decisions, not what happens when a model conditions on its own prior output.
Damage compounds over 8,000 words. This generates that text and measures it.

Deliberately judge-free by default. For two quantizations of one base model, an
LLM judge's noise floor and position bias both exceed the effect size, so the
primary instruments here are deterministic metrics plus your own blind reading.
See install-eval-bench_CreativeWriting.md.

Four phases:

    generate   plan -> characters -> N chapters against one served endpoint
    metrics    repetition, lexical diversity, entity drift, length adherence
    pair       blind randomized pairs for human review
    report     metric table plus win rate with a confidence interval

Fixed seeds throughout, derived from premise id and step index, so sampling
noise is not a confound: two checkpoints see identical prompts and identical
seeds, and any divergence is the quantization.

Usage:
    # once per checkpoint, against whichever model is currently served
    python tools/longform_ab.py generate \
        --model behemoth-nvfp4-omlp --run-id omlp \
        --out-dir /media/fmodels2/working_Model-Opt/longform

    python tools/longform_ab.py metrics \
        /media/fmodels2/working_Model-Opt/longform/omlp

    python tools/longform_ab.py pair \
        --runs .../longform/omlp .../longform/omlp_q \
        --out-dir .../longform/pairs

    python tools/longform_ab.py report \
        --runs .../longform/omlp .../longform/omlp_q \
        --pairs .../longform/pairs
"""

import argparse
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def line(label, value, note=""):
    print(f"  {label:<30} {value}{('   ' + note) if note else ''}")


def stable_seed(*parts):
    """Deterministic seed from arbitrary strings.

    Python's hash() is salted per process, so it cannot be used: the two
    checkpoints would get different seeds on different runs and the comparison
    would silently include sampling noise.
    """
    h = hashlib.sha1("\x00".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


# ---------------------------------------------------------------------------
# generate

PLAN_PROMPT = """You are writing a novella of {chapters} chapters, about \
{target_words} words each, from this premise:

{premise}

Write the plan only. Give the title, the setting, the central tension, and a \
one-paragraph outline for each of the {chapters} chapters, numbered. Do not \
write any prose from the story itself."""

CHARACTER_PROMPT = """Write the character profiles for this novella.

For each significant character give, on its own line and in this order: full \
name, age, one line of physical description including eye and hair colour, \
their occupation, what they want, and what they are hiding. Be specific and \
concrete -- these details must stay consistent for the whole novella.

Cover every character who appears in more than one chapter, and no others."""

CHAPTER_PROMPT = """Write chapter {n} of {chapters}, about {target_words} \
words.

Follow the plan and keep every established detail consistent with what you \
have already written. Write the chapter as finished prose. Do not summarise, \
do not include notes to the reader, and do not write a chapter heading beyond \
the chapter number and title."""


def chat(session, args, messages, seed, max_tokens):
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        # min_p is not an OpenAI field but vLLM honours it, and it is what the
        # EQ-Bench harnesses generate with. Kept so numbers stay comparable.
        "min_p": args.min_p,
    }
    resp = session.post(args.base_url.rstrip("/") + "/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {args.api_key}"},
                        timeout=args.timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} from server: {resp.text[:400]}")
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    finish = body["choices"][0].get("finish_reason")
    return text, finish


def generate_story(session, args, spec, premise):
    """Run the full pipeline for one premise, returning every artifact.

    Context is cumulative and verbatim: the plan, the profiles, and all prior
    chapters stay in the conversation. Summarising earlier chapters instead
    would hide exactly the failure being measured, since a model that has lost
    track of chapter 2 looks fine if something else re-states chapter 2 for it.
    """
    chapters = spec["chapters"]
    target = spec["target_words"]
    pid = premise["id"]

    # ~1.6 tokens/word plus headroom; chapters overrun their target routinely.
    chapter_tokens = int(target * 1.6 * 1.6)

    convo = []
    steps = []

    def step(name, prompt, max_tokens):
        convo.append({"role": "user", "content": prompt})
        text, finish = chat(session, args, convo,
                            stable_seed(pid, name, args.seed_salt), max_tokens)
        convo.append({"role": "assistant", "content": text})
        steps.append({"step": name, "prompt": prompt, "text": text,
                      "finish_reason": finish, "words": len(text.split())})
        return text

    step("plan", PLAN_PROMPT.format(chapters=chapters, target_words=target,
                                    premise=premise["premise"]),
         args.plan_tokens)
    step("characters", CHARACTER_PROMPT, args.plan_tokens)
    for n in range(1, chapters + 1):
        step(f"chapter{n:02d}",
             CHAPTER_PROMPT.format(n=n, chapters=chapters, target_words=target),
             chapter_tokens)

    return {
        "premise_id": pid,
        "bucket": premise.get("bucket"),
        "stress": premise.get("stress", []),
        "premise": premise["premise"],
        "model": args.model,
        "run_id": args.run_id,
        "sampling": {"temperature": args.temperature, "min_p": args.min_p,
                     "seed_salt": args.seed_salt},
        "chapters": chapters,
        "target_words": target,
        "steps": steps,
    }


def cmd_generate(args):
    import requests

    spec = json.loads(Path(args.premises).read_text(encoding="utf-8"))
    premises = spec["premises"]
    if args.only:
        wanted = set(args.only)
        premises = [p for p in premises if p["id"] in wanted]
    if args.bucket:
        premises = [p for p in premises if p.get("bucket") == args.bucket]
    if not premises:
        sys.exit("no premise selected")

    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== generate ===")
    line("model", args.model)
    line("run id", args.run_id)
    line("premises", len(premises))
    line("steps each", spec["chapters"] + 2,
         f"plan, characters, {spec['chapters']} chapters")
    line("sampling", f"temp {args.temperature}, min_p {args.min_p}",
         f"seed salt {args.seed_salt}")
    line("output", str(out))

    session = requests.Session()
    todo = []
    for premise in premises:
        dest = out / f"{premise['id']}.json"
        if dest.exists() and not args.overwrite:
            print(f"  skip {premise['id']} (exists)")
            continue
        todo.append((premise, dest))

    if not todo:
        print("\n  nothing to do; pass --overwrite to regenerate.")
        return

    def run(item):
        premise, dest = item
        story = generate_story(session, args, spec, premise)
        dest.write_text(json.dumps(story, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        words = sum(s["words"] for s in story["steps"]
                    if s["step"].startswith("chapter"))
        return premise["id"], words

    print()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for pid, words in pool.map(run, todo):
            print(f"  wrote {pid}  ({words} words of prose)", flush=True)

    print(f"\n  {len(todo)} stor{'y' if len(todo) == 1 else 'ies'} in {out}")
    print(f"  next: python tools/longform_ab.py metrics {out}")


# ---------------------------------------------------------------------------
# metrics

WORD_RE = re.compile(r"[a-z']+")


def words_of(text):
    return WORD_RE.findall(text.lower())


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(tokens, n):
    grams = ngrams(tokens, n)
    return len(set(grams)) / len(grams) if grams else 0.0


def mattr(tokens, window=500):
    """Moving-average type-token ratio.

    Plain TTR is unusable for comparing texts of different lengths because it
    falls mechanically as a text grows. MATTR holds the window fixed, so a
    difference between two runs is a difference in diversity, not in length.
    """
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    total = 0.0
    for i in range(len(tokens) - window + 1):
        total += len(set(tokens[i:i + window])) / window
    return total / (len(tokens) - window + 1)


def mtld(tokens, threshold=0.72):
    """Measure of Textual Lexical Diversity, bidirectional, standard threshold.

    Reported alongside MATTR because the two fail differently: MATTR is blind to
    diversity structure beyond its window, MTLD is sensitive to it.
    """
    def one_pass(seq):
        factors, types, count = 0, set(), 0
        for w in seq:
            count += 1
            types.add(w)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold)
        return len(seq) / factors if factors else float(len(seq))

    if len(tokens) < 50:
        return 0.0
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def repetition(tokens):
    """Degeneration signals, from mild to disqualifying."""
    from collections import Counter

    eight = Counter(ngrams(tokens, 8))
    dup8 = sum(c - 1 for c in eight.values() if c > 1)
    twenty = Counter(ngrams(tokens, 20))
    looped = {g: c for g, c in twenty.items() if c >= 3}

    return {
        "repeat_8gram_rate": dup8 / max(1, sum(eight.values())),
        "looped_20grams": len(looped),
        # A model that loops is disqualified regardless of any other score, so
        # this is reported as a flag rather than folded into a number.
        "loop_detected": bool(looped),
        "loop_example": " ".join(max(looped, key=looped.get)) if looped else None,
    }


NAME_NOISE = re.compile(r"^(mr|mrs|ms|miss|dr|sir|lady|lord|captain|the)\b\s*",
                        re.I)


def clean_name(text):
    # Looped, not a single sub: titles stack ("the Lady Halva"), and stripping
    # only the outermost would leave "Lady Halva" to be reported as a different
    # person from "Halva".
    text = text.strip()
    while True:
        stripped = NAME_NOISE.sub("", text)
        if stripped == text:
            break
        text = stripped
    return re.sub(r"['\u2019]s$", "", text).strip(" .,:;\"'\u2014-")


def entity_drift(nlp, profile_text, chapter_texts):
    """Cast consistency across chapters, via NER on the model's own profiles.

    The character profiles the model wrote are the ground truth: it committed to
    a cast, so anything else is its own drift. Two signals, both robust:

      unknown persons -- named people who were never established, i.e. the model
        inventing cast because it lost the thread
      name variants   -- near-misses on an established name (Elara -> Elera),
        which is the classic signature of a degraded checkpoint

    Deliberately not attempting general contradiction detection. "Her eyes were
    green in chapter 2 and blue in chapter 6" needs coreference plus attribute
    grounding, and a fragile implementation of it would produce noise that looks
    like signal.
    """
    from difflib import SequenceMatcher

    cast = set()
    for ent in nlp(profile_text).ents:
        if ent.label_ == "PERSON":
            name = clean_name(ent.text)
            if len(name) > 2:
                cast.add(name)

    # Individual tokens of established names count as known: "Corvane" should
    # match "Corvane Adler" without being reported as an invention.
    cast_tokens = {t for name in cast for t in name.split() if len(t) > 2}

    unknown, variants, seen = {}, {}, set()
    for text in chapter_texts:
        for ent in nlp(text).ents:
            if ent.label_ != "PERSON":
                continue
            name = clean_name(ent.text)
            if len(name) <= 2:
                continue
            if name in cast or name in cast_tokens:
                seen.add(name)
                continue
            near = max(cast_tokens | cast,
                       key=lambda c: SequenceMatcher(None, name.lower(),
                                                     c.lower()).ratio(),
                       default=None)
            ratio = (SequenceMatcher(None, name.lower(), near.lower()).ratio()
                     if near else 0.0)
            if 0.80 <= ratio < 1.0:
                variants[name] = variants.get(name, 0) + 1
            else:
                unknown[name] = unknown.get(name, 0) + 1

    return {
        "cast_size": len(cast),
        "cast_seen": len(seen),
        "cast_coverage": len(seen) / len(cast) if cast else 0.0,
        "unknown_persons": sum(unknown.values()),
        "unknown_distinct": len(unknown),
        "unknown_examples": sorted(unknown, key=unknown.get, reverse=True)[:8],
        "name_variants": sum(variants.values()),
        "name_variant_examples": sorted(variants, key=variants.get,
                                        reverse=True)[:8],
    }


def load_nlp():
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy is required for entity drift:\n"
                 "  pip install spacy\n"
                 "  python -m spacy download en_core_web_sm\n"
                 "Or pass --no-entities to skip it.")
    try:
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    except OSError:
        sys.exit("spaCy model missing:\n"
                 "  python -m spacy download en_core_web_sm")


def story_metrics(story, nlp):
    from collections import Counter

    steps = {s["step"]: s for s in story["steps"]}
    chapters = [s for k, s in sorted(steps.items()) if k.startswith("chapter")]
    prose = "\n\n".join(c["text"] for c in chapters)
    tokens = words_of(prose)

    target = story["target_words"]
    lengths = [c["words"] for c in chapters]
    truncated = sum(1 for c in chapters if c["finish_reason"] == "length")

    out = {
        "premise_id": story["premise_id"],
        "bucket": story.get("bucket"),
        "stress": story.get("stress", []),
        "total_words": len(prose.split()),
        "chapters": len(chapters),
        "chapter_words_mean": sum(lengths) / len(lengths) if lengths else 0,
        "chapter_words_min": min(lengths) if lengths else 0,
        "chapter_words_max": max(lengths) if lengths else 0,
        # Chronic undershoot is a real failure: a model losing the thread stops
        # early rather than writing badly.
        "length_mae_pct": (sum(abs(w - target) for w in lengths)
                           / max(1, len(lengths)) / target * 100),
        "truncated_chapters": truncated,
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "distinct_3": distinct_n(tokens, 3),
        "distinct_4": distinct_n(tokens, 4),
        "mattr_500": mattr(tokens),
        "mtld": mtld(tokens),
    }
    out.update(repetition(tokens))

    if nlp is not None:
        out["entities"] = entity_drift(nlp, steps["characters"]["text"],
                                       [c["text"] for c in chapters])

    # Saved so `report` can compute over-representation against the other run
    # rather than against an external slop list, which would date badly and is
    # not the comparison being made anyway.
    out["_top_4grams"] = Counter(ngrams(tokens, 4)).most_common(400)
    return out


def cmd_metrics(args):
    run = Path(args.run_dir)
    stories = sorted(run.glob("*.json"))
    stories = [p for p in stories if p.name != "metrics.json"]
    if not stories:
        sys.exit(f"no story JSON in {run}")

    nlp = None if args.no_entities else load_nlp()

    print("\n=== metrics ===")
    line("run", str(run))
    line("stories", len(stories))
    line("entity drift", "skipped" if nlp is None else "spaCy en_core_web_sm")

    results = []
    for path in stories:
        story = json.loads(path.read_text(encoding="utf-8"))
        results.append(story_metrics(story, nlp))
        print(f"  scored {story['premise_id']}", flush=True)

    dest = run / "metrics.json"
    dest.write_text(json.dumps({"run": run.name, "stories": results},
                               ensure_ascii=False, indent=1), encoding="utf-8")

    def avg(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n=== summary ===")
    line("total words", sum(r["total_words"] for r in results))
    line("mean chapter words", f"{avg('chapter_words_mean'):.0f}")
    line("length MAE", f"{avg('length_mae_pct'):.1f}%")
    line("truncated chapters", sum(r["truncated_chapters"] for r in results))
    line("MATTR-500", f"{avg('mattr_500'):.4f}")
    line("MTLD", f"{avg('mtld'):.1f}")
    line("distinct-4", f"{avg('distinct_4'):.4f}")
    line("repeat 8-gram rate", f"{avg('repeat_8gram_rate') * 100:.3f}%")

    looped = [r["premise_id"] for r in results if r["loop_detected"]]
    line("stories with loops", f"{len(looped)}", ", ".join(looped) or "none")

    if nlp is not None:
        line("unknown persons", sum(r["entities"]["unknown_persons"]
                                    for r in results))
        line("name variants", sum(r["entities"]["name_variants"]
                                 for r in results))
        line("cast coverage", f"{sum(r['entities']['cast_coverage'] for r in results) / len(results) * 100:.1f}%")

    if looped:
        print("\n  A repetition loop disqualifies a checkpoint on its own.")
        print("  Inspect loop_example in metrics.json before anything else.")

    print(f"\n  wrote {dest}")


# ---------------------------------------------------------------------------
# pair


def cmd_pair(args):
    """Write blinded side-by-side pairs and an empty verdict sheet.

    Blinding is the whole point. You know which checkpoint you would like to
    win, and reading labelled outputs will confirm it. Side assignment is
    randomized per premise from a fixed seed, so the mapping is reproducible but
    not visible in the files you read.
    """
    if len(args.runs) != 2:
        sys.exit("pair compares exactly two runs")

    runs = [Path(r) for r in args.runs]
    names = [r.name for r in runs]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    common = sorted(set.intersection(*[
        {p.stem for p in r.glob("*.json") if p.name != "metrics.json"}
        for r in runs]))
    if not common:
        sys.exit("the two runs share no premise ids")

    print("\n=== pair ===")
    line("run A pool", names[0], str(runs[0]))
    line("run B pool", names[1], str(runs[1]))
    line("shared premises", len(common))

    key, verdicts = {}, {}
    for pid in common:
        rng = random.Random(stable_seed(pid, args.seed_salt, *names))
        order = [0, 1]
        rng.shuffle(order)
        # left/right rather than A/B for the reader, so nothing in the filename
        # hints at run order.
        key[pid] = {"left": names[order[0]], "right": names[order[1]]}

        pdir = out / pid
        pdir.mkdir(exist_ok=True)
        for side, run_idx in (("left", order[0]), ("right", order[1])):
            story = json.loads((runs[run_idx] / f"{pid}.json")
                               .read_text(encoding="utf-8"))
            body = [f"# {pid} — {side}", "", f"> {story['premise']}", ""]
            for step in story["steps"]:
                if step["step"].startswith("chapter"):
                    body += [f"## {step['step']}", "", step["text"], ""]
            (pdir / f"{side}.md").write_text("\n".join(body), encoding="utf-8")

        verdicts[pid] = {"winner": None, "confidence": None, "note": ""}

    (out / "_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    sheet = out / "verdicts.json"
    if sheet.exists() and not args.overwrite:
        print(f"\n  kept existing {sheet} (pass --overwrite to reset)")
    else:
        sheet.write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
        print(f"\n  wrote {sheet}")

    print(f"""
  Read {out}/<premise>/left.md against right.md, then fill in verdicts.json:

    "winner":     "left", "right", or "tie"
    "confidence": "clear" or "slight"
    "note":       optional, why

  Do not open _key.json until you are done. Then:
    python tools/longform_ab.py report --runs {' '.join(args.runs)} \\
        --pairs {out}""")


# ---------------------------------------------------------------------------
# report


def wilson(wins, total, z=1.96):
    """Wilson score interval.

    Used instead of the normal approximation because these samples are small
    (a dozen premises) and often near 0 or 1, where the normal interval runs
    outside [0,1] and badly understates uncertainty.
    """
    if not total:
        return 0.0, 0.0, 0.0
    p = wins / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cmd_report(args):
    from collections import Counter

    runs = [Path(r) for r in args.runs]
    loaded = []
    for r in runs:
        mpath = r / "metrics.json"
        if not mpath.exists():
            sys.exit(f"{mpath} missing; run `metrics {r}` first")
        loaded.append(json.loads(mpath.read_text(encoding="utf-8")))

    print("\n=== deterministic metrics ===")
    fields = [
        ("total words", "total_words", "{:.0f}", 0),
        ("mean chapter words", "chapter_words_mean", "{:.0f}", 0),
        ("length MAE %", "length_mae_pct", "{:.1f}", -1),
        ("truncated chapters", "truncated_chapters", "{:.0f}", -1),
        ("MATTR-500", "mattr_500", "{:.4f}", 1),
        ("MTLD", "mtld", "{:.1f}", 1),
        ("distinct-2", "distinct_2", "{:.4f}", 1),
        ("distinct-4", "distinct_4", "{:.4f}", 1),
        ("repeat 8-gram %", "repeat_8gram_rate", "{:.3f}", -1),
        ("stories with loops", "loop_detected", "{:.0f}", -1),
    ]

    header = "  " + f"{'metric':<22}" + "".join(
        f"{d['run'][:14]:>16}" for d in loaded) + f"{'better':>9}"
    print(header)

    def mean(d, key):
        vals = [s[key] for s in d["stories"]]
        vals = [float(v) for v in vals if isinstance(v, (int, float, bool))]
        return sum(vals) / len(vals) if vals else 0.0

    for label, key, fmt, direction in fields:
        cells = []
        for d in loaded:
            v = mean(d, key)
            if key == "repeat_8gram_rate":
                v *= 100
            if key in ("loop_detected", "truncated_chapters", "total_words"):
                v = sum(float(s[key]) for s in d["stories"])
            cells.append(v)
        # ASCII only: this table gets piped through tee into logs, and a dash
        # that renders as mojibake on one console is not worth the typography.
        arrow = {1: "higher", -1: "lower", 0: "info"}[direction]
        print("  " + f"{label:<22}"
              + "".join(f"{fmt.format(c):>16}" for c in cells)
              + f"{arrow:>9}")

    if any("entities" in s for d in loaded for s in d["stories"]):
        print("\n=== entity drift ===")
        print("  " + f"{'metric':<22}" + "".join(
            f"{d['run'][:14]:>16}" for d in loaded) + f"{'better':>9}")
        for label, key, fmt in (("cast coverage %", "cast_coverage", "{:.1f}"),
                                ("unknown persons", "unknown_persons", "{:.0f}"),
                                ("name variants", "name_variants", "{:.0f}")):
            cells = []
            for d in loaded:
                vals = [s["entities"][key] for s in d["stories"]
                        if "entities" in s]
                v = sum(vals)
                if key == "cast_coverage":
                    v = sum(vals) / len(vals) * 100 if vals else 0.0
                cells.append(v)
            arrow = "higher" if key == "cast_coverage" else "lower"
            print("  " + f"{label:<22}"
                  + "".join(f"{fmt.format(c):>16}" for c in cells)
                  + f"{arrow:>9}")

    # Comparative slop: phrases one run leans on and the other does not. No
    # external slop list, which would date and is not this comparison anyway.
    if len(loaded) == 2:
        print("\n=== phrase over-representation ===")
        tables = []
        for d in loaded:
            c = Counter()
            for s in d["stories"]:
                for gram, n in s.get("_top_4grams", []):
                    c[tuple(gram)] += n
            tables.append(c)
        for i, j in ((0, 1), (1, 0)):
            extra = [(g, tables[i][g] - tables[j].get(g, 0))
                     for g in tables[i] if tables[i][g] >= 4]
            extra.sort(key=lambda kv: -kv[1])
            print(f"\n  leaned on by {loaded[i]['run']}:")
            for gram, delta in extra[:8]:
                if delta <= 0:
                    break
                print(f"    +{delta:<4} \"{' '.join(gram)}\"")

    # Blind human verdicts.
    if args.pairs:
        pairs = Path(args.pairs)
        sheet = json.loads((pairs / "verdicts.json").read_text(encoding="utf-8"))
        key = json.loads((pairs / "_key.json").read_text(encoding="utf-8"))

        tally, ties, unfilled = Counter(), 0, 0
        for pid, v in sheet.items():
            w = (v.get("winner") or "").strip().lower()
            if w in ("left", "right"):
                tally[key[pid][w]] += 1
            elif w == "tie":
                ties += 1
            else:
                unfilled += 1

        print("\n=== blind human A/B ===")
        line("premises judged", sum(tally.values()) + ties)
        if unfilled:
            line("not yet judged", unfilled)
        line("ties", ties)
        decided = sum(tally.values())
        for name in sorted(tally, key=lambda k: -tally[k]):
            p, lo, hi = wilson(tally[name], decided)
            line(name, f"{tally[name]}/{decided} wins",
                 f"{p * 100:.0f}%  95% CI [{lo * 100:.0f}%, {hi * 100:.0f}%]")
        if decided:
            _, lo, hi = wilson(max(tally.values()), decided)
            if lo <= 0.5 <= hi:
                print("\n  The interval spans 50%, so no preference was")
                print("  demonstrated. With near-identical checkpoints that is")
                print("  the expected and useful result: take the smaller one.")
            else:
                print("\n  The interval excludes 50%: a real preference.")

    print("\n=== reading ===")
    print("  A repetition loop or a truncation spike disqualifies a checkpoint")
    print("  regardless of everything else. Check those rows first.")
    print("  MATTR-500 and MTLD falling together means the distribution")
    print("  narrowed -- the failure token-level agreement cannot see.")


# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)

g = sub.add_parser("generate", help="Generate stories from one served model.")
g.add_argument("--base-url", default="http://localhost:8000/v1")
g.add_argument("--api-key", default="EMPTY")
g.add_argument("--model", required=True, help="Model name as served.")
g.add_argument("--run-id", required=True,
               help="Subdirectory name for this checkpoint's output.")
g.add_argument("--out-dir", required=True)
g.add_argument("--premises", default="data/longform_ab_premises.json")
g.add_argument("--only", nargs="*", help="Restrict to these premise ids.")
g.add_argument("--bucket", help="Restrict to one bucket.")
g.add_argument("--temperature", type=float, default=0.7)
g.add_argument("--min-p", type=float, default=0.1)
g.add_argument("--seed-salt", default="v1",
               help="Change to get a different but still reproducible draw. "
                    "Must match across checkpoints being compared.")
g.add_argument("--plan-tokens", type=int, default=3000)
g.add_argument("--concurrency", type=int, default=4)
g.add_argument("--timeout", type=int, default=1800)
g.add_argument("--overwrite", action="store_true")
g.set_defaults(func=cmd_generate)

t = sub.add_parser("metrics", help="Score one run's stories, judge-free.")
t.add_argument("run_dir", help="Directory of story JSON from `generate`.")
t.add_argument("--no-entities", action="store_true",
               help="Skip spaCy entity drift.")
t.set_defaults(func=cmd_metrics)

p = sub.add_parser("pair", help="Write blinded pairs for human review.")
p.add_argument("--runs", nargs=2, required=True, help="Two run directories.")
p.add_argument("--out-dir", required=True)
p.add_argument("--seed-salt", default="v1",
               help="Changes the left/right assignment reproducibly.")
p.add_argument("--overwrite", action="store_true",
               help="Reset verdicts.json, discarding any verdicts already given.")
p.set_defaults(func=cmd_pair)

o = sub.add_parser("report", help="Aggregate metrics and human verdicts.")
o.add_argument("--runs", nargs="+", required=True)
o.add_argument("--pairs", default=None,
               help="Pair directory with a filled-in verdicts.json.")
o.set_defaults(func=cmd_report)

args = ap.parse_args()
args.func(args)
