"""Check the truncation warning fires on the user's real shape and not otherwise."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

WORDS = ("the lantern guttered as she crossed the frozen weir and counted "
         "seven names carved beneath the mill wheel before dawn broke over "
         "the valley again while nobody spoke of what had happened there")


def make_run(root, n_premises, chapters, finish_reason, words_per_chapter=2088):
    root.mkdir(parents=True, exist_ok=True)
    base = WORDS.split()
    for p in range(n_premises):
        pid = f"cw{p:02d}"
        steps = [{"step": "plan", "prompt": "p", "text": "plan text",
                  "finish_reason": "stop", "words": 2},
                 {"step": "characters", "prompt": "c", "text": "Ada\nBram",
                  "finish_reason": "stop", "words": 2}]
        for c in range(1, chapters + 1):
            # Vary wording per chapter so lexical metrics are not degenerate.
            body = " ".join(base[(c + i) % len(base)] for i in range(words_per_chapter))
            steps.append({"step": f"chapter{c:02d}", "prompt": "x", "text": body,
                          "finish_reason": finish_reason, "words": words_per_chapter})
        (root / f"{pid}.json").write_text(json.dumps({
            "premise_id": pid, "bucket": "creative_writing", "model": "m",
            "chapters": chapters, "target_words": 1000, "premise": "x",
            "stress": [], "steps": steps}), encoding="utf-8")


def run(d):
    r = subprocess.run([sys.executable, "tools/longform_ab.py", "metrics", str(d),
                        "--no-entities"], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


tmp = Path(tempfile.mkdtemp())

# The user's actual situation: 12 premises x 8 chapters, all truncated.
bad = tmp / "all_truncated"
make_run(bad, 12, 8, "length")
code, log = run(bad)
assert code == 0, log
print("--- 96/96 truncated ---")
for l in log.splitlines():
    if any(k in l for k in ("length MAE", "truncated chapters", "finish_reason=length",
                            "--chapter-tokens", "mid-sentence", "ceiling")):
        print("   ", l.rstrip())
assert "96/96" in log and "100%" in log, log
assert "UNRELIABLE" in log, log
assert "finish_reason=length" in log, log

# Chapters that ended naturally must not trip it.
good = tmp / "clean"
make_run(good, 12, 8, "stop", words_per_chapter=1050)
code, log = run(good)
assert code == 0, log
print("\n--- 0/96 truncated ---")
for l in log.splitlines():
    if any(k in l for k in ("length MAE", "truncated chapters")):
        print("   ", l.rstrip())
assert "0/96" in log and "UNRELIABLE" not in log, log
assert "finish_reason=length" not in log, log

# Just under the 25% threshold stays quiet; just over speaks up.
for reason_count, expect in ((2, False), (3, True)):
    mixed = tmp / f"mixed{reason_count}"
    mixed.mkdir(parents=True, exist_ok=True)
    make_run(mixed, 1, 8, "stop")
    d = json.loads((mixed / "cw00.json").read_text())
    for s in [s for s in d["steps"] if s["step"].startswith("chapter")][:reason_count]:
        s["finish_reason"] = "length"
    (mixed / "cw00.json").write_text(json.dumps(d), encoding="utf-8")
    code, log = run(mixed)
    got = "UNRELIABLE" in log
    assert got == expect, f"{reason_count}/8 truncated: warned={got} want={expect}"
    print(f"\n{reason_count}/8 truncated ({reason_count / 8:.0%}) -> warned={got}")

print("\nall assertions passed")
