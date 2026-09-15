#!/usr/bin/env python3
"""Validate a calibration YAML against the Hub before building it.

Every source is checked for three things that otherwise only surface as a
silent shortfall an hour into build_calib_from_yaml.py:

  * the split exists
  * it holds at least num_samples rows
  * the declared columns exist in the schema

A wrong column name is the expensive one. The formatters return None for a
missing column and the builder just reports 0/N and moves on, so the sample
count quietly drops while the bucket percentages stay plausible.

Uses the datasets-server metadata API, so it downloads nothing. A 501 means
the repo is script-based and has no parquet conversion, which is also the
reason it cannot be loaded by datasets v4 at all.

Usage:
    python tools/check_calib_spec.py --yaml data/behemoth_r1_123b_calib.yaml
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

import yaml

API = "https://datasets-server.huggingface.co/info?dataset="

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--yaml", required=True, help="Calibration spec to validate.")
ap.add_argument("--timeout", type=int, default=40)
args = ap.parse_args()


def fetch(dataset):
    try:
        req = urllib.request.Request(API + dataset, headers={"User-Agent": "quant-toolkit"})
        return json.load(urllib.request.urlopen(req, timeout=args.timeout))
    except urllib.error.HTTPError as exc:
        return {"_err": f"HTTP {exc.code} {exc.reason}"}
    except Exception as exc:
        return {"_err": f"{type(exc).__name__}: {exc}"}


def main():
    with open(args.yaml, encoding="utf-8") as f:
        spec = yaml.safe_load(f)["calibration_set"]

    entries = spec.get("datasets", [])
    default_len = int(spec.get("max_seq_length", 4096))
    problems, buckets, lengths = [], {}, {}

    print(f"Validating {len(entries)} sources from {args.yaml}\n")
    for entry in entries:
        dataset = entry["dataset"]
        want = int(entry.get("num_samples", 0))
        split = entry.get("split", "train")
        columns = entry.get("columns") or []

        buckets[entry.get("bucket", "unlabelled")] = (
            buckets.get(entry.get("bucket", "unlabelled"), 0) + want)
        group_len = int(entry.get("max_seq_length", default_len))
        lengths[group_len] = lengths.get(group_len, 0) + want

        info = fetch(dataset)
        if "_err" in info:
            note = info["_err"]
            if "501" in note:
                note += " (script-based repo, no parquet conversion)"
            problems.append((dataset, note))
            print(f"  {dataset[:56]:<56} {'?':>8}  {note}")
            continue

        configs = info.get("dataset_info") or {}
        subset = entry.get("subset")
        key = subset if subset in configs else next(iter(configs), None)
        config = configs.get(key) or {}
        splits = config.get("splits") or {}
        schema = list(config.get("features") or {})
        rows = (splits.get(split) or {}).get("num_examples")

        notes = []
        if rows is None and splits:
            notes.append(f"split {split!r} not in {list(splits)}")
        elif rows is not None and rows < want:
            notes.append(f"TOO SMALL: want {want}, have {rows}")

        missing = [c for c in columns if c not in schema]
        if missing and schema:
            notes.append(f"columns {missing} not in {schema}")

        for note in notes:
            problems.append((dataset, note))
        rows_txt = "?" if rows is None else str(rows)
        print(f"  {dataset[:56]:<56} {rows_txt:>8}  {'; '.join(notes) or 'ok'}")

    total = sum(buckets.values())
    print(f"\nPlanned {total} samples across {len(entries)} sources")
    for name, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {count:>6}  {100 * count / total:5.1f}%")
    print(f"  by max_seq_length: {dict(sorted(lengths.items()))}")

    if problems:
        print(f"\n{len(problems)} problem(s) — fix these before building:")
        for dataset, note in problems:
            print(f"  {dataset}: {note}")
        sys.exit(1)
    print("\nAll sources resolve, hold enough rows, and declare valid columns.")


if __name__ == "__main__":
    main()
