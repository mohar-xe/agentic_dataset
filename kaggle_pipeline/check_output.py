#!/usr/bin/env python3
"""Inspect a downloaded Kaggle run (sentences_1000.jsonl).

Reports parse/wiki coverage and, when ``penman`` is importable, validates every
graph against Legal-AMR Lean Core v1.1 with ``legal_schema.check_legal_amr``.

Usage:
    python kaggle_pipeline/check_output.py data/kaggle_runs/<run>/out/sentences_1000.jsonl
"""

import argparse
import collections
import json
import sys

try:
    from legal_schema import check_legal_amr
except ImportError:  # running from another cwd
    sys.path.insert(0, ".")
    from legal_schema import check_legal_amr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="validate only the first N rows (0 = all)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    parsed = [r for r in rows if r.get("amr")]
    wiki = [r for r in rows if r.get("wiki")]

    print(f"rows              {len(rows)}")
    print(f"parsed            {len(parsed)}")
    print(f"parse failures    {len(rows) - len(parsed)}")
    print(f"rows with :wiki   {len(wiki)} "
          f"({100 * len(wiki) / max(len(rows), 1):.1f}%)")
    print(f"wiki nodes        {sum(len(r.get('wiki') or []) for r in rows)}")
    print(f"sources           {dict(collections.Counter(r.get('source') for r in rows))}")
    print(f"articles          {len({r.get('article_id') for r in rows})}")
    print(f"words             {sum(r.get('words') or 0 for r in rows)}")

    titles = collections.Counter(
        w["title"] for r in rows for w in (r.get("wiki") or []))
    print("top wiki titles   " + ", ".join(
        f"{t} ({n})" for t, n in titles.most_common(10)))

    subset = parsed[: args.limit] if args.limit else parsed
    ok = 0
    failures = []
    for r in subset:
        valid, errors, _ = check_legal_amr(r["amr"])
        if valid:
            ok += 1
        else:
            failures.append((r["para_id"], r["sent_index"], errors[:2]))
    print(f"schema-valid      {ok}/{len(subset)}")
    for pid, idx, errs in failures[:10]:
        print(f"  fail {pid}.{idx}: {errs}")


if __name__ == "__main__":
    main()
