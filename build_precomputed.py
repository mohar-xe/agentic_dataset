#!/usr/bin/env python3
"""Bundle the precomputed SPRING graphs, matched 1:1 to ``data/india.db``.

The Kaggle run (``kaggle_pipeline``) parsed a seeded sample of corpus
sentences and wrote ``data/precomputed_15k/``:

    picked.json   15000 picked rows (article_id, para_id, sent_index, sentence, tags)
    indices.json  graph index -> row in the full 20k ``picked.json``
    graphs.json   {"graphs": [...], "parsed": 15000}

Kaggle re-segmented paragraphs with SaT, so a picked row is not always one
corpus sentence: 2,299 rows are fragments of a longer sentence (or two merged).
This script joins every row to ``india.db`` by exact paragraph + text and keeps
only rows that match exactly one corpus sentence, so the output is keyed by the
same ``sentence_id`` the agent, tools, records and validator already use.

Output: one JSON object per line, in corpus order, resumable and validated:

    {"sentence_id": 2, "article_id": 1, "amr": "...", "status": "ok", ...}

    python build_precomputed.py            # writes data/precomputed/graphs.jsonl
    python build_precomputed.py --check    # report coverage, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from common import validate_amr
from config import load_config

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "data" / "precomputed_15k"
DEFAULT_OUT = ROOT / "data" / "precomputed" / "graphs.jsonl"

_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ",
})


def key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(_QUOTES)).strip()


def load_corpus(path: Path) -> dict[tuple[int, str], list[dict]]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    index: dict[tuple[int, str], list[dict]] = defaultdict(list)
    try:
        for row in db.execute(
                "SELECT s.id, s.paragraph_id, s.idx, s.text, "
                "p.article_id, p.para_index "
                "FROM sentences s JOIN paragraphs p ON p.id = s.paragraph_id"):
            index[(row["paragraph_id"], key(row["text"]))].append(dict(row))
    finally:
        db.close()
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--corpus", default=None, help="corpus sqlite path")
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    cfg = load_config()
    corpus_path = Path(args.corpus or cfg["corpus_db"])
    picked = json.loads((args.src / "picked.json").read_text(encoding="utf-8"))
    graphs = json.loads((args.src / "graphs.json").read_text(encoding="utf-8"))
    amrs = graphs.get("graphs") or []
    if len(amrs) != len(picked):
        raise SystemExit(f"graphs ({len(amrs)}) and picked ({len(picked)}) "
                         "are not aligned; refusing to guess")

    index = load_corpus(corpus_path)
    rows: dict[int, dict] = {}
    skipped = {"no_graph": 0, "no_match": 0, "ambiguous": 0, "duplicate": 0}
    invalid = 0
    for i, (row, amr) in enumerate(zip(picked, amrs)):
        if not (amr or "").strip():
            skipped["no_graph"] += 1
            continue
        hits = index.get((row.get("para_id"), key(row.get("sentence")))) or []
        if not hits:
            skipped["no_match"] += 1
            continue
        if len(hits) > 1:
            skipped["ambiguous"] += 1
            continue
        hit = hits[0]
        sid = int(hit["id"])
        if sid in rows:
            skipped["duplicate"] += 1
            continue
        ok, errors, info = validate_amr(amr)
        if not ok:
            # A base graph SPRING itself marks invalid is still the crutch the
            # agent starts from; keep it and record why.
            invalid += 1
        tags = row.get("tags") or {}
        rows[sid] = {
            "sentence_id": sid,
            "article_id": int(hit["article_id"]),
            "paragraph_id": int(hit["paragraph_id"]),
            "para_index": int(hit["para_index"]),
            "idx": int(hit["idx"]),
            "sentence": hit["text"],
            "amr": amr,
            "status": "ok" if ok else "invalid",
            "errors": [] if ok else errors,
            "n_nodes": info.get("n_nodes"),
            "n_edges": info.get("n_edges"),
            "source": "kaggle-spring",
            "tags": {k: tags.get(k) or [] for k in
                     ("statutes", "provisions", "precedents")},
            "picked_index": i,
        }

    ordered = [rows[sid] for sid in sorted(rows)]
    total = len(index)
    print(f"corpus sentences: {sum(len(v) for v in index.values())} "
          f"({total} distinct paragraph+text keys)", file=sys.stderr)
    print(f"matched: {len(ordered)} ({invalid} base graph(s) invalid but "
          f"kept)  skipped: "
          f"{', '.join(f'{k}={v}' for k, v in skipped.items())}", file=sys.stderr)
    if args.check:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"wrote {len(ordered)} graph(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
