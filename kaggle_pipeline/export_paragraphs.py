#!/usr/bin/env python3
"""Export the India corpus paragraphs for the Kaggle SaT+SPRING kernel.

The Kaggle kernel cannot read data/india.db directly, so this script writes a
compact gzip JSON payload (articles + paragraphs) that is uploaded once as a
private Kaggle dataset.  The database itself stays local.

Usage:
    python kaggle_pipeline/export_paragraphs.py [--db data/india.db]
                                                [--out kaggle_pipeline/upload]
"""

import argparse
import gzip
import json
import os
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/india.db")
    ap.add_argument("--out", default="kaggle_pipeline/upload")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # A plain connection dies if another process holds a write lock; the db is
    # read-only here, so retry briefly before giving up.
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row

    articles = {}
    for row in con.execute(
        "select id, url, title, source, section, author, words, n_paragraphs "
        "from articles order by id"
    ):
        articles[str(row["id"])] = {
            "url": row["url"],
            "title": row["title"],
            "source": row["source"],
            "section": row["section"],
            "author": row["author"],
            "words": row["words"],
            "n_paragraphs": row["n_paragraphs"],
        }

    paragraphs = []
    for row in con.execute(
        "select id, article_id, para_index, text, statutes, provisions, "
        "precedents from paragraphs order by article_id, para_index"
    ):
        tags = {}
        for col in ("statutes", "provisions", "precedents"):
            raw = row[col]
            try:
                tags[col] = json.loads(raw) if raw else []
            except (TypeError, ValueError):
                tags[col] = []
        paragraphs.append(
            {
                "para_id": row["id"],
                "article_id": row["article_id"],
                "para_index": row["para_index"],
                "text": row["text"],
                "tags": tags,
            }
        )

    payload = {
        "corpus": "india-legal-livelaw-scobserver",
        "publishable": 0,
        "n_articles": len(articles),
        "n_paragraphs": len(paragraphs),
        "articles": articles,
        "paragraphs": paragraphs,
    }
    path = os.path.join(args.out, "paragraphs.json.gz")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(payload, fh, ensure_ascii=False)
    size = os.path.getsize(path)
    print(f"wrote {path}: {size/1e6:.1f} MB "
          f"({len(articles)} articles, {len(paragraphs)} paragraphs)")

    meta = {
        "title": "Legal AMR India Paragraphs",
        "id": "mohar07/legal-amr-india-paragraphs",
        "licenses": [{"name": "other"}],
        "description": (
            "Research-only India legal corpus (LiveLaw + Supreme Court Observer) "
            "exported from data/india.db for sentence segmentation with SaT-12L-SM "
            "and AMR parsing. Non-publishable; do not redistribute."
        ),
    }
    with open(os.path.join(args.out, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote dataset-metadata.json")


if __name__ == "__main__":
    main()
