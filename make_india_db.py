#!/usr/bin/env python3
"""Build the India-only corpus database used by the agent.

``dataset_amr/data/legal.db`` mixes Indian legal commentary with German
material (``verfassungsblog``). The agent only needs India, so this script
copies the source database, deletes every article whose ``source`` is not in
the India set, and rebuilds the paragraph FTS index.

    python make_india_db.py --source /path/to/legal.db
    python make_india_db.py --source /path/to/legal.db --out data/india.db

The output has the same schema and article/paragraph/sentence ids as the
source, so existing caches keyed by sentence id keep working.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "india.db"

# Indian legal sources in the corpus. ``verfassungsblog`` is the German blog
# and is deliberately excluded.
INDIA_SOURCES = (
    "livelaw",
    "indiacorplaw",
    "scctimes",
    "scobserver",
    "barandbench",
)


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone() is not None


def build_india_db(source: Path, out: Path, sources: tuple[str, ...]) -> dict:
    if not source.exists():
        raise SystemExit(f"source database not found: {source}")
    if source.resolve() == out.resolve():
        raise SystemExit("refusing to overwrite the source database")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    try:
        src.backup(dst)
        dst.execute("PRAGMA foreign_keys = OFF")
        marks = ",".join("?" for _ in sources)
        keep = f"SELECT id FROM articles WHERE source IN ({marks})"

        old_total = dst.execute("SELECT count(*) FROM articles").fetchone()[0]
        n_articles = dst.execute(
            f"SELECT count(*) FROM articles WHERE source IN ({marks})", sources
        ).fetchone()[0]

        dst.execute(
            "DELETE FROM sentences WHERE paragraph_id IN ("
            "SELECT id FROM paragraphs WHERE article_id NOT IN ("
            f"{keep}))", sources,
        )
        dst.execute(
            "DELETE FROM paragraphs WHERE article_id NOT IN ("
            f"{keep})", sources,
        )
        dst.execute(f"DELETE FROM articles WHERE id NOT IN ({keep})", sources)
        if _table_exists(dst, "seen"):
            dst.execute(
                f"DELETE FROM seen WHERE source IS NOT NULL AND source NOT IN ({marks})",
                sources,
            )
        dst.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("scope", "india"),
        )
        dst.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("sources", json.dumps(list(sources))),
        )

        if _table_exists(dst, "paragraphs_fts"):
            dst.execute("INSERT INTO paragraphs_fts(paragraphs_fts) VALUES('rebuild')")

        dst.commit()
        dst.execute("VACUUM")
        dst.commit()

        n_paragraphs = dst.execute("SELECT count(*) FROM paragraphs").fetchone()[0]
        n_sentences = dst.execute("SELECT count(*) FROM sentences").fetchone()[0]
    finally:
        dst.close()
        src.close()

    if out.exists():
        out.unlink()
    tmp.replace(out)
    return {
        "out": str(out),
        "source": str(source),
        "sources": list(sources),
        "articles_before": old_total,
        "articles": n_articles,
        "paragraphs": n_paragraphs,
        "sentences": n_sentences,
        "bytes": out.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="full corpus database (read-only)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"India-only output (default: {DEFAULT_OUT})")
    parser.add_argument("--sources", default=",".join(INDIA_SOURCES),
                        help="comma-separated Indian corpus sources")
    args = parser.parse_args()
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    report = build_india_db(args.source, args.out, sources)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
