"""Event log, counters, checkpoints and output sinks.

All sinks are append-only JSONL / JSON and flushed after every write, so a
crash or a resumed run never loses an accepted batch. This is the audit trail
the example walkthrough is rendered from.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class EventLog:
    """Per-article, append-only event log (one file per article)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts: Counter = Counter()
        self._lock = threading.Lock()

    def log(self, article_id: int, kind: str, **data: Any) -> dict:
        event = {"ts": _now(), "article_id": article_id, "kind": kind, **data}
        with self._lock:
            _append_jsonl(self.root / f"article_{article_id}.jsonl", event)
            self.counts[kind] += 1
        return event

    def read(self, article_id: int) -> list[dict]:
        return list(_read_jsonl(self.root / f"article_{article_id}.jsonl"))


class OutputSink:
    """Shared, thread-safe sinks for records, rejections and checkpoints."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done_articles: set[int] = set()
        self._claimed: set[int] = set()
        self._load_done()

    def _load_done(self) -> None:
        path = self.out / "done.jsonl"
        for row in _read_jsonl(path):
            if row.get("article_id") is not None:
                self.done_articles.add(int(row["article_id"]))

    def is_done(self, article_id: int) -> bool:
        return article_id in self.done_articles

    def claim(self, article_id: int) -> bool:
        """Reserve an article for one worker; False if done or already taken."""
        with self._lock:
            if article_id in self.done_articles or article_id in self._claimed:
                return False
            self._claimed.add(article_id)
            return True

    def write_records(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            for row in rows:
                _append_jsonl(self.out / "records.jsonl", row)

    def write_rejected(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            for row in rows:
                _append_jsonl(self.out / "rejected.jsonl", row)

    def mark_done(self, article_id: int, stats: dict) -> None:
        row = {"article_id": article_id, "ts": _now(), **stats}
        with self._lock:
            _append_jsonl(self.out / "done.jsonl", row)
            self.done_articles.add(article_id)

    def save_checkpoint(self, article_id: int, state: dict) -> None:
        path = self.out / "checkpoints" / f"{article_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.flush()
        tmp.replace(path)

    def load_checkpoint(self, article_id: int) -> dict | None:
        path = self.out / "checkpoints" / f"{article_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def clear_checkpoint(self, article_id: int) -> None:
        path = self.out / "checkpoints" / f"{article_id}.json"
        if path.exists():
            path.unlink()

    def write_stats(self, stats: dict) -> None:
        path = self.out / "stats.json"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            f.flush()
        tmp.replace(path)


def merge_article_stats(all_stats: list[dict]) -> dict:
    agg = {
        "articles_total": 0,
        "articles_done": 0,
        "sentences_total": 0,
        "sentences_accepted": 0,
        "sentences_rejected": 0,
        "tool_calls": Counter(),
        "rewrites": 0,
        "rewrites_by_kind": Counter(),
        "reject_reasons": Counter(),
        "tool_rounds": 0,
        "repairs": 0,
    }
    for s in all_stats:
        agg["articles_total"] += 1
        agg["articles_done"] += 1
        agg["sentences_total"] += s.get("sentences_total", 0)
        agg["sentences_accepted"] += s.get("sentences_accepted", 0)
        agg["sentences_rejected"] += s.get("sentences_rejected", 0)
        agg["rewrites"] += s.get("rewrites", 0)
        agg["tool_rounds"] += s.get("tool_rounds", 0)
        agg["repairs"] += s.get("repairs", 0)
        agg["tool_calls"].update(s.get("tool_calls") or {})
        agg["rewrites_by_kind"].update(
            s.get("rewrite_kinds") or s.get("rewrites_by_kind") or {}
        )
        agg["reject_reasons"].update(s.get("reject_reasons") or {})
    for key in ("tool_calls", "rewrites_by_kind", "reject_reasons"):
        agg[key] = dict(agg[key])
    agg["generated_at"] = _now()
    return agg
