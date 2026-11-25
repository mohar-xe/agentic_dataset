#!/usr/bin/env python3
"""Annotate the bundled precomputed graphs.

    python main.py --agents 8 --tag arbitration --limit 100

One sentence is one unit of work. Its graph comes from
``data/precomputed/graphs.jsonl`` (matched 1:1 to ``data/india.db`` by
``build_precomputed.py``) and sits in the ``spring_parse`` cache, so SPRING is
never loaded. The agent, tools, validator, repair loop and sinks are the ones
in ``run.py``.

``--tag`` filters work (repeatable): source name, ``statute:...``,
``provision:...`` or ``precedent:...``; substring, case-insensitive.
``--agents`` is the number of parallel agents. Resume is by output: sentences
already in ``records.jsonl``/``rejected.jsonl`` are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from agent import ArticleAgent, _ledger_from, _merge_ledger
from build import build
from config import load_config, out_dir
from corpus import Corpus, Paragraph
from llm import OpenAIClient
from spring import SpringParser
from store import EventLog, OutputSink
from wiki import LiveWiki

ROOT = Path(__file__).resolve().parent
GRAPHS = ROOT / "data" / "precomputed" / "graphs.jsonl"


@dataclass
class Tally:
    """The four counters ``ArticleAgent._run_batch`` updates in place."""

    stats: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    kinds: Counter = field(default_factory=Counter)
    calls: Counter = field(default_factory=Counter)

    def add(self, other: "Tally") -> None:
        for mine, theirs in ((self.stats, other.stats),
                             (self.reasons, other.reasons),
                             (self.kinds, other.kinds),
                             (self.calls, other.calls)):
            mine.update(theirs)


def load_graphs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tags_of(row: dict, source: str) -> set[str]:
    tags = {source.lower()}
    for kind, values in (row.get("tags") or {}).items():
        tags.update(f"{kind[:-1]}:{v}".lower() for v in values or [] if v)
    return {t for t in tags if t}


def select(graphs: list[dict], articles: dict, wanted: list[str]):
    """(selected rows, per-tag counts) for the ``--tag`` filters."""
    counts: Counter = Counter()
    rows: list[dict] = []
    for row in graphs:
        tags = tags_of(row, articles[row["article_id"]].source)
        counts.update(tags)
        if not wanted or any(w.lower() in t for w in wanted for t in tags):
            rows.append(row)
    return rows, counts


def done_ids(out: Path) -> set[int]:
    done: set[int] = set()
    for name in ("records.jsonl", "rejected.jsonl"):
        path = out / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["sentence_id"]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
    return done


def ledger_for(article, out: Path) -> list[dict]:
    """Article ledger rebuilt from accepted records of earlier runs."""
    path = out / "records.jsonl"
    if not path.exists():
        return []
    by_id = {s.id: s for s in article.sentences}
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("article_id") == article.id and row.get("sentence_id") in by_id:
                rows.append(row)
    rows.sort(key=lambda r: article.sentence_index(r["sentence_id"]))
    ledger: list[dict] = []
    for row in rows:
        ledger = _merge_ledger(ledger, _ledger_from(row, by_id[row["sentence_id"]]))
    return ledger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", type=Path, default=GRAPHS)
    p.add_argument("--tag", action="append", default=[],
                   help="only sentences carrying this tag (repeatable)")
    p.add_argument("--agents", type=int, default=1, help="parallel agents")
    p.add_argument("--limit", type=int, default=0, help="max sentences (0 = all)")
    p.add_argument("--batch-size", type=int, default=0, help="sentences per call")
    p.add_argument("--out", default=str(ROOT / "data" / "runs" / "main"))
    p.add_argument("--model", default=None)
    p.add_argument("--api-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--rewrite", choices=["on", "off"], default=None)
    p.add_argument("--build", action="store_true", help="write dataset files")
    p.add_argument("--dry-run", action="store_true", help="show selection only")
    return p.parse_args()


def annotate(cfg: dict, corpus: Corpus, wiki: LiveWiki, sink: OutputSink,
             log: EventLog, client: OpenAIClient, spring: SpringParser,
             articles: dict, by_article: dict[int, list[dict]],
             agents: int, batch_size: int) -> Tally:
    """Run one agent per article group; articles are the parallel unit."""
    total = Tally()
    lock = threading.Lock()
    finished = 0

    def task(aid: int, rows: list[dict]) -> int:
        article = articles[aid]
        by_id = {s.id: s for s in article.sentences}
        with lock:  # records.jsonl is appended by the other agents
            ledger = ledger_for(article, sink.out)
        agent = ArticleAgent(cfg, corpus, wiki, sink, log, client, spring)
        local = Tally()
        for batch_no, start in enumerate(range(0, len(rows), batch_size)):
            chunk = [r for r in rows[start:start + batch_size]
                     if r["sentence_id"] in by_id]
            if not chunk:
                continue
            paragraphs = [
                Paragraph(id=by_id[r["sentence_id"]].paragraph_id,
                          para_index=by_id[r["sentence_id"]].para_index,
                          sentences=[by_id[r["sentence_id"]]])
                for r in chunk
            ]
            ledger = agent._run_batch(
                article, paragraphs, batch_no, ledger,
                local.stats, local.reasons, local.kinds, local.calls)["ledger"]
        with lock:
            total.add(local)
        return len(rows)

    with ThreadPoolExecutor(max_workers=agents) as pool:
        futures = {pool.submit(task, aid, rows): aid
                   for aid, rows in by_article.items()}
        for future in as_completed(futures):
            done = future.result()
            with lock:
                finished += 1
                print(f"[{finished}/{len(by_article)}] article {futures[future]}: "
                      f"{done} sentence(s) | accepted {total.stats['accepted']} "
                      f"rejected {total.stats['rejected']} "
                      f"repairs {total.stats['repairs']}", file=sys.stderr)
    return total


def main() -> None:
    args = parse_args()
    cfg = load_config()
    cfg["out"] = args.out
    if args.model:
        cfg["model"] = args.model
    if args.api_url:
        cfg["llm"]["api_url"] = args.api_url
    if args.api_key:
        cfg["llm"]["api_key"] = args.api_key
    if args.rewrite:
        cfg["rewrite"]["enabled"] = args.rewrite == "on"

    corpus = Corpus(cfg["corpus_db"])
    graphs = load_graphs(args.graphs)
    if not graphs:
        raise SystemExit(f"no graphs in {args.graphs}; run build_precomputed.py")
    articles = {aid: corpus.get_article(aid)
                for aid in {g["article_id"] for g in graphs}}

    out = out_dir(cfg)
    selected, counts = select(graphs, articles, args.tag)
    done = done_ids(out)
    remaining = [r for r in selected if r["sentence_id"] not in done]
    todo = remaining[:args.limit] if args.limit else remaining
    print(f"graphs {len(graphs)} | tag {args.tag or 'all'} -> "
          f"{len(selected)} | done {len(selected) - len(remaining)} | "
          f"this run {len(todo)}", file=sys.stderr)

    if args.dry_run:
        for tag, n in counts.most_common(40):
            print(f"  {tag:<48} {n}", file=sys.stderr)
        return
    if not todo:
        print("nothing to do", file=sys.stderr)
        if args.build:
            print(json.dumps(build(out), indent=1), file=sys.stderr)
        return
    if not cfg["llm"]["api_url"] or not cfg["model"]:
        raise SystemExit("no LLM: set llm.api_url/model in config.json or pass "
                         "--api-url/--model (or AMR_API_URL/AMR_MODEL)")

    by_article: dict[int, list[dict]] = defaultdict(list)
    for row in todo:
        by_article[row["article_id"]].append(row)
    for rows in by_article.values():
        rows.sort(key=lambda r: (r["para_index"], r["idx"]))

    tally = annotate(
        cfg, corpus, LiveWiki(cfg["wiki"]), OutputSink(out),
        EventLog(out / "events"), OpenAIClient(cfg),
        SpringParser(cache={g["sentence_id"]: g["amr"] for g in graphs}),
        articles, by_article, max(1, args.agents),
        args.batch_size or int(cfg["agent"]["batch_size"]))

    summary = {
        "graphs": len(graphs), "tags": args.tag, "selected": len(selected),
        "done_before": len(selected) - len(remaining),
        "annotated_this_run": len(todo), "agents": max(1, args.agents),
        "accepted": int(tally.stats["accepted"]),
        "rejected": int(tally.stats["rejected"]),
        "rewrites": int(tally.stats["rewrites"]),
        "repairs": int(tally.stats["repairs"]),
        "tool_rounds": int(tally.stats["tool_rounds"]),
        "tool_calls": dict(tally.calls),
        "reject_reasons": dict(tally.reasons),
        "rewrite_kinds": dict(tally.kinds),
        "out": str(out),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=1), file=sys.stderr)
    if args.build:
        print(json.dumps(build(out), indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
