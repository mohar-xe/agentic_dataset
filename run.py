#!/usr/bin/env python3
"""Run the per-article agent.

Each article is one independent task, so ``--workers`` runs them in parallel
and ``--shard i/n`` splits a corpus across machines. Resumption, repair and the
append-only sinks are unchanged by parallelism.

API/DB settings are inherited from the older ``dataset_amr`` project
(``pipeline.config.json``) when present, then from ``config.json``, then from
the environment, then from CLI flags.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# config must be imported first: it puts legacy scripts on sys.path
from config import LEGACY_SCRIPTS, load_config, out_dir  # noqa: F401

from agent import ArticleAgent, _paragraph_batches
from build import build
from corpus import Corpus, declared_names, kind_of_name
from examples import write_example
from llm import MockClient, OpenAIClient
from spring import SpringParser
from store import EventLog, OutputSink, merge_article_stats
from wiki import LiveWiki

PRINT_LOCK = threading.Lock()


def load_base_graphs(path: str | Path | None,
                     wanted: set[int] | None = None) -> dict[int, str]:
    """Warm cache for ``spring_parse``: graphs from an earlier SPRING run.

    The agent still asks the tool for every sentence; a cached graph only saves
    the model call. SPRING output can be large, so the file is streamed once
    and unrelated sentence ids are dropped instead of building a dict of the
    whole corpus.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[int, str] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid, amr = row.get("sentence_id"), row.get("amr")
            if sid is None or not amr:
                continue
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                continue
            if wanted is not None and sid not in wanted:
                continue
            out[sid] = amr
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--article", type=int, default=5772,
                   help="article id to run (default: 5772, has 'the Act')")
    p.add_argument("--all", action="store_true", help="run all eligible articles")
    p.add_argument("--limit", type=int, default=0, help="max articles")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel articles (default: config or 1)")
    p.add_argument("--shard", default="",
                   help="i/n, 1-based: process articles where id %% n == i-1")
    p.add_argument("--mock", action="store_true", help="scripted client, no API key")
    p.add_argument("--build", action="store_true", help="write dataset files after run")
    p.add_argument("--example", action="store_true", help="render walkthroughs")
    p.add_argument("--force", action="store_true", help="rerun even if done")
    p.add_argument("--rewrite", choices=["on", "off"], default=None)
    p.add_argument("--rewrite-kinds", default=None, help="comma list or 'all'")
    p.add_argument("--model", default=None)
    p.add_argument("--api-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--corpus", default=None, help="corpus sqlite path")
    p.add_argument("--base-graphs", default=None, help="SPRING graphs jsonl")
    p.add_argument("--no-inherit", action="store_true",
                   help="do not read the older pipeline.config.json")
    return p.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.rewrite == "off":
        cfg["rewrite"]["enabled"] = False
    elif args.rewrite == "on":
        cfg["rewrite"]["enabled"] = True
    if args.rewrite_kinds:
        cfg["rewrite"]["kinds"] = (
            "all" if args.rewrite_kinds == "all"
            else [k.strip() for k in args.rewrite_kinds.split(",") if k.strip()]
        )
    if args.model:
        cfg["model"] = args.model
    if args.api_url:
        cfg["llm"]["api_url"] = args.api_url
    if args.api_key:
        cfg["llm"]["api_key"] = args.api_key
    if args.corpus:
        cfg["corpus_db"] = args.corpus
    if args.base_graphs:
        cfg["base_graphs"] = args.base_graphs
    if args.workers:
        cfg["workers"] = args.workers
    return cfg


def choose_ids(corpus: Corpus, args: argparse.Namespace, cfg: dict) -> list[int]:
    if args.all:
        ids = corpus.article_ids(
            limit=0, min_sentences=int(cfg.get("min_sentences") or 5)
        )
    else:
        ids = [args.article]
    if args.shard:
        try:
            part, total = args.shard.split("/")
            part, total = int(part), int(total)
            if not (1 <= part <= total):
                raise ValueError
        except ValueError:
            raise SystemExit("--shard must look like 1/4")
        ids = [aid for aid in ids if aid % total == part - 1]
    limit = int(args.limit or 0)
    return ids[:limit] if limit else ids


def smart_mock(article, base_by_sid, batch, cfg=None):
    """Per-batch scripted client aware of declarations in the whole article.

    It plays the part of a careful model: first it splits the paragraph and
    asks ``spring_parse`` for every sentence in one turn (parallel calls), then
    it answers.
    """
    cfg = cfg or {"wiki": {"enabled": True}}
    decls = []
    for s in article.sentences:
        for name, alias in declared_names(s.text):
            decls.append((s, name, alias))

    script = []
    if batch:
        # one turn, one call per sentence -> the tool loop runs them together
        script.append({"tool_calls": [
            {"name": "spring_parse", "args": {"sentence": s.text}}
            for s in batch
        ]})
    for s in batch:
        t = s.text.lower()
        surf = None
        for needle, label in [(" the act", "the Act"), (" the board", "the Board"),
                              (" the code", "the Code"),
                              (" the commission", "the Commission")]:
            if needle in f" {t} " or t.endswith(needle.strip()):
                surf = label
                break
        if surf:
            script.append({"tool_calls": [
                {"name": "article_search",
                 "args": {"query": surf, "before": s.text, "limit": 5}},
                {"name": "resolve_mention",
                 "args": {"surface": surf, "sentence": s.text}},
            ]})
            if cfg.get("wiki", {}).get("enabled", True):
                for _, name, alias in decls:
                    if alias and alias.lower() == "act":
                        script.append({"tool_calls": [
                            {"name": "wiki_lookup",
                             "args": {"surface": name,
                                      "context": article.title}},
                        ]})
                        break
            break

    results = []
    for s in batch:
        original = s.text
        resolved = original
        edits = []
        entities = []
        for needle, alias_word in [(" the Act ", "act"),
                                   (" the Board ", "board"),
                                   (" the Code ", "code")]:
            if needle not in f" {original} ":
                continue
            target = None
            for prev in article.sentences:
                if prev.id >= s.id:
                    break
                for name, alias in declared_names(prev.text):
                    if alias and alias.lower() == alias_word:
                        target = name
                        break
                if target:
                    break
            if not target:
                for kind, name in article.metadata():
                    if name and alias_word in name.lower():
                        target = name
                        break
            if target and needle in resolved:
                resolved = resolved.replace(needle, f" {target} ", 1)
                edits.append({
                    "surface": needle.strip(), "occurrence": 1,
                    "replacement": target, "resolved_to": "x1",
                    "evidence": "article", "kind": kind_of_name(target),
                })
                entities.append({"name": target,
                                 "kind": kind_of_name(target),
                                 "alias": alias_word.title()})
            break
        amr = base_by_sid.get(s.id) or "(z / thing)"
        for e in edits:
            safe = e["replacement"].replace('"', '\\"')
            kind = e.get("kind") or "legal-instrument"
            amr = (f'(z / {kind} :legal-name "{safe}" '
                   f':wiki "{e["replacement"].replace(" ", "_")}")')
        results.append({
            "original": s.text, "status": "ok",
            "sentence": resolved, "amr": amr,
            "edits": edits, "entities": entities,
        })
    script.append({"content": {"results": results}})
    return MockClient(script)


def run_article(cfg: dict, corpus: Corpus, wiki: LiveWiki, sink: OutputSink,
                log: EventLog, article_id: int, base: dict[int, str],
                mock: bool, spring: SpringParser) -> dict:
    if not sink.claim(article_id):
        return {"article_id": article_id, "skipped": True}
    article = corpus.get_article(article_id)
    if mock:
        return run_article_mock(cfg, corpus, wiki, sink, log, article, base,
                                spring)
    agent = ArticleAgent(cfg, corpus, wiki, sink, log, OpenAIClient(cfg),
                         spring)
    return agent.run(article)


def run_article_mock(cfg, corpus, wiki, sink, log, article, base, spring):
    """Per-batch scripted clients so each batch sees real tool/ledger state."""
    from collections import Counter
    if sink.is_done(article.id):
        return {"article_id": article.id, "skipped": True}
    ckpt = sink.load_checkpoint(article.id) or {}
    start = int(ckpt.get("next_index") or 0)
    ledger = ckpt.get("ledger") or []
    stats = Counter(ckpt.get("stats") or {})
    reject_reasons = Counter(ckpt.get("reject_reasons") or {})
    rewrite_kinds = Counter(ckpt.get("rewrite_kinds") or {})
    tool_calls = Counter(ckpt.get("tool_calls") or {})
    log.log(article.id, "article_start", title=article.title,
            n_sentences=len(article.sentences), resume_from=start, mock=True)
    batch_size = int(cfg["agent"]["batch_size"])
    batches = _paragraph_batches(article, batch_size)
    agent = ArticleAgent(cfg, corpus, wiki, sink, log, None, spring)
    for batch_no, paragraphs in enumerate(batches):
        if batch_no < start:
            continue
        batch = [s for p in paragraphs for s in p.sentences]
        agent.client = smart_mock(article, base, batch, cfg)
        b = agent._run_batch(article, paragraphs, batch_no, ledger,
                             stats, reject_reasons, rewrite_kinds, tool_calls)
        ledger = b["ledger"]
        sink.save_checkpoint(article.id, {
            "next_index": batch_no + 1,
            "ledger": ledger, "stats": dict(stats),
            "reject_reasons": dict(reject_reasons),
            "rewrite_kinds": dict(rewrite_kinds),
            "tool_calls": dict(tool_calls),
        })
    summary = {
        "article_id": article.id,
        "sentences_total": len(article.sentences),
        "sentences_accepted": int(stats.get("accepted", 0)),
        "sentences_rejected": int(stats.get("rejected", 0)),
        "rewrites": int(stats.get("rewrites", 0)),
        "tool_rounds": int(stats.get("tool_rounds", 0)),
        "repairs": int(stats.get("repairs", 0)),
        "tool_calls": dict(tool_calls),
        "rewrite_kinds": dict(rewrite_kinds),
        "reject_reasons": dict(reject_reasons),
        "ledger_size": len(ledger),
    }
    log.log(article.id, "article_done",
            **{k: v for k, v in summary.items() if k != "article_id"})
    sink.mark_done(article.id, summary)
    sink.clear_checkpoint(article.id)
    return summary


def main() -> None:
    args = parse_args()
    overrides = {"inherit_old_config": False} if args.no_inherit else None
    cfg = apply_cli_overrides(load_config(args.config, overrides), args)
    out = out_dir(cfg)
    corpus = Corpus(cfg["corpus_db"])
    wiki = LiveWiki(cfg["wiki"])
    sink = OutputSink(out)
    log = EventLog(out / "events")
    ids = choose_ids(corpus, args, cfg)

    if not ids:
        raise SystemExit("no articles selected")
    if not args.mock:
        if not cfg["llm"]["api_url"] or not cfg["model"]:
            raise SystemExit(
                "real run needs an LLM: set --api-url/--model or inherit "
                "dataset_amr/pipeline.config.json (or use --mock)"
            )
    wanted: set[int] = set()
    for aid in ids:
        wanted.update(s.id for s in corpus.get_article(aid).sentences)
    # Optional warm cache: the agent always asks spring_parse, but a graph from
    # an earlier SPRING run saves the model call.
    base = load_base_graphs(cfg["base_graphs"], wanted)
    spring = SpringParser(cache=base, allow_placeholder=args.mock)
    if not args.mock:
        try:
            spring.load()
        except RuntimeError as exc:
            raise SystemExit(
                f"spring_parse needs the local SPRING parser: {exc}")
    if base:
        print(f"spring_parse warm cache: {len(base)} graph(s) for the selected "
              f"sentences", file=sys.stderr)

    workers = max(1, int(cfg.get("workers") or 1))
    summaries: list[dict] = []
    summary_lock = threading.Lock()

    def task(aid: int) -> dict:
        try:
            result = run_article(cfg, corpus, wiki, sink, log, aid, base,
                                 args.mock, spring)
            if args.example:
                try:
                    path = write_example(out, aid)
                    print(f"example: {path}", file=sys.stderr)
                except Exception:
                    pass
            return result
        except Exception as exc:  # one bad article must not kill the run
            log.log(aid, "error", error=f"{type(exc).__name__}: {exc}")
            return {"article_id": aid, "error": f"{type(exc).__name__}: {exc}"}

    def record(summary: dict) -> None:
        with summary_lock:
            summaries.append(summary)
            real = [s for s in summaries if not s.get("skipped")]
            sink.write_stats(merge_article_stats(real))
            if summary.get("error"):
                line = f"article {summary['article_id']}: ERROR {summary['error']}"
            elif summary.get("skipped"):
                line = f"article {summary['article_id']}: skipped"
            else:
                line = (f"article {summary['article_id']}: "
                        f"{summary.get('sentences_accepted', 0)} accepted, "
                        f"{summary.get('sentences_rejected', 0)} rejected, "
                        f"{summary.get('rewrites', 0)} rewrites, "
                        f"{summary.get('repairs', 0)} repairs")
            with PRINT_LOCK:
                print(line, file=sys.stderr)

    if workers == 1 or len(ids) == 1:
        for aid in ids:
            record(task(aid))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(task, aid): aid for aid in ids}
            for future in as_completed(futures):
                record(future.result())

    if args.build or args.mock:
        print(json.dumps(build(out), indent=2))
    real = [s for s in summaries if not s.get("skipped")]
    if real:
        print(json.dumps(merge_article_stats(real), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
