#!/usr/bin/env python3
"""Precompute the SPRING base graphs once, then run many agents without it.

``run.py`` asks the local SPRING parser for one graph per sentence. SPRING is
a 1.6 GB CPU model and is serialized by a lock, so a 100-thread agent run that
parses every sentence live is limited by that one model. If the graphs are
already in a warm cache, ``spring_parse`` is a dictionary hit and the agents
are limited by the LLM instead.

This script builds that cache:

    # all selected articles, resumable, one model load
    python precompute_spring.py --all --batch-size 32

    # a shard on each machine/key, then merge the jsonl files
    python precompute_spring.py --all --shard 1/4 --out data/spring/part1.jsonl

    # small smoke test
    python precompute_spring.py --article 5772 --max-sentences 8

Output format is the one ``run.py --base-graphs`` reads: one JSON object per
line with ``sentence_id`` and ``amr``. Existing sentences are skipped, so the
script can be stopped and restarted.

Batching is the main speedup. On a small CPU box, one sentence at a time with
beam 5 can take ~9 s; a batch of 32 is several times faster per sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import validate_amr
from config import load_config
from corpus import Corpus

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "amr_spring.jsonl"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "amrlib" / "model_parse_spring-v0_1_0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--corpus", default=None, help="corpus sqlite path")
    p.add_argument("--article", type=int, default=None,
                   help="one article id (default: all eligible articles)")
    p.add_argument("--all", action="store_true",
                   help="all eligible articles (the default)")
    p.add_argument("--limit", type=int, default=0,
                   help="max articles after sharding")
    p.add_argument("--shard", default="", help="i/n, 1-based")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beam", type=int, default=5,
                   help="SPRING beam size; 1 is faster, 5 is the usual quality")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (default: leave it to torch)")
    p.add_argument("--max-sentences", type=int, default=0,
                   help="stop after this many new sentences (smoke test)")
    p.add_argument("--force", action="store_true",
                   help="reparse sentences already in --out")
    return p.parse_args()


def choose_ids(corpus: Corpus, args: argparse.Namespace,
               cfg: dict) -> list[int]:
    if args.article is not None:
        ids = [args.article]
    else:
        ids = corpus.article_ids(
            limit=0, min_sentences=int(cfg.get("min_sentences") or 5))
    if args.shard:
        try:
            part, total = (int(x) for x in args.shard.split("/"))
            if not (1 <= part <= total):
                raise ValueError
        except ValueError:
            raise SystemExit("--shard must look like 1/4")
        ids = [aid for aid in ids if aid % total == part - 1]
    if args.limit:
        ids = ids[:args.limit]
    return ids


def load_done(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = int(row["sentence_id"])
                if row.get("amr"):
                    done.add(sid)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return done


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.corpus:
        cfg["corpus_db"] = args.corpus
    corpus = Corpus(cfg["corpus_db"])

    ids = choose_ids(corpus, args, cfg)
    if not ids:
        raise SystemExit("no articles selected")

    todo: list[tuple[int, int, str]] = []
    done = set() if args.force else load_done(args.out)
    for aid in ids:
        article = corpus.get_article(aid)
        for sentence in article.sentences:
            if sentence.id in done:
                continue
            text = " ".join(sentence.text.split())
            if text:
                todo.append((aid, sentence.id, text))
    if args.max_sentences:
        todo = todo[:args.max_sentences]
    if not todo:
        print(f"nothing to do: {len(done)} sentence(s) already cached",
              file=sys.stderr)
        return

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    import amrlib

    print(f"loading SPRING from {args.model_dir} on {args.device} "
          f"(batch {args.batch_size}, beam {args.beam}) ...", flush=True)
    started = time.time()
    stog = amrlib.load_stog_model(
        model_dir=str(args.model_dir), device=args.device,
        batch_size=args.batch_size, beam_size=args.beam)
    print(f"loaded in {time.time() - started:.1f}s; "
          f"{len(todo)} sentence(s) to parse", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.out.exists() and not args.force else "w"
    written = 0
    t0 = time.time()
    with args.out.open(mode, encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            texts = [text for _, _, text in chunk]
            batch_started = time.time()
            try:
                graphs = stog.parse_sents(texts, add_metadata=False)
            except Exception as exc:
                print(f"batch {start // args.batch_size + 1} failed: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                graphs = [None] * len(chunk)
            seconds = (time.time() - batch_started) / max(1, len(chunk))
            for (aid, sid, text), graph in zip(chunk, graphs):
                amr = str(graph) if graph is not None else None
                ok, errors, info = validate_amr(amr) if amr else (False, [], {})
                row = {
                    "sentence_id": sid,
                    "article_id": aid,
                    "sentence": text,
                    "amr": amr,
                    "status": "ok" if ok else ("error" if not amr else "invalid"),
                    "errors": errors,
                    "parser": "spring",
                    "model": "spring-v0_1_0",
                    "n_nodes": info.get("n_nodes"),
                    "n_edges": info.get("n_edges"),
                    "seconds": round(seconds, 3),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            done_now = start + len(chunk)
            elapsed = time.time() - t0
            rate = done_now / elapsed if elapsed else 0.0
            left = len(todo) - done_now
            eta = left / rate if rate else 0.0
            print(f"{done_now}/{len(todo)}  {rate:.2f} sent/s  "
                  f"eta {eta / 60:.1f} min", flush=True)

    print(f"wrote {written} new graph(s) to {args.out} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
