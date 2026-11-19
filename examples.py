"""Render an article's event log as a step-by-step Markdown walkthrough.

This is the "what each step actually does" artefact: it reads the saved
events and records and explains the tool calls, rewrites and rejections in
order. No LLM, no recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_article(out: Path, article_id: int) -> str:
    events_path = out / "events" / f"article_{article_id}.jsonl"
    events = []
    if events_path.exists():
        events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    records = _index(out / "records.jsonl", article_id)
    rejected = _index(out / "rejected.jsonl", article_id)
    lines = [f"# Article {article_id} - agent walkthrough", ""]
    for ev in events:
        kind = ev.get("kind")
        if kind == "article_start":
            lines += [f"**Start**: {ev.get('title','')} ({ev.get('n_sentences')} sentences)", ""]
        elif kind == "tool_call":
            r = ev.get("result") or {}
            summary = _tool_summary(ev.get("tool"), ev.get("args") or {}, r)
            lines.append(f"- tool `{ev.get('tool')}`: {summary}")
        elif kind == "sentence_accepted":
            sid = ev.get("sentence_id")
            rec = records.get(sid) or {}
            lines.append(f"- **accepted** sentence {sid} (edits={ev.get('n_edits',0)})")
            for edit in rec.get("edits") or []:
                lines.append(
                    f"  - rewrite `{edit.get('surface')}` -> `{edit.get('replacement')}` "
                    f"(evidence={edit.get('evidence')}, occurrence={edit.get('occurrence')})"
                )
        elif kind == "sentence_rejected":
            sid = ev.get("sentence_id")
            rec = rejected.get(sid) or {}
            lines.append(
                f"- **rejected** sentence {sid}: {ev.get('reason')} "
                f"(candidates={len(ev.get('candidates') or [])})"
            )
        elif kind == "article_done":
            lines += ["", f"**Done**: accepted={ev.get('sentences_accepted')} "
                      f"rejected={ev.get('sentences_rejected')} "
                      f"rewrites={ev.get('rewrites')} ledger={ev.get('ledger_size')}", ""]
    return "\n".join(lines).rstrip() + "\n"


def _index(path: Path, article_id: int) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("article_id", -1)) == article_id:
            out[int(row["sentence_id"])] = row
    return out


def _tool_summary(name: str, args: dict, result: dict) -> str:
    if name == "spring_parse":
        if not result.get("amr"):
            return f"no graph ({result.get('status')})"
        nodes = result.get("n_nodes")
        edges = result.get("n_edges")
        size = f"{nodes}n/{edges}e " if nodes is not None else ""
        return f"SPRING {size}({result.get('status')})"
    if name == "article_search":
        return f"query={args.get('query')!r} -> {result.get('count',0)} hits"
    if name == "resolve_mention":
        cands = result.get("candidates") or []
        top = cands[0]["name"] if cands else None
        return (f"surface={args.get('surface')!r} -> {len(cands)} candidates"
                + (f", top={top!r}" if top else "")
                + (", ambiguous" if result.get("ambiguous") else ""))
    if name == "wiki_lookup":
        cands = result.get("candidates") or []
        top = cands[0]["title"] if cands else None
        return f"surface={args.get('surface')!r} -> {len(cands)} candidates" + (f", top={top!r}" if top else "")
    if name == "check_amr":
        return f"ok={result.get('ok')} errors={result.get('n_errors',0)}"
    return str(args)[:80]


def write_example(out: Path, article_id: int) -> Path:
    md = render_article(out, article_id)
    path = out / "examples" / f"article_{article_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return path
