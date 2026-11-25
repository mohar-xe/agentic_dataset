#!/usr/bin/env python3
"""Run five real sentences through the real agent with hand-authored answers.

There is no LLM call here. The answers below were written the way the agent is
asked to answer (resolved sentence + explicit edits + a Legal-AMR graph +
evidence). Everything after that is the real pipeline: the same tools, the same
corpus, the same validator, the same accept/reject gates, the same sinks.

The six sentences are run one per batch so the article memory is visible: the
ledger is saved before and after every sentence. SPRING base graphs are the
real local model output from ``docs/spring_five.py``.

    python docs/five_sentences.py

Writes ``data/five/`` (records, rejections, events, report) and
``data/five_repair/`` (one deliberately broken first answer, repaired by the
real agent).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402

from agent import ArticleAgent  # noqa: E402
from check import check_graph  # noqa: E402
from common import STOPWORDS, TOKEN_RE, stem  # noqa: E402
from corpus import Corpus, Paragraph, _norm  # noqa: E402
from llm import MockClient  # noqa: E402
from rewrite import apply_edits  # noqa: E402
from spring import SpringParser  # noqa: E402
from store import EventLog, OutputSink  # noqa: E402
from wiki import LiveWiki  # noqa: E402

ARTICLE = 5772
SENTENCES = [360428, 360429, 360439, 360450, 360465, 360477]

ACT = "Arbitration and Conciliation Act, 1996"
WIKI = "Arbitration_and_Conciliation_Act,_1996"

GRAPHS = {
    360428: f"""(e / enable-01
   :ARG0 (s / statute
            :wiki "{WIKI}"
            :legal-name "{ACT}"
            :legal-year 1996
            :mod (i2 / india))
   :ARG1 (p / provision
            :legal-level "part"
            :legal-name "Part one"
            :part-of s
            :quant (sev / several))
   :ARG2 (c / gov-entity
            :legal-type "court"
            :mod (i / indian))
   :purpose (x / interfere-01
               :ARG0 c)
   :topic (a2 / arbitration))""",
    360429: f"""(a / applicable
   :polarity -
   :ARG1 (p / provision
            :legal-level "sub-section"
            :legal-name "section 2(2)"
            :part-of (s / statute
                              :wiki "{WIKI}"
                              :legal-name "{ACT}"))
   :location (i / india)
   :condition (pl / place
                 :mod (a2 / arbitration)))""",
    360450: f"""(c / challenge-01
   :ARG1 (a / award)
   :legal-cites (p / provision
                  :legal-level "section"
                  :legal-name "section 34"
                  :part-of (s / statute
                                    :wiki "{WIKI}"
                                    :legal-name "{ACT}"))
   :location (h / gov-entity
                :legal-type "court"
                :legal-name "Delhi High Court")
   :time (d / date-entity
            :year 2013
            :month 7))""",
    360465: """(n / note-01
   :ARG0 (c / gov-entity
            :legal-type "court"
            :legal-name "Supreme Court")
   :ARG1 (c2 / controversy
            :mod (s / say-01))
   :ARG2 (g / govern-01
            :ARG1 c2
            :ARG2 (p / principle
                     :mod (l / lay-01
                             :ARG1 (b / company
                                      :name (n2 / name
                                               :op1 "Bharat"
                                               :op2 "Aluminium"
                                               :op3 "Company")))))
   :location (j / judgment
                :quant (m / many)))""",
    360477: f"""(n / note-01
   :ARG0 (l / legislator)
   :ARG1 (p / provide-01
            :polarity -
            :ARG0 (a / statute
                     :wiki "{WIKI}"
                     :legal-name "{ACT}")
            :ARG1 (r / reference
                     :mod (c / concept
                             :mod (s / seat-or-venue
                                     :mod (a2 / arbitration))))
            :ARG2 (p2 / provide-01
                     :ARG0 a
                     :ARG1 (p3 / place
                              :mod (a3 / arbitration)))))""",
}

EDITS = {
    360429: [{"surface": "Act", "occurrence": 1, "replacement": ACT,
              "resolved_to": "a1", "evidence": "article", "kind": "statute"}],
    360450: [{"surface": "Act", "occurrence": 1, "replacement": ACT,
              "resolved_to": "a1", "evidence": "article", "kind": "statute"}],
    360465: [{"surface": "the Court", "occurrence": 1,
              "replacement": "the Supreme Court",
              "resolved_to": "c1", "evidence": "article", "kind": "gov-entity"}],
    360477: [{"surface": "Act", "occurrence": 1, "replacement": ACT,
              "resolved_to": "a1", "evidence": "article", "kind": "statute"}],
}

ENTITIES = {
    360428: [{"name": ACT, "kind": "statute", "alias": "Act"}],
    360429: [{"name": ACT, "kind": "statute", "alias": "Act"},
             {"name": "section 2(2)", "kind": "provision"}],
    360450: [{"name": ACT, "kind": "statute", "alias": "Act"},
             {"name": "section 34", "kind": "provision"},
             {"name": "Delhi High Court", "kind": "gov-entity"}],
    360465: [{"name": "Supreme Court", "kind": "gov-entity"}],
    360477: [{"name": ACT, "kind": "statute", "alias": "Act"}],
}

def tool_calls_for(article, sid: int) -> list[dict]:
    """The demo's tool turn: parse the sentence, then ask about its mentions.

    The agent has no ids: ``before`` and ``sentence`` are sentence text, which
    is what the real tools take now.
    """
    by_id = {s.id: s for s in article.sentences}
    text = by_id[sid].text
    calls = [{"name": "spring_parse", "args": {"sentence": text}}]
    for call in TOOL_CALLS.get(sid) or []:
        args = dict(call["args"])
        if "before_sentence_id" in args:
            args["before"] = by_id[args.pop("before_sentence_id")].text
        if "sentence_id" in args:
            args["sentence"] = by_id[args.pop("sentence_id")].text
        calls.append({"name": call["name"], "args": args})
    return calls


TOOL_CALLS = {
    360428: [{"name": "wiki_lookup",
              "args": {"surface": ACT, "context": "Indian arbitration"}}],
    360429: [
        {"name": "article_search",
         "args": {"query": "the Act", "before_sentence_id": 360429}},
        {"name": "resolve_mention",
         "args": {"surface": "the Act", "sentence_id": 360429}},
        {"name": "resolve_mention",
         "args": {"surface": "section 2(2)", "sentence_id": 360429}},
    ],
    360439: [{"name": "resolve_mention",
              "args": {"surface": "HEPL", "sentence_id": 360439}}],
    360450: [
        {"name": "resolve_mention",
         "args": {"surface": "the Act", "sentence_id": 360450}},
        {"name": "resolve_mention",
         "args": {"surface": "section 34", "sentence_id": 360450}},
    ],
    360465: [{"name": "resolve_mention",
              "args": {"surface": "the Court", "sentence_id": 360465}}],
    360477: [{"name": "resolve_mention",
              "args": {"surface": "the Act", "sentence_id": 360477}}],
}

ROLES = {
    360428: "a definition becomes memory",
    360429: "two mentions resolved in one sentence",
    360439: "rejected, because the name is cut off",
    360450: "a provision and a statute together",
    360465: "the same sentence settles the mention",
    360477: "a statute rewrite the old run could not do",
}

# What changed between the real SPRING graph and the agent's graph, in plain
# words. Every claim here is checked against data/five_spring/spring.jsonl and
# the accepted graph in data/five/records.jsonl.
GRAPH_CHANGES = {
    360428: [
        "SPRING found a `law` node named 'Arbitration and Conciliation Act' "
        "(no year) inside `legislate-01`.",
        "The agent made a `statute` node with :legal-name the full name, "
        ":legal-year 1996 and :wiki title, plus a `provision` node for "
        "'Part one' linked with :part-of.",
    ],
    360429: [
        "SPRING output is itself invalid: `:mode imperative` is unquoted, and "
        "the core validator says variables used but never defined: "
        "['imperative'].",
        "SPRING has `section :mod 2` and `law :name Act`. The agent made a "
        "`provision` node with :legal-name 'section 2(2)' and linked it to "
        "the statute with :part-of.",
    ],
    360439: [
        "SPRING also reads the company name as only 'Ltd.' - the rest of the "
        "name is missing from the sentence itself.",
        "The agent writes no graph: HEPL cannot be tied to a credible name.",
    ],
    360450: [
        "SPRING has `section :mod 34` and `law :name Act`, and a court name "
        "'Delhi High Court'.",
        "The agent made the section a `provision` with :legal-name "
        "'section 34', linked it to the full statute, and tagged the court "
        "as a `gov-entity` with :legal-type 'court'.",
    ],
    360465: [
        "SPRING already found the name 'Supreme Court'.",
        "The agent made it a `gov-entity` with :legal-type 'court' and "
        ":legal-name 'Supreme Court', and kept the judgment name.",
    ],
    360477: [
        "SPRING has a `law` node named only 'Act'.",
        "The agent made it a `statute` with the full :legal-name and :wiki "
        "title, and kept the negation (:polarity -).",
    ],
}

FORMULAS = {
    "resolve_mention": [
        "candidates are shown with evidence (no numeric score); the agent picks",
        "memory, exact name/alias: direct match already resolved in this article",
        "same sentence, name before mention: strongest text evidence in place",
        "earlier sentence (alias match/name match): earlier declaration, with distance",
        "article metadata: name listed for the whole article",
        "memory, same kind only: weak hint, not proof",
        "no clear single answer (none, or several equally plausible) -> reject, don't guess",
    ],
    "wiki_lookup": [
        "every number/year in the surface must appear in the candidate, or it is rejected",
        "equal names after normalization -> 1.000",
        "otherwise 0.6*word overlap + 0.35 same year",
        "  + 0.1 punctuation-insensitive match + 0.03 'India' in description",
        "capped at 0.99",
    ],
    "coverage": [
        "content words = words with 3+ letters or digits, minus stopwords",
        "a word is kept when it, or its light stem, appears in the graph",
        "coverage = kept content words / all content words",
        "this is a rough check, not a grammar score",
    ],
}


def answers(article) -> dict[int, dict]:
    by_id = {s.id: s for s in article.sentences}
    cfg = load_config()
    out: dict[int, dict] = {}
    for sid in SENTENCES:
        if sid == 360439:
            out[sid] = {
                "original": by_id[sid].text, "status": "rejected",
                "reason": "no_credible_source", "candidates": [],
            }
            continue
        edits = EDITS.get(sid) or []
        resolved = by_id[sid].text
        if edits:
            resolved, _, _ = apply_edits(resolved, edits, cfg)
        out[sid] = {
            "original": by_id[sid].text,
            "status": "ok",
            "sentence": resolved,
            "amr": GRAPHS[sid],
            "edits": edits,
            "entities": ENTITIES.get(sid) or [],
        }
    return out


def coverage_math(sentence: str, amr: str) -> dict:
    words = {w for w in TOKEN_RE.findall(sentence.lower())
             if w not in STOPWORDS}
    got = set(TOKEN_RE.findall(amr.lower()))
    got_stems = {stem(w) for w in got}
    kept = sorted(w for w in words if w in got or stem(w) in got_stems)
    missing = sorted(words - set(kept))
    return {
        "kept": len(kept),
        "total": len(words),
        "value": round(len(kept) / len(words), 3) if words else 1.0,
        "kept_words": kept,
        "missing": missing,
    }


def memory_lines(ledger: list[dict]) -> list[str]:
    if not ledger:
        return ["(empty)"]
    lines = []
    for entry in ledger:
        alias = entry.get("alias") or ", ".join(entry.get("aliases") or [])
        alias_part = f' alias "{alias}"' if alias else ""
        wiki = f' wiki {entry["wiki"]}' if entry.get("wiki") else ""
        lines.append(
            f'{entry["name"]} [{entry.get("kind") or "?"}]'
            f'{alias_part} from {entry.get("declared_at")}{wiki}')
    return lines


def _evidence_reason(cand: dict) -> str:
    match = cand.get("match") or ""
    source = cand.get("source") or ""
    alias = cand.get("alias")
    alias_part = f' (alias "{alias}")' if alias else ""
    distance = cand.get("distance")
    if source == "article_memory":
        return match + alias_part
    if source == "article_text":
        ev = cand.get("evidence_sentence_id")
        where = f"sentence {ev}" if ev is not None else "article text"
        dist = f", {distance} sentence(s) earlier" if distance else ""
        return f"{match}{alias_part} — {where}{dist}"
    if source == "article_metadata":
        return match
    return match or source or "evidence in this article"


def _short(text: object, limit: int = 34) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def tool_lines(event: dict, sid: int) -> list[str]:
    """Format one real tool result as evidence for the drawing."""
    tool, args, result = event["tool"], event.get("args") or {}, event["result"]
    if tool == "resolve_mention":
        surface = result.get("surface") or args.get("surface")
        lines = [f'resolve_mention("{surface}", "{_short(args.get("sentence"))}")']
        candidates = result.get("candidates") or []
        if not candidates:
            return lines + ["  -> candidates: []  (nothing found in this article)"]
        lines.append(f'  -> {len(candidates)} candidate(s), shown with evidence:')
        for i, cand in enumerate(candidates[:4], 1):
            lines.append(f'     {i}. {cand["name"]} [{cand.get("kind") or "?"}]')
            lines.append(f'        {_evidence_reason(cand)}')
            if cand.get("evidence_sentence_id"):
                lines.append(f'        evidence: "{_short(cand.get("evidence_text"), 60)}"')
        return lines
    if tool == "article_search":
        query = args.get("query")
        before = args.get("before")
        lines = [f'article_search("{query}", before "{_short(before)}")']
        hits = result.get("hits") or []
        if not hits:
            return lines + [
                "  -> 0 sentences (the short form in text may use different quotes/punctuation)"]
        lines += [f'  -> {len(hits)} sentence(s)']
        lines += [f'     {h["sentence_id"]}: {h["text"]}' for h in hits]
        return lines
    if tool == "wiki_lookup":
        surface = args.get("surface")
        lines = [f'wiki_lookup("{surface}")']
        candidates = result.get("candidates") or []
        if not candidates:
            return lines + ["  -> no candidates"]
        top = candidates[0]
        lines += [
            f'  -> {top["title"]}',
            f'     {top.get("qid") or "no qid"} · "{top.get("description")}"',
        ]
        if _norm(surface) == _norm(top["title"]):
            lines.append("     names equal after normalization; numbers/years match")
        else:
            lines.append(f'     score {top["score"]} (numbers/years must match)')
        return lines
    return [f"{tool}: {json.dumps(result, ensure_ascii=False)}"]


def run_main(cfg, corpus, wiki, article, base) -> dict:
    out = ROOT / "data" / "five"
    if out.exists():
        shutil.rmtree(out)
    sink = OutputSink(out)
    log = EventLog(out / "events")
    # The demo isolates one sentence at a time, so each "paragraph" it hands
    # over holds exactly that sentence. The graphs come from the same warm
    # cache the tool would read, so no model has to load.
    agent = ArticleAgent(cfg, corpus, wiki, sink, log, None,
                         SpringParser(cache=base))
    by_id = {s.id: s for s in article.sentences}
    answer_by_id = answers(article)

    stats: Counter = Counter()
    reasons: Counter = Counter()
    kinds: Counter = Counter()
    calls: Counter = Counter()
    ledger: list[dict] = []
    memory_before: dict[str, list[str]] = {}
    memory_after: dict[str, list[str]] = {}
    repairs: dict[str, int] = {}

    for batch_no, sid in enumerate(SENTENCES):
        memory_before[str(sid)] = memory_lines(ledger)
        steps = [
            {"tool_calls": tool_calls_for(article, sid)},
            {"content": {"results": [answer_by_id[sid]]}},
        ]
        agent.client = MockClient(steps)
        repairs_before = stats.get("repairs", 0)
        one = Paragraph(id=by_id[sid].paragraph_id,
                        para_index=by_id[sid].para_index,
                        sentences=[by_id[sid]])
        result = agent._run_batch(
            article, [one], batch_no, ledger,
            stats, reasons, kinds, calls)
        ledger = result["ledger"]
        repairs[str(sid)] = stats.get("repairs", 0) - repairs_before
        memory_after[str(sid)] = memory_lines(ledger)

    records = [json.loads(line) for line in
               (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    rejected = [json.loads(line) for line in
                (out / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()] if (out / "rejected.jsonl").exists() else []
    events = [json.loads(line) for line in
              (out / "events" / f"article_{ARTICLE}.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()]

    sid_by_batch = {i: sid for i, sid in enumerate(SENTENCES)}
    tools: dict[str, list[str]] = {str(sid): [] for sid in SENTENCES}
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        if event.get("tool") == "spring_parse":
            continue  # the graph is reported on its own, not as a lookup
        sid = sid_by_batch.get(event.get("batch"))
        if sid is not None:
            tools[str(sid)] += tool_lines(event, sid)

    checks: dict[str, dict] = {}
    for record in records:
        report = check_graph(
            record["sentence_original"], record["sentence"],
            record["amr"], record["edits"], cfg)
        checks[str(record["sentence_id"])] = {
            "ok": report["ok"],
            "errors": report["errors"],
            "coverage": report["coverage"],
            "coverage_math": coverage_math(
                record["sentence"], record["amr"]),
        }
    for record in rejected:
        checks[str(record["sentence_id"])] = {
            "ok": None, "errors": [], "coverage": None,
            "coverage_math": None,
        }

    spring = {}
    for line in (ROOT / "data" / "five_spring" / "spring.jsonl") \
            .read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spring[str(row["sentence_id"])] = {
            "amr": row["amr"], "status": row["status"],
            "errors": row["errors"], "n_nodes": row["n_nodes"],
            "n_edges": row["n_edges"], "model": row["model"],
        }

    return {
        "article": ARTICLE,
        "article_title": article.title,
        "sentences": SENTENCES,
        "accepted": len(records),
        "rejected": len(rejected),
        "reasons": dict(reasons),
        "stats": dict(stats),
        "tool_calls": dict(calls),
        "roles": {str(k): v for k, v in ROLES.items()},
        "formulas": FORMULAS,
        "graph_changes": {str(k): v for k, v in GRAPH_CHANGES.items()},
        "memory_before": memory_before,
        "memory_after": memory_after,
        "repairs": repairs,
        "tools": tools,
        "spring": spring,
        "checks": checks,
        "records": [
            {"sentence_id": r["sentence_id"],
             "sentence_original": r["sentence_original"],
             "sentence": r["sentence"],
             "edits": r["edits"],
             "amr": r["amr"],
             "base_amr": r["base_amr"],
             "coverage": r["coverage"],
             "legal_concepts": r["legal_concepts"],
             "legal_relations": r["legal_relations"],
             "legal_names": re.findall(r':legal-name "([^"]+)"', r["amr"]),
             "wiki_titles": re.findall(r':wiki "([^"]+)"', r["amr"])}
            for r in records
        ],
        "rejected_records": [
            {"sentence_id": r["sentence_id"], "reason": r["reason"],
             "original": r.get("original"), "candidates": r.get("candidates")}
            for r in rejected
        ],
    }


def run_repair_demo(cfg, corpus, wiki, article, base,
                    ledger_before: list[dict]) -> dict:
    """One real repair: a deliberately broken first answer, fixed by the agent."""
    out = ROOT / "data" / "five_repair"
    if out.exists():
        shutil.rmtree(out)
    sid = 360450
    by_id = {s.id: s for s in article.sentences}
    answer = answers(article)[sid]
    broken = answer["amr"].replace("(h / gov-entity", "(h / court")
    assert broken != answer["amr"], "repair demo did not break the graph"

    sink = OutputSink(out)
    log = EventLog(out / "events")
    agent = ArticleAgent(cfg, corpus, wiki, sink, log, None,
                         SpringParser(cache=base))
    agent.client = MockClient([
        {"tool_calls": tool_calls_for(article, sid)},
        {"content": {"results": [{**answer, "amr": broken}]}},
        {"content": {"results": [answer]}},
    ])
    stats: Counter = Counter()
    one = Paragraph(id=by_id[sid].paragraph_id,
                    para_index=by_id[sid].para_index,
                    sentences=[by_id[sid]])
    agent._run_batch(
        article, [one], 0, ledger_before,
        stats, Counter(), Counter(), Counter())

    events = [json.loads(line) for line in
              (out / "events" / f"article_{ARTICLE}.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    repair_start = next(e for e in events if e["kind"] == "repair_start")
    repair_result = next(e for e in events if e["kind"] == "repair_result")
    record = json.loads((out / "records.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    report = {
        "sentence_id": sid,
        "broken_amr": broken,
        "first_errors": repair_start["errors"][0],
        "fixed": repair_result["fixed"],
        "remaining": repair_result["remaining"],
        "attempt": repair_start["attempt"],
        "final_amr": record["amr"],
        "sentence": record["sentence"],
        "accepted": True,
        "stats": dict(stats),
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    cfg = load_config()
    corpus = Corpus(cfg["corpus_db"])
    article = corpus.get_article(ARTICLE)
    wiki = LiveWiki(cfg["wiki"])
    base = {}
    for line in (ROOT / "data" / "five_spring" / "spring.jsonl") \
            .read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            base[int(row["sentence_id"])] = row["amr"]

    report = run_main(cfg, corpus, wiki, article, base)
    (ROOT / "data" / "five" / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Run the repair demo with the memory the real run had after 360429.
    ledger_after_360429 = [
        {"name": ACT, "kind": "statute", "alias": "Act", "aliases": ["Act"],
         "wiki": None, "declared_at": 360428},
        {"name": "section 2(2)", "kind": "provision", "alias": None,
         "aliases": [], "wiki": None, "declared_at": 360429},
    ]
    repair = run_repair_demo(cfg, corpus, wiki, article, base,
                             ledger_after_360429)

    print(json.dumps({
        "main": {k: report[k] for k in
                 ("accepted", "rejected", "stats", "tool_calls")},
        "repair": {k: repair[k] for k in
                   ("sentence_id", "first_errors", "fixed", "remaining")},
        "spring": {sid: {"status": s["status"], "n_nodes": s["n_nodes"]}
                   for sid, s in report["spring"].items()},
    }, indent=1))


if __name__ == "__main__":
    main()
