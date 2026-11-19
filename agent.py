"""Per-article agent: paragraphs in, sentences split by the agent itself.

One article is one independent task. Paragraphs are processed in order so the
agent can see declarations before later generic mentions. The agent splits each
paragraph itself and asks the local SPRING tool for a base graph per sentence
(several calls per turn, executed in parallel); nothing is pre-split for it.
The only persistent memory inside the article is the ledger (resolved entities
and aliases), injected into every batch and rebuilt from accepted results.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from check import check_graph
from common import extract_json_payload
from corpus import (Article, Corpus, Paragraph, Sentence, declared_names,
                    kind_of_name)
from llm import Client
from prompt import batch_message, system_prompt
from rewrite import apply_edits
from spring import SpringParser
from store import EventLog, OutputSink
from tools import ToolBox, _key
from wiki import LiveWiki


def _paragraph_batches(article: Article, max_sentences: int
                       ) -> list[list[Paragraph]]:
    """Whole paragraphs, about ``max_sentences`` sentences per batch."""
    batches: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    count = 0
    for para in article.paragraphs():
        current.append(para)
        count += len(para.sentences)
        if count >= max_sentences:
            batches.append(current)
            current, count = [], 0
    if current:
        batches.append(current)
    return batches


def _parse_results(content: str | None, batch: list[Sentence]) -> list[dict]:
    """Attach the model's answers to corpus sentences by their text.

    The agent never sees corpus ids: it splits the paragraph itself and names
    each answer with the sentence text it passed to ``spring_parse``. Matching
    that text back to the corpus sentence (punctuation and case ignored) is how
    ``sentence_id`` gets attached, and it is deliberately strict — a sentence
    the agent paraphrased or merged lands nowhere and is reported as missing.
    """
    data = extract_json_payload(content or "")
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        by_text: dict[str, dict] = {}
        for item in data["results"]:
            if not isinstance(item, dict):
                continue
            key = _key(str(item.get("original") or ""))
            if key and key not in by_text:
                by_text[key] = item
        return [by_text.get(_key(s.text), {}) for s in batch]
    return [{} for _ in batch]


def _ledger_from(record: dict, sentence: Sentence) -> list[dict]:
    """Entities declared/resolved in an accepted record."""
    entries = []
    for ent in record.get("entities") or []:
        if not isinstance(ent, dict) or not ent.get("name"):
            continue
        entries.append({
            "name": ent["name"],
            "kind": ent.get("kind") or kind_of_name(ent["name"]),
            "alias": ent.get("alias"),
            "aliases": [a for a in [ent.get("alias")] if a],
            "wiki": ent.get("wiki"),
            "declared_at": sentence.id,
        })
    # Also learn from the original sentence text (explicit declarations).
    for name, alias in declared_names(sentence.text):
        entries.append({
            "name": name,
            "kind": kind_of_name(name),
            "alias": alias,
            "aliases": [a for a in [alias] if a],
            "wiki": None,
            "declared_at": sentence.id,
        })
    return entries


def _merge_ledger(ledger: list[dict], new_entries: list[dict]) -> list[dict]:
    by_name = {e["name"]: dict(e) for e in ledger}
    for entry in new_entries:
        cur = by_name.get(entry["name"])
        if cur is None:
            by_name[entry["name"]] = dict(entry)
            continue
        aliases = set(cur.get("aliases") or []) | set(entry.get("aliases") or [])
        cur["aliases"] = sorted(a for a in aliases if a)
        cur["alias"] = cur.get("alias") or entry.get("alias")
        cur["wiki"] = cur.get("wiki") or entry.get("wiki")
    # keep all resolved entities for the whole article
    return list(by_name.values())


def _reject(sentence: Sentence, reason: str, **extra: Any) -> dict:
    return {
        "sentence_id": sentence.id,
        "original": sentence.text,
        "sentence": sentence.text,
        "status": "rejected",
        "reason": reason,
        **extra,
    }


class ArticleAgent:
    def __init__(self, cfg: dict, corpus: Corpus, wiki: LiveWiki,
                 sink: OutputSink, log: EventLog, client: Client,
                 spring: SpringParser | None = None) -> None:
        self.cfg = cfg
        self.corpus = corpus
        self.wiki = wiki
        self.sink = sink
        self.log = log
        self.client = client
        self.spring = spring or SpringParser()

    def run(self, article: Article) -> dict:
        if self.sink.is_done(article.id):
            return {"article_id": article.id, "skipped": True}
        ckpt = self.sink.load_checkpoint(article.id) or {}
        start = int(ckpt.get("next_index") or 0)
        ledger: list[dict] = ckpt.get("ledger") or []
        stats = Counter(ckpt.get("stats") or {})
        reject_reasons = Counter(ckpt.get("reject_reasons") or {})
        rewrite_kinds = Counter(ckpt.get("rewrite_kinds") or {})
        tool_calls = Counter(ckpt.get("tool_calls") or {})
        self.log.log(article.id, "article_start", title=article.title,
                     n_sentences=len(article.sentences), resume_from=start)

        batch_size = int(self.cfg["agent"]["batch_size"])
        batches = _paragraph_batches(article, batch_size)
        for batch_no, paragraphs in enumerate(batches):
            if batch_no < start:
                continue
            bstats = self._run_batch(article, paragraphs, batch_no, ledger,
                                     stats, reject_reasons,
                                     rewrite_kinds, tool_calls)
            ledger = bstats["ledger"]
            state = {
                "next_index": batch_no + 1,
                "ledger": ledger,
                "stats": dict(stats),
                "reject_reasons": dict(reject_reasons),
                "rewrite_kinds": dict(rewrite_kinds),
                "tool_calls": dict(tool_calls),
            }
            self.sink.save_checkpoint(article.id, state)

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
        self.log.log(article.id, "article_done", **{k: v for k, v in summary.items() if k != "article_id"})
        self.sink.mark_done(article.id, summary)
        self.sink.clear_checkpoint(article.id)
        return summary

    def _run_batch(self, article: Article, paragraphs: list[Paragraph],
                   batch_no: int, ledger: list[dict],
                   stats: Counter, reject_reasons: Counter,
                   rewrite_kinds: Counter, tool_calls: Counter) -> dict:
        batch = [s for p in paragraphs for s in p.sentences]
        self.log.log(article.id, "batch_start", batch=batch_no,
                     para_indexes=[p.para_index for p in paragraphs],
                     sentence_ids=[s.id for s in batch],
                     n_chars=sum(len(p.text) for p in paragraphs))
        tools = ToolBox(self.corpus, article, self.wiki, ledger, self.cfg,
                        self.spring)
        system = system_prompt(self.cfg)
        user = batch_message(article, paragraphs, ledger)
        result = self.client.run(system, user, tools.run)
        self._record_tools(article.id, batch_no, result, stats, tool_calls)
        # Base graphs exist only because the agent asked spring_parse for them.
        base_by_sid = tools.base_graphs
        self.log.log(article.id, "spring_graphs", batch=batch_no,
                     parsed=len(base_by_sid),
                     sentences=len(batch),
                     calls=tools.counts.get("spring_parse", 0))
        self.log.log(article.id, "graphs_received", batch=batch_no,
                     rounds=result.rounds, chars=len(result.content or ""))

        items = _parse_results(result.content, batch)
        self.log.log(article.id, "answers_matched", batch=batch_no,
                     sentences=len(batch),
                     matched=sum(1 for item in items if item))
        outcomes = {
            s.id: self._evaluate(article, s, item, base_by_sid, ledger)
            for s, item in zip(batch, items)
        }

        # Repair is an inherent agent ability: broken graphs are sent back to
        # the same conversation with the validator's errors, exactly like a
        # coding agent repairs a failing build. Policy rejections (ambiguous /
        # no credible source) are never "repaired" into a guess.
        max_repairs = int(self.cfg["agent"].get("max_repairs") or 0)
        for repair_no in range(1, max_repairs + 1):
            failed = [
                (s, outcomes[s.id]) for s in batch
                if outcomes[s.id]["kind"] == "failed"
            ]
            if not failed:
                break
            self.log.log(article.id, "repair_start", batch=batch_no,
                         attempt=repair_no,
                         sentence_ids=[s.id for s, _ in failed],
                         errors=[o["errors"][:4] for _, o in failed])
            note = _repair_note(failed, tools.base_graphs)
            result = self.client.run(
                None, note, tools.run,
                messages=result.messages + [{"role": "user", "content": note}]
                if result.messages else None,
            )
            self._record_tools(article.id, batch_no, result, stats, tool_calls)
            stats["repairs"] += 1
            repaired = _parse_results(result.content, [s for s, _ in failed])
            fixed = 0
            for (sentence, _), item in zip(failed, repaired):
                outcome = self._evaluate(
                    article, sentence, item, tools.base_graphs, ledger)
                if outcome["kind"] == "accepted":
                    fixed += 1
                # Keep the newest outcome, including remaining errors.
                outcomes[sentence.id] = outcome
            stats["repaired"] += fixed
            self.log.log(article.id, "repair_result", batch=batch_no,
                         attempt=repair_no, fixed=fixed,
                         remaining=sum(
                             1 for s in batch
                             if outcomes[s.id]["kind"] == "failed"))

        records, rejected = [], []
        for sentence in batch:
            outcome = outcomes[sentence.id]
            rec = outcome.get("record")
            if rec is None:
                # Repairs exhausted: reject, never emit unchecked output.
                reason = outcome.get("reason") or "validation_failed"
                rec = _reject(sentence, reason, article_id=article.id,
                              errors=outcome.get("errors"),
                              sentence_attempt=outcome.get("attempt"))
                outcome = {"kind": "rejected", "record": rec}
            if outcome["kind"] == "accepted":
                records.append(rec)
                stats["accepted"] += 1
                ledger = _merge_ledger(ledger, _ledger_from(rec, sentence))
                for e in rec.get("edits") or []:
                    rewrite_kinds[e.get("kind") or "entity"] += 1
                    stats["rewrites"] += 1
                self.log.log(article.id, "sentence_accepted", batch=batch_no,
                             sentence_id=sentence.id,
                             n_edits=len(rec.get("edits") or []))
            else:
                rejected.append(rec)
                stats["rejected"] += 1
                reject_reasons[rec.get("reason") or "rejected"] += 1
                self.log.log(article.id, "sentence_rejected", batch=batch_no,
                             sentence_id=sentence.id, reason=rec.get("reason"),
                             candidates=(rec.get("candidates") or [])[:3])
        self.sink.write_records(records)
        self.sink.write_rejected(rejected)
        self.log.log(article.id, "batch_accepted", batch=batch_no,
                     accepted=len(records), rejected=len(rejected))
        return {"ledger": ledger}

    def _record_tools(self, article_id: int, batch_no: int, result,
                      stats: Counter, tool_calls: Counter) -> None:
        stats["tool_rounds"] += result.rounds
        for call in result.tool_trace:
            self.log.log(article_id, "tool_call", batch=batch_no, **call)
            tool_calls[call["tool"]] += 1

    def _evaluate(self, article: Article, sentence: Sentence, item: dict,
                  base_by_sid: dict[int, str], ledger: list[dict]) -> dict:
        """Classify one model answer: accepted, policy-rejected, or repairable."""
        if not item:
            return {"kind": "failed", "reason": "no_graph",
                    "errors": ["no result whose 'original' is this "
                               "sentence; return one entry per sentence"]}

        if sentence.id not in base_by_sid:
            # No crutch: the agent never asked spring_parse for this sentence.
            return {"kind": "failed", "reason": "no_base_graph",
                    "errors": ["no spring_parse call for this sentence; "
                               "call spring_parse with its exact text first"]}

        if (item.get("status") or "ok") == "rejected":
            reason = item.get("reason") or "rejected"
            if reason in {"ambiguous_antecedent", "no_credible_source"}:
                return {"kind": "rejected", "record": _reject(
                    sentence, reason, candidates=item.get("candidates") or [],
                    article_id=article.id)}
            return {"kind": "failed", "reason": reason,
                    "errors": [f"model reported: {reason}"]}

        original = sentence.text
        resolved = str(item.get("sentence") or original)
        amr = item.get("amr")
        edits = item.get("edits") or []
        if not isinstance(amr, str) or not amr.strip():
            return {"kind": "failed", "reason": "no_graph",
                    "errors": ["missing or empty amr"]}

        rewritten, _, edit_errors = apply_edits(original, edits, self.cfg)
        if rewritten is not None:
            resolved = rewritten
        report = check_graph(original, resolved, amr, edits, self.cfg)
        if not report["ok"]:
            joined = " ".join(edit_errors or report.get("errors") or [])
            if "no credible evidence" in joined:
                return {"kind": "rejected", "record": _reject(
                    sentence, "no_credible_source", article_id=article.id,
                    errors=report.get("errors"))}
            if "ambiguous" in joined:
                return {"kind": "rejected", "record": _reject(
                    sentence, "ambiguous_antecedent", article_id=article.id,
                    errors=report.get("errors"))}
            return {"kind": "failed", "reason": "validation_failed",
                    "errors": report.get("errors") or ["invalid graph"],
                    "attempt": {"sentence": resolved, "amr": amr,
                                "edits": edits}}

        record = {
            "id": f"legal-{sentence.id:06d}",
            "sentence_id": sentence.id,
            "article_id": article.id,
            "paragraph_id": sentence.paragraph_id,
            "para_index": sentence.para_index,
            "sentence_index": sentence.idx,
            "source": article.source,
            "article_title": article.title,
            "sentence_original": original,
            "sentence": resolved,
            "amr": amr,
            "edits": report.get("edits") or [],
            "entities": item.get("entities") or [],
            "status": "ok",
            "base_amr": base_by_sid.get(sentence.id),
            "coverage": report.get("coverage"),
            "legal_concepts": report.get("legal_concepts"),
            "legal_relations": report.get("legal_relations"),
        }
        return {"kind": "accepted", "record": record}


def _repair_note(failed: list[tuple[Sentence, dict]],
                 base_by_sid: dict[int, str]) -> str:
    payload = [
        {
            "original": sentence.text,
            "base_amr": base_by_sid.get(sentence.id),
            "previous": outcome.get("attempt"),
            "errors": (outcome.get("errors") or [])[:8],
        }
        for sentence, outcome in failed
    ]
    return (
        "REPAIR PASS: the answers below failed local validation. Fix them and "
        "return corrected results for ONLY these sentences, in the same JSON "
        "shape and with the same 'original' text. Each one still has its "
        "SPRING base graph in 'base_amr' first: repair from that graph (call "
        "spring_parse again if one is missing). Use the other tools if you "
        "need to re-check a name or an antecedent. Do not change sentences "
        "that were not listed.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
