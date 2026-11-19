"""Tool schemas and dispatch for the per-article agent.

The model is given a small, fixed set of tools. Each tool is a plain
function over this article's state plus the wiki cache; dispatch is a dict
lookup, not a framework. Every call/result is cheap to log.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

from check import check_graph
from corpus import Article, Corpus, Sentence, head_of, is_generic
from rewrite import apply_edits
from spring import SpringParser
from wiki import LiveWiki


def _unmatched(field: str) -> dict[str, Any]:
    return {
        "error": (f"'{field}' is not a sentence of this article: copy one "
                  f"sentence exactly as the paragraph splits it (punctuation "
                  f"may differ, wording may not)"),
        "hint": "call spring_parse with the same text first",
    }


def _key(text: str) -> str:
    """Punctuation-insensitive form used to give a split sentence its id."""
    text = " ".join((text or "").split()).lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return " ".join(text.split())


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "spring_parse",
            "description": (
                "Run the local SPRING parser on ONE sentence and get its "
                "classic AMR graph. This is a plain parser call and nothing "
                "else: give it a sentence, it gives you the graph. Split the "
                "paragraph yourself, then call this once per sentence; calls "
                "in the same turn run in parallel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentence": {
                        "type": "string",
                        "description": "One sentence, exactly as it appears "
                                       "in the paragraph.",
                    },
                },
                "required": ["sentence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "article_search",
            "description": (
                "Search sentences in THIS article. Use it to find where an "
                "entity was declared or defined (e.g. 'Arbitration and "
                "Conciliation Act, 1996 (the Act)'). 'before' is optional and "
                "is the sentence text you are reading from; it limits the "
                "search to what comes earlier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "before": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_mention",
            "description": (
                "Resolve a generic legal mention (e.g. 'the Act', 'the Board') "
                "to a specific entity using earlier sentences, article metadata "
                "and the running ledger. Returns ranked candidates with evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "sentence": {
                        "type": "string",
                        "description": "The sentence the mention occurs in, "
                                       "exactly as you split it.",
                    },
                },
                "required": ["surface", "sentence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_lookup",
            "description": (
                "Look up a legal name on live Wikipedia/Wikidata (cached). "
                "Returns candidate titles, descriptions, qids and kinds. "
                "Only use a result that credibly matches the name and context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["surface"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_amr",
            "description": (
                "Validate an AMR graph against core AMR, the legal schema and "
                "the sentence-rewrite contract. Call this before finalising."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "sentence": {"type": "string"},
                    "amr": {"type": "string"},
                    "edits": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["original", "sentence", "amr"],
            },
        },
    },
]


class ToolBox:
    def __init__(self, corpus: Corpus, article: Article, wiki: LiveWiki,
                 ledger: list[dict], cfg: dict,
                 spring: SpringParser | None = None) -> None:
        self.corpus = corpus
        self.article = article
        self.wiki = wiki
        self.ledger = ledger
        self.cfg = cfg
        self.spring = spring or SpringParser()
        self.counts: dict[str, int] = {}
        # sentence text -> corpus sentence, for spring_parse
        self._by_key: dict[str, Sentence] = {
            _key(s.text): s for s in article.sentences
        }
        # base graphs obtained THROUGH THE TOOL during this batch
        self.base_graphs: dict[int, str] = {}
        self.base_details: dict[int, dict] = {}
        self._lock = threading.Lock()

    def match_sentence(self, text: str) -> Sentence | None:
        key = _key(text)
        if not key:
            return None
        found = self._by_key.get(key)
        if found is not None:
            return found
        # tolerate a split that cut a quote or a trailing dash differently
        for candidate, sentence in self._by_key.items():
            if candidate.startswith(key) or key.startswith(candidate):
                return sentence
        return None

    def _spring_parse(self, args: dict[str, Any]) -> dict[str, Any]:
        """A plain SPRING call: sentence in, PENMAN graph out.

        The model gets the graph and nothing else — no sentence ids, no
        paragraph metadata. Matching the text back to the corpus sentence (so
        the record can be keyed by ``sentence_id``) happens here, off to the
        side, and never changes what the model sees.
        """
        text = " ".join(str(args.get("sentence") or "").split())
        sentence = self.match_sentence(text)
        parsed = self.spring.parse(text, sentence.id if sentence else None)
        amr = parsed.get("base_amr")
        result = {
            "amr": amr,
            "status": parsed.get("status"),
            "errors": parsed.get("errors") or [],
            "parser": parsed.get("parser"),
            "model": parsed.get("model"),
        }
        if parsed.get("n_nodes") is not None:
            result["n_nodes"] = parsed["n_nodes"]
            result["n_edges"] = parsed["n_edges"]
        if sentence is not None and amr:
            with self._lock:
                self.base_graphs[sentence.id] = amr
                self.base_details[sentence.id] = result
        return result

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + 1
        if name == "spring_parse":
            return self._spring_parse(args)
        if name == "article_search":
            before_id = None
            before = args.get("before") or args.get("before_sentence") or ""
            if before:
                anchor = self.match_sentence(str(before))
                if anchor is None:
                    return _unmatched("before")
                before_id = anchor.id
            hits = self.corpus.search(
                self.article, args.get("query", ""),
                before_sentence_id=before_id,
                limit=int(args.get("limit") or 5),
            )
            return {"hits": hits, "count": len(hits)}
        if name == "resolve_mention":
            anchor = self.match_sentence(str(args.get("sentence") or ""))
            if anchor is None:
                return _unmatched("sentence")
            return self.corpus.resolve_mention(
                self.article, args.get("surface", ""), anchor.id,
                self.ledger,
            )
        if name == "wiki_lookup":
            candidates = self.wiki.lookup(
                args.get("surface", ""), args.get("context", "")
            )
            return {"surface": args.get("surface", ""), "candidates": candidates}
        if name == "check_amr":
            return check_graph(
                args.get("original", ""), args.get("sentence", ""),
                args.get("amr", ""), args.get("edits") or [], self.cfg,
            )
        return {"error": f"unknown tool: {name}"}
