"""Read-only access to the legal corpus, one article at a time.

The agent works on a single article: its title, ordered sentences, the
article-level legal metadata, and two lookups used as tools:

* ``search``          - where does this text occur in this article?
* ``resolve_mention`` - what did "the Act" / "the Board" refer to here?

``resolve_mention`` is deliberately local and deterministic: earlier
sentences (explicit declarations such as ``... Act, 1996 (the 'Act')``),
article metadata, and the agent's own article ledger. It never guesses: tied
candidates are reported as ambiguous so the agent can reject the sentence.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import CREDIBLE_SOURCES

ACT_HEADS = (
    "Act|Code|Rules|Regulations|Regulation|Scheme|Order|Notification|"
    "Ordinance|Sanhita|Adhiniyam|Bill"
)
ENTITY_HEADS = (
    "Board|Commission|Tribunal|Court|Authority|Council|Ministry|Department|"
    "Committee|Company|Bank|Government|Institute|University|Corporation"
)
ALL_HEADS = ACT_HEADS + "|" + ENTITY_HEADS

_WORD = r"[A-Z][\w'’&().-]*"
_CONNECTOR = r"(?:of|and|for|the|on)"
NAME_RE = re.compile(
    rf"((?:{_WORD}(?:\s+{_CONNECTOR})?\s+){{0,8}}(?:{ALL_HEADS})"
    r"(?:\s*,?\s*(?:19|20)\d{2})?)\b"
)
ALIAS_RE = re.compile(
    r"\(\s*(?:the|said)?\s*['‘’\"“”]?\s*"
    rf"(?P<alias>{ACT_HEADS}|{ENTITY_HEADS})"
    r"(?:s)?\s*['‘’\"“”]?\s*\)",
    re.IGNORECASE,
)

GENERIC_WORDS = {
    "act", "code", "rule", "regulation", "scheme", "order", "notification",
    "ordinance", "bill", "board", "commission", "tribunal", "court",
    "authority", "council", "ministry", "department", "committee", "company",
    "bank", "government", "instrument", "statute", "provision",
}
_SINGULAR = {
    "acts": "act", "codes": "code", "rules": "rule",
    "regulations": "regulation", "schemes": "scheme", "orders": "order",
    "notifications": "notification", "boards": "board",
    "commissions": "commission", "tribunals": "tribunal", "courts": "court",
    "authorities": "authority", "councils": "council",
    "ministries": "ministry", "departments": "department",
    "committees": "committee", "companies": "company", "banks": "bank",
}
_DETERMINERS = {"the", "a", "an", "said", "aforesaid", "aforementioned",
                "this", "that", "these", "those"}
_STOP_START = {
    "in", "for", "however", "this", "that", "under", "section", "article",
    "but", "and", "or", "if", "when", "where", "while", "as", "it", "there",
    "these", "those", "such", "no", "not", "part", "case", "law", "courts",
}


def _norm(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s&/'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def head_of(surface: str) -> str:
    """The head noun of a mention: "the said Act" -> "act"."""
    words = [w for w in _norm(surface).split() if w not in _DETERMINERS]
    if not words:
        return ""
    word = words[-1].strip("'’-")
    return _SINGULAR.get(word, word)


def is_generic(surface: str) -> bool:
    words = [w for w in _norm(surface).split() if w not in _DETERMINERS]
    return bool(words) and len(words) <= 3 and words[-1] in (
        GENERIC_WORDS | set(_SINGULAR)
    )


def head_in(name: str, head: str) -> bool:
    """Whether ``head`` is one of the words of a legal name."""
    if not head:
        return False
    cleaned = re.sub(r"\((?:19|20)\d{2}\)", " ", name)
    cleaned = re.sub(r",?\s*(?:19|20)\d{2}\s*$", " ", cleaned)
    words = [_SINGULAR.get(w, w) for w in _norm(cleaned).split()]
    return head in words


def name_variants(name: str) -> list[str]:
    """Surface variants that count as declaring ``name`` in running text."""
    base = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", name).strip()
    no_year = re.sub(r",?\s*(?:19|20)\d{2}\s*$", "", base).strip()
    variants = {name, base, no_year}
    words = [
        w for w in re.findall(r"[A-Za-z0-9]+", no_year)
        if w.lower() not in {"the", "of", "and", "for", "in", "on", "at", "to"}
        and not w.isdigit()
    ]
    if len(words) >= 2:
        variants.add("".join(w[0].upper() for w in words))
    return [v for v in variants if len(v) >= 3]


def declared_names(text: str) -> list[tuple[str, str | None]]:
    """Legal names in a sentence, with an explicit parenthetical alias."""
    found: dict[str, str | None] = {}
    for match in NAME_RE.finditer(text):
        name = _clean_name(match.group(1))
        if not name:
            continue
        alias = None
        tail = text[match.end():match.end() + 60]
        alias_match = ALIAS_RE.match(tail.lstrip())
        if alias_match:
            alias = alias_match.group("alias")
        found[name] = found.get(name) or alias
    # Parenthetical aliases whose declaration sits before the match.
    for alias_match in ALIAS_RE.finditer(text):
        before = text[max(0, alias_match.start() - 240):alias_match.start()]
        for match in NAME_RE.finditer(before):
            name = _clean_name(match.group(1))
            if name:
                found.setdefault(name, alias_match.group("alias"))
    return list(found.items())


def _clean_name(name: str) -> str | None:
    name = re.sub(r"\s+", " ", name).strip(" ,.;:()'’\"“”")
    # Drop a trailing declaration fragment such as " (the 'Act'".
    name = re.sub(r"\s*\([^()]*\)?\s*$", "", name).strip()
    name = re.sub(
        r"^(?:" + "|".join(sorted(_STOP_START, key=len, reverse=True))
        + r")\s+", "", name, flags=re.IGNORECASE,
    )
    words = name.split()
    if len(words) < 2 or len(name) < 6:
        return None
    if not re.search(rf"\b(?:{ALL_HEADS})\b", name):
        return None
    # A sentence fragment, not a name: "Act. In Bhatia ...", "Act in Delhi ...".
    if ". " in name or " v. " in name:
        return None
    if head_of(words[0]) in GENERIC_WORDS:
        return None
    return name


@dataclass
class Sentence:
    id: int
    idx: int
    paragraph_id: int
    para_index: int
    text: str


@dataclass
class Paragraph:
    """One paragraph of an article, unsplit, plus its corpus sentences.

    The agent receives ``text`` and splits it itself; ``sentences`` is what the
    tool uses to give a split sentence back its ``sentence_id``.
    """

    id: int
    para_index: int
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass
class Article:
    id: int
    title: str
    source: str
    sentences: list[Sentence] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    provisions: list[str] = field(default_factory=list)
    precedents: list[str] = field(default_factory=list)

    def paragraphs(self) -> list[Paragraph]:
        """Sentences grouped by paragraph, in article order."""
        groups: dict[int, Paragraph] = {}
        for sentence in self.sentences:
            group = groups.get(sentence.paragraph_id)
            if group is None:
                group = Paragraph(id=sentence.paragraph_id,
                                  para_index=sentence.para_index)
                groups[sentence.paragraph_id] = group
            group.sentences.append(sentence)
        return [groups[k] for k in sorted(
            groups, key=lambda k: (groups[k].para_index, k))]

    def sentence_index(self, sentence_id: int) -> int:
        for i, sentence in enumerate(self.sentences):
            if sentence.id == sentence_id:
                return i
        return len(self.sentences)

    def metadata(self) -> list[tuple[str, str]]:
        """``(kind, surface)`` pairs from the article-level metadata."""
        rows: list[tuple[str, str]] = []
        rows += [("statute", name) for name in self.statutes]
        rows += [("provision", name) for name in self.provisions]
        rows += [("judgment", name) for name in self.precedents]
        clean: list[tuple[str, str]] = []
        for kind, name in rows:
            name = re.sub(r"\s+", " ", name or "").strip()
            # metadata sometimes carries a sentence fragment ("Act. In ...");
            # a name with an internal full stop is not a name.
            if len(name) < 3 or ". " in name or name.endswith("."):
                continue
            clean.append((kind, name))
        return clean


class Corpus:
    """SQLite reader with a connection per thread."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"corpus database not found: {self.path}")
        self._local = threading.local()

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
            )
            db.row_factory = sqlite3.Row
            self._local.db = db
        return db

    def article_ids(self, limit: int = 0, min_sentences: int = 1) -> list[int]:
        sql = """
            SELECT a.id
            FROM articles a
            JOIN paragraphs p ON p.article_id = a.id
            JOIN sentences s ON s.paragraph_id = p.id
            GROUP BY a.id
            HAVING count(s.id) >= ?
            ORDER BY a.id
        """
        args: list[Any] = [min_sentences]
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [row[0] for row in self._db().execute(sql, args)]

    def get_article(self, article_id: int) -> Article:
        db = self._db()
        row = db.execute(
            "SELECT id, title, source FROM articles WHERE id = ?",
            (article_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"article not found: {article_id}")
        sentences = [
            Sentence(
                id=r["id"], idx=r["idx"], paragraph_id=r["paragraph_id"],
                para_index=r["para_index"], text=r["text"],
            )
            for r in db.execute(
                """
                SELECT s.id, s.idx, s.paragraph_id, p.para_index, s.text
                FROM sentences s
                JOIN paragraphs p ON p.id = s.paragraph_id
                WHERE p.article_id = ?
                ORDER BY p.para_index, s.idx, s.id
                """,
                (article_id,),
            )
        ]
        meta: dict[str, list[str]] = {
            "statutes": [], "provisions": [], "precedents": []
        }
        seen: dict[str, set[str]] = {key: set() for key in meta}
        for r in db.execute(
            "SELECT statutes, provisions, precedents FROM paragraphs "
            "WHERE article_id = ?",
            (article_id,),
        ):
            for key in meta:
                try:
                    values = json.loads(r[key] or "[]")
                except (TypeError, json.JSONDecodeError):
                    continue
                for value in values:
                    if isinstance(value, str) and value not in seen[key]:
                        seen[key].add(value)
                        meta[key].append(value)
        return Article(
            id=row["id"], title=row["title"] or "", source=row["source"] or "",
            sentences=sentences, statutes=meta["statutes"],
            provisions=meta["provisions"], precedents=meta["precedents"],
        )

    # -- tools ------------------------------------------------------------- #

    def search(self, article: Article, query: str,
               before_sentence_id: int | None = None,
               limit: int = 5) -> list[dict]:
        query = (query or "").strip()
        if len(query) < 2:
            return []
        terms = [query]
        if query.lower().startswith("the "):
            terms.append(query[4:])
        pattern = "%" + terms[0].replace("\\", "\\\\").replace("%", "\\%") \
            .replace("_", "\\_") + "%"
        rows = self._db().execute(
            """
            SELECT s.id, s.idx, p.para_index, s.text
            FROM sentences s
            JOIN paragraphs p ON p.id = s.paragraph_id
            WHERE p.article_id = ? AND s.text LIKE ? ESCAPE '\\'
            """,
            (article.id, pattern),
        ).fetchall()
        if not rows and len(terms) > 1:
            pattern = "%" + terms[1].replace("%", "\\%") + "%"
            rows = self._db().execute(
                """
                SELECT s.id, s.idx, p.para_index, s.text
                FROM sentences s
                JOIN paragraphs p ON p.id = s.paragraph_id
                WHERE p.article_id = ? AND s.text LIKE ? ESCAPE '\\'
                """,
                (article.id, pattern),
            ).fetchall()
        hits = [
            {
                "sentence_id": r["id"], "idx": r["idx"],
                "para_index": r["para_index"], "text": r["text"],
                "distance": abs(r["id"] - before_sentence_id)
                if before_sentence_id is not None else 0,
            }
            for r in rows
        ]
        if before_sentence_id is not None:
            # "what did the article say before this point": earlier only,
            # nearest first, and never the sentence we are resolving.
            hits = [
                dict(h, distance=before_sentence_id - h["sentence_id"])
                for h in hits if h["sentence_id"] < before_sentence_id
            ]
        hits.sort(key=lambda h: (h["distance"], -h["sentence_id"]))
        return hits[:limit]

    def resolve_mention(self, article: Article, surface: str,
                        sentence_id: int,
                        ledger: Iterable[dict] | None = None
                        ) -> dict[str, Any]:
        """Evidence-backed candidates for a mention in this article.

        Candidates are presented with their evidence (no numeric score). The
        agent decides which one, if any, the mention refers to. When there is
        no clear single answer, the sentence should be rejected rather than
        guessed.
        """
        head = head_of(surface)
        generic = is_generic(surface)
        index = article.sentence_index(sentence_id)
        candidates: dict[str, dict[str, Any]] = {}

        # Lower rank = stronger evidence (used only to order the list).
        rank_order = {
            "memory_exact": 0,          # exact name/alias already resolved
            "same_sentence_before": 1,  # full name written before mention
            "earlier_alias": 2,         # declared earlier with an alias
            "earlier_name": 3,          # declared earlier by name
            "metadata": 4,              # listed in article metadata
            "memory_head": 5,           # same kind of thing in memory (hint)
        }

        def offer(name: str, kind: str, source: str, match: str,
                  rank: int, evidence: Sentence | None = None,
                  alias: str | None = None,
                  distance: int | None = None,
                  before_mention: bool = False) -> None:
            name = re.sub(r"\s+", " ", name).strip()
            name = name.strip(" ,.;:'’\"“”")
            # keep a closing paren that closes an earlier one: "section 2(2)"
            while name.endswith(")") and name.count("(") < name.count(")"):
                name = name[:-1].strip(" ,.;:'’\"“”")
            while name.startswith("(") and name.count("(") > name.count(")"):
                name = name[1:].strip(" ,.;:'’\"“”")
            if not name or len(name) < 3:
                return
            current = candidates.get(name)
            if current is not None:
                cur_rank = current.get("_rank", 99)
                cur_dist = current.get("distance")
                if cur_rank < rank:
                    return
                if cur_rank == rank and cur_dist is not None and distance is not None and cur_dist <= distance:
                    return
            candidates[name] = {
                "name": name,
                "kind": kind,
                "source": source,
                "match": match,
                "alias": alias,
                "evidence_sentence_id": evidence.id if evidence else None,
                "evidence_text": evidence.text if evidence else None,
                "distance": distance,
                "before_mention_in_sentence": before_mention,
                "_rank": rank,
            }

        # 1. Article memory (ledger): names already resolved in this article.
        #    An exact name/alias is a direct match. A generic mention (e.g.
        #    "the Court") that only shares the head with a remembered name is a
        #    weak hint, not proof — it must not outweigh a name stated in the
        #    sentence itself.
        for entry in ledger or []:
            names = [entry.get("name", "")] + list(entry.get("aliases") or [])
            exact = any(_norm(n) == _norm(surface) for n in names)
            head_only = generic and any(head_in(n, head) for n in names)
            if exact:
                offer(
                    entry["name"], entry.get("kind") or "legal-instrument",
                    "article_memory", "exact name or alias in memory",
                    rank_order["memory_exact"], None, entry.get("alias"),
                )
            elif head_only:
                offer(
                    entry["name"], entry.get("kind") or "legal-instrument",
                    "article_memory",
                    "generic mention matches same kind in memory (hint only)",
                    rank_order["memory_head"], None, entry.get("alias"),
                )

        # 2. Article sentences: explicit declarations and legal names.
        #    The current sentence counts too, but only a name that appears
        #    BEFORE the mention ("... before the Supreme Court, the Court ...").
        ceiling = max(0, index)
        here = article.sentences[ceiling].text if article.sentences else ""
        mention_at = here.find(surface)
        for position in range(ceiling + 1):
            sentence = article.sentences[position]
            same_sentence = position == ceiling
            for name, alias in declared_names(sentence.text):
                alias_hit = bool(alias and head_of(alias) == head)
                if not generic:
                    if _norm(surface) not in _norm(name):
                        continue
                elif not alias_hit and not head_in(name, head):
                    continue
                if same_sentence:
                    name_at = here.find(name)
                    if mention_at < 0 or name_at < 0 or name_at >= mention_at:
                        continue
                    offer(name, kind_of_name(name), "article_text",
                          "declared before the mention in the same sentence",
                          rank_order["same_sentence_before"], sentence, alias,
                          distance=0, before_mention=True)
                else:
                    distance = ceiling - position
                    if alias_hit:
                        match = "declared in an earlier sentence (alias match)"
                        rank = rank_order["earlier_alias"]
                    else:
                        match = "declared in an earlier sentence"
                        rank = rank_order["earlier_name"]
                    offer(name, kind_of_name(name), "article_text",
                          match, rank, sentence, alias, distance=distance)

        # 3. Article metadata: legal texts tagged for this whole article.
        for kind, name in article.metadata():
            if not generic:
                if _norm(surface) not in _norm(name):
                    continue
            elif not head_in(name, head):
                continue
            if is_generic(name):
                continue
            offer(name, kind, "article_metadata",
                  "listed in article metadata", rank_order["metadata"])

        listed = sorted(
            candidates.values(),
            key=lambda c: (c["_rank"], c.get("distance") if c.get("distance") is not None else 10**9, len(c["name"])),
        )
        for c in listed:
            c.pop("_rank", None)
        return {
            "surface": surface,
            "head": head,
            "generic": generic,
            "candidates": listed,
            "sentence_id": sentence_id,
        }


def kind_of_name(name: str) -> str:
    if re.search(rf"\b(?:{ACT_HEADS})\b", name):
        return "statute"
    if re.search(rf"\b(?:{ENTITY_HEADS})\b", name):
        return "gov-entity"
    return "legal-instrument"
