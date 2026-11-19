"""Live Wikipedia / Wikidata lookup for :wiki titles.

No local title database: every lookup goes to the live APIs and is written to a
SQLite cache, so parallel workers and reruns are reproducible and polite. The
result of a lookup is a ranked list of candidates with enough evidence
(`source`, `description`, `qid`, `kind`, `year`) for the agent and the
consistency checker to judge credibility.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corpus import _norm, head_of

USER_AGENT = "agentic-dataset-amr/0.1 (research; deterministic lookup)"

STOPWORDS = {
    "the", "and", "for", "of", "in", "on", "at", "to", "a", "an", "by", "with",
    "or", "into", "under", "over", "act", "code", "rules", "regulation",
    "order", "notification", "scheme", "court", "board", "commission",
    "tribunal", "authority", "ministry", "department", "government", "state",
}

ACT_HEADS = {
    "act", "code", "sanhita", "ordinance", "bill", "rules", "regulations",
    "scheme", "notification", "order",
}
COURT_HEADS = {"court", "tribunal"}
BODY_HEADS = {
    "commission", "board", "authority", "ministry", "department", "council",
    "government", "bank", "committee", "corporation", "institute",
}


def _wiki_title(title: str) -> str:
    return title.replace(" ", "_")


def _year(text: str) -> str | None:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text or "")
    return m.group(1) if m else None


def _number_tokens(text: str) -> list[str]:
    return re.findall(r"\d+", text or "")


def _numbers_ok(surface: str, title: str, description: str = "") -> bool:
    """Every number/year in the surface must appear in the candidate."""
    need = _number_tokens(surface)
    if not need:
        return True
    have = set(_number_tokens(title) + _number_tokens(description))
    return all(n in have for n in need)


def _kind(title: str, description: str = "", instance_labels: list[str] | None = None) -> str:
    hay = " ".join([title, description] + (instance_labels or [])).lower()
    words = set(re.findall(r"[a-z]+", title.lower()))
    if " v. " in title.lower() or " vs " in title.lower() or "judgment" in hay or "case" in hay:
        return "judgment"
    if words & COURT_HEADS or "court" in hay:
        return "court"
    if words & ACT_HEADS or "statute" in hay or "legislation" in hay:
        return "statute"
    if words & BODY_HEADS or "government agency" in hay or "ministry" in hay:
        return "gov-entity"
    if "company" in hay or "corporation" in hay:
        return "company"
    return "legal-instrument"


def _score(surface: str, title: str, description: str = "") -> float:
    ns, nt = _norm(surface), _norm(title)
    if not ns or not nt:
        return 0.0
    if not _numbers_ok(surface, title, description):
        return 0.0
    if ns == nt:
        return 1.0
    ys, yt = _year(surface), _year(title)
    st = {t for t in re.findall(r"[a-z0-9]+", ns) if t not in STOPWORDS and not t.isdigit()}
    tt = {t for t in re.findall(r"[a-z0-9]+", nt) if t not in STOPWORDS and not t.isdigit()}
    overlap = len(st & tt) / max(len(st), 1)
    score = 0.6 * overlap
    if ys and yt and ys == yt:
        score += 0.35
    if _norm(surface).replace(" ", "") == nt.replace(" ", ""):
        score += 0.1
    if "india" in (description or "").lower():
        score += 0.03
    return round(min(score, 0.99), 4)


class WikiCache:
    """SQLite cache keyed by a normalized query; safe across threads."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._init()

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=30000")
            self._local.db = db
        return db

    def _init(self) -> None:
        with self._init_lock:
            self._db().execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, kind TEXT NOT NULL, "
                "at TEXT NOT NULL)"
            )
            self._db().commit()

    def get(self, key: str) -> Any | None:
        row = self._db().execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any, kind: str = "wiki") -> Any:
        payload = json.dumps(value, ensure_ascii=False)
        at = datetime.now(timezone.utc).isoformat()
        db = self._db()
        db.execute(
            "INSERT OR REPLACE INTO cache(key,value,kind,at) VALUES(?,?,?,?)",
            (key, payload, kind, at),
        )
        db.commit()
        return value


class LiveWiki:
    """Live lookup against the English Wikipedia / Wikidata APIs."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enabled = cfg.get("enabled", True)
        self.lang = cfg.get("lang", "en")
        self.timeout = cfg.get("timeout", 10)
        self.max_results = cfg.get("max_results", 5)
        self.cache = WikiCache(cfg["cache"])

    def _fetch(self, host: str, params: dict[str, Any]) -> dict:
        params = {**params, "format": "json"}
        url = host + "?" + urllib.parse.urlencode(params)
        cache_key = url
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network issues are not fatal to the run
            data = {"_error": str(exc)}
        return self.cache.set(cache_key, data, kind="api")

    def lookup(self, surface: str, context: str = "") -> list[dict[str, Any]]:
        """Ranked candidates for ``surface`` (live, cached)."""
        surface = (surface or "").strip()
        if not surface or not self.enabled:
            return []
        candidates: dict[str, dict[str, Any]] = {}

        def offer(c: dict[str, Any]) -> None:
            key = c["wiki_title"]
            old = candidates.get(key)
            if old is None or c["score"] > old["score"]:
                candidates[key] = c

        api = f"https://{self.lang}.wikipedia.org/w/api.php"
        # 1. Exact title / redirect.
        data = self._fetch(api, {
            "action": "query", "titles": surface, "redirects": 1,
            "prop": "description|pageprops|info", "inprop": "url",
        })
        for page in (data.get("query") or {}).get("pages", {}).values():
            if int(page.get("pageid", 0)) < 0:
                continue
            title = page.get("title", "")
            desc = page.get("description") or ""
            if not _numbers_ok(surface, title, desc):
                continue
            offer({
                "title": title, "wiki_title": _wiki_title(title),
                "description": desc, "kind": _kind(title, desc),
                "qid": (page.get("pageprops") or {}).get("wikibase_item"),
                "score": max(_score(surface, title, desc), 0.85),
                "source": "wiki_title", "credible": True,
            })

        # 2. Search (ranked, with snippets).
        search = self._fetch(api, {
            "action": "query", "list": "search", "srsearch": surface,
            "srlimit": self.max_results, "srprop": "snippet",
        })
        for i, row in enumerate((search.get("query") or {}).get("search") or []):
            title = row.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", html.unescape(row.get("snippet") or ""))
            if not _numbers_ok(surface, title, snippet):
                continue
            s = _score(surface, title, snippet) - i * 0.02
            offer({
                "title": title, "wiki_title": _wiki_title(title),
                "description": snippet, "kind": _kind(title, snippet),
                "qid": None, "score": round(max(s, 0.0), 4),
                "source": "wiki_search", "credible": s >= 0.55,
            })

        # 3. Wikidata entity search (good for statutes/bodies by alias).
        wd = self._fetch("https://www.wikidata.org/w/api.php", {
            "action": "wbsearchentities", "search": surface,
            "language": self.lang, "limit": self.max_results, "type": "item",
        })
        for row in wd.get("search") or []:
            label = row.get("label") or row.get("match", {}).get("text") or ""
            desc = row.get("description") or ""
            title = label
            if not _numbers_ok(surface, label, desc):
                continue
            s = max(_score(surface, label, desc), 0.5)
            offer({
                "title": title, "wiki_title": _wiki_title(title),
                "description": desc, "kind": _kind(label, desc),
                "qid": row.get("id"), "score": round(s, 4),
                "source": "wikidata", "credible": s >= 0.55,
            })

        ranked = sorted(
            candidates.values(), key=lambda c: (-c["score"], len(c["title"]))
        )[: self.max_results]
        # Top must clearly lead to be treated as unambiguous later.
        return ranked
