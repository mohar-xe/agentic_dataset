"""Stage: attach :wiki from Wikipedia.

Runs with the notebook's venv interpreter (penman + network).
Inputs:  /kaggle/working/picked.json, /kaggle/working/graphs.json
Writes:  /kaggle/working/linked.json  list of rows with "amr" (wiki-linked)
         and "wiki" ([{node, title, mention}]).

Linking strategy, precision over recall:
1. exact/normalised title lookups, batched 50 mentions per API call;
2. search-API fallback for the remaining mentions;
3. a title is kept only if it restates the mention (and carries no extra
   parenthetical qualifier);
4. disambiguation pages are dropped.

Resolutions are cached in wiki_cache.json, so a rerun makes almost no requests.
"""

import json
import os
import re
import sys
import time
import traceback
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import penman
from penman.models.amr import model as amr_model

WORK = os.environ.get("AMR_WORK", "/kaggle/working")
PICKED_JSON = os.environ.get("AMR_PICKED", f"{WORK}/picked.json")
GRAPHS_JSON = os.environ.get("AMR_GRAPHS", f"{WORK}/graphs.json")
LINKED_JSON = os.environ.get("AMR_LINKED", f"{WORK}/linked.json")
CACHE_JSON = os.environ.get("AMR_WIKI_CACHE", f"{WORK}/wiki_cache.json")
WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "legal-amr-research/1.0 (academic corpus annotation)"
WORKERS = 8
BATCH = 50


def _get(url, tries=4):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:          # 429s and hiccups: back off, retry
            last = exc
            time.sleep(min(1.5 * (attempt + 1), 6))
    raise last


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()


def _confident(mention, title):
    """Accept only titles that restate the mention."""
    if "(" in (title or "") and "(" not in (mention or ""):
        return False                      # "Mercury (planet)" for "Mercury"
    m, t = _norm(mention), _norm(title)
    if not m or not t:
        return False
    return m == t or (len(m) >= 2 and set(m) <= set(t))


def exact_titles(mentions):
    """mention -> title via batched exact title lookups (50 per request)."""
    found = {}
    for i in range(0, len(mentions), BATCH):
        chunk = mentions[i:i + BATCH]
        q = urllib.parse.urlencode({"action": "query", "format": "json",
                                    "redirects": 1,
                                    "titles": "|".join(chunk)})
        try:
            data = _get(f"{WIKI_API}?{q}")
        except Exception:
            traceback.print_exc()
            continue
        query = data.get("query", {})
        norm = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in query.get("redirects", [])}
        by_title = {}
        for m in chunk:
            t = redir.get(norm.get(m, m), norm.get(m, m))
            by_title.setdefault(t.lower(), []).append(m)
        for page in query.get("pages", {}).values():
            if "missing" in page:
                continue
            title = page.get("title") or ""
            for m in by_title.get(title.lower(), []):
                if _confident(m, title):
                    found[m] = title
                    break
    return found


def search_wiki(mention):
    q = urllib.parse.urlencode({"action": "query", "format": "json",
                                "list": "search", "srsearch": mention,
                                "srlimit": 5, "redirects": 1})
    try:
        hits = _get(f"{WIKI_API}?{q}")["query"]["search"]
    except Exception:                     # network hiccup: leave unlinked
        return None
    for hit in hits:
        if _confident(mention, hit["title"]):
            return hit["title"]
    return None


def disambiguation_titles(titles):
    out = set()
    titles = sorted(titles)
    for i in range(0, len(titles), BATCH):
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "redirects": 1,
            "prop": "pageprops", "ppprop": "disambiguation",
            "titles": "|".join(titles[i:i + BATCH])})
        try:
            pages = _get(f"{WIKI_API}?{q}")["query"]["pages"]
            for page in pages.values():
                if "disambiguation" in (page.get("pageprops") or {}):
                    out.add(page["title"])
        except Exception:
            traceback.print_exc()
    return out


def _var(node):
    """Penman triples may carry (variable, alignment) tuples."""
    return node[0] if isinstance(node, tuple) else node


def name_mentions(amr):
    """(variable, mention) for :name nodes without :wiki yet."""
    if not amr:
        return []
    try:
        g = penman.decode(amr, model=amr_model)
    except Exception:                     # malformed graph from a backoff
        return []
    has_wiki = {_var(s) for s, r, _ in g.triples if r == ":wiki"}
    ops = {}
    for s, r, t in g.triples:
        m = re.fullmatch(r":op(\d+)", r)
        if m:
            ops.setdefault(_var(s), []).append(
                (int(m.group(1)), str(t).strip('"')))
    out = []
    for s, r, t in g.triples:
        if r != ":name":
            continue
        var, words = _var(s), ops.get(_var(t), [])
        if words and var not in has_wiki:
            out.append((var, " ".join(w for _, w in sorted(words))))
    return out


def add_wiki(amr, links):
    g = penman.decode(amr, model=amr_model)
    triples = list(g.triples)
    for node, title in links:
        triples.append(penman.Triple(node, ":wiki", f'"{title}"'))
    return penman.encode(penman.Graph(triples, top=g.top, metadata=g.metadata),
                         model=amr_model)


def main():
    picked = json.load(open(PICKED_JSON))
    graphs = json.load(open(GRAPHS_JSON))["graphs"]
    cache = {}
    if os.path.isfile(CACHE_JSON):
        try:
            cache = json.load(open(CACHE_JSON))
        except Exception:
            cache = {}

    mentions_by_row = [name_mentions(a) for a in graphs]
    all_mentions = sorted({m for row in mentions_by_row for _, m in row})
    missing = [m for m in all_mentions if m not in cache]
    print(f"{len(all_mentions)} unique entity mentions, "
          f"{len(missing)} not cached", flush=True)

    if missing:
        t0 = time.time()
        found = exact_titles(missing)
        print(f"exact-title hits: {len(found)}/{len(missing)} "
              f"({time.time()-t0:.0f}s)", flush=True)

        rest = [m for m in missing if m not in found]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            searched = dict(zip(rest, pool.map(search_wiki, rest)))
        print(f"search fallback: {sum(1 for v in searched.values() if v)}"
              f"/{len(rest)} ({time.time()-t0:.0f}s)", flush=True)

        for m in missing:
            cache[m] = found.get(m) or searched.get(m)
        json.dump(cache, open(CACHE_JSON, "w"))

    titles = {t for t in cache.values() if t}
    disambig = disambiguation_titles(titles) if titles else set()
    print(f"{len(disambig)} disambiguation pages dropped", flush=True)

    linked_rows = linked_nodes = 0
    out_rows = []
    for row, amr, mentions in zip(picked, graphs, mentions_by_row):
        links, wiki = [], []
        for node, mention in mentions:
            title = cache.get(mention)
            if title and title not in disambig:
                links.append((node, title))
                wiki.append({"node": node, "title": title, "mention": mention})
        if links:
            try:
                row["amr"] = add_wiki(amr, links)
                linked_rows += 1
                linked_nodes += len(links)
            except Exception:
                traceback.print_exc()
                row["amr"] = amr
        else:
            row["amr"] = amr
        row["wiki"] = wiki
        out_rows.append(row)

    json.dump(out_rows, open(LINKED_JSON, "w"))
    print(f"{linked_nodes} wiki nodes in {linked_rows}/{len(out_rows)} "
          f"sentences", flush=True)
    for r in out_rows:
        if r.get("wiki"):
            print("\n", r["sentence"][:100], "->", r["wiki"])
            print(r["amr"][:300])
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
