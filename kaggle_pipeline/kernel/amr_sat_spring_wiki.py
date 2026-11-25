"""Kaggle GPU kernel: SaT-12L-SM segmentation + SPRING (amrlib) + wiki linker.

Input  : paragraphs.json (auto-discovered under /kaggle/input)
Output : /kaggle/working/out/sentences_2000.jsonl
         /kaggle/working/out/sentences_2000.amr
         /kaggle/working/out/split_stats.json
         /kaggle/working/out/run_meta.json

Why amrlib instead of native SPRING: the official Sapienza checkpoints are
served only from nlp.uniroma1.it, which currently answers HTTP 503 on every
path (AMR2/AMR3, parsing/generation) and has no public mirror.  amrlib ships
the same SPRING AMR3 weights converted to a HF BART seq2seq checkpoint
(bjascob/amrlib-models release model_parse_spring-v0_1_0).  Those weights were
trained with `remove_wiki: "remove"`, so they never emit `:wiki`; the linker
below adds it afterwards from Wikipedia.

Pipeline
  1. split every paragraph with segment-any-text/sat-12l-sm on the GPU
     (wtpsplit/torch, child process);
  2. keep sentences with >= MIN_WORDS words and draw N_SENTENCES with
     random.Random(SEED).sample over the whole corpus;
  3. parse with amrlib's SPRING checkpoint (beam 5);
  4. for every named-entity node (:name), look the mention up on Wikipedia,
     drop disambiguation pages, attach `:wiki "Title"` (post-hoc, not native).
"""

import glob
import gzip
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SEED = 42
N_SENTENCES = 2000
MIN_WORDS = 4

OUTDIR = "/kaggle/working/out"
WORK = "/kaggle/working"
SENTS_JSON = f"{WORK}/sentences_all.json"
AMRLIB_DIR = f"{WORK}/amrlib_models"
SPRING_MODEL = f"{AMRLIB_DIR}/model_parse_spring-v0_1_0"
SPRING_URL = ("https://github.com/bjascob/amrlib-models/releases/download/"
              "model_parse_spring-v0_1_0/model_parse_spring-v0_1_0.tar.gz")
WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "legal-amr-research/1.0 (academic corpus annotation)"

os.makedirs(OUTDIR, exist_ok=True)


def sh(cmd, timeout=3600):
    print(f"$ {cmd}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    print((r.stdout + r.stderr)[-2500:], flush=True)
    print(f"[rc={r.returncode} {time.time()-t0:.0f}s]", flush=True)
    return r.returncode


# ------------------------------------------------------------------ input ---
print("run type:", os.environ.get("KAGGLE_KERNEL_RUN_TYPE"), flush=True)
print("input dirs:",
      sorted(os.listdir("/kaggle/input"))
      if os.path.isdir("/kaggle/input") else "NO /kaggle/input", flush=True)
HITS = sorted(glob.glob("/kaggle/input/**/paragraphs.json*", recursive=True),
              key=os.path.getsize, reverse=True)
if not HITS:
    for root, dirs, files in os.walk("/kaggle/input"):
        print("  tree:", root, dirs[:5], files[:5], flush=True)
    sys.exit("FATAL: paragraphs.json not mounted under /kaggle/input; "
             "add dataset mohar07/legal-amr-india-paragraphs as input")
DATA = HITS[0]
print("data:", DATA, os.path.getsize(DATA), "bytes", flush=True)

with (gzip.open(DATA, "rt", encoding="utf-8") if DATA.endswith(".gz")
      else open(DATA, encoding="utf-8")) as fh:
    payload = json.load(fh)
articles = payload["articles"]
paragraphs = payload["paragraphs"]
print(f"{len(articles)} articles / {len(paragraphs)} paragraphs", flush=True)

# ------------------------------------------------------------ segmentation ---
# Full wtpsplit (torch) in a child process: ~3 min on a T4 for the corpus.
sh("pip install -q wtpsplit", 1800)

SEG_JOB = r'''
import json, os, sys, time
import torch
from wtpsplit import SaT

data, out = sys.argv[1], sys.argv[2]
if data.endswith(".gz"):
    import gzip
    payload = json.load(gzip.open(data, "rt", encoding="utf-8"))
else:
    payload = json.load(open(data, encoding="utf-8"))
texts = [p["text"] for p in payload["paragraphs"]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "",
      flush=True)
t0 = time.time()
sat = SaT("sat-12l-sm")
if dev == "cuda":
    sat.half().to("cuda")
print(f"SaT loaded {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
sents = list(sat.split(texts, batch_size=32, outer_batch_size=1000,
                       stride=128, block_size=256, verbose=True))
print(f"segmented {len(texts)} paragraphs -> {sum(map(len, sents))} "
      f"sentences in {time.time()-t0:.0f}s", flush=True)
json.dump({"device": dev, "sentences": sents}, open(out, "w"))
'''
with open(f"{WORK}/seg_job.py", "w") as fh:
    fh.write(SEG_JOB)
if sh(f"{sys.executable} {WORK}/seg_job.py {DATA} {SENTS_JSON}", 7200) != 0:
    sys.exit("FATAL: SaT segmentation failed (see log above)")

seg = json.load(open(SENTS_JSON))
split = seg["sentences"]
print(f"segmentation device: {seg['device']}", flush=True)

sentences = []
for para, sents in zip(paragraphs, split):
    art = articles.get(str(para["article_id"]), {})
    for i, sent in enumerate(sents):
        sent = (sent or "").strip()
        if len(sent.split()) < MIN_WORDS:
            continue
        sentences.append({
            "article_id": para["article_id"],
            "para_id": para["para_id"],
            "para_index": para["para_index"],
            "sent_index": i,
            "sentence": sent,
            "words": len(sent.split()),
            "title": art.get("title"),
            "url": art.get("url"),
            "source": art.get("source"),
            "section": art.get("section"),
            "tags": para.get("tags", {}),
        })
print(f"{len(sentences)} usable sentences after MIN_WORDS={MIN_WORDS}",
      flush=True)

rng = random.Random(SEED)
picked = rng.sample(sentences, min(N_SENTENCES, len(sentences)))
picked.sort(key=lambda s: (s["article_id"], s["para_index"], s["sent_index"]))
print(f"sampled {len(picked)} sentences with seed={SEED}", flush=True)

stats = {
    "seed": SEED,
    "n_requested": N_SENTENCES,
    "n_picked": len(picked),
    "min_words": MIN_WORDS,
    "n_articles": len(articles),
    "n_paragraphs": len(paragraphs),
    "n_sentences_total": sum(map(len, split)),
    "n_sentences_usable": len(sentences),
    "n_articles_picked": len({s["article_id"] for s in picked}),
    "splitter": "segment-any-text/sat-12l-sm",
    "splitter_device": seg["device"],
    "parser": "amrlib model_parse_spring-v0_1_0 (SPRING AMR3 weights)",
    "wiki": "post-hoc Wikipedia search linker",
}
with open(f"{OUTDIR}/split_stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)
print(json.dumps(stats, indent=2), flush=True)

# ------------------------------------------------------------------ amrlib ---
# amrlib needs a 4.x transformers; Kaggle ships 5.x, which breaks the old
# seq2seq loading code.  Pin a 4.x release that matches the installed torch.
sh("pip install -q amrlib 'transformers==4.44.2'", 1800)
sh("python -c \"import torch, transformers, amrlib; "
   "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
   "print('transformers', transformers.__version__); "
   "print('amrlib', amrlib.__version__)\"")

os.makedirs(AMRLIB_DIR, exist_ok=True)
if not os.path.isfile(f"{SPRING_MODEL}/model.pt"):
    tgz = f"{WORK}/spring_amrlib.tar.gz"
    ok = False
    for attempt in range(1, 4):
        sh(f"curl -sL --fail --retry 3 -o {tgz} '{SPRING_URL}'", 3600)
        size = os.path.getsize(tgz) if os.path.exists(tgz) else 0
        print(f"amrlib checkpoint attempt {attempt}: {size/1e6:.1f} MB",
              flush=True)
        if size > 1_000_000_000:
            ok = True
            break
        time.sleep(30)
    if not ok:
        json.dump({"status": "error", "stage": "amrlib-checkpoint"},
                  open(f"{OUTDIR}/run_meta.json", "w"))
        sys.exit("FATAL: amrlib SPRING checkpoint unavailable")
    sh(f"tar xzf {tgz} -C {AMRLIB_DIR}", 1800)
    os.remove(tgz)

import amrlib                                                   # noqa: E402

t0 = time.time()
stog = amrlib.load_stog_model(model_dir=SPRING_MODEL)
print(f"amrlib SPRING loaded {time.time()-t0:.0f}s on {stog.device}",
      flush=True)

t0 = time.time()
graphs = stog.parse_sents([s["sentence"] for s in picked],
                          disable_progress=False)
print(f"parsed {len(graphs)} sentences in {time.time()-t0:.0f}s", flush=True)
for row, amr in zip(picked, graphs):
    amr = (amr or "").strip()
    # amrlib returns "()" for a graph it had to back off on.
    row["amr"] = "" if amr == "()" else amr
    row["status"] = "OK" if row["amr"] else "error"

# ------------------------------------------------------------ wiki linker ---
import penman                                                   # noqa: E402
from penman.models.noop import model as noop_model              # noqa: E402


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()


def _confident(mention, title):
    """Accept a Wikipedia hit only if it restates the mention."""
    m, t = _norm(mention), _norm(title)
    if not m or not t:
        return False
    if m == t:
        return True
    return len(m) >= 2 and set(m) <= set(t)


def search_wiki(mention):
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "list": "search",
        "srsearch": mention, "srlimit": 5, "redirects": 1,
    })
    try:
        hits = _get(f"{WIKI_API}?{q}")["query"]["search"]
    except Exception:  # noqa: BLE001 -- network hiccup: leave unlinked
        return None
    for hit in hits:
        if _confident(mention, hit["title"]):
            return hit["title"]
    return None


def _var(node):
    """Penman triples may carry (variable, alignment) tuples."""
    return node[0] if isinstance(node, tuple) else node


def name_mentions(amr):
    """(variable, mention) for named entities that have no :wiki yet."""
    try:
        g = penman.decode(amr, model=noop_model)
    except Exception:  # noqa: BLE001
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
        var = _var(s)
        words = ops.get(_var(t), [])
        if words and var not in has_wiki:
            out.append((var, " ".join(w for _, w in sorted(words))))
    return out


def add_wiki(amr, links):
    """links: list of (node, title); rewrite the PENMAN string."""
    g = penman.decode(amr, model=noop_model)
    triples = list(g.triples)
    for node, title in links:
        triples.append(penman.Triple(node, ":wiki", f'"{title}"'))
    return penman.encode(penman.Graph(triples, top=g.top), model=noop_model)


t0 = time.time()
mentions_by_row = [name_mentions(r["amr"]) if r["amr"] else [] for r in picked]
all_mentions = sorted({m for row in mentions_by_row for _, m in row})
print(f"{len(all_mentions)} unique entity mentions", flush=True)

with ThreadPoolExecutor(max_workers=6) as pool:
    resolved = dict(zip(all_mentions,
                        pool.map(search_wiki, all_mentions)))

# drop disambiguation pages in bulk (50 titles per query)
titles = sorted({t for t in resolved.values() if t})
disambig = set()
for i in range(0, len(titles), 40):
    chunk = titles[i:i + 40]
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "redirects": 1,
        "prop": "pageprops", "ppprop": "disambiguation",
        "titles": "|".join(chunk),
    })
    try:
        pages = _get(f"{WIKI_API}?{q}")["query"]["pages"]
        for page in pages.values():
            if "disambiguation" in (page.get("pageprops") or {}):
                disambig.add(page["title"])
    except Exception:  # noqa: BLE001
        traceback.print_exc()

linked_rows = 0
linked_nodes = 0
for row, mentions in zip(picked, mentions_by_row):
    links, wiki = [], []
    for node, mention in mentions:
        title = resolved.get(mention)
        if not title or title in disambig:
            continue
        links.append((node, title))
        wiki.append({"node": node, "title": title, "mention": mention})
    if links:
        try:
            row["amr"] = add_wiki(row["amr"], links)
            row["wiki"] = wiki
            linked_rows += 1
            linked_nodes += len(links)
        except Exception:  # noqa: BLE001 -- keep the parse, drop the wikis
            traceback.print_exc()
            row["wiki"] = []
    else:
        row["wiki"] = []
print(f"wiki linker: {linked_nodes} nodes in {linked_rows} sentences "
      f"({time.time()-t0:.0f}s), {len(disambig)} disambiguation pages dropped",
      flush=True)

# ------------------------------------------------------------------ output ---
order = ("article_id", "para_id", "para_index", "sent_index", "sentence",
         "words", "amr", "status", "wiki", "title", "url", "source",
         "section", "tags")
with open(f"{OUTDIR}/sentences_2000.jsonl", "w", encoding="utf-8") as fh:
    for r in picked:
        fh.write(json.dumps({k: r.get(k) for k in order},
                            ensure_ascii=False) + "\n")

with open(f"{OUTDIR}/sentences_2000.amr", "w", encoding="utf-8") as fh:
    for r in picked:
        fh.write(f"# ::id {r['para_id']}.{r['sent_index']}\n")
        fh.write(f"# ::snt {r['sentence']}\n")
        fh.write((r.get("amr") or "") + "\n\n")

parsed = sum(1 for r in picked if r.get("amr"))
stats.update({
    "status": "ok",
    "n_parsed": parsed,
    "n_parse_failed": len(picked) - parsed,
    "n_entities": len(all_mentions),
    "n_sentences_with_wiki": linked_rows,
    "n_wiki_nodes": linked_nodes,
    "n_disambiguation_dropped": len(disambig),
})
with open(f"{OUTDIR}/run_meta.json", "w") as fh:
    json.dump(stats, fh, indent=2)
print(json.dumps(stats, indent=2), flush=True)
print("DONE", flush=True)
