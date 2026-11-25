#!/usr/bin/env python3
"""Build the manual-run Kaggle notebook: SaT-12L-SM -> SPRING (amrlib) -> wiki.

Cells, in order:
  0  title / instructions (markdown)
  1  load corpus from /kaggle/input
  2  create the venv (reuses image torch via --system-site-packages)
  3  install wtpsplit in the venv, check CUDA
  4  segment all paragraphs on GPU                    -> sentences_all.json
  5  sample 20,000 sentences (seed 42)                -> picked.json
  6  install amrlib + transformers==4.44.2 in the venv
  7  download SPRING AMR3 weights (1.5 GB)            -> amrlib_models/
  8  parse with the venv python (resumable)           -> graphs.json
  9  Wikipedia linker (venv python, cached)           -> linked.json
 10  write out/*.jsonl, *.amr, run_meta.json + zip

Everything is installed inside `/kaggle/working/venv` so nothing fights the
image's transformers 5.x. The stage scripts are embedded as base64 so the
notebook needs only ONE input dataset (the corpus).

Notebook validity matters: Kaggle refuses to import an .ipynb whose markdown
cells carry `outputs`/`execution_count`, so `md()` and `code()` build separate
shapes and every cell gets an id.
"""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(HERE, "notebook_stages")
CELLS = []
N_SENTENCES = 20000


def md(text, idx):
    CELLS.append({
        "cell_type": "markdown",
        "id": f"md-{idx:02d}",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text, idx):
    CELLS.append({
        "cell_type": "code",
        "id": f"code-{idx:02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


def stage_b64(name):
    with open(os.path.join(STAGES, name), "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def build():
    n = 0

    def next_i():
        nonlocal n
        n += 1
        return n

    md(f"""# Legal AMR: SaT-12L-SM → SPRING → Wikipedia `:wiki`

Run the cells top to bottom. Nothing here needs a GPU queue.

**Settings (right panel):**
- Accelerator: **GPU T4**
- Internet: **On**
- Add Input → **legal-amr-india-paragraphs** (only input needed)

Stages: load corpus → create venv → segment (SaT-12L-SM) → sample
{N_SENTENCES:,} → amrlib + SPRING weights → parse → `:wiki` linker →
outputs + zip.
""", next_i())

    code('''
import glob, gzip, json, os, random, re, subprocess, sys, time, traceback

SEED, N_SENTENCES, MIN_WORDS = 42, %d, 4
WORK, OUTDIR = "/kaggle/working", "/kaggle/working/out"
SENTS_JSON = f"{WORK}/sentences_all.json"
PICKED_JSON = f"{WORK}/picked.json"
GRAPHS_JSON = f"{WORK}/graphs.json"
GRAPHS_JSONL = f"{WORK}/graphs.jsonl"
LINKED_JSON = f"{WORK}/linked.json"
VENV = f"{WORK}/venv"
PY = f"{VENV}/bin/python"
SPRING_MODEL = f"{WORK}/amrlib_models/model_parse_spring-v0_1_0"
OUT_NAME = f"sentences_{N_SENTENCES}"
os.makedirs(OUTDIR, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print("python:", sys.version.split()[0], sys.executable)
print("input dirs:", sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else "NO /kaggle/input")

hits = sorted(glob.glob("/kaggle/input/**/paragraphs.json*", recursive=True),
              key=os.path.getsize, reverse=True)
assert hits, ("no paragraphs.json under /kaggle/input - use Add Input and attach "
              "dataset mohar07/legal-amr-india-paragraphs")
DATA = hits[0]
print("data:", DATA, os.path.getsize(DATA), "bytes")

with (gzip.open(DATA, "rt", encoding="utf-8") if DATA.endswith(".gz")
      else open(DATA, encoding="utf-8")) as fh:
    payload = json.load(fh)
articles, paragraphs = payload["articles"], payload["paragraphs"]
print(len(articles), "articles /", len(paragraphs), "paragraphs")
''' % N_SENTENCES, next_i())

    md("""## 1 · Create the environment

Everything is installed into `/kaggle/working/venv`, never into the image.
The venv is created **with `--system-site-packages`** so it inherits the
image's torch/CUDA (a fresh torch wheel would be several GB and can shadow the
GPU build), and every package we install lands inside the venv.

Kaggle's `/usr/bin/python3` has no `ensurepip`, so plain `python3 -m venv`
fails there. The cell tries, in order:

1. `python3 -m venv --system-site-packages`
2. `python3 -m venv --without-pip` and reuse the system pip
3. `virtualenv` (ships its own pip wheels, does not need `ensurepip`)
4. the system interpreter, with a warning

It prints `sys.prefix` and the pip location so you can see where installs land.
""", next_i())

    code('''
import shutil
t0 = time.time()
VENV_PY = f"{VENV}/bin/python"


def run(cmd, quiet=False):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} failed:\\n{(res.stderr or res.stdout)[-600:]}")
    if not quiet:
        print(res.stdout.strip())


def pip_ok(py):
    """True only when pip itself lives inside the venv (not inherited)."""
    try:
        res = subprocess.run([py, "-c", "import pip; print(pip.__file__)"],
                             capture_output=True, text=True)
    except Exception:
        return False
    return res.returncode == 0 and res.stdout.strip().startswith(VENV)


shutil.rmtree(VENV, ignore_errors=True)   # drop a half-built venv from an earlier try
ok = False

try:                                       # 1. plain venv
    run([sys.executable, "-m", "venv", "--system-site-packages", VENV], quiet=True)
    ok = pip_ok(VENV_PY)
    print("venv:", "ok" if ok else "created but pip is missing")
except Exception as exc:
    print("venv failed:", str(exc).splitlines()[0])

if not ok:                                 # 2. venv without pip, reuse system pip
    shutil.rmtree(VENV, ignore_errors=True)
    try:
        run([sys.executable, "-m", "venv", "--system-site-packages",
             "--without-pip", VENV], quiet=True)
        ok = pip_ok(VENV_PY)
        print("venv --without-pip:", "ok" if ok else
              "rejected - pip would still be the system one")
    except Exception as exc:
        print("venv --without-pip failed:", str(exc).splitlines()[0])

if not ok:                                 # 3. virtualenv ships its own pip wheels
    shutil.rmtree(VENV, ignore_errors=True)
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], quiet=True)
        run([sys.executable, "-m", "virtualenv", "-q",
             "--system-site-packages", VENV], quiet=True)
        ok = pip_ok(VENV_PY)
        print("virtualenv:", "ok" if ok else "no usable pip")
    except Exception as exc:
        print("virtualenv failed:", str(exc).splitlines()[0])

if ok:
    PY = VENV_PY
    print("env ready:", PY, f"({time.time()-t0:.0f}s)")
else:
    PY = sys.executable
    print("WARNING: no venv could be built; installing into the image interpreter",
          PY, "- segmentation must run before amrlib downgrades transformers")

run([PY, "-c", "import sys; print('prefix', sys.prefix); print('base', sys.base_prefix)"])
run([PY, "-m", "pip", "--version"])
subprocess.run([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])
''', next_i())

    md("""## 2 · Install `wtpsplit` in the venv + segment with `sat-12l-sm`

Segmentation of all 26,272 paragraphs takes ~3 min on a T4. The model runs in a
child process so a model-load problem cannot wedge the kernel.
""", next_i())

    code('''
t0 = time.time()
subprocess.run([PY, "-m", "pip", "install", "-q", "wtpsplit"], check=True)
print(f"wtpsplit installed ({time.time()-t0:.0f}s)")
subprocess.run([PY, "-c",
                "import torch, wtpsplit; "
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
               check=True)
''', next_i())

    code('''
SEG_JOB = r\'\'\'
import json, sys, time
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
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
sat = SaT("sat-12l-sm")
if dev == "cuda":
    sat.half().to("cuda")
t0 = time.time()
sents = list(sat.split(texts, batch_size=32, outer_batch_size=1000,
                       stride=128, block_size=256, verbose=True))
print(f"segmented {len(texts)} paragraphs -> {sum(map(len, sents))} sentences "
      f"in {time.time()-t0:.0f}s", flush=True)
json.dump({"device": dev, "sentences": sents}, open(out, "w"))
\'\'\'
open(f"{WORK}/seg_job.py", "w").write(SEG_JOB)
t0 = time.time()
subprocess.run([PY, f"{WORK}/seg_job.py", DATA, SENTS_JSON], check=True)
print(f"segmentation stage done in {time.time()-t0:.0f}s")
''', next_i())

    md(f"""## 3 · Sample {N_SENTENCES:,} sentences — `random.Random(42)`, ≥4 words""", next_i())

    code('''
seg = json.load(open(SENTS_JSON))
split = seg["sentences"]
assert len(split) == len(paragraphs), (len(split), len(paragraphs))

sentences = []
for para, sents in zip(paragraphs, split):
    art = articles.get(str(para["article_id"]), {})
    for i, sent in enumerate(sents):
        sent = (sent or "").strip()
        if len(sent.split()) < MIN_WORDS:
            continue
        sentences.append({
            "article_id": para["article_id"], "para_id": para["para_id"],
            "para_index": para["para_index"], "sent_index": i,
            "sentence": sent, "words": len(sent.split()),
            "title": art.get("title"), "url": art.get("url"),
            "source": art.get("source"), "section": art.get("section"),
            "tags": para.get("tags", {}),
        })
print(len(sentences), "usable sentences")

picked = random.Random(SEED).sample(sentences, min(N_SENTENCES, len(sentences)))
picked.sort(key=lambda s: (s["article_id"], s["para_index"], s["sent_index"]))
json.dump(picked, open(PICKED_JSON, "w"))

stats = {
    "seed": SEED, "n_requested": N_SENTENCES, "n_picked": len(picked),
    "min_words": MIN_WORDS, "n_articles": len(articles),
    "n_paragraphs": len(paragraphs),
    "n_sentences_total": sum(map(len, split)),
    "n_sentences_usable": len(sentences),
    "n_articles_picked": len({s["article_id"] for s in picked}),
    "splitter": "segment-any-text/sat-12l-sm",
    "splitter_device": seg["device"],
}
json.dump(stats, open(f"{OUTDIR}/split_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
''', next_i())

    md("""## 4 · Install `amrlib` + `transformers==4.44.2` in the same venv

`amrlib` needs `transformers` 4.x and the image ships 5.x. SaT has already run,
so downgrading `transformers` inside the venv now is safe.
""", next_i())

    code('''
t0 = time.time()
# transformers 4.44.2 pins tokenizers<0.20, and tokenizers 0.19 has no cp313
# wheels, so it cannot install on Kaggle's Python 3.13. 4.46.3 needs
# tokenizers 0.20.x, which does. Fall through the ladder until one sticks.
CANDIDATES = ["transformers==4.46.3", "transformers==4.48.3", "transformers==4.44.2"]
installed = None
for cand in CANDIDATES:
    print("trying", cand, flush=True)
    res = subprocess.run([PY, "-m", "pip", "install", "-q",
                          "amrlib==0.8.1", cand, "cached-property", "networkx",
                          "--only-binary", "tokenizers"],
                         capture_output=True, text=True)
    if res.returncode == 0:
        installed = cand
        print("installed", cand, f"({time.time()-t0:.0f}s)")
        break
    print("failed:", (res.stderr or res.stdout).strip().splitlines()[-1][:300])
assert installed, "amrlib could not be installed with any transformers 4.x"
# amrlib forgets to declare cached_property/networkx: import the parser module
# now so a missing dependency fails before the 1.5 GB checkpoint download.
subprocess.run([PY, "-c",
                "import amrlib.models.parse_spring.inference; "
                "print('parse_spring import ok')"], check=True)
subprocess.run([PY, "-c",
                "import sys, torch, transformers, amrlib, penman\\n"
                "print('python', sys.executable)\\n"
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\\n"
                "print('transformers', transformers.__version__)\\n"
                "print('amrlib', amrlib.__version__)"], check=True)
''', next_i())

    md("""## 5 · Download SPRING AMR3 weights — 1.5 GB

`model_parse_spring-v0_1_0` is the SPRING AMR3 checkpoint converted to HF
seq2seq (`facebook/bart-large` architecture). Verified size: 1,519,328,023 bytes.
""", next_i())

    code('''
SPRING_URL = ("https://github.com/bjascob/amrlib-models/releases/download/"
              "model_parse_spring-v0_1_0/model_parse_spring-v0_1_0.tar.gz")
os.makedirs(f"{WORK}/amrlib_models", exist_ok=True)
if os.path.isfile(f"{SPRING_MODEL}/model.pt"):
    print("already extracted:", SPRING_MODEL)
else:
    tgz = f"{WORK}/spring_amrlib.tar.gz"
    t0 = time.time()
    subprocess.run(["curl", "-L", "--fail", "--retry", "3",
                    "-o", tgz, SPRING_URL], check=True)
    size = os.path.getsize(tgz)
    print(f"downloaded {size/1e9:.2f} GB in {time.time()-t0:.0f}s")
    assert size > 1_400_000_000, f"checkpoint download truncated: {size} bytes"
    subprocess.run(["tar", "xzf", tgz, "-C", f"{WORK}/amrlib_models"], check=True)
    os.remove(tgz)
print(sorted(os.listdir(SPRING_MODEL)))
''', next_i())

    md(f"""## 6 · Parse {N_SENTENCES:,} sentences — writes `graphs.json`

Smoke-parses 5 sentences first, then works in chunks of 500 and appends to
`graphs.jsonl`, so a session restart resumes instead of starting over.
""", next_i())

    code(f'''
import base64, pathlib
STAGE_B64 = {{
    "amr_stage.py": "{stage_b64("amr_stage.py")}",
    "wiki_stage.py": "{stage_b64("wiki_stage.py")}",
}}
for name, blob in STAGE_B64.items():
    p = pathlib.Path(WORK) / name
    p.write_bytes(base64.b64decode(blob))
    print("wrote", p, p.stat().st_size, "bytes")
''', next_i())

    code('''
t0 = time.time()
subprocess.run([PY, f"{WORK}/amr_stage.py"], check=True)
print(f"parse stage done in {time.time()-t0:.0f}s")
''', next_i())

    md("""## 7 · Wikipedia linker — writes `linked.json`

Links only `:name` nodes, keeps a title only if it restates the mention, drops
disambiguation pages. Resolutions are cached in `wiki_cache.json`, so a rerun
does not hit the API again.
""", next_i())

    code('''
t0 = time.time()
subprocess.run([PY, f"{WORK}/wiki_stage.py"], check=True)
print(f"wiki stage done in {time.time()-t0:.0f}s")
''', next_i())

    md("""## 8 · Write outputs + zip""", next_i())

    code('''
import zipfile

rows = json.load(open(LINKED_JSON))
graphs = json.load(open(GRAPHS_JSON))["graphs"]
assert len(rows) == len(graphs) == len(json.load(open(PICKED_JSON)))

order = ("article_id", "para_id", "para_index", "sent_index", "sentence",
         "words", "amr", "status", "wiki", "title", "url", "source",
         "section", "tags")
with open(f"{OUTDIR}/{OUT_NAME}.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps({k: r.get(k) for k in order},
                            ensure_ascii=False) + "\\n")

with open(f"{OUTDIR}/{OUT_NAME}.amr", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(f"# ::id {r['para_id']}.{r['sent_index']}\\n")
        fh.write(f"# ::snt {r['sentence']}\\n")
        fh.write((r.get("amr") or "") + "\\n\\n")

stats = json.load(open(f"{OUTDIR}/split_stats.json"))
stats.update({
    "status": "ok",
    "n_rows": len(rows),
    "n_parsed": sum(1 for r in rows if r.get("amr")),
    "n_parse_failed": sum(1 for r in rows if not r.get("amr")),
    "n_sentences_with_wiki": sum(1 for r in rows if r.get("wiki")),
    "n_wiki_nodes": sum(len(r.get("wiki") or []) for r in rows),
    "spring_model": "amrlib model_parse_spring-v0_1_0 (SPRING AMR3)",
})
json.dump(stats, open(f"{OUTDIR}/run_meta.json", "w"), indent=2)
print(json.dumps(stats, indent=2))

zip_path = f"{WORK}/legal_amr_output.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in (f"{OUT_NAME}.jsonl", f"{OUT_NAME}.amr",
                 "run_meta.json", "split_stats.json"):
        z.write(f"{OUTDIR}/{name}", f"out/{name}")
print("zip:", zip_path, os.path.getsize(zip_path), "bytes")
''', next_i())

    md(f"""## 9 · Get the results out

The zip is at `/kaggle/working/legal_amr_output.zip` — download it from the
**Output** panel on the right (no "Save version" needed).

Then locally:

```bash
unzip legal_amr_output.zip -d data/kaggle_runs/manual-<date>
python kaggle_pipeline/check_output.py \\
    data/kaggle_runs/manual-<date>/out/sentences_{N_SENTENCES}.jsonl
```
""", next_i())

    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "datasetSources": ["mohar07/legal-amr-india-paragraphs"],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(HERE, "legal_amr_sat_spring_wiki.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(CELLS)} cells, N={N_SENTENCES})")


if __name__ == "__main__":
    build()
