#!/usr/bin/env python3
"""Build the CPU batch kernel: picked.json -> AMR + :wiki -> out/ + zip.

The kernel is a plain script (`kernel_type: script`) so it can be pushed with
`kaggle kernels push` and started without a GPU queue. Inputs: the private
dataset `mohar07/legal-amr-picked` (picked.json, 20k rows). The two stage
scripts are embedded as base64 so the kernel folder stays self-contained.
"""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(HERE, "notebook_stages")
OUT = os.path.join(HERE, "kernel_cpu")

TEMPLATE = '''"""CPU batch kernel: picked.json -> AMR graphs + Wikipedia :wiki -> out/.

No GPU is used, so this skips Kaggle's GPU queue entirely. Parsing is
checkpointed every 500 sentences and stops after AMR_MAX_HOURS (default 10) so
the run always finishes inside the batch limit with partial results.

Env knobs (kernel-metadata.json "kernel_sources" not needed):
  AMR_BEAMS      beam width for generation (default 2 on CPU, 5 on GPU)
  AMR_BATCH      comma-separated batch sizes to try (default 16,8,4)
  AMR_MAX_HOURS  wall-clock budget for parsing (default 10)
"""

import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

WORK, OUTDIR = "/kaggle/working", "/kaggle/working/out"
os.makedirs(OUTDIR, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("AMR_BEAMS", "2")
os.environ.setdefault("AMR_MAX_HOURS", "10")

print("python:", sys.version.split()[0], sys.executable, flush=True)

hits = sorted(glob.glob("/kaggle/input/**/picked.json", recursive=True),
              key=os.path.getsize, reverse=True)
assert hits, "attach dataset mohar07/legal-amr-picked as input"
shutil.copy(hits[0], f"{WORK}/picked.json")
picked = json.load(open(f"{WORK}/picked.json"))
print(len(picked), "sentences from", hits[0], flush=True)

# ---------------------------------------------------------------- dependencies
CANDIDATES = ["transformers==4.46.3", "transformers==4.48.3", "transformers==4.44.2"]
installed = None
for cand in CANDIDATES:
    print("installing amrlib +", cand, flush=True)
    res = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                          "amrlib==0.8.1", cand, "cached-property", "networkx",
                          "--only-binary", "tokenizers"],
                         capture_output=True, text=True)
    if res.returncode == 0:
        installed = cand
        break
    print("failed:", (res.stderr or res.stdout).strip().splitlines()[-1][:300],
          flush=True)
assert installed, "amrlib could not be installed with any transformers 4.x"
# amrlib 0.8.1 forgets to declare cached_property and networkx; import the
# parser module now so a missing dependency fails in seconds, not after the
# 1.5 GB checkpoint download.
subprocess.run([sys.executable, "-c",
                "import amrlib.models.parse_spring.inference; "
                "print('parse_spring import ok')"], check=True)
subprocess.run([sys.executable, "-c",
                "import torch, transformers, amrlib; "
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                "'transformers', transformers.__version__, "
                "'amrlib', amrlib.__version__)"], check=True)

# --------------------------------------------------------------- SPRING weights
SPRING_MODEL = f"{WORK}/amrlib_models/model_parse_spring-v0_1_0"
SPRING_URL = ("https://github.com/bjascob/amrlib-models/releases/download/"
              "model_parse_spring-v0_1_0/model_parse_spring-v0_1_0.tar.gz")
os.makedirs(f"{WORK}/amrlib_models", exist_ok=True)
if not os.path.isfile(f"{SPRING_MODEL}/model.pt"):
    tgz = f"{WORK}/spring.tar.gz"
    t0 = time.time()
    subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-o", tgz,
                    SPRING_URL], check=True)
    size = os.path.getsize(tgz)
    print(f"downloaded {size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    assert size > 1_400_000_000, f"truncated checkpoint: {size}"
    subprocess.run(["tar", "xzf", tgz, "-C", f"{WORK}/amrlib_models"], check=True)
    os.remove(tgz)

# --------------------------------------------------------------- stage scripts
STAGE_B64 = {STAGES}

for name, blob in STAGE_B64.items():
    with open(os.path.join(WORK, name), "wb") as fh:
        fh.write(base64.b64decode(blob))
    print("wrote", name, flush=True)

# ------------------------------------------------------------------- parse
t0 = time.time()
subprocess.run([sys.executable, f"{WORK}/amr_stage.py"], check=True)
print(f"parse stage done in {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------- wiki
t0 = time.time()
subprocess.run([sys.executable, f"{WORK}/wiki_stage.py"], check=True)
print(f"wiki stage done in {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------- outputs
rows = json.load(open(f"{WORK}/linked.json"))
graphs = json.load(open(f"{WORK}/graphs.json"))["graphs"]
assert len(rows) == len(graphs) == len(picked), (len(rows), len(graphs), len(picked))

OUT_NAME = f"sentences_{len(rows)}"
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

stats = {
    "status": "ok",
    "n_rows": len(rows),
    "n_requested": len(picked),
    "n_parsed": sum(1 for r in rows if r.get("amr")),
    "n_parse_failed": sum(1 for r in rows if not r.get("amr")),
    "n_sentences_with_wiki": sum(1 for r in rows if r.get("wiki")),
    "n_wiki_nodes": sum(len(r.get("wiki") or []) for r in rows),
    "n_articles": len({r["article_id"] for r in rows}),
    "amr_beams": int(os.environ["AMR_BEAMS"]),
    "amr_max_hours": float(os.environ["AMR_MAX_HOURS"]),
    "transformers": installed,
    "spring_model": "amrlib model_parse_spring-v0_1_0 (SPRING AMR3)",
    "device": "cpu",
}
json.dump(stats, open(f"{OUTDIR}/run_meta.json", "w"), indent=2)
print(json.dumps(stats, indent=2), flush=True)

zip_path = f"{WORK}/legal_amr_output.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in (f"{OUT_NAME}.jsonl", f"{OUT_NAME}.amr", "run_meta.json"):
        z.write(f"{OUTDIR}/{name}", f"out/{name}")
print("zip:", zip_path, os.path.getsize(zip_path), "bytes", flush=True)
'''

META = {
    "id": "mohar07/legal-amr-parse-cpu",
    "title": "legal-amr-parse-cpu",
    "code_file": "parse_cpu.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": ["mohar07/legal-amr-picked"],
    "competition_sources": [],
    "kernel_sources": [],
}


def main():
    os.makedirs(OUT, exist_ok=True)
    stages = {}
    for name in ("amr_stage.py", "wiki_stage.py"):
        with open(os.path.join(STAGES, name), "rb") as fh:
            stages[name] = base64.b64encode(fh.read()).decode("ascii")

    src = TEMPLATE.replace("{STAGES}", json.dumps(stages, indent=4))
    with open(os.path.join(OUT, "parse_cpu.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(META, fh, indent=2)
    print("wrote", os.path.join(OUT, "parse_cpu.py"),
          len(src), "bytes")


if __name__ == "__main__":
    main()
