"""CPU batch kernel: picked.json -> AMR graphs + Wikipedia :wiki -> out/.

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
STAGE_B64 = {
    "amr_stage.py": "IiIiU3RhZ2U6IHBhcnNlIHRoZSBzYW1wbGVkIHNlbnRlbmNlcyB3aXRoIFNQUklORyAoYW1ybGliKS4KClJ1bnMgd2l0aCB0aGUgbm90ZWJvb2sncyB2ZW52IGludGVycHJldGVyLiBJbnB1dHM6CiAgL2thZ2dsZS93b3JraW5nL3BpY2tlZC5qc29uICAgICAgc2FtcGxlZCBzZW50ZW5jZXMgKHN0YWdlIDMpCiAgL2thZ2dsZS93b3JraW5nL2FtcmxpYl9tb2RlbHMvbW9kZWxfcGFyc2Vfc3ByaW5nLXYwXzFfMAoKV3JpdGVzOgogIC9rYWdnbGUvd29ya2luZy9ncmFwaHMuanNvbmwgICAgIG9uZSB7ImkiOiBpZHgsICJhbXIiOiBzdHJ9IHBlciBsaW5lLCBhcHBlbmRlZAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGluIGNodW5rcyBvZiBDSFVOSyBzbyBhIHJlc3RhcnQgcmVzdW1lcwogIC9rYWdnbGUvd29ya2luZy9ncmFwaHMuanNvbiAgICAgIHsiZ3JhcGhzIjogWy4uLl0sICJwYXJzZWQiOiBpbnR9CgphbXJsaWIgcmV0dXJucyAiKCkiIHdoZW4gaXQgaGFzIHRvIGJhY2sgb2ZmOyB0aG9zZSBiZWNvbWUgZW1wdHkgc3RyaW5ncy4KIiIiCgppbXBvcnQganNvbgppbXBvcnQgb3MKaW1wb3J0IHN5cwppbXBvcnQgdGltZQppbXBvcnQgdHJhY2ViYWNrCgpXT1JLID0gIi9rYWdnbGUvd29ya2luZyIKUElDS0VEX0pTT04gPSBmIntXT1JLfS9waWNrZWQuanNvbiIKR1JBUEhTX0pTT04gPSBmIntXT1JLfS9ncmFwaHMuanNvbiIKR1JBUEhTX0pTT05MID0gZiJ7V09SS30vZ3JhcGhzLmpzb25sIgpTUFJJTkdfTU9ERUwgPSBmIntXT1JLfS9hbXJsaWJfbW9kZWxzL21vZGVsX3BhcnNlX3NwcmluZy12MF8xXzAiCgpDSFVOSyA9IDUwMCAgICAgICAgICAgIyBzZW50ZW5jZXMgcGVyIGNoZWNrcG9pbnQgZmx1c2gKQkFUQ0hFUyA9IHR1cGxlKGludCh4KSBmb3IgeCBpbiBvcy5lbnZpcm9uLmdldCgiQU1SX0JBVENIIiwgIjE2LDgsNCIpLnNwbGl0KCIsIikpCkJFQU1TID0gaW50KG9zLmVudmlyb24uZ2V0KCJBTVJfQkVBTVMiLCAiNSIpKQpNQVhfSE9VUlMgPSBmbG9hdChvcy5lbnZpcm9uLmdldCgiQU1SX01BWF9IT1VSUyIsICIwIikpICAjIDAgPSBubyBidWRnZXQKCgpkZWYgcGFyc2VfY2h1bmsoc3RvZywgdG9yY2gsIHNlbnRzKToKICAgIGxhc3QgPSBOb25lCiAgICBmb3IgYnMgaW4gQkFUQ0hFUzoKICAgICAgICBzdG9nLmJhdGNoX3NpemUgPSBicwogICAgICAgIHRyeToKICAgICAgICAgICAgcmV0dXJuIHN0b2cucGFyc2Vfc2VudHMoc2VudHMsIGFkZF9tZXRhZGF0YT1GYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZGlzYWJsZV9wcm9ncmVzcz1UcnVlKQogICAgICAgIGV4Y2VwdCB0b3JjaC5jdWRhLk91dE9mTWVtb3J5RXJyb3IgYXMgZXhjOgogICAgICAgICAgICBsYXN0ID0gZXhjCiAgICAgICAgICAgIHRvcmNoLmN1ZGEuZW1wdHlfY2FjaGUoKQogICAgICAgICAgICBwcmludChmIiAgQ1VEQSBPT00gYXQgYmF0Y2hfc2l6ZT17YnN9LCByZXRyeWluZyBzbWFsbGVyIiwgZmx1c2g9VHJ1ZSkKICAgIHJhaXNlIGxhc3QKCgpkZWYgbWFpbigpOgogICAgcGlja2VkID0ganNvbi5sb2FkKG9wZW4oUElDS0VEX0pTT04pKQogICAgc2VudHMgPSBbc1sic2VudGVuY2UiXSBmb3IgcyBpbiBwaWNrZWRdCiAgICBuID0gbGVuKHNlbnRzKQogICAgcHJpbnQoZiJ7bn0gc2VudGVuY2VzIHRvIHBhcnNlIiwgZmx1c2g9VHJ1ZSkKCiAgICBkb25lID0ge30KICAgIGlmIG9zLnBhdGguaXNmaWxlKEdSQVBIU19KU09OTCk6CiAgICAgICAgd2l0aCBvcGVuKEdSQVBIU19KU09OTCwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZmg6CiAgICAgICAgICAgIGZvciBsaW5lIGluIGZoOgogICAgICAgICAgICAgICAgbGluZSA9IGxpbmUuc3RyaXAoKQogICAgICAgICAgICAgICAgaWYgbm90IGxpbmU6CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICByZWMgPSBqc29uLmxvYWRzKGxpbmUpCiAgICAgICAgICAgICAgICAgICAgZG9uZVtpbnQocmVjWyJpIl0pXSA9IHJlY1siYW1yIl0KICAgICAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICBwcmludChmInJlc3VtaW5nOiB7bGVuKGRvbmUpfS97bn0gYWxyZWFkeSBpbiB7R1JBUEhTX0pTT05MfSIsIGZsdXNoPVRydWUpCgogICAgaW1wb3J0IGFtcmxpYgogICAgaW1wb3J0IHRvcmNoCgogICAgc3RvZyA9IGFtcmxpYi5sb2FkX3N0b2dfbW9kZWwobW9kZWxfZGlyPVNQUklOR19NT0RFTCkKICAgICMgYW1ybGliJ3MgbW9kZWxfdXRpbHMgc2V0cyBjb25maWcub3V0cHV0X2F0dGVudGlvbnMgPSBUcnVlLiBGcm9tCiAgICAjIHRyYW5zZm9ybWVycyA0LjQ2IG9uLCBnZW5lcmF0ZSgpIHNlZXMgdGhhdCBmbGFnIGFuZCByZXR1cm5zIGEKICAgICMgR2VuZXJhdGVCZWFtT3V0cHV0IGRpY3QgaW5zdGVhZCBvZiBhIExvbmdUZW5zb3IsIHNvIGFtcmxpYidzIG93bgogICAgIyBkZWNvZGVfYW1yKHRva2sudG9saXN0KCkpIGdldHMgdGhlIHN0cmluZyAic2VxdWVuY2VzIiBhbmQgZGllcy4gVHVybgogICAgIyBldmVyeSBkaWN0LXJldHVybmluZyBmbGFnIG9mZiBiZWZvcmUgcGFyc2luZy4KICAgIGZvciBjZmcgaW4gKGdldGF0dHIoc3RvZywgImNvbmZpZyIsIE5vbmUpLAogICAgICAgICAgICAgICAgZ2V0YXR0cihzdG9nLm1vZGVsLCAiY29uZmlnIiwgTm9uZSksCiAgICAgICAgICAgICAgICBnZXRhdHRyKHN0b2cubW9kZWwsICJnZW5lcmF0aW9uX2NvbmZpZyIsIE5vbmUpKToKICAgICAgICBpZiBjZmcgaXMgTm9uZToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBmb3IgYXR0ciBpbiAoIm91dHB1dF9hdHRlbnRpb25zIiwgIm91dHB1dF9zY29yZXMiLCAib3V0cHV0X2hpZGRlbl9zdGF0ZXMiKToKICAgICAgICAgICAgaWYgaGFzYXR0cihjZmcsIGF0dHIpOgogICAgICAgICAgICAgICAgc2V0YXR0cihjZmcsIGF0dHIsIEZhbHNlKQogICAgICAgIGlmIGhhc2F0dHIoY2ZnLCAicmV0dXJuX2RpY3RfaW5fZ2VuZXJhdGUiKToKICAgICAgICAgICAgY2ZnLnJldHVybl9kaWN0X2luX2dlbmVyYXRlID0gRmFsc2UKICAgIHN0b2cubnVtX2JlYW1zID0gQkVBTVMKICAgIHByaW50KCJTUFJJTkcgbG9hZGVkIG9uIiwgc3RvZy5kZXZpY2UsICJiYXRjaCIsIHN0b2cuYmF0Y2hfc2l6ZSwKICAgICAgICAgICJiZWFtcyIsIEJFQU1TLCBmbHVzaD1UcnVlKQoKICAgIHRvZG8gPSBbaSBmb3IgaSBpbiByYW5nZShuKSBpZiBpIG5vdCBpbiBkb25lXQogICAgaWYgdG9kbzoKICAgICAgICAjIHNtb2tlIHRlc3QgYmVmb3JlIGNvbW1pdHRpbmcgdG8gdGhlIGxvbmcgcnVuCiAgICAgICAgc21va2UgPSBwYXJzZV9jaHVuayhzdG9nLCB0b3JjaCwgc2VudHNbdG9kb1swXTp0b2RvWzBdICsgNV0pCiAgICAgICAgYXNzZXJ0IGFsbChpc2luc3RhbmNlKGcsIHN0cikgZm9yIGcgaW4gc21va2UpLCBcCiAgICAgICAgICAgIGYicGFyc2Vfc2VudHMgcmV0dXJuZWQge3R5cGUoc21va2VbMF0pLl9fbmFtZV9ffSwgbm90IHN0ciIKICAgICAgICBmb3IgaSwgZyBpbiB6aXAodG9kb1s6NV0sIHNtb2tlKToKICAgICAgICAgICAgcHJpbnQoIlxuU05UOiIsIHNlbnRzW2ldWzoxMTBdKQogICAgICAgICAgICBwcmludCgoZyBvciAiIilbOjQwMF0sIGZsdXNoPVRydWUpCgogICAgICAgIHQwID0gdGltZS50aW1lKCkKICAgICAgICB3aXRoIG9wZW4oR1JBUEhTX0pTT05MLCAiYSIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIG91dDoKICAgICAgICAgICAgZm9yIHN0YXJ0IGluIHJhbmdlKDAsIGxlbih0b2RvKSwgQ0hVTkspOgogICAgICAgICAgICAgICAgaWR4cyA9IHRvZG9bc3RhcnQ6c3RhcnQgKyBDSFVOS10KICAgICAgICAgICAgICAgIGdyYXBocyA9IHBhcnNlX2NodW5rKHN0b2csIHRvcmNoLCBbc2VudHNbaV0gZm9yIGkgaW4gaWR4c10pCiAgICAgICAgICAgICAgICBmb3IgaSwgYW1yIGluIHppcChpZHhzLCBncmFwaHMpOgogICAgICAgICAgICAgICAgICAgIGFtciA9IChhbXIgb3IgIiIpLnN0cmlwKCkKICAgICAgICAgICAgICAgICAgICBpZiBhbXIgPT0gIigpIjogICAgICAgIyBhbXJsaWIgYmFja29mZgogICAgICAgICAgICAgICAgICAgICAgICBhbXIgPSAiIgogICAgICAgICAgICAgICAgICAgIGRvbmVbaV0gPSBhbXIKICAgICAgICAgICAgICAgICAgICBvdXQud3JpdGUoanNvbi5kdW1wcyh7ImkiOiBpLCAiYW1yIjogYW1yfSkgKyAiXG4iKQogICAgICAgICAgICAgICAgb3V0LmZsdXNoKCkKICAgICAgICAgICAgICAgIGVsYXBzZWQgPSB0aW1lLnRpbWUoKSAtIHQwCiAgICAgICAgICAgICAgICByYXRlID0gKHN0YXJ0ICsgbGVuKGlkeHMpKSAvIG1heChlbGFwc2VkLCAxZS05KQogICAgICAgICAgICAgICAgcHJpbnQoZiIgIHtsZW4oZG9uZSl9L3tufSBwYXJzZWQsIHtyYXRlOi4xZn0gc2VudC9zLCAiCiAgICAgICAgICAgICAgICAgICAgICBmIntlbGFwc2VkLzYwOi4xZn0gbWluIGVsYXBzZWQiLCBmbHVzaD1UcnVlKQogICAgICAgICAgICAgICAgaWYgTUFYX0hPVVJTIGFuZCBlbGFwc2VkID4gTUFYX0hPVVJTICogMzYwMDoKICAgICAgICAgICAgICAgICAgICBwcmludChmInRpbWUgYnVkZ2V0IHtNQVhfSE9VUlN9aCByZWFjaGVkLCBzdG9wcGluZyBlYXJseSIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgICAgICBicmVhawogICAgZWxzZToKICAgICAgICBwcmludCgibm90aGluZyB0byBkbywgYWxsIHNlbnRlbmNlcyBhbHJlYWR5IHBhcnNlZCIsIGZsdXNoPVRydWUpCgogICAgZ3JhcGhzID0gW2RvbmUuZ2V0KGksICIiKSBmb3IgaSBpbiByYW5nZShuKV0KICAgIHBhcnNlZCA9IHN1bSgxIGZvciBnIGluIGdyYXBocyBpZiBnKQogICAganNvbi5kdW1wKHsiZ3JhcGhzIjogZ3JhcGhzLCAicGFyc2VkIjogcGFyc2VkfSwgb3BlbihHUkFQSFNfSlNPTiwgInciKSkKICAgIHByaW50KGYiXG5wYXJzZWQge3BhcnNlZH0ve259OyBlbXB0eSAoYmFja29mZikge24gLSBwYXJzZWR9IiwgZmx1c2g9VHJ1ZSkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgdHJ5OgogICAgICAgIG1haW4oKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICB0cmFjZWJhY2sucHJpbnRfZXhjKCkKICAgICAgICBzeXMuZXhpdCgxKQo=",
    "wiki_stage.py": "IiIiU3RhZ2U6IGF0dGFjaCA6d2lraSBmcm9tIFdpa2lwZWRpYS4KClJ1bnMgd2l0aCB0aGUgbm90ZWJvb2sncyB2ZW52IGludGVycHJldGVyIChwZW5tYW4gKyBuZXR3b3JrKS4KSW5wdXRzOiAgL2thZ2dsZS93b3JraW5nL3BpY2tlZC5qc29uLCAva2FnZ2xlL3dvcmtpbmcvZ3JhcGhzLmpzb24KV3JpdGVzOiAgL2thZ2dsZS93b3JraW5nL2xpbmtlZC5qc29uICBsaXN0IG9mIHJvd3Mgd2l0aCAiYW1yIiAod2lraS1saW5rZWQpCiAgICAgICAgIGFuZCAid2lraSIgKFt7bm9kZSwgdGl0bGUsIG1lbnRpb259XSkuCgpMaW5raW5nIHN0cmF0ZWd5LCBwcmVjaXNpb24gb3ZlciByZWNhbGw6CjEuIGV4YWN0L25vcm1hbGlzZWQgdGl0bGUgbG9va3VwcywgYmF0Y2hlZCA1MCBtZW50aW9ucyBwZXIgQVBJIGNhbGw7CjIuIHNlYXJjaC1BUEkgZmFsbGJhY2sgZm9yIHRoZSByZW1haW5pbmcgbWVudGlvbnM7CjMuIGEgdGl0bGUgaXMga2VwdCBvbmx5IGlmIGl0IHJlc3RhdGVzIHRoZSBtZW50aW9uIChhbmQgY2FycmllcyBubyBleHRyYQogICBwYXJlbnRoZXRpY2FsIHF1YWxpZmllcik7CjQuIGRpc2FtYmlndWF0aW9uIHBhZ2VzIGFyZSBkcm9wcGVkLgoKUmVzb2x1dGlvbnMgYXJlIGNhY2hlZCBpbiB3aWtpX2NhY2hlLmpzb24sIHNvIGEgcmVydW4gbWFrZXMgYWxtb3N0IG5vIHJlcXVlc3RzLgoiIiIKCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgcmUKaW1wb3J0IHN5cwppbXBvcnQgdGltZQppbXBvcnQgdHJhY2ViYWNrCmltcG9ydCB1cmxsaWIucGFyc2UKaW1wb3J0IHVybGxpYi5yZXF1ZXN0CmZyb20gY29uY3VycmVudC5mdXR1cmVzIGltcG9ydCBUaHJlYWRQb29sRXhlY3V0b3IKCmltcG9ydCBwZW5tYW4KZnJvbSBwZW5tYW4ubW9kZWxzLmFtciBpbXBvcnQgbW9kZWwgYXMgYW1yX21vZGVsCgpXT1JLID0gIi9rYWdnbGUvd29ya2luZyIKUElDS0VEX0pTT04sIEdSQVBIU19KU09OLCBMSU5LRURfSlNPTiA9IChmIntXT1JLfS9waWNrZWQuanNvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZiJ7V09SS30vZ3JhcGhzLmpzb24iLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYie1dPUkt9L2xpbmtlZC5qc29uIikKQ0FDSEVfSlNPTiA9IGYie1dPUkt9L3dpa2lfY2FjaGUuanNvbiIKV0lLSV9BUEkgPSAiaHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3cvYXBpLnBocCIKVUEgPSAibGVnYWwtYW1yLXJlc2VhcmNoLzEuMCAoYWNhZGVtaWMgY29ycHVzIGFubm90YXRpb24pIgpXT1JLRVJTID0gOApCQVRDSCA9IDUwCgoKZGVmIF9nZXQodXJsLCB0cmllcz00KToKICAgIGxhc3QgPSBOb25lCiAgICBmb3IgYXR0ZW1wdCBpbiByYW5nZSh0cmllcyk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgaGVhZGVycz17IlVzZXItQWdlbnQiOiBVQX0pCiAgICAgICAgICAgIHdpdGggdXJsbGliLnJlcXVlc3QudXJsb3BlbihyZXEsIHRpbWVvdXQ9MjUpIGFzIHI6CiAgICAgICAgICAgICAgICByZXR1cm4ganNvbi5sb2FkcyhyLnJlYWQoKS5kZWNvZGUoInV0Zi04IikpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6ICAgICAgICAgICMgNDI5cyBhbmQgaGljY3VwczogYmFjayBvZmYsIHJldHJ5CiAgICAgICAgICAgIGxhc3QgPSBleGMKICAgICAgICAgICAgdGltZS5zbGVlcChtaW4oMS41ICogKGF0dGVtcHQgKyAxKSwgNikpCiAgICByYWlzZSBsYXN0CgoKZGVmIF9ub3JtKHRleHQpOgogICAgcmV0dXJuIHJlLnN1YihyIlteYS16MC05IF0rIiwgIiAiLCAodGV4dCBvciAiIikubG93ZXIoKSkuc3BsaXQoKQoKCmRlZiBfY29uZmlkZW50KG1lbnRpb24sIHRpdGxlKToKICAgICIiIkFjY2VwdCBvbmx5IHRpdGxlcyB0aGF0IHJlc3RhdGUgdGhlIG1lbnRpb24uIiIiCiAgICBpZiAiKCIgaW4gKHRpdGxlIG9yICIiKSBhbmQgIigiIG5vdCBpbiAobWVudGlvbiBvciAiIik6CiAgICAgICAgcmV0dXJuIEZhbHNlICAgICAgICAgICAgICAgICAgICAgICMgIk1lcmN1cnkgKHBsYW5ldCkiIGZvciAiTWVyY3VyeSIKICAgIG0sIHQgPSBfbm9ybShtZW50aW9uKSwgX25vcm0odGl0bGUpCiAgICBpZiBub3QgbSBvciBub3QgdDoKICAgICAgICByZXR1cm4gRmFsc2UKICAgIHJldHVybiBtID09IHQgb3IgKGxlbihtKSA+PSAyIGFuZCBzZXQobSkgPD0gc2V0KHQpKQoKCmRlZiBleGFjdF90aXRsZXMobWVudGlvbnMpOgogICAgIiIibWVudGlvbiAtPiB0aXRsZSB2aWEgYmF0Y2hlZCBleGFjdCB0aXRsZSBsb29rdXBzICg1MCBwZXIgcmVxdWVzdCkuIiIiCiAgICBmb3VuZCA9IHt9CiAgICBmb3IgaSBpbiByYW5nZSgwLCBsZW4obWVudGlvbnMpLCBCQVRDSCk6CiAgICAgICAgY2h1bmsgPSBtZW50aW9uc1tpOmkgKyBCQVRDSF0KICAgICAgICBxID0gdXJsbGliLnBhcnNlLnVybGVuY29kZSh7ImFjdGlvbiI6ICJxdWVyeSIsICJmb3JtYXQiOiAianNvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJyZWRpcmVjdHMiOiAxLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAidGl0bGVzIjogInwiLmpvaW4oY2h1bmspfSkKICAgICAgICB0cnk6CiAgICAgICAgICAgIGRhdGEgPSBfZ2V0KGYie1dJS0lfQVBJfT97cX0iKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHRyYWNlYmFjay5wcmludF9leGMoKQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHF1ZXJ5ID0gZGF0YS5nZXQoInF1ZXJ5Iiwge30pCiAgICAgICAgbm9ybSA9IHtuWyJmcm9tIl06IG5bInRvIl0gZm9yIG4gaW4gcXVlcnkuZ2V0KCJub3JtYWxpemVkIiwgW10pfQogICAgICAgIHJlZGlyID0ge3JbImZyb20iXTogclsidG8iXSBmb3IgciBpbiBxdWVyeS5nZXQoInJlZGlyZWN0cyIsIFtdKX0KICAgICAgICBieV90aXRsZSA9IHt9CiAgICAgICAgZm9yIG0gaW4gY2h1bms6CiAgICAgICAgICAgIHQgPSByZWRpci5nZXQobm9ybS5nZXQobSwgbSksIG5vcm0uZ2V0KG0sIG0pKQogICAgICAgICAgICBieV90aXRsZS5zZXRkZWZhdWx0KHQubG93ZXIoKSwgW10pLmFwcGVuZChtKQogICAgICAgIGZvciBwYWdlIGluIHF1ZXJ5LmdldCgicGFnZXMiLCB7fSkudmFsdWVzKCk6CiAgICAgICAgICAgIGlmICJtaXNzaW5nIiBpbiBwYWdlOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgdGl0bGUgPSBwYWdlLmdldCgidGl0bGUiKSBvciAiIgogICAgICAgICAgICBmb3IgbSBpbiBieV90aXRsZS5nZXQodGl0bGUubG93ZXIoKSwgW10pOgogICAgICAgICAgICAgICAgaWYgX2NvbmZpZGVudChtLCB0aXRsZSk6CiAgICAgICAgICAgICAgICAgICAgZm91bmRbbV0gPSB0aXRsZQogICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICByZXR1cm4gZm91bmQKCgpkZWYgc2VhcmNoX3dpa2kobWVudGlvbik6CiAgICBxID0gdXJsbGliLnBhcnNlLnVybGVuY29kZSh7ImFjdGlvbiI6ICJxdWVyeSIsICJmb3JtYXQiOiAianNvbiIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImxpc3QiOiAic2VhcmNoIiwgInNyc2VhcmNoIjogbWVudGlvbiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAic3JsaW1pdCI6IDUsICJyZWRpcmVjdHMiOiAxfSkKICAgIHRyeToKICAgICAgICBoaXRzID0gX2dldChmIntXSUtJX0FQSX0/e3F9IilbInF1ZXJ5Il1bInNlYXJjaCJdCiAgICBleGNlcHQgRXhjZXB0aW9uOiAgICAgICAgICAgICAgICAgICAgICMgbmV0d29yayBoaWNjdXA6IGxlYXZlIHVubGlua2VkCiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGZvciBoaXQgaW4gaGl0czoKICAgICAgICBpZiBfY29uZmlkZW50KG1lbnRpb24sIGhpdFsidGl0bGUiXSk6CiAgICAgICAgICAgIHJldHVybiBoaXRbInRpdGxlIl0KICAgIHJldHVybiBOb25lCgoKZGVmIGRpc2FtYmlndWF0aW9uX3RpdGxlcyh0aXRsZXMpOgogICAgb3V0ID0gc2V0KCkKICAgIHRpdGxlcyA9IHNvcnRlZCh0aXRsZXMpCiAgICBmb3IgaSBpbiByYW5nZSgwLCBsZW4odGl0bGVzKSwgQkFUQ0gpOgogICAgICAgIHEgPSB1cmxsaWIucGFyc2UudXJsZW5jb2RlKHsKICAgICAgICAgICAgImFjdGlvbiI6ICJxdWVyeSIsICJmb3JtYXQiOiAianNvbiIsICJyZWRpcmVjdHMiOiAxLAogICAgICAgICAgICAicHJvcCI6ICJwYWdlcHJvcHMiLCAicHBwcm9wIjogImRpc2FtYmlndWF0aW9uIiwKICAgICAgICAgICAgInRpdGxlcyI6ICJ8Ii5qb2luKHRpdGxlc1tpOmkgKyBCQVRDSF0pfSkKICAgICAgICB0cnk6CiAgICAgICAgICAgIHBhZ2VzID0gX2dldChmIntXSUtJX0FQSX0/e3F9IilbInF1ZXJ5Il1bInBhZ2VzIl0KICAgICAgICAgICAgZm9yIHBhZ2UgaW4gcGFnZXMudmFsdWVzKCk6CiAgICAgICAgICAgICAgICBpZiAiZGlzYW1iaWd1YXRpb24iIGluIChwYWdlLmdldCgicGFnZXByb3BzIikgb3Ige30pOgogICAgICAgICAgICAgICAgICAgIG91dC5hZGQocGFnZVsidGl0bGUiXSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICB0cmFjZWJhY2sucHJpbnRfZXhjKCkKICAgIHJldHVybiBvdXQKCgpkZWYgX3Zhcihub2RlKToKICAgICIiIlBlbm1hbiB0cmlwbGVzIG1heSBjYXJyeSAodmFyaWFibGUsIGFsaWdubWVudCkgdHVwbGVzLiIiIgogICAgcmV0dXJuIG5vZGVbMF0gaWYgaXNpbnN0YW5jZShub2RlLCB0dXBsZSkgZWxzZSBub2RlCgoKZGVmIG5hbWVfbWVudGlvbnMoYW1yKToKICAgICIiIih2YXJpYWJsZSwgbWVudGlvbikgZm9yIDpuYW1lIG5vZGVzIHdpdGhvdXQgOndpa2kgeWV0LiIiIgogICAgaWYgbm90IGFtcjoKICAgICAgICByZXR1cm4gW10KICAgIHRyeToKICAgICAgICBnID0gcGVubWFuLmRlY29kZShhbXIsIG1vZGVsPWFtcl9tb2RlbCkKICAgIGV4Y2VwdCBFeGNlcHRpb246ICAgICAgICAgICAgICAgICAgICAgIyBtYWxmb3JtZWQgZ3JhcGggZnJvbSBhIGJhY2tvZmYKICAgICAgICByZXR1cm4gW10KICAgIGhhc193aWtpID0ge192YXIocykgZm9yIHMsIHIsIF8gaW4gZy50cmlwbGVzIGlmIHIgPT0gIjp3aWtpIn0KICAgIG9wcyA9IHt9CiAgICBmb3IgcywgciwgdCBpbiBnLnRyaXBsZXM6CiAgICAgICAgbSA9IHJlLmZ1bGxtYXRjaChyIjpvcChcZCspIiwgcikKICAgICAgICBpZiBtOgogICAgICAgICAgICBvcHMuc2V0ZGVmYXVsdChfdmFyKHMpLCBbXSkuYXBwZW5kKAogICAgICAgICAgICAgICAgKGludChtLmdyb3VwKDEpKSwgc3RyKHQpLnN0cmlwKCciJykpKQogICAgb3V0ID0gW10KICAgIGZvciBzLCByLCB0IGluIGcudHJpcGxlczoKICAgICAgICBpZiByICE9ICI6bmFtZSI6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgdmFyLCB3b3JkcyA9IF92YXIocyksIG9wcy5nZXQoX3Zhcih0KSwgW10pCiAgICAgICAgaWYgd29yZHMgYW5kIHZhciBub3QgaW4gaGFzX3dpa2k6CiAgICAgICAgICAgIG91dC5hcHBlbmQoKHZhciwgIiAiLmpvaW4odyBmb3IgXywgdyBpbiBzb3J0ZWQod29yZHMpKSkpCiAgICByZXR1cm4gb3V0CgoKZGVmIGFkZF93aWtpKGFtciwgbGlua3MpOgogICAgZyA9IHBlbm1hbi5kZWNvZGUoYW1yLCBtb2RlbD1hbXJfbW9kZWwpCiAgICB0cmlwbGVzID0gbGlzdChnLnRyaXBsZXMpCiAgICBmb3Igbm9kZSwgdGl0bGUgaW4gbGlua3M6CiAgICAgICAgdHJpcGxlcy5hcHBlbmQocGVubWFuLlRyaXBsZShub2RlLCAiOndpa2kiLCBmJyJ7dGl0bGV9IicpKQogICAgcmV0dXJuIHBlbm1hbi5lbmNvZGUocGVubWFuLkdyYXBoKHRyaXBsZXMsIHRvcD1nLnRvcCwgbWV0YWRhdGE9Zy5tZXRhZGF0YSksCiAgICAgICAgICAgICAgICAgICAgICAgICBtb2RlbD1hbXJfbW9kZWwpCgoKZGVmIG1haW4oKToKICAgIHBpY2tlZCA9IGpzb24ubG9hZChvcGVuKFBJQ0tFRF9KU09OKSkKICAgIGdyYXBocyA9IGpzb24ubG9hZChvcGVuKEdSQVBIU19KU09OKSlbImdyYXBocyJdCiAgICBjYWNoZSA9IHt9CiAgICBpZiBvcy5wYXRoLmlzZmlsZShDQUNIRV9KU09OKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIGNhY2hlID0ganNvbi5sb2FkKG9wZW4oQ0FDSEVfSlNPTikpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgY2FjaGUgPSB7fQoKICAgIG1lbnRpb25zX2J5X3JvdyA9IFtuYW1lX21lbnRpb25zKGEpIGZvciBhIGluIGdyYXBoc10KICAgIGFsbF9tZW50aW9ucyA9IHNvcnRlZCh7bSBmb3Igcm93IGluIG1lbnRpb25zX2J5X3JvdyBmb3IgXywgbSBpbiByb3d9KQogICAgbWlzc2luZyA9IFttIGZvciBtIGluIGFsbF9tZW50aW9ucyBpZiBtIG5vdCBpbiBjYWNoZV0KICAgIHByaW50KGYie2xlbihhbGxfbWVudGlvbnMpfSB1bmlxdWUgZW50aXR5IG1lbnRpb25zLCAiCiAgICAgICAgICBmIntsZW4obWlzc2luZyl9IG5vdCBjYWNoZWQiLCBmbHVzaD1UcnVlKQoKICAgIGlmIG1pc3Npbmc6CiAgICAgICAgdDAgPSB0aW1lLnRpbWUoKQogICAgICAgIGZvdW5kID0gZXhhY3RfdGl0bGVzKG1pc3NpbmcpCiAgICAgICAgcHJpbnQoZiJleGFjdC10aXRsZSBoaXRzOiB7bGVuKGZvdW5kKX0ve2xlbihtaXNzaW5nKX0gIgogICAgICAgICAgICAgIGYiKHt0aW1lLnRpbWUoKS10MDouMGZ9cykiLCBmbHVzaD1UcnVlKQoKICAgICAgICByZXN0ID0gW20gZm9yIG0gaW4gbWlzc2luZyBpZiBtIG5vdCBpbiBmb3VuZF0KICAgICAgICB0MCA9IHRpbWUudGltZSgpCiAgICAgICAgd2l0aCBUaHJlYWRQb29sRXhlY3V0b3IobWF4X3dvcmtlcnM9V09SS0VSUykgYXMgcG9vbDoKICAgICAgICAgICAgc2VhcmNoZWQgPSBkaWN0KHppcChyZXN0LCBwb29sLm1hcChzZWFyY2hfd2lraSwgcmVzdCkpKQogICAgICAgIHByaW50KGYic2VhcmNoIGZhbGxiYWNrOiB7c3VtKDEgZm9yIHYgaW4gc2VhcmNoZWQudmFsdWVzKCkgaWYgdil9IgogICAgICAgICAgICAgIGYiL3tsZW4ocmVzdCl9ICh7dGltZS50aW1lKCktdDA6LjBmfXMpIiwgZmx1c2g9VHJ1ZSkKCiAgICAgICAgZm9yIG0gaW4gbWlzc2luZzoKICAgICAgICAgICAgY2FjaGVbbV0gPSBmb3VuZC5nZXQobSkgb3Igc2VhcmNoZWQuZ2V0KG0pCiAgICAgICAganNvbi5kdW1wKGNhY2hlLCBvcGVuKENBQ0hFX0pTT04sICJ3IikpCgogICAgdGl0bGVzID0ge3QgZm9yIHQgaW4gY2FjaGUudmFsdWVzKCkgaWYgdH0KICAgIGRpc2FtYmlnID0gZGlzYW1iaWd1YXRpb25fdGl0bGVzKHRpdGxlcykgaWYgdGl0bGVzIGVsc2Ugc2V0KCkKICAgIHByaW50KGYie2xlbihkaXNhbWJpZyl9IGRpc2FtYmlndWF0aW9uIHBhZ2VzIGRyb3BwZWQiLCBmbHVzaD1UcnVlKQoKICAgIGxpbmtlZF9yb3dzID0gbGlua2VkX25vZGVzID0gMAogICAgb3V0X3Jvd3MgPSBbXQogICAgZm9yIHJvdywgYW1yLCBtZW50aW9ucyBpbiB6aXAocGlja2VkLCBncmFwaHMsIG1lbnRpb25zX2J5X3Jvdyk6CiAgICAgICAgbGlua3MsIHdpa2kgPSBbXSwgW10KICAgICAgICBmb3Igbm9kZSwgbWVudGlvbiBpbiBtZW50aW9uczoKICAgICAgICAgICAgdGl0bGUgPSBjYWNoZS5nZXQobWVudGlvbikKICAgICAgICAgICAgaWYgdGl0bGUgYW5kIHRpdGxlIG5vdCBpbiBkaXNhbWJpZzoKICAgICAgICAgICAgICAgIGxpbmtzLmFwcGVuZCgobm9kZSwgdGl0bGUpKQogICAgICAgICAgICAgICAgd2lraS5hcHBlbmQoeyJub2RlIjogbm9kZSwgInRpdGxlIjogdGl0bGUsICJtZW50aW9uIjogbWVudGlvbn0pCiAgICAgICAgaWYgbGlua3M6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHJvd1siYW1yIl0gPSBhZGRfd2lraShhbXIsIGxpbmtzKQogICAgICAgICAgICAgICAgbGlua2VkX3Jvd3MgKz0gMQogICAgICAgICAgICAgICAgbGlua2VkX25vZGVzICs9IGxlbihsaW5rcykKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHRyYWNlYmFjay5wcmludF9leGMoKQogICAgICAgICAgICAgICAgcm93WyJhbXIiXSA9IGFtcgogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHJvd1siYW1yIl0gPSBhbXIKICAgICAgICByb3dbIndpa2kiXSA9IHdpa2kKICAgICAgICBvdXRfcm93cy5hcHBlbmQocm93KQoKICAgIGpzb24uZHVtcChvdXRfcm93cywgb3BlbihMSU5LRURfSlNPTiwgInciKSkKICAgIHByaW50KGYie2xpbmtlZF9ub2Rlc30gd2lraSBub2RlcyBpbiB7bGlua2VkX3Jvd3N9L3tsZW4ob3V0X3Jvd3MpfSAiCiAgICAgICAgICBmInNlbnRlbmNlcyIsIGZsdXNoPVRydWUpCiAgICBmb3IgciBpbiBvdXRfcm93czoKICAgICAgICBpZiByLmdldCgid2lraSIpOgogICAgICAgICAgICBwcmludCgiXG4iLCByWyJzZW50ZW5jZSJdWzoxMDBdLCAiLT4iLCByWyJ3aWtpIl0pCiAgICAgICAgICAgIHByaW50KHJbImFtciJdWzozMDBdKQogICAgICAgICAgICBicmVhawoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICB0cnk6CiAgICAgICAgbWFpbigpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHRyYWNlYmFjay5wcmludF9leGMoKQogICAgICAgIHN5cy5leGl0KDEpCg=="
}

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
                            ensure_ascii=False) + "\n")

with open(f"{OUTDIR}/{OUT_NAME}.amr", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(f"# ::id {r['para_id']}.{r['sent_index']}\n")
        fh.write(f"# ::snt {r['sentence']}\n")
        fh.write((r.get("amr") or "") + "\n\n")

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
