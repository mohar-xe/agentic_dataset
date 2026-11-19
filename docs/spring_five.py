#!/usr/bin/env python3
"""Parse the six demo sentences with the real local SPRING model.

SPRING is the base-graph parser used by the old pipeline. This script runs it
on CPU for exactly the six sentences in ``docs/five_sentences.py`` and writes
``data/five_spring/spring.jsonl`` in the format ``run.py --base-graphs`` reads.

    python docs/spring_five.py

No API is used: the model files are local (amrlib / SPRING v0.1.0).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "dataset_amr" / "scripts"))

from config import load_config  # noqa: E402
from corpus import Corpus  # noqa: E402
from common import validate_amr  # noqa: E402

ARTICLE = 5772
SENTENCES = [360428, 360429, 360439, 360450, 360465, 360477]
MODEL_DIR = Path.home() / ".cache" / "amrlib" / "model_parse_spring-v0_1_0"
OUT = ROOT / "data" / "five_spring"


def main() -> None:
    import amrlib

    cfg = load_config()
    corpus = Corpus(cfg["corpus_db"])
    article = corpus.get_article(ARTICLE)
    by_id = {s.id: s for s in article.sentences}
    batch = [by_id[sid] for sid in SENTENCES]

    print(f"loading SPRING from {MODEL_DIR} on cpu ...", flush=True)
    started = time.time()
    stog = amrlib.load_stog_model(
        model_dir=str(MODEL_DIR), device="cpu", batch_size=6, beam_size=5)
    print(f"loaded in {time.time() - started:.1f}s", flush=True)

    t0 = time.time()
    graphs = stog.parse_sents([s.text for s in batch], add_metadata=False)
    elapsed = time.time() - t0
    print(f"parsed {len(graphs)} sentences in {elapsed:.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with (OUT / "spring.jsonl").open("w", encoding="utf-8") as f:
        for sentence, graph in zip(batch, graphs):
            amr = str(graph)
            ok, errors, info = validate_amr(amr)
            row = {
                "sentence_id": sentence.id,
                "article_id": article.id,
                "sentence": sentence.text,
                "amr": amr,
                "status": "ok" if ok else "invalid",
                "errors": errors,
                "parser": "spring",
                "model": "spring-v0_1_0",
                "n_nodes": info.get("n_nodes"),
                "n_edges": info.get("n_edges"),
                "seconds": round(elapsed / len(batch), 2),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"\n--- {sentence.id} {row['status']} "
                  f"{info.get('n_nodes')} nodes ---\n{amr}")

    print(f"\nwrote {OUT / 'spring.jsonl'}")
    print(json.dumps([
        {"sentence_id": r["sentence_id"], "status": r["status"],
         "n_nodes": r["n_nodes"], "n_edges": r["n_edges"]}
        for r in rows], indent=1))


if __name__ == "__main__":
    main()
