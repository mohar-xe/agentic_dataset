#!/usr/bin/env python3
"""Turn a wiki-linked ``linked.json`` into the published dataset files.

Writes three files next to ``--out``:

  sentences_<N>.jsonl  one row per sentence, fields in the pipeline order
  sentences_<N>.amr    # ::id / # ::snt + graph blocks
  run_meta.json        counts and provenance

The linker (``notebook_stages/wiki_stage.py``) writes ``linked.json``; this
script is the single place that decides what the published row looks like, so
the notebook, the CPU kernel and a local rebuild all stay byte-compatible.

Usage:
  python kaggle_pipeline/export_linked.py data/precomputed_15k/linked.json \
      --out data/precomputed_15k/out
"""

import argparse
import json
import os
import zipfile

ORDER = ("article_id", "para_id", "para_index", "sent_index", "sentence",
         "words", "amr", "status", "wiki", "title", "url", "source",
         "section", "tags")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("linked", help="linked.json produced by wiki_stage.py")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--zip", action="store_true", help="also write a zip bundle")
    ap.add_argument("--source", default="local",
                    help="provenance note stored in run_meta.json")
    args = ap.parse_args()

    rows = json.load(open(args.linked, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    name = f"sentences_{len(rows)}"

    with open(os.path.join(args.out, f"{name}.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in ORDER},
                                ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, f"{name}.amr"), "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"# ::id {r['para_id']}.{r['sent_index']}\n")
            fh.write(f"# ::snt {r['sentence']}\n")
            fh.write((r.get("amr") or "") + "\n\n")

    meta = {
        "n_rows": len(rows),
        "n_parsed": sum(1 for r in rows if r.get("amr")),
        "n_parse_failed": sum(1 for r in rows if not r.get("amr")),
        "n_sentences_with_wiki": sum(1 for r in rows if r.get("wiki")),
        "n_wiki_nodes": sum(len(r.get("wiki") or []) for r in rows),
        "n_articles": len({r["article_id"] for r in rows}),
        "n_words": sum(int(r.get("words") or 0) for r in rows),
        "sources": args.source,
        "spring_model": "amrlib model_parse_spring-v0_1_0 (SPRING AMR3)",
    }
    json.dump(meta, open(os.path.join(args.out, "run_meta.json"), "w"),
              indent=2)
    print(json.dumps(meta, indent=2))

    if args.zip:
        with zipfile.ZipFile(os.path.join(args.out, "legal_amr_output.zip"),
                             "w", zipfile.ZIP_DEFLATED) as z:
            for f in (f"{name}.jsonl", f"{name}.amr", "run_meta.json"):
                z.write(os.path.join(args.out, f), f"out/{f}")
        print("zip written")


if __name__ == "__main__":
    main()
