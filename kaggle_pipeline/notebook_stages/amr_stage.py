"""Stage: parse the sampled sentences with SPRING (amrlib).

Runs with the notebook's venv interpreter. Inputs:
  /kaggle/working/picked.json      sampled sentences (stage 3)
  /kaggle/working/amrlib_models/model_parse_spring-v0_1_0

Writes:
  /kaggle/working/graphs.jsonl     one {"i": idx, "amr": str} per line, appended
                                   in chunks of CHUNK so a restart resumes
  /kaggle/working/graphs.json      {"graphs": [...], "parsed": int}

amrlib returns "()" when it has to back off; those become empty strings.
"""

import json
import os
import sys
import time
import traceback

WORK = "/kaggle/working"
PICKED_JSON = f"{WORK}/picked.json"
GRAPHS_JSON = f"{WORK}/graphs.json"
GRAPHS_JSONL = f"{WORK}/graphs.jsonl"
SPRING_MODEL = f"{WORK}/amrlib_models/model_parse_spring-v0_1_0"

CHUNK = 500           # sentences per checkpoint flush
BATCHES = tuple(int(x) for x in os.environ.get("AMR_BATCH", "16,8,4").split(","))
BEAMS = int(os.environ.get("AMR_BEAMS", "5"))
MAX_HOURS = float(os.environ.get("AMR_MAX_HOURS", "0"))  # 0 = no budget


def parse_chunk(stog, torch, sents):
    last = None
    for bs in BATCHES:
        stog.batch_size = bs
        try:
            return stog.parse_sents(sents, add_metadata=False,
                                    disable_progress=True)
        except torch.cuda.OutOfMemoryError as exc:
            last = exc
            torch.cuda.empty_cache()
            print(f"  CUDA OOM at batch_size={bs}, retrying smaller", flush=True)
    raise last


def main():
    picked = json.load(open(PICKED_JSON))
    sents = [s["sentence"] for s in picked]
    n = len(sents)
    print(f"{n} sentences to parse", flush=True)

    done = {}
    if os.path.isfile(GRAPHS_JSONL):
        with open(GRAPHS_JSONL, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[int(rec["i"])] = rec["amr"]
                except Exception:
                    continue
        print(f"resuming: {len(done)}/{n} already in {GRAPHS_JSONL}", flush=True)

    import amrlib
    import torch

    stog = amrlib.load_stog_model(model_dir=SPRING_MODEL)
    # amrlib's model_utils sets config.output_attentions = True. From
    # transformers 4.46 on, generate() sees that flag and returns a
    # GenerateBeamOutput dict instead of a LongTensor, so amrlib's own
    # decode_amr(tokk.tolist()) gets the string "sequences" and dies. Turn
    # every dict-returning flag off before parsing.
    for cfg in (getattr(stog, "config", None),
                getattr(stog.model, "config", None),
                getattr(stog.model, "generation_config", None)):
        if cfg is None:
            continue
        for attr in ("output_attentions", "output_scores", "output_hidden_states"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, False)
        if hasattr(cfg, "return_dict_in_generate"):
            cfg.return_dict_in_generate = False
    stog.num_beams = BEAMS
    print("SPRING loaded on", stog.device, "batch", stog.batch_size,
          "beams", BEAMS, flush=True)

    todo = [i for i in range(n) if i not in done]
    if todo:
        # smoke test before committing to the long run
        smoke = parse_chunk(stog, torch, sents[todo[0]:todo[0] + 5])
        assert all(isinstance(g, str) for g in smoke), \
            f"parse_sents returned {type(smoke[0]).__name__}, not str"
        for i, g in zip(todo[:5], smoke):
            print("\nSNT:", sents[i][:110])
            print((g or "")[:400], flush=True)

        t0 = time.time()
        with open(GRAPHS_JSONL, "a", encoding="utf-8") as out:
            for start in range(0, len(todo), CHUNK):
                idxs = todo[start:start + CHUNK]
                graphs = parse_chunk(stog, torch, [sents[i] for i in idxs])
                for i, amr in zip(idxs, graphs):
                    amr = (amr or "").strip()
                    if amr == "()":       # amrlib backoff
                        amr = ""
                    done[i] = amr
                    out.write(json.dumps({"i": i, "amr": amr}) + "\n")
                out.flush()
                elapsed = time.time() - t0
                rate = (start + len(idxs)) / max(elapsed, 1e-9)
                print(f"  {len(done)}/{n} parsed, {rate:.1f} sent/s, "
                      f"{elapsed/60:.1f} min elapsed", flush=True)
                if MAX_HOURS and elapsed > MAX_HOURS * 3600:
                    print(f"time budget {MAX_HOURS}h reached, stopping early",
                          flush=True)
                    break
    else:
        print("nothing to do, all sentences already parsed", flush=True)

    graphs = [done.get(i, "") for i in range(n)]
    parsed = sum(1 for g in graphs if g)
    json.dump({"graphs": graphs, "parsed": parsed}, open(GRAPHS_JSON, "w"))
    print(f"\nparsed {parsed}/{n}; empty (backoff) {n - parsed}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
