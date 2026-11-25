# Data files

The agent needs one corpus database and, for a live run, the local SPRING
model. Neither is fetched automatically.

## Corpus: `india.db`

`data/india.db` is the India-only corpus (sources `indiacorplaw`, `scctimes`,
`scobserver`, `livelaw`, `barandbench`). It is built from a full corpus that
contains the German `verfassungsblog` material:

```bash
python make_india_db.py --source /path/to/legal.db --out data/india.db
```

The source database is not part of this repository. If you clone the repo,
either:

1. keep `data/india.db` in the clone (about 54 MB, under GitHub's 100 MB file
   limit, or use Git LFS), or
2. publish `data/india.db.gz` (about 19 MB) somewhere and document the
   download, or
3. provide a source `legal.db` and run the command above.

Check it before a run:

```bash
python - <<'PY'
import sqlite3
db = sqlite3.connect("data/india.db")
print(db.execute("select source, count(*) from articles group by source").fetchall())
print(db.execute("select count(*) from sentences").fetchone())
PY
```

## SPRING model

`spring_parse` uses `amrlib` and the `model_parse_spring-v0_1_0` checkpoint at:

```
~/.cache/amrlib/model_parse_spring-v0_1_0
```

The checkpoint is about 1.6 GB. Download and unpack it once:

```bash
mkdir -p ~/.cache/amrlib
curl -L -o /tmp/model_parse_spring-v0_1_0.tar.gz \
  https://github.com/bjascob/amrlib-models/releases/download/model_parse_spring-v0_1_0/model_parse_spring-v0_1_0.tar.gz
tar -xzf /tmp/model_parse_spring-v0_1_0.tar.gz -C ~/.cache/amrlib
rm /tmp/model_parse_spring-v0_1_0.tar.gz
ls ~/.cache/amrlib/model_parse_spring-v0_1_0
```

Use `requirements.txt` as-is: `amrlib==0.8.1` does not work with
`transformers` 5.x.

## Warm cache

A real run can compute base graphs on demand, but that is the slow part on CPU.
Precompute once with `precompute_spring.py`, then pass the jsonl to the agent:

```bash
python precompute_spring.py --all --batch-size 32 --out data/amr_spring.jsonl

python run.py --all --workers 100 --out data/runs/full \
  --base-graphs data/amr_spring.jsonl --build
```

When every selected sentence is in the cache, `run.py` does not load the SPRING
model at all.
