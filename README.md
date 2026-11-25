# Agentic Legal-AMR dataset

One thin agent per article. The agent receives **whole paragraphs, unsplit**, in
article order. It splits each paragraph into sentences itself, calls
`spring_parse` once per sentence (all calls for a paragraph in one turn, so they
run in parallel), and edits the graph that comes back. The agent never sees a
corpus sentence id: the harness matches its sentences back to the corpus text
when the answer arrives. It keeps an article-level ledger of resolved entities
and can call tools while it works:

| Tool | What it does |
| --- | --- |
| `spring_parse` | run the local SPRING parser on one sentence: sentence in, classic AMR out |
| `article_search` | find where something was declared/used in this article (`before` is sentence text) |
| `resolve_mention` | resolve “the Act”/“the Board” from earlier declarations, article metadata and the ledger (`sentence` is sentence text) |
| `wiki_lookup` | live Wikipedia/Wikidata lookup (always cached) |
| `check_amr` | validate a graph + the sentence-rewrite contract |

SPRING is no longer run ahead of time: it is a tool the agent calls, and a
sentence the agent never parsed cannot be annotated. `--base-graphs` is now only
a warm cache for `spring_parse` (graphs from an earlier run). |

**Repair is built in.** When a graph fails validation the same conversation is
sent back with the validator's errors and up to `agent.max_repairs` corrected
attempts are requested — the way a coding agent repairs a failing build.
Genuine resolution failures (no credible source, tied candidates) are rejected,
never guessed.

## Run

```bash
cd agentic_dataset
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# no API key: scripted pass over a real article
python run.py --article 5772 --mock --build --example

# real model, after config.json or AMR_* environment variables
python run.py --article 5772 --build --example

# many articles, in parallel, split across machines
python run.py --all --limit 500 --workers 8 --shard 1/4 --out data/runs/part1
```

`python run.py --all --workers N` runs articles in parallel; `--shard i/n`
selects articles by `article_id % n == i-1`. `--out` chooses the output
directory; use a fresh one for each run so the tracked demo files in
`data/agent/` are not mixed with a new dataset. Every article is independent
and resumable, so workers and machines never need to coordinate.

## Handover run: bundled graphs, `main.py`

`data/precomputed/graphs.jsonl` is the Kaggle SPRING output matched 1:1 to
`data/india.db` sentence ids (12,701 graphs, one JSON row per sentence). The
agent runs on those sentences with the graph already in the `spring_parse`
cache, so the 1.6 GB SPRING model is never loaded.

Use the SPRING venv (it has `penman`, `amrlib`, `transformers`):

```bash
~/.cache/amrlib/venv/bin/python main.py --dry-run      # coverage + tag counts
~/.cache/amrlib/venv/bin/python main.py --agents 4 --limit 100
~/.cache/amrlib/venv/bin/python main.py --agents 8     # all remaining graphs
```

- `--agents N` — how many agents run in parallel; one article's sentences stay
  on one agent so its ledger stays coherent.
- `--tag TAG` — repeatable work filter. A sentence carries its source
  (`livelaw`, `indiacorplaw`, `scctimes`, `scobserver`, `barandbench`) and
  `statute:<name>`, `provision:<name>`, `precedent:<name>` from the picked row.
  Matching is case-insensitive substring: `--tag arbitration`,
  `--tag statute:companies act`, `--tag scctimes`.
- `--limit N` — annotate at most N more sentences in this invocation.
- `--out DIR` — output directory (default `data/runs/main`).
- `--build` — write `dataset.all.jsonl` / `dataset.all.amr` at the end.

Resume is by output, not by counter: a sentence already in `records.jsonl` or
`rejected.jsonl` is skipped. Rerun the same command to continue; raising
`--limit` or adding tags only adds work. Outputs: `records.jsonl`,
`rejected.jsonl`, `events/`, `run_summary.json`.

`build_precomputed.py` rebuilds the bundle from `data/precomputed_15k/`
(`picked.json` + `graphs.json`): each picked row is joined to a corpus sentence
by exact paragraph + text, and only unambiguous matches are kept. 2,299 Kaggle
rows are SaT re-segmentations of a longer corpus sentence, so they cannot be
keyed by `sentence_id`; they are skipped instead of being mis-matched.
`--check` reports without writing. 359 matched graphs that SPRING itself marks
invalid are kept: the agent prompt treats an invalid SPRING graph as the
starting point, not a reason to drop the sentence.

## First run from a fresh clone

1. **Python and packages.** Python 3.11 is recommended. Install
   `requirements.txt` as shown above. Do not upgrade `transformers` past the
   pin: `amrlib` 0.8.1 still uses a tokenizer attribute that 5.x removed.
2. **Settings.** Copy `config.example.json` to `config.json`, then set
   `llm.api_key` and `model`. `config.json` is ignored by git. Environment
   variables (`AMR_API_URL`, `AMR_API_KEY`, `AMR_MODEL`) win over the file.
3. **Corpus.** `data/india.db` is the India-only database. It is not built
   automatically; see `data/README.md`. A clone must either include the file,
   download it, or build it from a source `legal.db` with
   `python make_india_db.py --source /path/to/legal.db`.
4. **SPRING.** Download the 1.6 GB checkpoint to
   `~/.cache/amrlib/model_parse_spring-v0_1_0`; the exact `curl` command is in
   `data/README.md`. A mock run does not need it.
5. **Smoke test.** `python run.py --article 5772 --mock --build --example`
   should finish with 53 accepted sentences and write
   `data/agent/examples/article_5772.md`.

## Running 100 article agents in parallel

`--workers` is a thread pool inside **one** process. Each thread runs one
article, so LLM requests overlap. There is exactly one local SPRING model per
process, and `spring_parse` is protected by a lock. That has two consequences:

- If the base graphs are precomputed (`--base-graphs`), `spring_parse` is a
  cache hit and `--workers 100` is limited by the LLM. `run.py` skips loading
  the model when every selected sentence is cached.
- Without a cache, all 100 threads queue on one parser. On a small CPU box a
  single sentence at beam 5 can take several seconds, so precompute before a
  large run.

```bash
# one-time, batched and resumable; can be sharded across machines
python precompute_spring.py --all --batch-size 32 --out data/amr_spring.jsonl

# 100 concurrent article agents, no local parser load when the cache is complete
python run.py --all --workers 100 --out data/runs/full \
  --base-graphs data/amr_spring.jsonl --build
```

If you need several parser processes instead, shard across machines and give
each one its own `--shard`, `--out` and `--base-graphs`. Do not start 100
processes on one machine: the checkpoint is 1.6 GB and each process loads its
own copy.

100 concurrent LLM calls must also fit the provider's rate limit. `llm.py`
retries HTTP failures with backoff, but a lower `--workers` or a higher-tier
key is simpler than fighting 429s. Live wiki lookups are cached; for a cold
large run consider `"wiki": {"enabled": false}` or warm the cache first.

The cache is keyed by **corpus sentence id**. `spring_parse` matches the
agent's sentence text to a corpus sentence (punctuation-insensitive), then
looks up that id. The corpus already has one row per sentence, so you do not
need a separate splitter to precompute. If you replace the splitter, make its
boundaries the canonical ones and rebuild the `sentences` rows/ids; otherwise
some calls miss the cache and fall back to the live model. Each
`spring_parse` result now says `"source": "cache"` or `"source": "model"`, so a
pilot run shows immediately whether the cache is being hit.

## Wikipedia lookups

`wiki.py` talks to two read-only endpoints:

```
https://en.wikipedia.org/w/api.php    action=query (title, redirects, search)
https://www.wikidata.org/w/api.php    action=wbsearchentities
```

Responses are cached in SQLite, so a repeated mention is free. Since the 2026
Wikimedia API rate limits, unauthenticated clients with a compliant
`User-Agent` get **200 requests/minute**; unidentified traffic gets 10.
Wikimedia also asks for **at most 3 concurrent requests** and a `User-Agent`
with contact details. `wiki.py` therefore defaults to 3 concurrent requests
and a 180/minute client-side pace, honours `Retry-After` on 429/503, and does
not cache failures.

Set your contact in `config.json` (or `AMR_WIKI_UA`):

```json
"wiki": {"user_agent": "agentic-dataset-amr/0.1 (https://example.org/you; you@example.org)"}
```

200/minute is about 12,000 lookups an hour (the default 180/minute pace is
about 10,800). For a cold large run, either accept the pacing, warm the cache
with a pilot first, use `"wiki": {"enabled": false}`, or authenticate the
client. Requests that exceed the limit receive HTTP 429 with `Retry-After`;
raising `--workers` will not raise the Wikimedia limit.

## Configuration and resources

Defaults are in `config.py`. A local `config.json` (copy
`config.example.json`) overrides them. Priority:

```
defaults < config.json < environment < CLI
```

Environment: `AMR_API_URL`, `AMR_API_KEY`, `AMR_MODEL`.

DBs and inputs:

- corpus: `data/india.db` (India-only: `livelaw`, `indiacorplaw`,
  `scctimes`, `scobserver`, `barandbench`; build it from a source DB with
  `python make_india_db.py --source /path/to/legal.db`; see `data/README.md`)
- SPRING model: `~/.cache/amrlib/model_parse_spring-v0_1_0` (local, CPU,
  1.6 GB; download link in `data/README.md`)
- optional warm cache for `spring_parse`: `data/amr_spring.jsonl`
  (`--base-graphs`, built by `precompute_spring.py`)
- wiki cache: `data/agent/wiki_cache.sqlite` (or the `wiki.cache` you set)

Nothing is imported from, or read out of, the older `dataset_amr` project.

Rewriting is configurable: `rewrite.enabled`, `rewrite.kinds` (`"all"` or a
list such as `["statute","provision","gov-entity"]`), or `--rewrite on|off`,
`--rewrite-kinds statute,provision`.

## Outputs

The default output directory is `data/agent/`. `--out` or `"out"` in
`config.json` changes it; use a fresh directory for a new dataset.

| Path | Contents |
| --- | --- |
| `records.jsonl` | accepted sentences: resolved `sentence`, `sentence_original`, `edits`, graph |
| `rejected.jsonl` | sentence-level rejections with reason, candidates, attempted graph |
| `dataset.all.jsonl`, `dataset.all.amr` | dataset + PENMAN (`# ::snt`, `# ::snt_orig`, `# ::edits`) |
| `events/article_<id>.jsonl` | every tool call, repair, accept/reject, ledger update |
| `checkpoints/<id>.json` | last completed batch, ledger and counters (resume point) |
| `done.jsonl` | completed articles |
| `stats.json` | aggregate counts (rewrites by kind, tool calls, reject reasons, repairs) |
| `examples/article_<id>.md` | step-by-step walkthrough rendered from the events |
| `wiki_cache.sqlite` | live wiki responses, always cached |

The dataset keeps both sentences: `sentence` is the resolved text used by
`::snt`/`::tok`, `sentence_original` preserves the corpus text. A specific name
in the graph and a rewrite edit must match in both directions, or the sentence
is rejected.

## Layout

```
config.py   defaults, CLI overrides
common.py   JSON extraction, core AMR validation, coverage score
legal_schema.py  Legal-AMR Lean Core v1.1 schema checker
make_india_db.py India-only corpus builder
precompute_spring.py  batch SPRING warm cache (resumable, shardable)
corpus.py   legal.db reader, article search, antecedent resolution
wiki.py     live Wikipedia/Wikidata + SQLite cache
rewrite.py  span edits + graph<->sentence invariant
check.py    core AMR + legal schema + rewrite checks
spring.py   SPRING as a tool call (local model, one sentence in, graph out)
tools.py    tool schemas + dispatch
prompt.py   system prompt + per-batch payload (paragraphs, never sentences)
llm.py      OpenAI-compatible tool loop + mock client
agent.py    per-article batches, ledger, repair, accept/reject
store.py    event log, sinks, checkpoints, stats
run.py      CLI, parallel workers, shards
build.py    dataset JSONL + PENMAN
examples.py event log -> Markdown walkthrough
```

The pipeline is self-contained: core AMR validation, the Legal-AMR v1.1 schema
checker, and all helpers live in this directory. The corpus is the India-only
`data/india.db`; no file or module is imported from the older project.

## Plan drawing

`agentic_amr_dataset_final.excalidraw` walks one real sentence through ten
moments: the corpus text, the SPRING graph, the two unclear names, the
decision, the edits, the final graph, the rewritten sentence, the check, the
memory, and the saved result. Beside it is the agent's tool layout.

To the right of that drawing there is a separate **Legal-AMR Lean Core v1.1**
panel. It explains the 7 concepts, 12 relations and 11 attributes in plain
language, gives one example for each, lists what was removed and why, and shows
a worked example with the v1.1 shape. `docs/build_lean_extension.py` adds or
replaces only that panel.

The older `agentic_amr_dataset.excalidraw` is kept unchanged as the original
sketch plus the longer seven-step pipeline. Its worked example still shows the
v1.0 labels; the current demos and the final drawing use the v1.1 shape.

Below the pipeline, **Six real sentences** has two halves. First sentence 360450
of article 5772, followed through nine moments in time: split the paragraph and
parse the sentence, recall, ask, rewrite, draw, repair, check, remember, save.
Each card carries the real tool results with their score arithmetic, the
sentence as it stands at that moment, both graphs (the SPRING graph the tool
returned and the agent's graph), the validator output and the memory before and
after. Then all six sentences of the article side by side: the same nine
moments as a row-per-moment, column-per-sentence grid, five accepted and one
rejected, with a card that explains the rejection (`no_credible_source`: the
corpus sentence starts at "Ltd", the short form "HEPL" is never spelled out, and
no candidate reached the credibility line). Two closing boxes list what changed
and what stayed the same for 360450.

Where the SPRING graphs come from: `data/five_spring/spring.jsonl`, produced
locally for these six sentences by `docs/spring_five.py` (amrlib,
`spring-v0_1_0`, CPU) before the demo run, so the demo does not depend on the
model being loadable on this machine. In a run, exactly those graphs arrive
through `spring_parse`; `records.jsonl` keeps each one as `base_amr`. The older
project's `amr_wiki.jsonl` is an LLM-edited graph, not a SPRING graph.

Every score is shown with its calculation. For `resolve_mention`: exact
name/alias in memory = 1.000; same sentence, name before the mention = 0.970;
earlier sentence = (1.000 with alias, else 0.900) − min(distance,40)*0.001;
article metadata = 0.800; generic head-only match in memory = 0.860 (a hint,
below a name stated in the sentence); credible if ≥ 0.700; ambiguous if the top
two differ by < 0.030. Coverage = content words kept (same token or light stem)
/ content words (3+ letters/digits, stopwords removed).

```bash
python docs/spring_five.py           # local SPRING -> data/five_spring/spring.jsonl
python docs/five_sentences.py        # real tools+validator -> data/five/, data/five_repair/
python docs/build_lean_extension.py  # add/replace the v1.1 panel beside the drawing
```

No model API is called. The answers are written by hand the way the agent is
asked to answer, then run through the real agent, corpus, resolver, cached wiki
and validator. The drawing says exactly what is real and what was written by
hand. `build_plan.py` replaces only its own tagged elements.
