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

# no API key: scripted pass over a real article
python run.py --article 5772 --mock --build --example

# real model, inherited OpenRouter settings (needs the local SPRING model)
python run.py --article 5772 --build --example

# many articles, in parallel, split across machines
python run.py --all --limit 500 --workers 8 --shard 1/4
```

`python run.py --all --workers N` runs articles in parallel; `--shard i/n`
selects articles by `article_id % n == i-1`. Every article is independent and
resumable, so workers and machines never need to coordinate.

## Configuration and inherited resources

Defaults are in `config.py`. A local `config.json` overrides them. The older
project's `dataset_amr/pipeline.config.json` is inherited automatically when it
exists (OpenRouter `api_url`/`api_key`/`model`, the export limit, and the DB and
SPRING paths). Priority:

```
defaults < dataset_amr/pipeline.config.json < config.json < environment < CLI
```

Environment: `AMR_API_URL`, `AMR_API_KEY`, `AMR_MODEL`.

DBs and inputs:

- corpus: `dataset_amr/data/legal.db` (`--corpus`)
- SPRING model: `~/.cache/amrlib/model_parse_spring-v0_1_0` (local, CPU)
- warm cache for `spring_parse`: `dataset_amr/data/work/amr_spring.jsonl` (`--base-graphs`)
- wiki cache: `agentic_dataset/data/agent/wiki_cache.sqlite`

Rewriting is configurable: `rewrite.enabled`, `rewrite.kinds` (`"all"` or a
list such as `["statute","provision","gov-entity"]`), or `--rewrite on|off`,
`--rewrite-kinds statute,provision`.

## Outputs (`data/agent/`)

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
config.py   defaults, inheritance, CLI overrides
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

The tested `legal_schema` / core validators in `dataset_amr/scripts` are
imported by path, never copied.

## Plan drawing

`agentic_amr_dataset.excalidraw` keeps the original sketch at the top and a
step-by-step pipeline below it. The pipeline is seven steps and starts with the
paragraph: whole paragraphs reach the agent unsplit, the agent splits them, and
its `spring_parse` calls (one per sentence, in parallel) return the graphs it
then edits. A real run preflights the local SPRING model and refuses to start
without it.

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
through `spring_parse`; `records.jsonl` keeps each one as `base_amr`.
`data/work/amr_wiki.jsonl` is an LLM-edited graph, not a SPRING graph.

Every score is shown with its calculation. For `resolve_mention`: exact
name/alias in memory = 1.000; same sentence, name before the mention = 0.970;
earlier sentence = (1.000 with alias, else 0.900) − min(distance,40)*0.001;
article metadata = 0.800; generic head-only match in memory = 0.860 (a hint,
below a name stated in the sentence); credible if ≥ 0.700; ambiguous if the top
two differ by < 0.030. Coverage = content words kept (same token or light stem)
/ content words (3+ letters/digits, stopwords removed).

```bash
python docs/spring_five.py      # local SPRING -> data/five_spring/spring.jsonl
python docs/five_sentences.py   # real tools+validator -> data/five/, data/five_repair/
python docs/build_plan.py       # redraw tagged sections (agent-pipeline, agent-one)
```

No model API is called. The answers are written by hand the way the agent is
asked to answer, then run through the real agent, corpus, resolver, cached wiki
and validator. The drawing says exactly what is real and what was written by
hand. `build_plan.py` replaces only its own tagged elements.
