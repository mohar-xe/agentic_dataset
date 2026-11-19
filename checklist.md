# Agentic AMR dataset – checklist

Working dir: `agentic_dataset/`. Plan: per-article thin agent, sequential sentences, article ledger, on-demand tools, evidence-based rewrites, reject when not credible.

## Milestone 1 — plumbing, no real LLM (mock/scripted)

- [x] `config.py` – defaults, JSON load, CLI overrides, rewrite/wiki toggles
- [x] `corpus.py` – article loader (ordered sentences), metadata, `article_search`, `resolve_mention` from declarations/metadata/ledger
- [x] `wiki.py` – live Wikipedia/Wikidata lookup, SQLite cache (no local title DB)
- [x] `store.py` – per-article event log, append-only records/rejections, checkpoints, stats
- [x] `rewrite.py` – span-based edits, recomputed offsets, graph↔sentence invariant
- [x] `check.py` – `penman` + `legal_schema` (reused by path) + rewrite checks
- [x] `tools.py` – schemas + dispatch: `article_search`, `resolve_mention`, `wiki_lookup`, `check_amr`
- [x] `prompt.py` – system prompt with tool contract, rewrite contract, lean-core rules
- [x] `llm.py` – OpenAI-compatible tool loop + `MockClient` (script replay)
- [x] `agent.py` – per-article sequential batches, running ledger, accept/reject, checkpoint after each batch
- [x] `run.py` – CLI `--article/--all/--mock/--build/--example/--rewrite-*`, per-batch mock, resume
- [x] `build.py` – dataset JSONL + PENMAN (`# ::snt` resolved, `# ::snt_orig`, `# ::edits`)
- [x] `examples.py` – event log -> Markdown walkthrough
- [x] Verify: imports + resolver on article 5772 ("the Act" declared -> credible candidate)
- [x] Verify: `--mock` run produces records, events, stats, checkpoint (cleared on done), example; rewrite span applied and backed by AMR (6 rewrites on 5772)
- [x] Verify: ambiguous/no-source case -> rejected sentence with candidates (not guessed)
- [x] Verify: broken edit/graph mismatch -> rejected with errors, not written as ok
- [x] Verify: rerun on done article is skipped (no reprocessing)
- [x] Plan drawing: `agentic_amr_dataset.excalidraw` has a seven-step plain-language pipeline section with a worked example at every step and the five tools, starting from whole paragraphs; the drawing shows the agent splitting a paragraph and issuing parallel `spring_parse` calls (`docs/build_plan.py` regenerates it)
- [x] Five-sentence demo (linear, one sentence per batch): real corpus, resolver, cached wiki, validator and local SPRING (`docs/spring_five.py` -> `data/five_spring/spring.jsonl`) on article 5772 -> 5 accepted, 1 rejected (`no_credible_source`), 4 rewrites, 0 repairs; memory before/after, tool results with score arithmetic, coverage math, full SPRING+change notes per sentence; real repair demo (`data/five_repair/`, broken first answer -> validator error -> fixed); the drawing shows this on **one** sentence (360450) as a `t0`-`t8` timeline of how it transforms, then all six sentences side by side on the same nine moments (rows = moments, columns = sentences, 360439 tinted and stopping at rejection with the reason given), built by `docs/build_plan.py` from `data/five/report.json`, `data/five/rejected.jsonl` and `data/five_repair/report.json`

## Contracts (locked)

- [x] Base graphs always provided for real runs (`--base-graphs`, SPRING); mock uses base when present else minimal
- [x] Wiki: live lookup only, always cached (`wiki_cache.sqlite`)
- [x] All entity kinds rewriteable, configurable (`rewrite.kinds`, `--rewrite-kinds`, on/off)
- [x] Resolved sentence primary; `sentence_original` always kept
- [x] Reject when no credible source or ambiguous (never guess)
- [x] Graph⇔sentence invariant enforced mechanically
- [x] Count and save after every major step (events + checkpoint + stats)

## Milestone 2 — real model & quality

- [x] Paragraph-in agent: the batch carries whole paragraphs (no sentence list, no pre-computed graphs); the agent splits them, calls `spring_parse` once per sentence (parallel calls in one turn, `llm.run_tool_calls`), and edits the returned graph; it never sees a corpus sentence id, the harness matches answers back by sentence text; a sentence it did not parse is rejected as `no_base_graph`; `--base-graphs` is only a warm cache; verified end to end on article 5772 with 53 `spring_parse` calls (live SPRING blocked by an amrlib/transformers version skew in this venv, see `docs/changes_2026-11-17.md`)
- [x] Repair turns on validation failure (bounded by `agent.max_repairs`, same conversation + validator errors + tools); policy rejections are never repaired into guesses
- [x] Per-article parallelism (`--workers`, `--shard i/n`), thread-safe sinks (`claim`), per-thread corpus + WAL wiki cache
- [x] Inherit older `dataset_amr/pipeline.config.json` (OpenRouter API URL/key/model, export limit, DB/SPRING paths); env + CLI overrides
- [x] Verified: parallel mock run of 3 articles with 3 workers (177 accepted, rewrites counted, examples rendered)
- [x] Verified: repair path (broken graph -> repair pass -> accepted, events `repair_start`/`repair_result`); documented with a real worked example in `data/five_repair/report.json` and the drawing
- [x] Verified: completed articles are skipped on rerun
- [ ] Single article with real LLM client (tool loop)
- [ ] Tune prompt / ledger size from real traces
- [ ] Pilot on N articles, compare to route-2 (valid rate, relations, wiki, rewrites, rejections, tokens)
- [ ] `build.py` split/dedupe compatible with existing consumers where possible
