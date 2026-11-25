# Kaggle pipeline: SaT-12L-SM → SPRING → Wikipedia `:wiki`

Turns `data/india.db` articles into sentence-segmented AMR graphs on a Kaggle
GPU, using:

1. **`segment-any-text/sat-12l-sm`** on the T4 (full `wtpsplit`/torch, in a
   subprocess) to split every paragraph into sentences;
2. **SPRING AMR3 weights via amrlib** (`model_parse_spring-v0_1_0`) to parse a
   seeded random sample of 20,000 sentences, then a **post-hoc Wikipedia
   linker** that adds `:wiki` to named-entity nodes.

`:`wiki` is no longer native: the official Sapienza checkpoint host
(`nlp.uniroma1.it`) answers HTTP 503 on every path and has no mirror, so the
kernel uses amrlib's converted copy of the same SPRING AMR3 weights. Those were
trained with `remove_wiki: "remove"` and never emit `:wiki`; the kernel adds it
afterwards from the Wikipedia search API, dropping disambiguation pages.

Everything runs on Kaggle because amrlib needs a 4.x `transformers` while the
image ships 5.x, and the segmentation model wants a different stack again.

## Pieces

| file | role |
| --- | --- |
| `export_paragraphs.py` | dumps articles + paragraphs from `data/india.db` to `upload/paragraphs.json.gz` |
| `upload/` | the Kaggle dataset payload (`mohar07/legal-amr-india-paragraphs`) |
| `kernel/amr_sat_spring_wiki.py` | the kernel: split → sample → parse → link wiki → JSONL |
| `kernel/kernel-metadata.json` | kernel id, GPU, dataset mount |
| `fetch_output.sh` | waits for the run, downloads `/kaggle/working/out` |
| `build_notebook.py` | generates the manual-run notebook from the same stages |
| `legal_amr_sat_spring_wiki.ipynb` | notebook to upload and run cell by cell |
| `check_output.py` | coverage + Legal-AMR Lean Core v1.1 validation |
| `build_cpu_kernel.py` | generates `kernel_cpu/`: CPU-only parse kernel (no GPU queue) |
| `kernel_cpu/` | script kernel `mohar07/legal-amr-parse-cpu` + metadata |

## Manual notebook (no batch queue)

Kaggle's batch GPU queue can sit for an hour; an interactive session starts at
once. Upload `legal_amr_sat_spring_wiki.ipynb` to a new Kaggle notebook
(*File → Import Notebook*), set **GPU T4** + **Internet on**, and add one input:
`legal-amr-india-paragraphs` (the corpus). Then run the cells in order. The last
cell zips `/kaggle/working/out` into `legal_amr_output.zip`, downloadable from
the right-hand Output panel.

The stage scripts (`notebook_stages/amr_stage.py`, `wiki_stage.py`) are embedded
in the notebook as base64, so there is no second dataset to attach. They are
also kept in the private dataset `mohar07/legal-amr-nb-stages` for reference.

Notes on the notebook build:
- markdown and code cells must not share keys — a markdown cell carrying
  `outputs`/`execution_count` makes the file invalid nbformat and Kaggle
  refuses the import;
- the venv is created with `--system-site-packages` so it reuses the image's
  torch/CUDA instead of downloading another multi-GB wheel, then installs
  `amrlib==0.8.1` + `transformers==4.44.2` on top;
- Kaggle's `/usr/bin/python3` has no `ensurepip`, so `python3 -m venv` fails
  there. The cell retries with `--without-pip` (reusing the system pip) and
  then with `virtualenv`, which ships its own pip wheels; only if all three
  fail does it fall back to the image interpreter (with a warning).

Regenerate it after editing the stage scripts:

```bash
python kaggle_pipeline/build_notebook.py
```

## CPU kernel (no GPU queue)

When the GPU queue is the bottleneck, parse on a CPU batch kernel instead:

```bash
python kaggle_pipeline/build_cpu_kernel.py
kaggle datasets status mohar07/legal-amr-picked      # picked.json input
kaggle kernels push -p kaggle_pipeline/kernel_cpu
kaggle kernels status mohar07/legal-amr-parse-cpu
kaggle kernels output mohar07/legal-amr-parse-cpu -p data/kaggle_runs/cpu-<date>
python kaggle_pipeline/check_output.py \
    data/kaggle_runs/cpu-<date>/out/sentences_20000.jsonl
```

The kernel installs `amrlib==0.8.1` plus the first installable transformers 4.x
(`4.46.3` → `4.48.3` → `4.44.2`), downloads the SPRING weights, parses with
`AMR_BEAMS=2` (CPU is ~5x slower per beam), checkpoints every 500 sentences and
stops after `AMR_MAX_HOURS=10` so it always finishes inside the batch limit.
Override with `kaggle kernels push` after editing the env defaults at the top of
`kernel_cpu/parse_cpu.py`.

## Reproduce

```bash
python kaggle_pipeline/export_paragraphs.py     # refresh payload
kaggle datasets version -p kaggle_pipeline/upload -m "refresh corpus"
kaggle kernels push -p kaggle_pipeline/kernel
./kaggle_pipeline/fetch_output.sh               # waits, downloads to data/kaggle_runs/
python kaggle_pipeline/check_output.py data/kaggle_runs/<run>/out/sentences_20000.jsonl
```

## Output contract

`sentences_20000.jsonl`, one row per sentence:

```json
{"article_id": 1, "para_id": 3, "para_index": 2, "sent_index": 4,
 "sentence": "...", "words": 31,
 "amr": "(p / parse ...)", "status": "OK",
 "wiki": [{"node": "p", "title": "Supreme Court of India",
           "mention": "Supreme Court"}],
 "title": "...", "url": "...", "source": "livelaw", "section": "...",
 "tags": {"statutes": [], "provisions": [], "precedents": []}}
```

`sentences_20000.amr` holds the same graphs as PENMAN with `# ::id`/`# ::snt`
comments. `split_stats.json` / `run_meta.json` record the seed, counts, and
wiki coverage.

## Notes

- Kaggle transparently unpacks an uploaded `.gz`, so the mounted file is
  `paragraphs.json` (26 MB); the kernel accepts either name.
- Sampling is `random.Random(42).sample` of 20,000 sentences with ≥4 words, so
  it is reproducible only while the corpus, splitter and `MIN_WORDS` stay
  fixed.
- Everything is installed in `/kaggle/working/venv` (created with
  `--system-site-packages`, so the image's CUDA torch is reused). `wtpsplit` is
  installed first and segmentation runs before transformers 4.x is installed for
  amrlib, so the two never share an interpreter.
- Kaggle runs Python 3.13 and `transformers==4.44.2` pins `tokenizers<0.20`,
  which has no cp313 wheels (source build needs Rust). The notebook and the CPU
  kernel therefore install `transformers==4.46.3` first (needs `tokenizers`
  0.20.x, which ships cp313 wheels) and only fall back to 4.48.3 / 4.44.2.
- Parsing checkpoints every 500 sentences into `graphs.jsonl`; rerunning the
  parse cell resumes instead of starting over. Wikipedia resolutions are cached
  in `wiki_cache.json` for the same reason.
- The ONNX CPU path (`wtpsplit-lite`) needs ~0.3 s/paragraph (~2.3 h for the
  corpus); the GPU path made that step take minutes. The ONNX variant remains
  available for CPU-only retries.
- The Sapienza checkpoint host (`nlp.uniroma1.it`) intermittently returns 503;
  the kernel retries 5× alternating http/https and refuses a tarball under
  100 MB (an error page would otherwise reach `tar`).
- `:wiki` is AMR 3.0 metadata, not part of Legal-AMR Lean Core v1.1; the schema
  only warns when the tag is malformed.
- The linker links only `:name` nodes, requires the Wikipedia title to restate
  the mention, and drops pages flagged `disambiguation`. Precision over recall:
  unmatched entities stay unlinked and are still visible in `amr`.
- Corpus is non-publishable; the dataset is private and research-only.
