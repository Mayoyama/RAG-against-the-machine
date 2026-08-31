# Collaboration Log — RAG against the machine

Living notes for cross-session continuity. Updated only when explicitly prompted.

**Read "Next steps" at the bottom first.** Everything above it (Project state, Decisions made, etc.) is a dated historical snapshot from around 2026-07-25 to 07-31 and has not been kept in sync with the actual codebase since — treat it as background/rationale, not current status. "Next steps" is the one section kept accurate as of the date in its own heading.

## Collaboration rules

Moved to `collaboration_rules.md`, same folder — read that file first,
every session, before this one. Kept separate and short deliberately.

## Project state (as of 2026-08-25)

`src/` skeleton: `__init__.py`, `__main__.py`, `cli.py`, `api.py`,
`models.py` at top level; `indexing/`, `retrieval/`, `generation/`,
`evaluation/` as subpackages, each with `__init__.py`. `cli.py`/`api.py`/
`__main__.py` still empty stubs. `pyproject.toml` has 9 dependencies
(`bm25s`, `fastapi`, `fire`, `pydantic`, `sentence-transformers`, `tqdm`,
`torch`, `transformers`, `uvicorn` — dev group has `flake8`, `mypy`);
`uv sync` has been run, `.venv` populated. `Makefile` has all required
rules (`install`, `run`, `debug`, `clean`, `lint`, `lint-strict`) plus
`uninstall`; `run`/`debug` support `make <rule> ARGS="..."` pass-through.
Git repo initialized, `admin/` gitignored.

`src/models.py`: **done**. All 8 PDF-given models (VI.4) plus an internal
`Chunk` model (composition, not inheritance: `Chunk.source: MinimalSource`,
plus `content: str` and `index: int`) — deliberately not reusing
`MinimalSource` via inheritance, to keep the subject's own output schema
untouched. Verified by actually running it (not just reading it) —
round-tripped a `RagDataset` with mixed `AnsweredQuestion`/`UnansweredQuestion`
through `model_dump_json()`/`model_validate_json()` and confirmed pydantic's
union resolution picks the right subclass back out, not just the simpler
`UnansweredQuestion` for everything.

`src/indexing/chunking.py`: in progress, Python-chunking half only
(Markdown/text chunking not started). Five working, individually-verified
functions so far: `get_utf8_content` (reads a file as UTF-8, returns `""`
and logs to stderr on `IOError`/`UnicodeDecodeError` rather than crashing),
`parse_content` (ast-parses source, returns `(lineno, end_lineno)` for each
top-level `FunctionDef`/`ClassDef`, returns `[]` and logs to stderr on
`SyntaxError`), `get_line_char_counts` (line number → character offset
lookup table; has an off-by-one-safe sentinel appended at the end so files
without a trailing newline don't cause an `IndexError` later), `get_start_
end_char_nums` (converts `(lineno, end_lineno)` pairs to `(first_character_
index, last_character_index)` pairs — note the asymmetry: start uses
`line_char_counts[lineno - 1]`, end uses `line_char_counts[end_lineno]`
with no `-1`, since the end boundary needs the start of the *next* line,
not the start of the last line itself), and `get_chunk` (slices actual
chunk text out of `content` given one `(first, last)` pair).

Still needed to finish Python chunking: the orchestrating function
(`chunk_python_file(pathname) -> list[Chunk]`, calls the five above in
sequence) and the `max_chunk_size` fallback — splitting any function/class
whose sliced content exceeds the 2000-char default cap, since structural
chunking alone can't guarantee that. Then Markdown/text chunking is a
separate, not-yet-started strategy.

## Project summary (from RAG.pdf)

Build a RAG system answering questions about the vLLM 0.10.1 codebase:
- **Index**: chunk `data/raw/` (vLLM repo) two ways — Python code chunking vs Markdown/text chunking. Persist under `data/processed/`. Must finish in ≤5 min. `--max_chunk_size` CLI arg, default 2000 chars (hard cap — moulinette rejects longer sources).
- **Retrieve**: TF-IDF or BM25 (at least one, lexical). Returns ranked `file_path` + char range sources. `file_path` must match corpus path verbatim.
- **Generate**: answer via `Qwen/Qwen3-0.6B`, structured JSON via provided pydantic models.
- **Data models**: `MinimalSource`, `UnansweredQuestion`/`AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`/`MinimalAnswer`, `StudentSearchResults`/`StudentSearchResultsAndAnswer` — all pydantic.
- **CLI** (Python Fire, `uv run python -m src <command>`): `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`. Never call the moulinette from within the solution.
- **Required layout**: `src/`, `pyproject.toml` + `uv.lock` at root, `README.md` at root, `data/raw/`, `data/processed/`, `data/datasets/{UnansweredQuestions,AnsweredQuestions}/`, `data/output/search_results/<Scope>/`, `data/output/search_results_and_answer/<Scope>/`.
- **Thresholds**: Recall@5 ≥ 80% docs, ≥ 50% code. Retrieval ≤ 90s/200 questions. Index ≤ 5 min.
- **Tooling**: Python 3.10+, uv only (reviewer runs `uv sync`), flake8, mypy (`--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`; `--strict` optional), tqdm progress bars, type hints everywhere, PEP 257 docstrings, Makefile rules: `install run debug clean lint [lint-strict]`.
- **README.md** (project deliverable) needs: italic first line with 42 logins, Description, Instructions, Resources (incl. AI usage disclosure), System architecture, Chunking strategy, Retrieval method, Performance analysis, Design decisions, Challenges faced, Example usage — in English.
- **Bonus** (1 pt each, only graded once mandatory fully passes): semantic embeddings (e.g. all-MiniLM-L6-v2), hybrid retrieval, incremental indexing, caching, local HTTP API.
- **Recode step**: be ready to make a small live modification during evaluation to prove understanding.

## Known wrinkle

`datasets_public.zip` extracts to `datasets_public/public/{AnsweredQuestions,UnansweredQuestions}/...` — one level deeper than the `data/datasets/{...}/` layout the PDF expects. Will need flattening at setup time (ask permission first).

## Open questions (unresolved as of 2026-07-30)

1. Solo project, or group? (affects README login line)
2. Is `Qwen/Qwen3-0.6B` (and `all-MiniLM-L6-v2` if bonus attempted) already available locally, or does the dev environment need internet access to pull from Hugging Face?
3. ~~Target scope for now — mandatory only, or bonus too?~~ Resolved 2026-07-31, see Decisions made.
4. Deferred (explicitly, by Shane): does `tests/` (696 .py files) count as "files you judge useful" for indexing, or is the corpus just `vllm/` (+ `docs/`)? Decide once building the indexer.

## Decisions made

- Repo initialized (`git init` + `.git` at RAG root, per earlier discussion on cloning into an existing non-empty folder).
- All admin/reference material (`RAG.pdf`, moulinette's own `README.md`, the three zips, `COLLAB_LOG.md`) moved into `admin/` at repo root, kept separate from actual project code.
- `.gitignore` added at root, currently: `admin/`, `.vs`, `.vscode/`, `.venv/`, `__pycache__/`, `*.pyc`, `.mypy_cache/`, `.pytest_cache`.
  - Note: `admin/` being gitignored means `RAG.pdf` and the zips won't be committed — fine, since they're reference material, not deliverables. Just make sure this doesn't accidentally also exclude something that should ship (e.g. if `src/` or the project's real `README.md` ever end up nested under `admin/` by mistake).
- Local dev machine only has the Windows build of `uv`. `moulinette-ubuntu`/`-fedora` are Linux ELF binaries and won't run under Git Bash (no WSL locally) — Shane will run the actual `moulinette` checks on the 42 school PCs, not locally. Local loop is limited to the project's own `evaluate` command; final/official recall@k numbers come from school-PC moulinette runs.
- Scope (2026-07-31): attempting all 5 bonuses. Semantic embeddings, hybrid
  retrieval, caching, and local HTTP API are committed. Incremental indexing
  is a stretch goal — may be cut if it proves too difficult, since no
  reference implementation exists in any repo we've researched (see peer
  research notes below). Also planning to work through non-bonus
  optimizations seen in other repos (per-type index routing, AST chunking,
  query expansion, reranking) as discussion points with Claude, adapted
  rather than copied, not treated as bonus-scoring items.
- `pyproject.toml` dependencies added 2026-07-31: `bm25s`, `fire`,
  `tqdm`, `torch`, `transformers` (versions checked against PyPI for
  Python 3.10-3.12 compatibility at the time). Followed up same day with
  `sentence-transformers`, `fastapi`, `uvicorn` once embeddings/hybrid and
  local-API approach were settled (also version-checked for 3.10-3.12).
- Embeddings/hybrid bonus: using `sentence-transformers` (model:
  `all-MiniLM-L6-v2`, per the subject PDF's own example) with a plain
  numpy/dict-based cosine-similarity search — no vector DB (ChromaDB
  considered and explicitly rejected). Corpus is small enough (~thousands
  of chunks) that a vector DB adds dependency weight and a new API to
  learn for no functional gain over a single matrix-multiply similarity
  search. Also means genuine incremental updates on the embeddings side
  are still possible later without ChromaDB — a persisted
  `chunk_id -> embedding` dict, selectively updated, gets the same effect.
- Local HTTP API bonus: FastAPI + uvicorn. Chosen over Flask/Streamlit
  because FastAPI takes pydantic models directly as request/response
  types, which lines up with the project's existing pydantic data models
  (no schema duplication); Streamlit is a UI tool, not really an API.
- Makefile: fixed `debug` (was pointing at a nonexistent `main.py`, now
  `uv run python -m pdb -m src $(ARGS)`); added an `ARGS` variable so
  `run`/`debug` accept `make run ARGS="search foo --k 5"` style
  pass-through. `lint`/`lint-strict` already matched the PDF's exact
  required flags.
- README: written at the end (after mandatory + attempted bonuses are
  done), not incrementally. Shane's usual workflow — Claude drafts the
  majority from the finished project's actual architecture/decisions,
  Shane tweaks and adds his own information, AI usage disclosed in the
  Resources section. Confirmed against the PDF's own AI Instructions
  chapter (p.5): "Use AI to reduce repetitive or tedious tasks" is
  listed as the main message's first bullet, and the README's AI-usage
  requirement is disclosure, not restriction.

## File inspection notes (2026-07-30)

- `README.md` (moulinette usage doc) matches the PDF: same CLI, same pass criteria, same JSON schemas.
- `datasets_public.zip` → `datasets_public/public/{AnsweredQuestions,UnansweredQuestions}/{dataset_docs_public.json,dataset_code_public.json}`. 100 docs Qs, 99 code Qs, every question has exactly 1 source, all `difficulty: "synthetic"`, all `is_valid: true`. Unanswered files share `question_id`s with answered files (self-check possible).
- `moulinette.zip` → `moulinette-ubuntu`/`moulinette-fedora`, stripped x86-64 ELF, Fire-based. `evaluate_student_search_results` has an undocumented (vs PDF/README) `--threshold` flag: if set, prints PASS/FAIL against that Recall@5 value.
- `vllm-0.10.1.zip` unpacked = 38M. `vllm/` 17M / 1763 .py / ~500K lines (largest: `third_party/pynvml.py` 234K, `config/__init__.py` 173K, `v1/worker/gpu_model_runner.py` 155K). `docs/` 7.1M / 178 .md / ~22K lines. `tests/` 6.6M / 696 .py. `csrc/` 3.5M, C/CUDA — out of scope for the two required chunkers (Python + Markdown only).

## Peer repo research — bonus approaches & reference points (2026-07-31)

Scanned ~25 real public GitHub repos of this same 42 project (research only,
not copied) to gauge how others approached BM25 and the 5 bonuses. Verified
directly rather than trusting secondhand (e.g. ChatGPT) summaries — some of
those turned out inaccurate on inspection.

**Incremental indexing bonus**: no verified repo does true file-level
incremental updates. Every `indexer.py`/`indexing.py` checked (including
jessesmidt/angry_at_the_robot, auroreRakoto/RAG-against-the-machine) does a
full-corpus rebuild + persist-to-disk on every `index` run (pickle, or
`bm25s`'s own `.save()`), with no file-hash/mtime delta logic anywhere. If we
attempt this bonus, treat it as genuinely original work, not an adapted
pattern — nobody's showing a template for it.

**Hybrid retrieval bonus** — 2 confirmed real BM25+dense implementations,
both fused with Reciprocal Rank Fusion (RRF):
- `antoine71/42-rag-against-the-machine` — BM25 (`bm25s`) + dense
  (`sentence-transformers/msmarco-bert-base-dot-v5` via ChromaDB), RRF with
  BM25 weighted 1.9 vs vector 1.0, k*4 candidates from each side before merge.
- `Sabofxx/rag-against-the-machine` — BM25 + dense MiniLM embeddings, RRF.

**Caching bonus** — 2 confirmed, both answer/query-level (skip regenerating
an answer for a repeated question) rather than index-level:
- `mpouillo/42-rag-against-the-machine`
- `II-SARIP-II/42-RAG-against-the-machine`

**Worth reading for design ideas even without a clean bonus claim**:
`cdonairevillen/RAG_Against_the-machine_42_project` — classifier-routed
per-type BM25 indices (doc/code/general) + cross-encoder reranking, AST
chunking with symbols stored separately from BM25 corpus text. Notably,
they tested semantic embeddings fused via RRF and found it *reduced* docs
recall on this corpus, so removed them entirely — useful reminder that
hybrid retrieval isn't automatically better and needs measuring, not assuming.

**Caveats for using any of this**:
- Subject/moulinette recall thresholds appear to drift across cohort
  versions (55%/45% and 70%/95% seen in other repos vs our PDF's 80%/50%)
  — don't treat another repo's stated thresholds or behavior as authoritative
  for us.
- A repo mentioning a high intra score isn't reliable evidence of what that
  score covers — found one case (Sabofxx) where a "125/100" figure quoted in
  a README was actually for an unrelated earlier project (Born2BeRoot) in a
  monorepo description, not this one.

## IMPORTANT — indexing architecture requirement (2026-07-31)

Whoever (whichever chat/session) writes the `index` command / indexing
pipeline: file-level chunking must be its own discrete, reusable unit —
e.g. a function that chunks a single file, called in a loop over the repo
— not one function that processes the whole repo as an inseparable batch.
This is a hard requirement for how the mandatory code gets built, not a
nice-to-have.

Why: incremental indexing (5th bonus, stretch goal, may get cut — see
scope decision above) is planned as a true add-on pass built *after*
everything else (mandatory + other 4 bonuses) is finished. The plan is a
hash/mtime-based manifest that diffs against the previous run and calls
the *existing* per-file chunking function only on changed files. If the
base pipeline is written as one non-separable batch process instead, this
becomes a refactor instead of a clean addition. Keeping file-level
processing decoupled from day one is what makes the 5th bonus
"plug-and-play"-able at the end rather than a redesign.

Note: this only affects file-level chunking/embedding granularity. BM25
indexing itself stays full-rebuild regardless of this decision — `bm25s`
has no incremental-update support at all; that's a library limitation,
independent of how the rest of the pipeline is structured (see peer
research notes above).

## Build plan (agreed 2026-07-31)

Sequenced to avoid the refactor problem Shane wants to avoid: bonus-compatible
architecture (pluggable retrieval interface, per-file chunking, one reusable
answer-generation function, CLI/logic separation) gets built INTO the
mandatory code from the start. Writing the actual bonus implementations is
gated behind full mandatory validation — not because the architecture isn't
bonus-ready, but because a late-discovered mandatory fix (e.g. a chunk
metadata change) would otherwise ripple into already-written bonus code.

Order (tracked as tasks in this session's task list too, may not carry over
to a fresh session — this is the durable record):

1. Pydantic data models (`src/models.py`)
2. Per-file chunking, Python + Markdown (`src/indexing/chunking.py`)
3. BM25 indexing + persistence (`src/indexing/index_store.py`, `index` CLI)
4. Retrieval interface + BM25 retriever (`src/retrieval/`, `search`/`search_dataset` CLI)
5. Answer generation (`src/generation/`, `answer`/`answer_dataset` CLI)
6. Evaluation / Recall@k (`src/evaluation/`, `evaluate` CLI)
7. Edge-case hardening pass (empty query, k=0, missing files, malformed JSON, no unhandled tracebacks)
8. **GATE**: full mandatory pipeline validated end-to-end against the real
   moulinette on a school PC (recall thresholds, timing constraints). Nothing
   bonus-related is evaluated by the subject until this passes — see PDF
   Bonus Part.
9. Bonus: semantic embeddings
10. Bonus: hybrid retrieval (needs embeddings done first)
11. Bonus: caching
12. Bonus: local HTTP API
13. Bonus: incremental indexing (stretch — may be cut, no reference
    implementation exists anywhere researched)
14. README.md (written last, see Decisions made)

## Next steps (updated 2026-08-31)

Mandatory is complete: recall measured at 85.0% docs / 51.5% code, both
above threshold (80%/50%). Full-scale indexing/retrieval timing against
the mandatory limits (≤5 min index, ≤90s/200 questions retrieval) hasn't
been measured yet — needs a run on a school PC.

4 of 5 bonuses are done and wired end-to-end: semantic embeddings
(`src/indexing/embedding.py`, `src/retrieval/semantic_retrieving.py`),
hybrid retrieval via RRF (`src/retrieval/hybrid_retrieving.py`), caching
(`src/caching.py`, scoped to the BM25 path of `search` only), and the
local HTTP API (`src/api.py`). `README.md`'s Bonus Features section has
the actual writeups for each — this file doesn't duplicate that.

**Only remaining bonus: incremental indexing** — "when a file changes,
re-index only that file instead of rebuilding the whole index" (PDF
Bonus Part, item 3).

Research recap (see "Peer repo research" above, still accurate as of
2026-07-31): none of the ~25 repos scanned has a real file-level
incremental implementation — every one does a full-corpus rebuild on
every `index` run. This is genuinely original work, not an adapted
pattern.

Hard constraint: `bm25s` has no incremental-update API — the BM25 index
structure itself must be rebuilt from the full chunk list every time,
no way around that. The actual opportunity is skipping the *expensive*
work for unchanged files: re-parsing/chunking via `ast`, and especially
re-embedding via `sentence-transformers` (the slowest step by far on a
large corpus).

Current idea (not set in stone — decide fresh next session):
- A persisted manifest (e.g. `data/processed/index_manifest.json`)
  mapping `file_path -> {mtime, chunks, embedding}` from the last
  successful `index` run.
- On a new `index` call: for each file, compare current mtime against
  the manifest. Unchanged files reuse cached chunks + embeddings;
  changed/new files get re-chunked and re-embedded via the existing
  per-file functions; files in the old manifest but missing from disk
  get dropped.
- Reassemble the full `chunks` list and embedding matrix (cached +
  fresh, same order as today) and continue through the existing
  `build_bm25_index`/`save_index`/`save_semantic_index` pipeline
  unchanged. BM25 itself still fully rebuilds every time either way.
- Open questions to settle with Shane: mtime vs. content hash for
  change detection (mtime is simpler but can false-negative after a
  git checkout resets timestamps; hash is more correct but means
  reading every file's full bytes regardless); whether a
  `--force_reindex` flag is worth adding for manifest-corruption
  recovery.

This matches the architecture decision from 2026-07-31 ("IMPORTANT —
indexing architecture requirement" above) — chunking was deliberately
kept as a per-file, reusable unit specifically so this bonus could be
added without a refactor.
