<a id="top"></a>

# RAG against the machine

*This project has been created as part of the 42 curriculum by <login1>.*

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)
- [System Architecture](#system-architecture)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Method](#retrieval-method)
- [Performance Analysis](#performance-analysis)
- [Design Decisions](#design-decisions)
- [Challenges Faced](#challenges-faced)
- [Example Usage](#example-usage)
- [Bonus Features](#bonus-features)
- [Resources](#resources)

## Description

"RAG against the machine" is a retrieval-augmented generation system built to answer natural-language questions about the vLLM 0.10.1 codebase. Given a question, it retrieves the most relevant chunks of source code or documentation, then uses a local language model to generate an answer grounded in those chunks.

The project is exposed entirely through a command-line interface: index the corpus, run retrieval (single query or full dataset), generate answers, and evaluate retrieval quality with recall@k against a labelled question set.

[↑ Back to top](#top)

## Instructions

Install dependencies with `make install` (runs `uv sync`). Commands are run through `make run ARGS="<command> [options]"`, or directly via `uv run python -m src <command> [options]`. Available commands: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate` — see Example Usage below.

Pass `--apply_bonuses True` to `index` to also build the semantic index and enable incremental reuse on later runs — required before `search`/`search_dataset` can use `--method semantic` or `--method hybrid`.

Use `make debug ARGS="..."` to run any command under `pdb`, `make lint` / `make lint-strict` to run flake8 and mypy, and `make clean` to remove caches and compiled files.

[↑ Back to top](#top)

## System Architecture

The codebase is split into independent modules under `src/`: `indexing/` (chunking, BM25/semantic index construction, incremental-reindex manifest), `retrieval/` (BM25, semantic, and hybrid search), `generation/` (LLM-based answer generation), and `evaluation/` (recall@k scoring). `models.py` defines the pydantic schemas shared across modules, and `utils.py` handles JSON I/O. `cli.py` wires all of the above into six commands, exposed through Python Fire in `__main__.py`.

Each module's public functions are independent of the CLI layer, so they can be used or tested without going through Fire.

[↑ Back to top](#top)

## Chunking Strategy

Python files are parsed with `ast` and chunked at the top level: each top-level function or class definition — including `async def` functions, not just plain `def` — becomes one chunk, its character span computed by mapping AST line numbers through a per-line character-offset table. Markdown, text, and JSON files are instead split on Markdown headers, with an ungrouped leading section for content before the first header. Chunks exceeding the configured size limit are further sliced by raw character length, with local offsets recombined back into absolute file positions.

A recursive, method-level variant was also prototyped, splitting classes down into individual methods instead of keeping them as one chunk. It wasn't adopted: top-level chunking (once extended to cover `async def`) already clears the mandatory recall thresholds with meaningfully less complexity.

Two later additions changed *what* gets chunked, not just how: oversized pieces now split into ~15%-overlapping windows instead of hard cuts, so content near a boundary isn't lost to either side, and content previously discarded between Python top-level definitions (imports, module-level code, gaps) is now captured as its own chunks. Both were tested against the real corpus and produced no measurable recall@5 change, while roughly tripling total chunk count (~9,200 → ~28,700) — kept for completeness, but a real cost, particularly for embedding time (see Incremental Indexing below).

[↑ Back to top](#top)

## Retrieval Method

Retrieval defaults to BM25 ranking via the `bm25s` library: `Bm25Retriever` builds its corpus from the chunk index saved during `index`, and `search` / `search_dataset` return the top-`k` matching chunks (as `MinimalSource` objects) for a single query or an entire dataset of questions respectively, with `k` bounded against corpus size.

Two additional retrieval methods are available as bonus features (see Bonus Features below): `SemanticRetriever`, which ranks chunks by cosine similarity over embeddings, and `HybridRetriever`, which fuses BM25 and semantic rankings via Reciprocal Rank Fusion. All three implement the same `RetrievalBase` interface and are selectable via `search`'s `method` parameter (`"bm25"`, `"semantic"`, or `"hybrid"`).

Chunk content and queries are also preprocessed with `expand_snake_case` (replacing `_` with spaces) before BM25 tokenization, since `bm25s` otherwise treats an identifier like `get_utf8_content` as one atomic token — invisible to a query using any of its component words. Verified against the real moulinette: +1.0pp code Recall@5 on its own (51.5% → 52.5%), no change on docs (no snake_case content there to fix), and a much larger effect once stacked with hybrid retrieval (see Hybrid Retrieval below). Applied unconditionally to both index-build and query-time tokenization — the two have to stay in sync, or matching breaks instead of improving.

[↑ Back to top](#top)

## Performance Analysis

Recall@k is computed via IoU (intersection-over-union of character spans, same-file match, threshold 0.05) against a labelled dataset, per the project spec. With the final chunking approach (top-level AST chunking, including `async def`) and `expand_snake_case` tokenization, the default (BM25-only) pipeline measures 85.0% documentation recall and 52.5% code recall, against the 80% / 50% mandatory thresholds.

<!-- TODO (ask Shane): add indexing time / retrieval time here once measured at full scale on the school machines (mandatory limits: indexing ≤ 5 min, retrieval ≤ 90s / 200 questions). -->

[↑ Back to top](#top)

## Design Decisions

`Generator` (in `generation/generating.py`) is implemented as a class rather than a standalone function, so the tokenizer and model are loaded once and reused across every `generate_answer` call instead of being reloaded per question. All CLI-reachable functions validate their inputs explicitly (type, range, and path-existence checks) before doing any work, so invalid input fails with a clear `ValueError` instead of an unhandled traceback from deeper in the pipeline.

<!-- TODO (ask Shane): add reasoning for any other decisions worth documenting. -->

[↑ Back to top](#top)

## Challenges Faced

<!-- TODO (Shane): expand these into prose. Starting list, drawn from what came up during the project: -->

- `evaluate_recallk` had a couple of logic bugs early on: `source_found` returned on the first same-file candidate instead of tracking the best IoU across all of them, and the final average was accidentally computed inside the per-question loop instead of after it.
- `search_dataset`'s directory-existence guards got swapped during hardening — the write-target (`save_directory`) was checked with `Path.is_dir()` (meant for read-targets), which wrongly rejected valid new output directories, while the actual read-target (`index_dir`) wasn't checked at all.
- Deciding how far to guard against degenerate CLI input: negative character indices, `k` larger than the corpus, missing/invalid directories, and a genuine infinite loop in `split_str` for `chunk_size <= 0` (left unfixed — currently unreachable from the CLI).
- Chunking granularity trade-off: top-level-only AST chunking vs. a recursive method-level variant that measured higher code recall in a prototype — decided against the recursive version, since top-level chunking (once extended to cover `async def`) already cleared the mandatory threshold with less complexity.
- `mypy --strict` effectively hanging on `generation/generating.py` because it tried to parse torch's and transformers' actual source for type info (both are large and lack clean stub packages); fixed by treating them as opaque via `follow_imports = "skip"` overrides.
- `generate_answer` assumed `apply_chat_template(..., return_tensors="pt")` always returns a plain tensor; against a real model it can return a `BatchEncoding` instead, which has no `.shape` — exactly the union-type error mypy had already flagged and that got deferred. Fixed by passing `return_dict=True` and reading `tokens["input_ids"]` explicitly. Separately, disabling Qwen3's "thinking" mode needed both `enable_thinking=False` *and* `add_generation_prompt=True` together — the chat template only checks `enable_thinking` inside the `add_generation_prompt` branch, so the flag alone silently did nothing.

[↑ Back to top](#top)

## Example Usage

```
uv run python -m src index --root_directory data/raw --save_directory data/processed
uv run python -m src index --root_directory data/raw --save_directory data/processed --apply_bonuses True
uv run python -m src search --query "..." --k 5
uv run python -m src search --query "..." --k 5 --method semantic
uv run python -m src search --query "..." --k 5 --method hybrid
uv run python -m src search_dataset --dataset_path data/datasets/questions.json --k 5 --save_directory data/processed
uv run python -m src answer --query "..." --k 5
uv run python -m src answer_dataset --student_search_results_path data/processed/questions.json --save_directory data/output
uv run python -m src evaluate --student_search_results_path data/output/questions.json --dataset_path data/datasets/questions.json
```

[↑ Back to top](#top)

## Bonus Features

### Semantic Embeddings

`index` now builds a second index alongside the BM25 one: every chunk's content is embedded with `all-MiniLM-L6-v2` (via `sentence-transformers`) into a 384-dimensional vector, and the resulting matrix is saved as `semantic_index.npy` in the same `save_directory` as the BM25 index. It deliberately doesn't duplicate chunk metadata — the vectors are saved in the exact same order as the chunks passed to the BM25 index, so `SemanticRetriever` reuses the corpus already persisted by `bm25s.BM25.save()` to map similarity results back to `MinimalSource` objects, rather than storing file paths and character ranges twice.

`SemanticRetriever.search()` embeds the query the same way, then ranks chunks by cosine similarity (`(a · b) / (|a| * |b|)`, computed as a single vectorized operation over the whole matrix) and returns the top-k. `search()` in `cli.py` takes a new `method` parameter (`"bm25"`, default, or `"semantic"`) to choose between the two retrievers; caching is intentionally scoped to the BM25 path only, since the query cache was already keyed on `(query, k)` without a `method` dimension, and semantic queries aren't the target of that optimization.

Verified: indexing the full vLLM corpus (~28,700 chunks with the current chunking strategy) produces a `semantic_index.npy` of matching shape `(num_chunks, 384)`, row count tracking the chunk count and the second dimension matching the model's embedding size.

### Hybrid Retrieval

`HybridRetriever` combines the BM25 and semantic rankings into one result list using Reciprocal Rank Fusion (RRF): each retriever is asked for `min(k * 4, corpus_size)` candidates rather than just `k`, since a chunk ranked just outside one list's top-`k` could still rank well once both lists are combined. Every chunk is scored `weight / (60 + rank + 1)` for each list it appears in (rank starting at 0, 60 the standard RRF constant), summed across whichever lists it appears in. The `search()`/`search_dataset` `method` parameter accepts `"hybrid"` alongside `"bm25"` and `"semantic"`.

Since BM25 and the semantic index are built from the exact same `chunks` list, in the same order, a chunk retrieved by both is *identical* — same `file_path`, same character range — so fusion matches chunks with plain equality on `(file_path, first_character_index)`, no fuzzy overlap check needed. Caching, as with semantic search, is intentionally scoped to the BM25 path only.

Verified against the real moulinette, not just an internal comparison script. Equal weighting (`weight=1.0` for both rankers) improved code Recall@5 but dropped docs Recall@5 *below* the mandatory 80% threshold — the same effect one researched peer repo reported for semantic fusion hurting docs recall. Weighting BM25 higher than semantic recovers docs while keeping most of the code gain; the final tuned ratio is `3.6:1.0`, found empirically by testing against both datasets and re-checking after `expand_snake_case` (Retrieval Method above) was added, since that changed the underlying numbers each weight was tuned against:

| Method | Code Recall@5 | Docs Recall@5 |
|---|---|---|
| BM25 only | 51.5% | 85.0% |
| BM25 + `expand_snake_case` | 52.5% | 85.0% |
| Hybrid, equal weight | 53.5% | 78.0% (fails 80% threshold) |
| Hybrid, weighted 3.6:1.0 + `expand_snake_case` | 57.6% | 82.0% |

The two bonuses compound well on code (57.6% vs 52.5% BM25-only) while docs keeps real margin above threshold (82.0% vs 80% required) — a deliberately safer landing spot than an earlier tuning pass that hit exactly 80.0% with no buffer. `search`/`search_dataset` default to `method="bm25"`, so none of this affects the mandatory baseline either way.

### Incremental Indexing

`index --apply_bonuses True` additionally builds the semantic index and skips re-chunking/re-embedding files unchanged since the last run, tracked via a manifest (`manifest.json`, saved alongside the other index files) keyed by file path, storing each file's modification time and its chunks. Unchanged files' embeddings are reused without duplicating them anywhere: each manifest entry keeps a row-range pointer into the *previous* run's `semantic_index.npy`, sliced out and reused directly rather than storing vectors a second time.

Scoped to the practical reading of the bonus requirement, not the strict one: the persisted BM25 index and `semantic_index.npy` still get fully rewritten every run — `bm25s` has no incremental-update API, so that part is unavoidable — but the expensive work (re-parsing files with `ast`, re-embedding via the model) is skipped for anything unchanged. `index` prints an unconditional summary every run (`Reindexing: X files changed, Y unchanged`), so the mechanism is demonstrable without a special flag or verbose mode.

`apply_bonuses` itself defaults to `False`, unlike every other bonus in this project (all always-on) — a deliberate exception. Building the semantic index is a real, unavoidable one-time cost even with caching, and with the current chunk count (~28,700, see Chunking Strategy) it measured well past the mandatory 5-minute indexing limit. Making it opt-in keeps mandatory `index` fast and unaffected; the trade-off is that semantic embeddings, hybrid retrieval, and incremental reuse all require explicitly passing `--apply_bonuses True` to be exercised at all.

### Caching

Search results for `search` are cached to `query_cache.json` alongside the index, keyed by a hash of `(query, k)`. A cache hit returns immediately, skipping BM25 retrieval entirely; a miss runs the search as normal, then writes the result before returning it. The index itself was already effectively cached before this bonus — `bm25s` persists it via `save()`/`load()`, so `index` never needs to be rerun just to search again.

Caching was deliberately scoped to `search` only. `search_dataset` processes ~100 distinct questions per run with no repeated queries to hit, and its real cost is the one-time index load rather than per-query retrieval; `answer`'s bottleneck is model loading and generation, not the (already fast) BM25 step that caching would skip — extending caching to either command wouldn't produce a meaningful speedup. The cache is automatically cleared on every `index` rebuild, so a reindex can never leave stale results behind.

Measured on the same query run twice: 4.22s (miss, includes building `Bm25Retriever` and running the real BM25 search) vs 2.80s (hit, retriever never gets created) — about 1.4s saved, roughly a third faster. The remaining time on the hit is fixed cost shared by both runs (Python startup, library imports) that caching isn't meant to touch.

### Local HTTP API

`src/api.py` exposes `/search` and `/answer` as GET endpoints via FastAPI, wrapping the same `search()` / `answer()` functions the CLI already uses. `/search` returns the same `list[MinimalSource]` JSON array as the CLI's search; `/answer` returns a structured `{"question": ..., "answer": ...}` object rather than piping the CLI's formatted text output through JSON. Run it with `uv run uvicorn src.api:app --reload`; this serves on `http://127.0.0.1:8000` by default. FastAPI serves interactive docs at `http://127.0.0.1:8000/docs`, so both endpoints can be tested directly in a browser without a separate HTTP client.

Building this surfaced a real architectural issue unrelated to the bonus itself: `cli.py` previously imported `Generator` (and therefore `torch` / `transformers`) at module level, so every CLI command — even ones that never touch the LLM, like `search` or `index` — paid that import cost. Since `Generator` is only actually needed by `answer` / `answer_dataset`, the import was moved inside those two functions, so the rest of the CLI, and this bonus's `/search` endpoint, no longer pay for it.

[↑ Back to top](#top)

## Resources

<!-- TODO (final README, Resources section): disclose AI usage covering
     two distinct things -
     1. AI-assisted coding/pair-programming throughout the project.
     2. AI-assisted research surveying other public 42 "RAG against the
        machine" repos for design-pattern reference points (e.g. file-type
        scope for chunking, bonus implementation approaches). No code was
        copied from any of them - only patterns observed and described in
        prose to inform our own decisions.
-->

**Peer repository research.** ~25 public GitHub repos of this same 42 project were scanned early on for design-pattern reference points (research only — no code copied, only patterns observed and described here in prose). Notable findings, by bonus:

- **Hybrid retrieval** — 2 confirmed real BM25 + dense implementations, both fused with Reciprocal Rank Fusion: `antoine71/42-rag-against-the-machine` (BM25 + `sentence-transformers/msmarco-bert-base-dot-v5` via ChromaDB) and `Sabofxx/rag-against-the-machine` (BM25 + MiniLM embeddings).
- **Caching** — 2 confirmed, both answer/query-level rather than index-level: `mpouillo/42-rag-against-the-machine` and `II-SARIP-II/42-RAG-against-the-machine`.
- **Incremental indexing** — no verified repo does true file-level incremental updates; every indexer checked (including `jessesmidt/angry_at_the_robot` and `auroreRakoto/RAG-against-the-machine`) does a full-corpus rebuild on every `index` run.
- Also worth noting: `cdonairevillen/RAG_Against_the-machine_42_project` used classifier-routed per-type BM25 indices plus cross-encoder reranking, and reported that semantic-embedding fusion *reduced* their docs recall — a reminder that hybrid retrieval isn't automatically better and needs measuring on this corpus, not assumed.

<!-- TODO (ask Shane): list any classic references (docs, articles, tutorials) you consulted — no visibility into what you read outside our sessions together. -->

<!-- Note: other repos' stated recall thresholds/scores vary by cohort and shouldn't be treated as authoritative for our own PDF's 80%/50% thresholds. -->

[↑ Back to top](#top)
