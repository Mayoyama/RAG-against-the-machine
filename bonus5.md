# Bonus 5 — Incremental Indexing: Handoff Notes

Combines yesterday's school-session design notes with decisions settled
in a follow-up session. Feed this whole file to the next Claude session
as context — nothing here is committed code yet except `manifest.py`'s
skeleton (not filled in).

## Goal (PDF Chapter IX, bonus #3)

> "when a file changes, re-index only that file instead of rebuilding
> the whole index"

Complete this without building a large new subsystem, and without
silently overclaiming what the implementation actually does in the
README.

## Key constraint: `bm25s` can't do incremental updates

Confirmed via the library's own documentation: it's suited to
mostly-static corpora; document additions/deletions require partial or
full reindexing, and incremental updates are described as nontrivial.
This sets the scope for the whole feature — the BM25 index structure
itself gets fully rebuilt from the complete chunk list on every
`index()` run, no way around that without replacing `bm25s` entirely
(not worth it for one bonus point).

## Reading the PDF's wording: practical, not strict

Two possible readings:
1. **Strict** — the persisted index files themselves shouldn't be
   touched for unchanged files.
2. **Practical** — the expensive *work* (chunking, embedding)
   shouldn't be redone for unchanged files, even though the final
   on-disk structures get rebuilt from the (cheaply-assembled) full
   chunk list every run.

**Decision: build to the practical reading, and say so explicitly in
the README's bonus write-up** rather than let the bonus's name imply
more than what's built.

### Why the strict reading was rejected

Satisfying it properly would mean sharding the index (one index per
file or batch), rebuilding only the changed shard, and merging results
at query time. The blocker: BM25 scoring depends on IDF (inverse
document frequency), a corpus-wide statistic. A per-shard index only
sees its own shard when computing IDF, so scores from different shards
aren't directly comparable — merging them naively produces subtly
wrong rankings, not just a slower system. Correctly reconciling IDF
across shards is a real distributed-IR problem, not a shortcut.
Rejected for scope reasons, not because it's impossible.

## Settled design decisions

**Change detection: mtime, not content hash.** Measured against the
real corpus (2372 files, ~36MB) in a sandbox with unusually slow I/O:
mtime-only (`stat()`) took 6.6s, full read+sha256 hash took 35.4s —
roughly 5x slower, though both numbers are almost certainly inflated
by that sandbox's slow filesystem, not representative of real
hardware. mtime's only failure mode is a file's content changing
without its mtime updating (rare — some tooling preserves timestamps).
Given this is a school project and the failure mode is rare and low
stakes, mtime was chosen over hash for simplicity.

**The manifest is self-contained** — it stores each file's `mtime` and
its actual chunk data (text + character ranges), not just a pointer
into `bm25s`'s own saved corpus. The alternative (manifest holds only
`mtime`, chunks reused by loading and scanning the previous run's
`bm25s` corpus) was rejected: it requires loading the entire old
corpus every run, scanning it to figure out which chunks belong to
which file (the corpus is a flat list, not organized by file), and it
couples two independently-written persistence formats — if a previous
run were interrupted between writing the manifest and writing the BM25
index, they could disagree. Self-contained costs a bit of duplicated
disk space; that's a fine trade for the robustness.

**Embeddings are NOT duplicated into the manifest.** Storing raw float
vectors as JSON would be slow to read/write and bloat the file badly.
Instead: each file's manifest entry stores `embedding_row_start` /
`embedding_row_end` — two ints pointing at where that file's rows
lived in the *previous* run's `semantic_index.npy`. On a new
`index()` run: load the old `semantic_index.npy` once before anything
gets overwritten. Unchanged files slice their remembered row range out
of that old matrix and reuse those vectors directly. Changed files get
embedded fresh via `model.encode()`. Once the new matrix is assembled
(reused rows + fresh rows, in the same order as the final chunk list)
and saved — overwriting the old file — the manifest gets rewritten
with each file's row range updated to match *this* run's matrix.

**This doesn't change search results.** Embedding models are
deterministic at inference — encoding the same text through the same
model always produces the same vector, whether computed now or reused
from last time. Reuse and recompute are identical for genuinely
unchanged content. BM25 results are unaffected in either direction,
since its index is always fully rebuilt from the complete current
chunk list regardless of caching.

**`SemanticRetriever` needs zero changes.** The final
`semantic_index.npy` is still a complete matrix, one row per current
chunk, every time — the caching only changes *how* it gets assembled
internally, never what it contains once saved. Downstream retrieval
code doesn't know or care whether any given row was reused or fresh.

## What this does and doesn't do (for the README write-up)

Be explicit about this in the eventual README section, rather than let
the bonus's name imply more than what's built:

- **Does**: skip redundant chunking and embedding for files that
  haven't changed — the computationally expensive part.
- **Doesn't**: leave the persisted index structures themselves
  untouched for unchanged files. Both `semantic_index.npy` and the
  BM25 index get fully rewritten on every `index()` run regardless,
  because `bm25s` has no incremental-update API. The win is entirely
  in the work skipped to produce the inputs to that rebuild, not in
  the rebuild itself.

## Why cache the chunk text at all, not just a changed/unchanged flag

Worth spelling out for whoever implements this: the final BM25 index
and embedding matrix get rebuilt from the *complete* chunk list every
run, including unchanged files' chunks — there's no way to "leave a
file out" of the rebuild just because it's unchanged. So even knowing
a file is unchanged doesn't help unless you also have something to
contribute for it. If the manifest only stored `{file_path: mtime}`
and nothing else, you'd correctly know to skip re-chunking — but then
have no chunk data left for that file, and the only way to get it back
would be to run the chunker again, defeating the purpose. Caching the
actual chunk content (and, via the row-range trick, the actual
embeddings) is what makes "skip the chunker" mean something real.

## A sequencing note for whoever implements the embedding reuse

The old `semantic_index.npy` must be fully loaded into memory *before*
the new one gets written to the same path — read-then-later-write
within a single `index()` call, never interleaved. This project is a
single-user CLI workflow (`index` runs to completion, then `search`
runs separately) rather than a server handling concurrent requests, so
there's no scenario where another process reads the file mid-rewrite —
worth stating that assumption explicitly if it ever comes up, rather
than over-engineering for concurrent access this project doesn't have.

## Demonstrating it actually works

PDF's literal bonus grading criterion: "implemented and working, not
merely described in the README.md, and you may be asked to
demonstrate it." No stated performance threshold, but "demonstrate"
means the mechanism needs to be provably engaging correctly.

**Decision: an unconditional one-line summary printed at the end of
every `index` run** — no CLI flag required, e.g.:

```
Reindexing: 3 files changed, 2369 unchanged (reused)
```

Rejected gating proof behind an opt-in flag — a mechanism that only
proves itself when you remember an extra switch is fragile exactly
when it matters (same risk class as forgetting a manual setup step
before evaluation). Rejected a full file-by-file printed list as the
primary proof — unnecessary; the summary line alone is sufficient.

## Remaining implementation checklist

1. **`src/indexing/manifest.py`** (skeleton written, not filled in —
   see below) — `load_manifest(save_dir) -> dict` /
   `save_manifest(manifest, save_dir) -> None`. Mirrors `caching.py`'s
   `get_cached`/`set_cached` pattern: same try/except shapes, and
   handles `Chunk` serialization (`.model_dump()`) /
   reconstruction (`Chunk(**data)`) the same way `caching.py` does for
   `MinimalSource`.

2. **Small addition to `src/indexing/embedding.py`**: a way to load
   the previous run's `semantic_index.npy` before it gets overwritten,
   e.g. `load_semantic_index(save_dir) -> np.ndarray | None` (`None`
   on first run). `build_semantic_index` itself probably doesn't need
   to change — calling it with just the *changed* files' chunks
   already does what's needed for the fresh side.

3. **The actual per-file decision loop** — the biggest remaining
   piece. For each path from `build_pathlist`: check mtime against the
   manifest; unchanged → reuse cached chunks + sliced embedding rows;
   changed/new → run `chunk_file` fresh and mark for embedding.
   Assemble the final chunk list, final embedding matrix, and updated
   manifest all in lockstep (same order, chunk-for-chunk — see "why
   order matters" note below). Track changed/unchanged counts for the
   summary line. Files in the old manifest but missing from disk
   (deletions) just get dropped — never processed, never carried
   forward. This is substantial enough it might deserve its own file
   (e.g. `src/indexing/incremental.py`) rather than being folded into
   `indexing.py` — decide once you see how big it gets.

4. **Rewrite `index_corpus` in `src/indexing/indexing.py`** to call
   the new flow instead of today's two fully separate passes
   (`build_chunks` then `build_semantic_index(chunks)`), and print the
   summary line at the end.

5. **`cli.py` likely needs no changes** — the manifest lives in
   `save_directory` the same way `query_cache.json` does. A
   `--force_reindex` flag (bypass the manifest, full rebuild) was
   floated as optional; not committed either way.

### Why chunk/embedding order matters (context for whoever builds #3)

`semantic_index.npy` row `i` must correspond to the same chunk as
`chunks[i]` (the list that becomes the BM25 corpus), because
`SemanticRetriever` finds "row 7 has the highest similarity" then
looks up `corpus[7]` for that chunk's metadata. If the embedding
matrix's rows don't positionally match the chunk list — e.g. by
appending fresh embeddings at the end instead of interleaving them
back into position — row 7's vector could belong to one chunk while
`corpus[7]` describes a completely different one. The fix is
structural: build both lists in a single per-file loop, appending each
file's contribution (chunks + its embedding rows) to both outputs at
the same iteration, rather than "chunk everything, then separately
embed everything" like today's code does.

## Proposed skeleton: `src/indexing/manifest.py`

Already written to disk, not filled in:

```python
from json import load, dump, JSONDecodeError
from pathlib import Path
from sys import stderr
from src import Chunk


def load_manifest(save_dir: str) -> dict:
    """Load the incremental-indexing manifest from disk.

    Args:
        save_dir: Directory the manifest was saved under (same
            directory as the BM25/semantic indexes).

    Returns:
        A dict keyed by file path. Each value holds at least "mtime"
        (float) and "chunks" (list[Chunk], already reconstructed —
        not raw dicts). Empty dict if no manifest exists yet (e.g.
        first run, or after a fresh clean).
    """

    # TODO: same try/except (IOError, JSONDecodeError) pattern as
    # get_cached in caching.py — build the manifest's path the same
    # way save_semantic_index does (Path(save_dir) / "manifest.json"),
    # return {} on any read failure.
    #
    # Once loaded, each file entry's "chunks" list will be raw dicts
    # (that's all JSON can hold) — reconstruct them into real Chunk
    # objects before returning, same idea as get_cached doing
    # `MinimalSource(**data)` for each cached result.
    ...


def save_manifest(manifest: dict, save_dir: str) -> None:
    """Persist the incremental-indexing manifest to disk.

    Args:
        manifest: The full manifest to write — already rebuilt for
            this index() run, not merged with whatever was on disk
            before. Each file entry's "chunks" should be actual
            Chunk objects at this point, same as what load_manifest
            returns.
        save_dir: Directory to save the manifest under.
    """

    # TODO: same try/except (IOError, TypeError) pattern as
    # set_cached in caching.py, printing to stderr on failure.
    #
    # Chunk objects aren't JSON-serializable directly — convert each
    # one via .model_dump() before writing, same idea as set_cached
    # doing `src.model_dump()` for each MinimalSource. Unlike
    # set_cached, there's no read-and-merge step here: index_corpus
    # already builds the complete manifest before calling this, so
    # it's a straight overwrite, not a merge into existing data.
    ...
```

## Still open / not yet started

- Not yet measured: how much of total indexing time embedding actually
  accounts for on the real corpus, at full scale. Worth timing
  `make index` (and ideally each stage within it) before/after
  building this, both to know whether the real-world payoff is
  meaningful and to write the README's performance claim honestly.
- Not yet started: the actual per-file loop (#3 above) and the
  `index_corpus` rewrite (#4) — this document only covers design and
  a persistence-layer skeleton, not the orchestration logic itself.
- Whether `manifest.py` deserves splitting further (e.g. a separate
  file for the per-file decision loop) — flagged above, not decided.
