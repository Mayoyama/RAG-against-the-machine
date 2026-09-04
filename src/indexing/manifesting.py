"""Loading, saving, and diffing the incremental-indexing manifest."""

from pydantic import ValidationError
from pathlib import Path
from sys import stderr
from src import Manifest, ManifestItem, Chunk


def load_manifest(save_dir: str) -> Manifest:
    """Load the incremental-indexing manifest from disk.

    Args:
        save_dir: Directory the manifest was saved under (same
            directory as the BM25/semantic indexes).

    Returns:
        The saved Manifest, or an empty one (max_chunk_size=None,
        files={}) if none exists yet - first run, or after a clean.
    """

    try:
        pathname = Path(save_dir) / "manifest.json"
        with open(pathname, "r") as f:
            content = f.read()
            manifesto = Manifest.model_validate_json(content)
            return manifesto
    except (IOError, ValidationError) as e:
        print(f"load_manifest: Error loading manifest: {e}. Returning "
              "manifest with default values", file=stderr)
        return Manifest()


def save_manifest(manifest: Manifest, save_dir: str) -> None:
    """Persist the incremental-indexing manifest to disk.

    Args:
        manifest: The full Manifest to write - already rebuilt for
            this index() run, not merged with whatever was on disk
            before.
        save_dir: Directory to save the manifest under.
    """

    try:
        pathname = Path(save_dir) / "manifest.json"
        with open(pathname, "w") as f:
            content = manifest.model_dump_json(indent=4)
            f.write(content)
    except (IOError, TypeError) as e:
        print(f"save_manifest: Error writing manifest: {e}.", file=stderr)


def file_has_changed(path: str, manifest: Manifest) -> tuple[bool, float]:
    """Check whether a file's manifest entry is still valid.

    Args:
        path: File path to check.
        manifest: The loaded Manifest to check against.

    Returns:
        A (changed, mod_time) pair. changed is False if path has an
    entry whose stored mod_time matches the file's current
    mod_time, True if there's no entry at all or the mod_time no
    longer matches. mod_time is the current file modification time.
    """

    mod_time = Path(path).stat().st_mtime

    item = manifest.files.get(path)
    if item is None:
        return True, mod_time

    if mod_time != item.mod_time:
        return True, mod_time

    return False, item.mod_time


def build_manifest_entries(chunks: list[Chunk],
                           mod_times: dict[str, float]
                           ) -> dict[str, ManifestItem]:
    """Build fresh ManifestItem entries from a chunk list assembled
    for this index() run.

    Args:
        chunks: The full chunk list for this run, grouped by source
            file - chunks from the same file are never interleaved
            with another file's chunks elsewhere in the list.
        mod_times: Each file's current modification time, keyed by
            path.

    Returns:
        A dict keyed by file path, one ManifestItem per file, with
        row ranges matching this list's actual order.
    """

    new_files: dict[str, ManifestItem] = {}
    current_path: str | None = None
    block_start = 0

    for row, chunk in enumerate(chunks):
        path = chunk.source.file_path

        if path != current_path:
            if current_path is not None:
                new_files[current_path] = ManifestItem(
                    mod_time=mod_times[current_path],
                    chunks=chunks[block_start:row],
                    embedding_row_start=block_start,
                    embedding_row_end=row
                )
            block_start = row
            current_path = path

    if len(chunks) > block_start:
        assert current_path is not None, "Error creating Manifest item"
        new_files[current_path] = ManifestItem(
            mod_time=mod_times[current_path],
            chunks=chunks[block_start:],
            embedding_row_start=block_start,
            embedding_row_end=len(chunks)
        )

    return new_files
