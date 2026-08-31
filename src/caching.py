from json import load, dump, JSONDecodeError
from hashlib import sha256
from src import MinimalSource
from sys import stderr


def _cache_key(query: str, k: int) -> str:
    """Build a stable, JSON-key-safe cache key for a given query + k.

    Args:
        query: The search query.
        k: Number of results requested.

    Returns:
        A sha256 hex digest of the query and k, usable as a JSON
        dict key.
    """

    to_hash = query + str(k)
    encoded = to_hash.encode()
    hashed = sha256(encoded)

    return hashed.hexdigest()


def get_cached(cache_path: str, query: str,
               k: int) -> list[MinimalSource] | None:
    """Look up a previously cached result for (query, k).

    Args:
        cache_path: Path to the cache JSON file.
        query: The search query.
        k: Number of results requested.

    Returns:
        The cached list of MinimalSource, or None on a cache miss
        (including when the cache file doesn't exist yet).
    """

    try:
        with open(cache_path, "r") as f:
            cache_dict = load(f)
    except (IOError, JSONDecodeError):
        return None

    res = cache_dict.get(_cache_key(query, k))

    if res is not None:
        return [MinimalSource(**data) for data in res]

    return None


def set_cached(cache_path: str, query: str, k: int,
               results: list[MinimalSource]) -> None:
    """Store a search result under (query, k) for future reuse.

    Args:
        cache_path: Path to the cache JSON file.
        query: The search query.
        k: Number of results requested.
        results: The sources retrieved for (query, k).
    """

    try:
        with open(cache_path, "r") as f:
            cache_dict = load(f)
    except (IOError, JSONDecodeError):
        cache_dict = {}

    cache_dict[_cache_key(query, k)] = [src.model_dump() for src in results]

    try:
        with open(cache_path, "w") as f:
            dump(cache_dict, f)
    except (IOError, TypeError) as e:
        print(f"set_cached: Error writing cache: {e}.", file=stderr)
