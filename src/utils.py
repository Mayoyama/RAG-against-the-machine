from json import dump
from sys import stderr, exit
from typing import Any
from .models import RagDataset, StudentSearchResults, MinimalSource
from pydantic import ValidationError
from typing import Callable, TypeVar
from tqdm import tqdm
import time
import threading


T = TypeVar("T")


def create_json(pathname: str, content: dict[str, Any]) -> None:
    """Write a dict to disk as pretty-printed JSON.

    Args:
        pathname: Path to write the JSON file to.
        content: JSON-serializable data to write.
    """

    try:
        with open(pathname, "w") as f:
            dump(content, f, indent=4)
    except (IOError, UnicodeDecodeError, TypeError) as e:
        print(f"create_json: Error writing data: {e}.",
              file=stderr)


def read_dataset(pathname: str) -> RagDataset:
    """Read and validate a RagDataset from a JSON file.

    Args:
        pathname: Path to the dataset JSON file.

    Returns:
        The parsed RagDataset.
    """

    try:
        with open(pathname, "r") as f:
            content = RagDataset.model_validate_json(f.read())
            return content
    except (IOError, UnicodeDecodeError, ValidationError) as e:
        print(f"read_dataset: Error reading data: {e}.",
              file=stderr)
        exit(1)


def read_search_results(pathname: str) -> StudentSearchResults:
    """Read and validate a StudentSearchResults from a JSON file.

    Args:
        pathname: Path to the search-results JSON file.

    Returns:
        The parsed StudentSearchResults.
    """

    try:
        with open(pathname, "r") as f:
            content = StudentSearchResults.model_validate_json(f.read())
            return content
    except (IOError, UnicodeDecodeError, ValidationError) as e:
        print(f"read_search_results: Error reading data: {e}.",
              file=stderr)
        exit(1)


def print_srcs(min_srcs: list[MinimalSource]) -> None:
    """Print each source's file path and character range, one per line.

    Args:
        min_srcs: Sources to print.
    """

    for src in min_srcs:
        print(f"{src.file_path} [{src.first_character_index}:"
              f"{src.last_character_index}]")


def output_progress_bar(task: Callable[[], T], description: str) -> T:

    returns: list[T] = []
    errors: list[BaseException] = []

    def run_task() -> None:
        try:
            returns.append(task())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_task)
    thread.start()

    with tqdm(desc=description, bar_format="{desc}: {elapsed}") as progbar:
        while thread.is_alive():
            progbar.update(1)
            time.sleep(0.2)

    thread.join()

    if errors:
        raise errors[0]

    return returns[0]
