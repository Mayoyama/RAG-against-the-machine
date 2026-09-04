from .cli import (index, search, search_dataset,
                  answer, answer_dataset, evaluate)
from typing import Any
import signal
import fire
import sys


def handle_termination(signum: int, frame: Any) -> Any:
    """Signal handler for Ctrl+C/Ctrl+Z: exits immediately with a
    clean message, rather than a raw traceback or a multi-minute
    wait on a running background thread.

    Registered for SIGINT and SIGTSTP before main() runs, so both
    keys terminate the same way. Replaces Python's default handling
    for both signals - once registered, KeyboardInterrupt is never
    raised for SIGINT again.

    Args:
        signum: The signal number that triggered this handler.
        frame: The interrupted stack frame, unused - see del below.
    """

    del frame
    sig_name = "Ctrl+C" if signum == signal.SIGINT else "Ctrl+Z"
    print(f"{sig_name} detected. Terminating program.", file=sys.stderr)
    sys.exit(0)


def main() -> None:
    """Entry point for `uv run python -m src <command> [options]`."""

    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate
    })


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, handle_termination)
        signal.signal(signal.SIGTSTP, handle_termination)

        main()
    except ValueError as e:
        print(f'ValueError: {e}')
    except IOError as e:
        print(f"File error: {e}")
    except KeyError as e:
        print(f"KeyError error: {e}")
    except KeyboardInterrupt:
        print("Ctrl + C detected. Terminating program")
    except AssertionError as e:
        print(f"Internal system error: {e}")
    except Exception as e:
        print(f"Unexpected general error: {e}")
