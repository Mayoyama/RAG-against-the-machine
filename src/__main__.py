from .cli import (index, search, search_dataset,
                  answer, answer_dataset, evaluate)
import fire


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
        main()
    except ValueError as e:
        print(f'ValueError: {e}')
    except IOError as e:
        print(f"File error: {e}")
    except KeyError as e:
        print(f"KeyError error: {e}")
    except AssertionError as e:
        print(f"Internal system error: {e}")
    except Exception as e:
        print(f"Unexpected general error: {e}")
