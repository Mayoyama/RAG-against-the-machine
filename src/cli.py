from .indexing import index_corpus
from .retrieval import Bm25Retriever
from .models import (MinimalSource, MinimalSearchResults, MinimalAnswer,
                     StudentSearchResults, StudentSearchResultsAndAnswer)
from .evaluation import evaluate_recallk
from .generation import Generator
from .utils import create_json, read_dataset, read_search_results
from pathlib import Path
from tqdm import tqdm


def index(max_chunk_size: int = 2000, root_directory: str = "data/raw",
          save_directory: str = "data/processed") -> None:
    """Build and persist a BM25 index over the codebase.

    Args:
        max_chunk_size: Maximum length of each chunk's content.
        root_directory: Directory to index.
        save_directory: Directory to save the resulting index under.
    """

    if not isinstance(max_chunk_size, int) or max_chunk_size < 1:
        raise ValueError("index: Invalid value for [max_chunk_size]")
    elif not Path(root_directory).is_dir():  # Read dir
        raise ValueError("index: Invalid value for [root_directory]")
    elif not isinstance(save_directory, str):  # Write dir
        raise ValueError("index: Invalid value for [save_directory]")

    index_corpus(root_directory, max_chunk_size, save_directory)


def search(query: str, k: int,
           save_directory: str = "data/processed") -> list[MinimalSource]:
    """Search the index for a single query and print the results.

    Args:
        query: The search query.
        k: Number of top results to return.
        save_directory: Directory the index was saved under.

    Returns:
        The top-k matching sources.
    """

    if not isinstance(query, str):
        raise ValueError("search: Invalid value for [query]")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("search: Invalid value for [k]")
    elif not Path(save_directory).is_dir():  # Read dir
        raise ValueError("search: Invalid value for [save_directory]")

    bm25_retriever = Bm25Retriever(save_directory)
    min_srcs = bm25_retriever.search(query, k)

    for src in min_srcs:
        print(f"{src.file_path} [{src.first_character_index}:"
              f"{src.last_character_index}]")

    return min_srcs


def search_dataset(dataset_path: str, k: int, save_directory: str,
                   index_dir: str = "data/processed") -> None:
    """Run search over every question in a dataset and save the results.

    Args:
        dataset_path: Path to the dataset JSON file.
        k: Number of top results to return per question.
        save_directory: Directory to write the output JSON file under.
        index_dir: Directory the index was saved under.
    """

    if not Path(dataset_path).is_file():  # Read file
        raise ValueError("search_dataset: Invalid value for [dataset_path]")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("search_dataset: Invalid value for [k]")
    elif not isinstance(save_directory, str):  # Write dir
        raise ValueError("search_dataset: Invalid value for [save_directory]")
    elif not Path(index_dir).is_dir():  # Read dir
        raise ValueError("search_dataset: Invalid value for [index_dir]")

    dataset = read_dataset(dataset_path)
    bm25_retriever = Bm25Retriever(index_dir)

    search_res: list[MinimalSearchResults] = []

    for data in tqdm(dataset.rag_questions):
        min_srcs = bm25_retriever.search(data.question, k)
        search_res.append(MinimalSearchResults(
            question_id=data.question_id,
            question=data.question,
            retrieved_sources=min_srcs
        ))

    student_res = StudentSearchResults(search_results=search_res, k=k)

    Path(save_directory).mkdir(parents=True, exist_ok=True)
    filename = Path(dataset_path).name
    pathname = str(Path(save_directory) / filename)

    create_json(pathname, student_res.model_dump())


def answer(query: str, k: int,
           save_directory: str = "data/processed") -> None:
    """Answer a single query using the retrieved context.

    Args:
        query: The question to answer.
        k: Number of sources to retrieve as context.
        save_directory: Directory the index was saved under.
    """
    if not isinstance(query, str):
        raise ValueError("answer: Invalid value for [query]")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("answer: Invalid value for [k]")
    elif not Path(save_directory).is_dir():  # Read dir
        raise ValueError("answer: Invalid value for [save_directory]")

    bm25_retriever = Bm25Retriever(save_directory)
    min_srcs = bm25_retriever.search(query, k)

    generator = Generator()
    answer_str = generator.generate_answer(query, min_srcs)

    print(f"QUERY: {query}\nANSWER: {answer_str}")


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:
    """Generate answers for every entry in a StudentSearchResults file.

    Args:
        student_search_results_path: Path to the StudentSearchResults
            JSON file.
        save_directory: Directory to write the output JSON file under.
    """

    if not Path(student_search_results_path).is_file():  # Read file
        raise ValueError("answer_dataset: Invalid value for "
                         "[student_search_results_path]")
    elif not isinstance(save_directory, str):  # Write dir
        raise ValueError("answer_dataset: Invalid value for [save_directory]")

    generator = Generator()
    res_n_ans: list[MinimalAnswer] = []

    search_res = read_search_results(student_search_results_path)

    for res in tqdm(search_res.search_results):
        ans = generator.generate_answer(res.question, res.retrieved_sources)
        min_ans = MinimalAnswer(
            question_id=res.question_id,
            question=res.question,
            retrieved_sources=res.retrieved_sources,
            answer=ans
        )
        res_n_ans.append(min_ans)

    student_res_n_ans = StudentSearchResultsAndAnswer(
            search_results=res_n_ans,
            k=search_res.k
        )

    Path(save_directory).mkdir(parents=True, exist_ok=True)
    filename = Path(student_search_results_path).name
    pathname = str(Path(save_directory) / filename)

    create_json(pathname, student_res_n_ans.model_dump())


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Report recall@k against a ground-truth dataset.

    Args:
        student_search_results_path: Path to the StudentSearchResults
            JSON file.
        dataset_path: Path to the ground-truth AnsweredQuestions dataset.
    """

    if not Path(student_search_results_path).is_file():  # Read file
        raise ValueError("evaluate: Invalid value for "
                         "[student_search_results_path]")
    elif not Path(dataset_path).is_file():  # Read file
        raise ValueError("evaluate: Invalid value for [dataset_path]")

    search_res = read_search_results(student_search_results_path)
    dataset = read_dataset(dataset_path)

    print(f"Recall@k score: {round(evaluate_recallk(search_res, dataset), 3)}")
