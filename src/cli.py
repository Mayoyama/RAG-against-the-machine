from .indexing import index_corpus
from .retrieval import (Bm25Retriever, SemanticRetriever,
                        HybridRetriever, RetrievalBase)
from .models import (MinimalSource, MinimalSearchResults, MinimalAnswer,
                     StudentSearchResults, StudentSearchResultsAndAnswer)
from .evaluation import evaluate_recallk
from .utils import (create_json, read_dataset, read_search_results,
                    print_srcs, output_progress_bar)
from .caching import load_cached, save_cached
from pathlib import Path
from tqdm import tqdm


def index(max_chunk_size: int = 2000, root_directory: str = "data/raw",
          save_directory: str = "data/processed",
          apply_bonuses: bool = False) -> None:
    """Build and persist a BM25 index over the codebase.

    Args:
        max_chunk_size: Maximum length of each chunk's content.
        root_directory: Directory to index.
        save_directory: Directory to save the resulting index under.
        apply_bonuses: If True, also builds a semantic index and
            enables incremental reuse of unchanged files' chunks and
            embeddings across runs.
    """

    if not isinstance(max_chunk_size, int) or max_chunk_size < 1:
        raise ValueError("index: Invalid value for [max_chunk_size]")
    elif not Path(root_directory).is_dir():  # Read dir
        raise ValueError("index: Invalid value for [root_directory]"
                         " or directory does not exist")
    elif not isinstance(save_directory, str):  # Write dir
        raise ValueError("index: Invalid value for [save_directory]")

    Path(save_directory).mkdir(parents=True, exist_ok=True)

    index_corpus(root_directory, max_chunk_size, save_directory, apply_bonuses)

    cache_path = Path(save_directory) / "query_cache.json"
    cache_path.unlink(missing_ok=True)


def search(query: str, k: int, save_directory: str = "data/processed",
           method: str = "bm25") -> list[MinimalSource]:
    """Search the index for a single query and print the results.

    Args:
        query: The search query.
        k: Number of top results to return.
        save_directory: Directory the index was saved under.
        method: Retrieval method to use, "bm25", "semantic", or "hybrid".

    Returns:
        The top-k matching sources.
    """

    if not isinstance(query, str):
        raise ValueError("search: Invalid value for [query]")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("search: Invalid value for [k]")
    elif not isinstance(save_directory, str):
        raise ValueError("search: Invalid value for [save_directory]")
    elif method not in ("bm25", "semantic", "hybrid"):
        raise ValueError("search: Invalid value for [method]")

    try:
        if method == "bm25":
            cache_path = str(Path(save_directory) / "query_cache.json")
            cache = load_cached(cache_path, query, k)

            if cache is not None:
                print_srcs(cache)
                return cache

            bm25_retriever = Bm25Retriever(save_directory)
            min_srcs = bm25_retriever.search(query, k)

            save_cached(cache_path, query, k, min_srcs)

        elif method == "semantic":
            semantic_retriever = SemanticRetriever(save_directory)
            min_srcs = semantic_retriever.search(query, k)

        else:
            hybrid_retriever = HybridRetriever(save_directory)
            min_srcs = hybrid_retriever.search(query, k)

        print_srcs(min_srcs)

        return min_srcs

    except FileNotFoundError:
        print(f"search: no index found at {save_directory}. Run index() first")
        return []


def search_dataset(dataset_path: str, k: int, save_directory: str,
                   index_dir: str = "data/processed",
                   method: str = "bm25") -> None:
    """Run search over every question in a dataset and save the results.

    Args:
        dataset_path: Path to the dataset JSON file.
        k: Number of top results to return per question.
        save_directory: Directory to write the output JSON file under.
        index_dir: Directory the index was saved under.
        method: Retrieval method to use, "bm25", "semantic", or "hybrid".
    """

    if not Path(dataset_path).is_file():  # Read file
        raise ValueError("search_dataset: Invalid value for [dataset_path]"
                         " or file does not exist")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("search_dataset: Invalid value for [k]")
    elif not isinstance(save_directory, str):  # Write dir
        raise ValueError("search_dataset: Invalid value for [save_directory]")
    elif not isinstance(index_dir, str):
        raise ValueError("search_dataset: Invalid value for [index_dir]")

    dataset = read_dataset(dataset_path)

    try:
        if method == "bm25":
            retriever: RetrievalBase = Bm25Retriever(index_dir)
        elif method == "semantic":
            retriever = SemanticRetriever(index_dir)
        else:
            retriever = HybridRetriever(index_dir)
    except FileNotFoundError:
        raise ValueError("search_dataset: no index found at "
                         f"{index_dir}. Run index() first")

    search_res: list[MinimalSearchResults] = []

    for data in tqdm(dataset.rag_questions):
        min_srcs = retriever.search(data.question, k)
        search_res.append(MinimalSearchResults(
            question_id=data.question_id,
            question=data.question,
            retrieved_sources=min_srcs
        ))
    print(flush=True)
    
    student_res = StudentSearchResults(search_results=search_res, k=k)

    Path(save_directory).mkdir(parents=True, exist_ok=True)
    filename = Path(dataset_path).name
    pathname = str(Path(save_directory) / filename)

    create_json(pathname, student_res.model_dump())


def answer(query: str, k: int,
           save_directory: str = "data/processed") -> str:
    """Answer a single query using the retrieved context.

    Args:
        query: The question to answer.
        k: Number of sources to retrieve as context.
        save_directory: Directory the index was saved under.

    Returns:
        The model's generated answer text.
    """
    if not isinstance(query, str):
        raise ValueError("answer: Invalid value for [query]")
    elif not isinstance(k, int) or k < 0:
        raise ValueError("answer: Invalid value for [k]")
    elif not isinstance(save_directory, str):
        raise ValueError("answer: Invalid value for [save_directory]")

    try:
        from .generation import Generator

        bm25_retriever = Bm25Retriever(save_directory)
        min_srcs = bm25_retriever.search(query, k)
    except FileNotFoundError:
        print(f"answer: no index found at {save_directory}. Run index() first")
        return ""

    generator = Generator()

    def run_generator() -> str:
        return generator.generate_answer(query, min_srcs)

    answer_str = output_progress_bar(run_generator, "Generating answer")
    print(flush=True)

    res = f"QUERY: {query}\nANSWER: {answer_str}"

    print(res)
    return answer_str


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
                         "[student_search_results_path]"
                         " or file does not exist")
    elif not isinstance(save_directory, str):  # Write dir
        raise ValueError("answer_dataset: Invalid value for [save_directory]")

    from .generation import Generator

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
    print(flush=True)
        
    student_res_n_ans = StudentSearchResultsAndAnswer(
            search_results=res_n_ans,
            k=search_res.k
        )

    Path(save_directory).mkdir(parents=True, exist_ok=True)
    filename = Path(student_search_results_path).name
    pathname = str(Path(save_directory) / filename)

    create_json(pathname, student_res_n_ans.model_dump())


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Report recall@k against a dataset.

    Args:
        student_search_results_path: Path to the StudentSearchResults
            JSON file.
        dataset_path: Path to the ground-truth AnsweredQuestions dataset.
    """

    if not Path(student_search_results_path).is_file():  # Read file
        raise ValueError("evaluate: Invalid value for "
                         "[student_search_results_path]"
                         " or file does not exist")
    elif not Path(dataset_path).is_file():  # Read file
        raise ValueError("evaluate: Invalid value for [dataset_path]"
                         " or file does not exist")

    search_res = read_search_results(student_search_results_path)
    dataset = read_dataset(dataset_path)

    print(f"Recall@k score: {round(evaluate_recallk(search_res, dataset), 3)}")
