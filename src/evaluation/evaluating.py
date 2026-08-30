from src import (MinimalSource,
                 StudentSearchResults,
                 RagDataset,
                 AnsweredQuestion)
from sys import stderr


def calculate_iou(min_src_a: MinimalSource, min_src_b: MinimalSource) -> float:
    """Compute intersection-over-union between two character spans.

    Args:
        min_src_a: One source location.
        min_src_b: Another source location.

    Returns:
        IoU of the two spans as a float in [0, 1]. 0 if they don't
        overlap, or if either span has zero length.
    """

    if min_src_a.file_path != min_src_b.file_path:
        return 0.0

    iou_inter = (
        max(0, (min(min_src_a.last_character_index,
                    min_src_b.last_character_index)
                - max(min_src_a.first_character_index,
                      min_src_b.first_character_index)))
    )

    iou_union = (
        max(min_src_a.last_character_index, min_src_b.last_character_index)
        - min(min_src_a.first_character_index, min_src_b.first_character_index)
    )

    try:
        return iou_inter/iou_union
    except ZeroDivisionError:
        return 0.0


def source_found(reference_src: MinimalSource,
                 retrieved_srcs: list[MinimalSource],
                 iou_goal: float = 0.05) -> bool:
    """Check whether a ground-truth source counts as retrieved.

    Args:
        reference_src: The correct source location for a question.
        retrieved_srcs: Sources actually retrieved for that question.
        iou_goal: Minimum IoU to count as a match (PDF: 0.05).

    Returns:
        True if any retrieved source is in the same file and overlaps
        the reference span by at least iou_goal.
    """

    best: float = 0.0

    for src in retrieved_srcs:
        if src.file_path == reference_src.file_path:
            iou = calculate_iou(reference_src, src)
            best = iou if iou > best else best

    return best >= iou_goal


def calculate_q_recallk(reference_srcs: list[MinimalSource],
                        retrieved_srcs: list[MinimalSource],
                        iou_goal: float = 0.05) -> float:
    """Compute recall@k for a single question.

    Args:
        reference_srcs: All correct sources for the question.
        retrieved_srcs: The top-k sources retrieved for the question.
        iou_goal: Passed through to source_found.

    Returns:
        Fraction of reference_srcs found among retrieved_srcs.
    """
    success = 0

    for src in reference_srcs:
        success += 1 if source_found(src, retrieved_srcs, iou_goal) else 0

    try:
        return success / len(reference_srcs)
    except ZeroDivisionError:
        return 0.0


def evaluate_recallk(search_results: StudentSearchResults,
                     dataset: RagDataset,
                     iou_goal: float = 0.05) -> float:
    """Compute average recall@k across a dataset.

    Args:
        search_results: This solution's retrieved sources per question.
        dataset: Ground-truth dataset to score against.
        iou_goal: Passed through to calculate_q_recallk.

    Returns:
        Mean recall@k across all matched questions.
    """
    unanswered_qs = 0
    invalid_q_ids = 0
    valid_qs = 0
    total_iou: float = 0.0

    results_dict = {result.question_id: result
                    for result in search_results.search_results}

    for rag_q in dataset.rag_questions:
        if not isinstance(rag_q, AnsweredQuestion):
            unanswered_qs += 1
            continue

        res = results_dict.get(rag_q.question_id)
        if not res:
            print(f"Error: {rag_q} not in search results. Skipping.",
                  file=stderr)
            invalid_q_ids += 1
            continue
        else:
            valid_qs += 1
            total_iou += calculate_q_recallk(rag_q.sources,
                                             res.retrieved_sources,
                                             iou_goal)

    if unanswered_qs > 0:
        print(f"Warning: {unanswered_qs} unanswered questions detected and "
              "skipped during the evaluation process", file=stderr)

    if invalid_q_ids > 0:
        print(f"Warning: {invalid_q_ids} counts of mismatched question IDs "
              "detected and skipped during the evaluation process",
              file=stderr)

    try:
        return total_iou / valid_qs
    except ZeroDivisionError:
        return 0.0
