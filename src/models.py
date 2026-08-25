from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """A single source location: file path and character span."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question with a unique ID, not yet answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """An UnansweredQuestion with its ground-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of answered and/or unanswered RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for a single question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A MinimalSearchResults with a generated answer attached."""

    answer: str


class StudentSearchResults(BaseModel):
    """A student's search results for a dataset, at a given k."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """A student's search results and answers for a dataset, at a given k."""

    search_results: list[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """An indexed piece of content with its source location."""

    source: MinimalSource
    content: str
    index: int
