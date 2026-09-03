from pydantic import BaseModel, Field, model_validator
import uuid


class MinimalSource(BaseModel):
    """A single source location: file path and character span."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def check_valid_end(self) -> "MinimalSource":
        if self.first_character_index > self.last_character_index:
            raise ValueError("MinimalSource model validation error: Invalid "
                             "value [first > last]")

        return self


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
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(BaseModel):
    """A student's search results and answers for a dataset, at a given k."""

    search_results: list[MinimalAnswer]
    k: int = Field(ge=0)


class Chunk(BaseModel):
    """An indexed piece of content with its source location."""

    source: MinimalSource
    content: str
    index: int = Field(ge=0)


class ManifestItem(BaseModel):
    """One file's cached state for incremental indexing."""

    mod_time: float
    chunks: list[Chunk]
    embedding_row_start: int = Field(ge=0)
    embedding_row_end: int = Field(ge=0)

    @model_validator(mode="after")
    def check_valid_end(self) -> "ManifestItem":
        if self.embedding_row_start > self.embedding_row_end:
            raise ValueError("Manifest model validation error: Invalid "
                             "value [start > end]")

        return self


class Manifest(BaseModel):

    max_chunk_size: int | None = Field(gt=0, default=None)
    files: dict[str, ManifestItem] = {}
