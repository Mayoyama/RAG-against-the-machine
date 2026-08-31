from transformers import AutoTokenizer, AutoModelForCausalLM
from src import MinimalSource
from src.indexing.chunking import get_utf8_content, get_chunk


class Generator:
    """Generates grounded answers from retrieved sources using a local LLM."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 max_new_tokens: int = 1500) -> None:
        """Load the tokenizer and model once, for reuse across many calls.

        Args:
            model_name: Hugging Face model name to load.
            max_new_tokens: Maximum number of tokens to generate per answer.
        """

        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Generator: Invalid value for [model_name]")
        elif not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("Generator: Invalid value for [max_new_tokens]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _get_source_content(source: MinimalSource) -> str:
        """Re-read the actual text a retrieved source points to.

        Args:
            source: A retrieved source location (file + character range).

        Returns:
            The text at that location in the file.
        """

        content = get_utf8_content(source.file_path)

        if not content:
            return "[Source unavailable]"

        return get_chunk(content, (
            source.first_character_index, source.last_character_index))

    def _build_messages(self, question: str,
                        sources: list[MinimalSource]) -> list[dict[str, str]]:
        """Build the chat-format prompt grounding the question in retrieved
        content.

        Args:
            question: The user's question.
            sources: Retrieved source locations to ground the answer in.

        Returns:
            Role-tagged message dicts, ready for tokenizer.apply_chat_template.
        """

        src_list: list[str] = []

        for src in sources:
            src_list.append(self._get_source_content(src))

        src_str = "---\n\n".join(f"Source {i + 1}:\n\n{src}\n\n"
                                 for i, src in enumerate(src_list))

        instructions = (
            "TASK: Using ONLY the context provided in numbered order, answer "
            "the question provided at the end of the user prompt. If the "
            "answer is not found in the context, respond 'Answer not found'."
            "\nThe user prompt will be several numbered sources under "
            "'CONTEXT:', followed by the question under 'QUESTION:'."
            "\nEnsure that the response is relevant to the question."
        )

        context = f"CONTEXT: {src_str}\n\n\n\nQUESTION: {question}"
        messages_for_LLM = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": context}
        ]

        return messages_for_LLM

    def generate_answer(self, question: str,
                        sources: list[MinimalSource]) -> str:
        """Generate a grounded answer for a question from its retrieved
        sources.

        Args:
            question: The user's question.
            sources: Retrieved source locations to ground the answer in.

        Returns:
            The model's generated answer text.
        """

        if not sources:
            return "No sources were retrieved for this question"

        messages = self._build_messages(question, sources)

        tokens = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True)

        output = self.model.generate(
            tokens["input_ids"], max_new_tokens=self.max_new_tokens)

        minus_prompt = output[0][tokens["input_ids"].shape[-1]:]
        res = self.tokenizer.decode(minus_prompt, skip_special_tokens=True)

        return str(res)
