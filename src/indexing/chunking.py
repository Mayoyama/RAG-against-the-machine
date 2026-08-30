from sys import stderr
from src import Chunk, MinimalSource
import ast


def get_utf8_content(pathname: str) -> str:
    """Read a file's content as UTF-8 text.

    Args:
        pathname: Path to the file to read.

    Returns:
        The file's content, or an empty string if it could not be read.
    """

    try:
        with open(pathname, 'r', encoding="utf-8") as f:
            content = f.read()
            return content
    except (IOError, UnicodeDecodeError) as e:
        print(f"get_utf8_content: {e}. Skipping file...", file=stderr)
        return ""


def parse_py_content(content: str) -> list[tuple[int, int]]:
    """Find top-level function and class definitions in Python source.

    Args:
        content: The full text of a Python source file.

    Returns:
        A list of (start_line, end_line) pairs, one per top-level
        function or class definition. Empty if content has invalid syntax.
    """

    try:
        parsed = ast.parse(content)

        func_class_list: list[tuple[int, int]] = []

        for item in parsed.body:
            if isinstance(item, ast.FunctionDef | ast.ClassDef):
                if item.end_lineno:
                    func_class_list.append((item.lineno, item.end_lineno))

        return func_class_list
    except SyntaxError as e:
        print(f"parse_py_content: {e}", file=stderr)
        return []


def get_line_char_counts(content: str) -> list[int]:
    """Compute the character offset where each line of content starts.

    Args:
        content: The full text to compute line offsets for.

    Returns:
        A list where index i is the character offset of line i+1.
        Includes a trailing sentinel so the last line is always
        safely indexable, even without a final newline.
    """

    line_start_char_nums: list[int] = []
    char_count = 0

    for line in content.split('\n'):
        line_start_char_nums.append(char_count)
        char_count += len(line) + 1

    #  Guards against files without \n at the end
    line_start_char_nums.append(char_count)

    return line_start_char_nums


def get_start_end_char_nums(
        func_class_list: list[tuple[int, int]], line_char_counts: list[int]
        ) -> list[tuple[int, int]]:
    """Convert line-number ranges to character-index ranges.

    Args:
        func_class_list: (start_line, end_line) pairs from parsed content.
        line_char_counts: Line-to-character-offset lookup from
            get_line_char_counts.

    Returns:
        A list of (first_character_index, last_character_index) pairs.
    """

    char_indexes: list[tuple[int, int]] = []

    for start, end in func_class_list:
        start_num = line_char_counts[start - 1]
        end_num = line_char_counts[end]
        char_indexes.append((start_num, end_num))

    return char_indexes


def get_chunk(content: str, char_indices: tuple[int, int]) -> str:
    """Slice a chunk's text out of the full file content.

    Args:
        content: The full text of the file.
        char_indices: (first_character_index, last_character_index)
            to slice.

    Returns:
        The text between the given character indices.
    """

    if char_indices[0] < 0 or char_indices[1] < 0:
        raise ValueError("get_chunk: Invalid value for [char_indices]")

    return content[char_indices[0]:char_indices[1]]


def split_str(content: str, chunk_size: int) -> list[tuple[str, int, int]]:
    """Split oversized content into pieces no larger than chunk_size.

    Args:
        content: The text to split.
        chunk_size: Maximum length of each returned piece.

    Returns:
        A list of (piece, start, end) tuples. start/end are local
        character offsets relative to the start of the given content,
        not absolute file positions.
    """

    if not len(content):
        return []

    new_chunks: list[tuple[str, int, int]] = []
    char_count = 0

    while content:
        new_chunk = content[:chunk_size]
        start = char_count
        char_count += len(new_chunk)
        end = char_count
        content = content[chunk_size:]
        new_chunks.append((new_chunk, start, end))

    return new_chunks


def calc_abs_loc(cut_chunks: list[tuple[str, int, int]], offset: int
                 ) -> list[tuple[str, int, int]]:
    """Shift local character offsets into absolute file positions.

    Args:
        cut_chunks: (piece, start, end) tuples with offsets local to
            wherever the piece was cut from (e.g. split_str's output).
        offset: The absolute character index the local offsets are
            relative to (the original oversized chunk's own start).

    Returns:
        The same (piece, start, end) tuples with start/end shifted to
        real file-wide character positions.
    """

    abs_locs: list[tuple[str, int, int]] = []

    for text, start, end in cut_chunks:
        abs_locs.append((text, start + offset, end + offset))

    return abs_locs


def parse_md_content(content: str) -> list[tuple[int, int]]:
    """Find section boundaries in Markdown content, split on headers.

    Args:
        content: The full text of a Markdown file.

    Returns:
        A list of (start_line, end_line) pairs, one per section. Header
        lines inside fenced code blocks are ignored. Content before the
        first header forms its own leading section.
    """

    sections: list[tuple[int, int]] = []
    in_code_block = False
    section_start: int = 1

    lines = content.split('\n')
    for line_number, line in enumerate(lines, start=1):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        if not in_code_block and line.strip().startswith('#'):
            if (line.strip().lstrip('#').startswith(' ')
                    and (line_number - 1 >= section_start)):
                sections.append((section_start, line_number - 1))
                section_start = line_number

    sections.append((section_start, len(lines)))

    return sections


def chunk_file(pathname: str, chunk_size_limit: int) -> list[Chunk]:
    """Chunk a Python, Markdown, text, or JSON file into size-bounded Chunks.

    Args:
        pathname: Path to the file to chunk. Must end in ".py", ".md",
            ".txt", or ".json".
        chunk_size_limit: Maximum length of each chunk's content.

    Returns:
        A list of Chunks covering the file's content, in order. Empty
        if the file has an unsupported extension.
    """

    if chunk_size_limit < 1:
        raise ValueError("chunk_file: chunk_size_limit must be >= 1")

    if pathname.endswith(".py"):
        content = get_utf8_content(pathname)
        parsed = parse_py_content(content)

    elif pathname.endswith((".md", ".txt", ".json")):
        content = get_utf8_content(pathname)
        parsed = parse_md_content(content)

    else:
        print(f"chunk_file: Invalid file path: {pathname}. Skipping file.",
              file=stderr)
        return []

    line_pos_list = get_line_char_counts(content)

    chunk_pos_list = get_start_end_char_nums(parsed, line_pos_list)

    to_chunk: list[tuple[str, int, int]] = []

    for pos in chunk_pos_list:
        chunk = get_chunk(content, pos)
        if len(chunk) > chunk_size_limit:
            to_chunk.extend(
                calc_abs_loc(split_str(chunk, chunk_size_limit), pos[0]))
        else:
            to_chunk.append((chunk, *pos))

    chunk_lst: list[Chunk] = []
    for i, (text, start, end) in enumerate(to_chunk):
        min_src = MinimalSource(
            file_path=pathname,
            first_character_index=start,
            last_character_index=end
        )
        chunk_lst.append(Chunk(
            source=min_src,
            content=text,
            index=i
        ))

    return chunk_lst
