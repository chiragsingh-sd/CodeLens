import time

from backend.core.llm import call_llm


SYSTEM_PROMPT = """
You are a codebase intelligence assistant.

Answer ONLY using the provided repository context.

Rules:
- Cite file paths and function names
- Be concrete and technical
- Never invent code
- Keep answers concise unless user asks otherwise
- If context is insufficient, say so
"""


MAX_CHUNKS = 3
MAX_CHARS_PER_CHUNK = 600


def format_context(chunks: list[dict]) -> str:

    sections = []

    for i, chunk in enumerate(chunks[:MAX_CHUNKS]):

        meta = chunk["metadata"]

        chunk_text = chunk["text"][:MAX_CHARS_PER_CHUNK]

        sections.append(
            f"""
[{i+1}]
FILE: {meta.get("file_path")}
SYMBOL: {meta.get("symbol_name")}

{chunk_text}
"""
        )

    return "\n".join(sections)


def generate_answer(
    user_query: str,
    retrieved_chunks: list[dict],
) -> str:

    total_start = time.time()

    format_start = time.time()

    context = format_context(retrieved_chunks)

    print(
        f"[TIMING] format_context: "
        f"{time.time() - format_start:.2f}s"
    )

    user_prompt = f"""
QUESTION:
{user_query}

REPOSITORY CONTEXT:
{context}
"""

    llm_start = time.time()

    answer = call_llm(
        SYSTEM_PROMPT,
        user_prompt,
        json_mode=False,
    )

    print(
        f"[TIMING] responder_llm: "
        f"{time.time() - llm_start:.2f}s"
    )

    print(
        f"[TIMING] generate_answer total: "
        f"{time.time() - total_start:.2f}s"
    )

    return answer
