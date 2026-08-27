import time

from backend.core.llm import call_llm


SYSTEM_PROMPT = """
You are a senior software architect.

Analyze the repository context and generate a structured technical report.

Rules:
- Be concrete
- Mention actual files and symbols
- Infer architecture carefully
- Never invent nonexistent systems
- If uncertain, explicitly say so
"""


def format_context(chunks: list[dict]) -> str:

    sections = []

    for i, chunk in enumerate(chunks[:25]):

        meta = chunk["metadata"]

        sections.append(
            f"""
[{i+1}]
FILE: {meta.get("file_path")}
SYMBOL: {meta.get("symbol_name")}
TYPE: {meta.get("chunk_type")}

{chunk["text"][:1500]}
"""
        )

    return "\n".join(sections)


def generate_report(
    repo_name: str,
    retrieved_chunks: list[dict],
) -> str:

    prompt_start = time.perf_counter()
    context = format_context(retrieved_chunks)

    user_prompt = f"""
Generate a repository architecture report for:

REPOSITORY:
{repo_name}

CONTEXT:
{context}

Report Structure:

1. Repository Purpose
2. Core Architecture
2.1. Architecture Diagram
3. Main Components
4. Retrieval / AI Systems
5. Data Flow
5.1.Data flow diagram
6. Key Technologies
7. Important Workflows
7.1. Workflow diagrams
8. Potential Weaknesses
9. Scalability Observations
10. Overall Technical Assessment
"""

    print(f"[TIMING] report prompt: {time.perf_counter() - prompt_start:.2f}s")

    return call_llm(
        SYSTEM_PROMPT,
        user_prompt,
        json_mode=False,
    )
