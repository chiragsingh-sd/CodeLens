import time

from backend.core.llm import call_llm


SYSTEM_PROMPT = """You are analyzing a code repository using ONLY the evidence provided below.
You have NO other knowledge about this specific repository beyond what appears in the evidence.

STRICT RULES:
1. Every factual claim about this repository must be tagged with its status:
   - CONFIRMED: directly shown in the evidence below.
   - INFERRED: not directly shown, but reasonably derived from MULTIPLE pieces of the evidence below
     (never from general knowledge about how similar apps are usually built).
   - UNKNOWN: the evidence does not establish this either way.

2. NEVER infer a technology, library, or behavior exists just because it is common in similar
   applications. Example: seeing SQLAlchemy does NOT mean PostgreSQL is used — SQLAlchemy supports
   many databases. If the specific database driver/connection string is not in the evidence, the
   database backend is UNKNOWN.

3. NEVER state something is absent ("no authentication", "no caching") unless the evidence explicitly
   shows the relevant code path in full and it is missing. If you only saw a fragment (e.g. a schema
   definition, not the full endpoint function), say the information "could not be verified from the
   retrieved context" instead of asserting absence.

4. BAD: "Redis is probably used for session caching."
   GOOD: "No session caching implementation was observed in the retrieved context."
   BAD: "The application uses PostgreSQL."
   GOOD: "SQLAlchemy is observed; the specific database backend cannot be determined from the retrieved
         context."

5. For every claim, cite the file path (and symbol/line if available) it came from. If you cannot cite
   a specific file for a claim, do not make the claim — mark it UNKNOWN instead.

Evidence below is everything you know about this repository. Nothing outside it is true unless stated.
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
