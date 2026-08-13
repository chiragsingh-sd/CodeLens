import time

from backend.tools.search_tool import hybrid_search
from backend.tools.file_tool import read_file


def retrieve_context(
    repo_id: str,
    repo_path: str,
    retrieval_query: str,
    mode: str = "chat",
    target_file: str | None = None,
) -> list[dict]:
    """
    Retrieves relevant chunks from vector DB.
    """

    total_start = time.time()

    retrieval_start = time.time()

    chunks = hybrid_search(
        repo_id=repo_id,
        query=retrieval_query,
        k=4 if mode == "report" else 3,
    )

    print(
        f"[TIMING] hybrid_search: "
        f"{time.time() - retrieval_start:.2f}s"
    )

    if mode == "review" and target_file:

        file_read_start = time.time()

        file_content = read_file(
            repo_path,
            target_file
        )

        print(
            f"[TIMING] file_read: "
            f"{time.time() - file_read_start:.2f}s"
        )

        chunks.insert(0, {
            "text": file_content,
            "metadata": {
                "file_path": target_file,
                "symbol_name": target_file,
                "chunk_type": "file",
            },
            "score": 1.0,
        })

    print(f"[retriever] chunks={len(chunks)}")

    print(
        f"[TIMING] retrieve_context total: "
        f"{time.time() - total_start:.2f}s"
    )

    return chunks
