"""
Retrieval variant implementations for CodeLens benchmark evaluation.

Variant A: Dense Only (ChromaDB vector search using all-MiniLM-L6-v2)
Variant B: Hybrid Search (BM25Okapi + ChromaDB vector search via RRF)
"""

from typing import Any
from backend.core.vector_store import get_collection
from backend.tools.search_tool import _get_embed_model, hybrid_search


def normalize_chunk(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    """
    Normalizes a retrieved chunk into a uniform dictionary representation
    for comparison against ground truth and failure analysis.
    """
    metadata = chunk.get("metadata", {})
    raw_file = metadata.get("file_path", "")
    norm_file = raw_file.replace("\\", "/").strip() if raw_file else ""

    text = chunk.get("text", "")
    snippet = text[:300].strip() if text else ""

    return {
        "rank": rank,
        "file_path": norm_file,
        "symbol_name": metadata.get("symbol_name", "").strip(),
        "chunk_type": metadata.get("chunk_type", "").strip(),
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("end_line"),
        "score": chunk.get("score"),
        "text_snippet": snippet,
    }


def run_dense_retrieval(repo_id: str, query: str, k: int = 3) -> list[dict[str, Any]]:
    """
    VARIANT A: Dense-Only Retrieval Pipeline.

    Flow:
    query -> embedding (all-MiniLM-L6-v2) -> ChromaDB vector search -> top K ranked results
    """
    collection = get_collection(repo_id)

    q_embedding = _get_embed_model().encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=q_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = semantic_results["documents"][0] if semantic_results.get("documents") else []
    metadatas = semantic_results["metadatas"][0] if semantic_results.get("metadatas") else []
    distances = semantic_results["distances"][0] if semantic_results.get("distances") else [None] * len(documents)

    results = []
    for rank_idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        score = round(1.0 - dist, 4) if dist is not None else None
        chunk_obj = {
            "text": doc,
            "metadata": meta,
            "score": score,
        }
        results.append(normalize_chunk(chunk_obj, rank=rank_idx))

    return results


def run_hybrid_retrieval(repo_id: str, query: str, k: int = 3) -> list[dict[str, Any]]:
    """
    VARIANT B: Production Hybrid Retrieval Pipeline.

    Flow:
    query -> dense vector search + query -> BM25Okapi search -> RRF fusion (K=60) -> top K
    """
    raw_results = hybrid_search(repo_id=repo_id, query=query, k=k)

    results = []
    for rank_idx, chunk_obj in enumerate(raw_results, start=1):
        results.append(normalize_chunk(chunk_obj, rank=rank_idx))

    return results
